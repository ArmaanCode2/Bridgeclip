import asyncio
import socket
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from network_guard import _guard_async_connect, _public_address


class NetworkGuardTests(unittest.TestCase):
    def test_rejects_private_addresses_at_connect(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            for host in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "192.168.1.2"):
                with self.subTest(host=host), self.assertRaises(OSError):
                    _public_address(sock, (host, 443))
            self.assertEqual(_public_address(sock, ("8.8.8.8", 443)), ("8.8.8.8", 443))
        finally:
            sock.close()

    def test_rejects_private_dns_result_even_after_initial_url_validation(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            private = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 443))]
            with patch.object(socket, "getaddrinfo", return_value=private), self.assertRaises(OSError):
                _public_address(sock, ("public.example", 443))
            public = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]
            with patch.object(socket, "getaddrinfo", return_value=public):
                self.assertEqual(_public_address(sock, ("public.example", 443)), ("8.8.8.8", 443))
        finally:
            sock.close()

    def test_async_connect_checks_and_pins_destination(self):
        addresses = []

        async def original(_loop, _sock, address):
            addresses.append(address)
            return address

        guarded = _guard_async_connect(original)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with self.assertRaisesRegex(OSError, "Local network destinations"):
                asyncio.run(guarded(None, sock, ("127.0.0.1", 443)))
            self.assertEqual(addresses, [])

            public = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]
            with patch.object(socket, "getaddrinfo", return_value=public):
                result = asyncio.run(guarded(None, sock, ("public.example", 443)))
            self.assertEqual(result, ("8.8.8.8", 443))
            self.assertEqual(addresses, [("8.8.8.8", 443)])
        finally:
            sock.close()

    @unittest.skipUnless(sys.platform == "win32", "Windows Proactor test")
    def test_windows_proactor_rejects_loopback_sock_connect(self):
        # Install in a child process because the guard patches sockets and the
        # event loop for its lifetime.
        script = """
import asyncio
import socket
import network_guard

network_guard.install()
async def check():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.setblocking(False)
            try:
                await asyncio.get_running_loop().sock_connect(client, listener.getsockname())
            except OSError as error:
                assert 'Local network destinations are not allowed' in str(error), error
            else:
                raise AssertionError('Proactor connected to a local TCP destination')

asyncio.run(check())
"""
        done = subprocess.run(
            [sys.executable, "-c", script], cwd=Path(__file__).parent,
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_asyncio_socketpair_works_without_allowing_local_destinations(self):
        # Force the Windows TCP fallback on every platform. A subprocess keeps
        # the process-wide socket guard out of other tests.
        if not hasattr(socket, "_fallback_socketpair"):
            self.skipTest("Python does not expose its TCP socketpair fallback")
        script = """
import asyncio
import socket
import threading
import network_guard as guard

socket.socketpair = socket._fallback_socketpair
guard.install()
assert asyncio.run(asyncio.sleep(0, result=1)) == 1
assert asyncio.run(asyncio.sleep(0, result=2)) == 2

# Hold the private pair at its own connect and prove that an ordinary socket
# cannot reach even that temporary listener while construction is in progress.
original_connect = guard._original_connect
ready = threading.Event()
proceed = threading.Event()
state = {}
def delayed_connect(sock, address):
    state['address'] = address
    ready.set()
    assert proceed.wait(5)
    return original_connect(sock, address)
guard._original_connect = delayed_connect
def make_pair():
    try:
        state['pair'] = socket.socketpair()
    except BaseException as error:
        state['error'] = error
worker = threading.Thread(target=make_pair)
worker.start()
try:
    assert ready.wait(5)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.connect(state['address'])
        except OSError as error:
            assert 'Local network destinations are not allowed' in str(error)
        else:
            raise AssertionError('The guard allowed an arbitrary loopback connection')
finally:
    proceed.set()
    worker.join(5)
assert not worker.is_alive()
assert 'error' not in state, state.get('error')
for sock in state['pair']:
    sock.close()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.connect(('127.0.0.1', 443))
    except OSError as error:
        assert 'Local network destinations are not allowed' in str(error)
    else:
        raise AssertionError('The guard allowed an arbitrary loopback connection')
"""
        done = subprocess.run(
            [sys.executable, "-c", script], cwd=Path(__file__).parent,
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(done.returncode, 0, done.stderr)


    def configured(self, **env):
        """Run init_allowed_endpoints_from_env with ``env`` as the whole environment."""
        import network_guard
        network_guard._allowed_local_endpoints.clear()
        with patch.dict("os.environ", env, clear=True):
            network_guard.init_allowed_endpoints_from_env()
        return network_guard

    def tearDown(self):
        import network_guard
        network_guard._allowed_local_endpoints.clear()

    def reachable(self, host, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            _public_address(sock, (host, port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def test_nothing_local_is_allowed_by_default(self):
        self.configured()
        self.assertFalse(self.reachable("127.0.0.1", 8000))
        self.assertFalse(self.reachable("192.168.1.10", 8000))

    def test_configured_whisper_server_is_allowed_on_its_exact_port_only(self):
        self.configured(TRANSCRIPTION_SOURCE="local", LOCAL_WHISPER_MODE="server",
                        LOCAL_WHISPER_URL="http://127.0.0.1:8000/v1")
        self.assertTrue(self.reachable("127.0.0.1", 8000))
        self.assertFalse(self.reachable("127.0.0.1", 8080))
        self.assertFalse(self.reachable("127.0.0.1", 22))
        self.assertFalse(self.reachable("192.168.1.1", 8000))

    def test_builtin_whisper_opens_no_port(self):
        self.configured(TRANSCRIPTION_SOURCE="local", LOCAL_WHISPER_MODE="builtin",
                        LOCAL_WHISPER_URL="http://127.0.0.1:8000/v1")
        self.assertFalse(self.reachable("127.0.0.1", 8000))

    def test_cloud_transcription_ignores_a_stale_whisper_url(self):
        self.configured(TRANSCRIPTION_SOURCE="cloud", LOCAL_WHISPER_MODE="server",
                        LOCAL_WHISPER_URL="http://127.0.0.1:8000/v1")
        self.assertFalse(self.reachable("127.0.0.1", 8000))

    def test_openai_compatible_endpoint_is_allowed_only_for_that_provider(self):
        self.configured(AI_PROVIDER="openai-compatible", OPENAI_COMPATIBLE_URL="http://192.168.1.50:11434/v1")
        self.assertTrue(self.reachable("192.168.1.50", 11434))
        self.assertFalse(self.reachable("192.168.1.50", 22))
        self.configured(AI_PROVIDER="gemini", OPENAI_COMPATIBLE_URL="http://192.168.1.50:11434/v1")
        self.assertFalse(self.reachable("192.168.1.50", 11434))

    def test_default_ports_follow_the_scheme(self):
        self.configured(AI_PROVIDER="openai-compatible", OPENAI_COMPATIBLE_URL="https://10.0.0.5/v1")
        self.assertTrue(self.reachable("10.0.0.5", 443))
        self.assertFalse(self.reachable("10.0.0.5", 80))

    def test_cloud_metadata_and_link_local_are_never_allowed(self):
        for url in ("http://169.254.169.254/v1", "http://169.254.169.254:80/v1", "http://[fe80::1]:8000/v1",
                    "http://0.0.0.0:8000/v1", "http://224.0.0.1:8000/v1"):
            with self.subTest(url=url):
                network_guard = self.configured(AI_PROVIDER="openai-compatible", OPENAI_COMPATIBLE_URL=url)
                self.assertEqual(network_guard._allowed_local_endpoints, set())
        self.assertFalse(self.reachable("169.254.169.254", 80))

    def test_a_hostname_must_not_be_able_to_reach_extra_local_addresses(self):
        self.configured(AI_PROVIDER="openai-compatible", OPENAI_COMPATIBLE_URL="http://127.0.0.1:11434/v1")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            mixed = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 11434)),
                     (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.9.9.9", 11434))]
            with patch.object(socket, "getaddrinfo", return_value=mixed), self.assertRaises(OSError):
                _public_address(sock, ("rebind.example", 11434))
            allowed = mixed[:1]
            with patch.object(socket, "getaddrinfo", return_value=allowed):
                self.assertEqual(_public_address(sock, ("localhost", 11434)), ("127.0.0.1", 11434))
        finally:
            sock.close()

    def test_malformed_urls_allow_nothing(self):
        for url in ("", "not a url", "ftp://127.0.0.1:21/", "http://127.0.0.1:notaport/v1", "http:///v1"):
            with self.subTest(url=url):
                network_guard = self.configured(AI_PROVIDER="openai-compatible", OPENAI_COMPATIBLE_URL=url)
                self.assertEqual(network_guard._allowed_local_endpoints, set())

    def test_ipv4_mapped_loopback_matches_the_allowed_address(self):
        self.configured(AI_PROVIDER="openai-compatible", OPENAI_COMPATIBLE_URL="http://127.0.0.1:11434/v1")
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        try:
            self.assertEqual(_public_address(sock, ("::ffff:127.0.0.1", 11434, 0, 0))[0], "::ffff:127.0.0.1")
        finally:
            sock.close()


if __name__ == "__main__":
    unittest.main()

