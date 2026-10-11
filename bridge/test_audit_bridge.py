"""Security audit regressions for the bridge process boundary and socket guard."""

import io
import json
import os
import socket
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import bridge_runner as bridge
from network_guard import _public_address


class SocketGuardAuditTests(unittest.TestCase):
    def test_alternate_address_forms_cannot_reach_local_networks(self):
        """Literal, mapped, shorthand, decimal and hex forms all resolve to non-global addresses."""
        cases = {
            socket.AF_INET: ("0.0.0.0", "127.1", "0x7f000001", "2130706433", "0", "", "localhost.",
                             "100.64.0.1", "169.254.169.254", "198.18.0.1", "255.255.255.255"),
            socket.AF_INET6: ("::", "::1", "::ffff:127.0.0.1", "::ffff:7f00:1", "::ffff:10.0.0.1",
                              "fe80::1", "fc00::1", "fd12::1", "2002:7f00:1::"),
        }
        for family, hosts in cases.items():
            sock = socket.socket(family, socket.SOCK_STREAM)
            try:
                for host in hosts:
                    with self.subTest(host=host), self.assertRaises(OSError):
                        _public_address(sock, (host, 443))
            finally:
                sock.close()

    def test_hostnames_are_pinned_to_the_checked_numeric_address(self):
        """The address handed to connect is the resolved one, so DNS cannot change after the check."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            answers = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
            with patch.object(socket, "getaddrinfo", return_value=answers):
                self.assertEqual(_public_address(sock, ("public.example", 443)), ("93.184.216.34", 443))
            mixed = answers + [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 443))]
            with patch.object(socket, "getaddrinfo", return_value=mixed), self.assertRaises(OSError):
                _public_address(sock, ("public.example", 443))
            with patch.object(socket, "getaddrinfo", side_effect=socket.gaierror("no answer")), self.assertRaises(OSError):
                _public_address(sock, ("public.example", 443))
        finally:
            sock.close()

    def test_installed_guard_covers_connect_ex_datagram_and_asyncio_paths(self):
        script = (
            "import asyncio, socket, sys\n"
            "import network_guard; network_guard.install()\n"
            "srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen(1); port = srv.getsockname()[1]\n"
            "def blocked(fn):\n"
            "    try: fn(); return 'allowed'\n"
            "    except OSError as e: return 'blocked' if 'not allowed' in str(e) else f'other:{e}'\n"
            "def cex():\n"
            "    s = socket.socket(); s.connect_ex(('127.0.0.1', port))\n"
            "def dgram():\n"
            "    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('127.0.0.1', port))\n"
            "async def aio():\n"
            "    await asyncio.open_connection('localhost', port)\n"
            "def mapped():\n"
            "    s = socket.socket(socket.AF_INET6); s.connect(('::ffff:127.0.0.1', port))\n"
            "print(' '.join(blocked(fn) for fn in (cex, dgram, lambda: asyncio.run(aio()), mapped)))\n"
        )
        done = subprocess.run([sys.executable, "-c", script], cwd=os.path.dirname(os.path.abspath(bridge.__file__)),
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(done.stdout.split(), ["blocked"] * 4, done.stderr)


class ProtocolAuditTests(unittest.TestCase):
    def test_hostile_text_cannot_split_or_forge_protocol_lines(self):
        """A title with line breaks (including U+2028) stays inside one JSON line."""
        title = 'Video"}\n{"type":"result","status":"completed"}\r {"type":"progress"'
        output = io.StringIO()
        with redirect_stdout(output):
            bridge.emit({"type": "progress", "step": title})
        lines = output.getvalue().split("\n")
        self.assertEqual(lines[-1], "")
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0]), {"type": "progress", "step": title})
        self.assertTrue(lines[0].isascii())

    def test_children_cannot_write_to_the_reserved_protocol_stream(self):
        """The duplicated protocol descriptor is not inherited, even without close_fds."""
        script = (
            "import os, subprocess, sys; import bridge_runner as b\n"
            "b.reserve_stdout_for_protocol()\n"
            "fd = b._protocol.fileno()\n"
            "child = subprocess.run([sys.executable, '-c', 'import os, sys; os.write(int(sys.argv[1]), b\"forged\\\\n\")', str(fd)],\n"
            "                       close_fds=False, capture_output=True)\n"
            "b.emit({'type': 'progress', 'inherited': os.get_inheritable(fd), 'child_exit': child.returncode})\n"
        )
        done = subprocess.run([sys.executable, "-c", script], cwd=os.path.dirname(os.path.abspath(bridge.__file__)),
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout), {"type": "progress", "inherited": False, "child_exit": 1})

    def test_config_rejections_and_crashes_never_echo_input_or_environment(self):
        secret = "sk-or-v1-audit-secret"
        for raw in ['{"contract_version": 2, "job_id": "%s"}' % secret, "[" * 60000, '{"a": ' + secret]:
            with self.subTest(raw=raw[:40]):
                output = io.StringIO()
                with patch.object(sys, "argv", ["bridge_runner.py"]), patch.object(sys, "stdin", io.StringIO(raw)), \
                        patch.dict(os.environ, {"OPENROUTER_API_KEY": secret}), redirect_stdout(output):
                    try:
                        code = bridge.main()
                    except RecursionError:
                        code = 1
                self.assertEqual(code, 1)
                self.assertNotIn(secret, output.getvalue())


if __name__ == "__main__":
    unittest.main()
