"""Security audit regressions for engine infrastructure: media tools, child environment, network policy."""
import asyncio
import os
import shutil
import socket
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from clip_engine import network_policy
from clip_engine.services.media_process import MEDIA_INPUT_OPTIONS, media_child_env, run_media


def _answer(address, family=socket.AF_INET):
    return [(family, socket.SOCK_STREAM, 6, "", (address, 443) if family == socket.AF_INET else (address, 443, 0, 0))]


def test_media_children_drop_secrets_proxies_and_loader_hooks(monkeypatch):
    for name, value in {
        "OPENROUTER_API_KEY": "secret", "AWS_SECRET_ACCESS_KEY": "secret", "BRIDGECLIP_API_KEY": "secret",
        "HTTP_PROXY": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1", "ALL_PROXY": "socks5://127.0.0.1:1",
        "PYTHONPATH": "/tmp/site", "PYTHONSTARTUP": "/tmp/hook.py", "DYLD_INSERT_LIBRARIES": "/tmp/evil.dylib",
        "LD_PRELOAD": "/tmp/evil.so", "SSL_CERT_FILE": "/tmp/ca.pem", "REQUESTS_CA_BUNDLE": "/tmp/ca.pem",
        "BRIDGECLIP_WORK_ROOT": "/tmp/work", "LOCAL_OUTPUT_DIR": "/tmp/out",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = media_child_env()
    assert set(env) <= {k for k in env if k.upper() in {
        "PATH", "HOME", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "WINDIR", "USERPROFILE", "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH", "FONTCONFIG_PATH", "FONTCONFIG_FILE", "LANG", "LC_ALL", "LC_CTYPE", "TZ"}}
    assert "OPENROUTER_API_KEY" not in env and "HTTP_PROXY" not in env and "DYLD_INSERT_LIBRARIES" not in env
    result = run_media([sys.executable, "-c", "import os; print(sorted(k for k in os.environ if 'PROXY' in k or 'KEY' in k or 'PRELOAD' in k or 'INSERT' in k))"])
    assert result.stdout.strip() == b"[]"


@pytest.mark.parametrize("payload", [
    "#EXTM3U\n#EXT-X-VERSION:3\n#EXTINF:1.0,\nprivate.wav\n#EXT-X-ENDLIST\n",
    "ffconcat version 1.0\nfile 'private.wav'\n",
    "[playlist]\nFile1=private.wav\n",
    "http://127.0.0.1/stream.ts\n",
])
def test_source_media_options_reject_playlists_and_remote_references(tmp_path, payload):
    if not shutil.which("ffprobe"):
        pytest.skip("FFmpeg tools required")
    disguised = tmp_path / "source.mp4"
    disguised.write_text(payload)
    result = run_media(["ffprobe", "-v", "error", *MEDIA_INPUT_OPTIONS, "-show_entries", "format=format_name",
                        "-of", "csv=p=0", str(disguised)], timeout=10)
    assert result.returncode != 0
    assert not result.stdout.strip()


def test_public_destination_pins_ip_and_rejects_mapped_and_special_ranges():
    with patch.object(network_policy.socket, "getaddrinfo", return_value=_answer("93.184.216.34")):
        pinned = network_policy.resolve_public_destination("https://Video.Example.com/a/b?c=d#frag")
    assert pinned.url == "https://93.184.216.34/a/b?c=d"
    assert pinned.hostname == "video.example.com" and pinned.host_header == "video.example.com"
    for address, family in [("::ffff:127.0.0.1", socket.AF_INET6), ("::ffff:169.254.169.254", socket.AF_INET6),
                            ("fe80::1", socket.AF_INET6), ("fc00::1", socket.AF_INET6), ("::", socket.AF_INET6),
                            ("2002:7f00:1::", socket.AF_INET6), ("100.64.0.1", socket.AF_INET), ("0.0.0.0", socket.AF_INET),
                            ("198.18.0.1", socket.AF_INET), ("192.0.0.8", socket.AF_INET)]:
        with patch.object(network_policy.socket, "getaddrinfo", return_value=_answer(address, family)):
            assert not network_policy.public_source_url("https://video.example.com/movie.mp4"), address
    mixed = _answer("93.184.216.34") + _answer("10.0.0.5")
    with patch.object(network_policy.socket, "getaddrinfo", return_value=mixed):
        assert not network_policy.public_source_url("https://video.example.com/movie.mp4")
    for url in ["https://a.b.localhost/x", "https://nas.local/x", "https://db.internal/x", "https://[::ffff:7f00:1]/x",
                "https://2130706433/x", "https://0x7f000001/x", "https://127.1/x", "https://example.com:8443/x",
                "https://example.com/x\r\nHost: evil", "ftp://example.com/x", "https://" + "a" * 8200 + ".com/x"]:
        assert not network_policy.public_source_url(url), url


def test_download_guard_follows_work_into_threads_and_executors():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    outcomes = {}

    def attempt(label):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            outcomes[label] = "allowed"
        except OSError as exc:
            outcomes[label] = "blocked" if "not public" in str(exc) else f"other: {exc}"

    async def via_loop():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, attempt, "default-executor")
        await asyncio.to_thread(attempt, "to_thread")

    loop = asyncio.new_event_loop()
    try:
        with network_policy.guarded_public_connections():
            attempt("direct")
            thread = threading.Thread(target=attempt, args=("thread",))
            thread.start()
            thread.join()
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(attempt, "pool").result()
            loop.run_until_complete(via_loop())
            with pytest.raises(OSError, match="not public"):
                socket.socket(socket.AF_INET6).connect(("::ffff:127.0.0.1", port))
        attempt("outside-guard")
    finally:
        loop.close()
        server.close()
    assert outcomes == {"direct": "blocked", "thread": "blocked", "pool": "blocked", "default-executor": "blocked",
                        "to_thread": "blocked", "outside-guard": "allowed"}


def test_external_downloaders_are_refused_while_guarded():
    from yt_dlp.downloader.external import FFmpegFD
    network_policy._install_socket_guard()
    downloader = FFmpegFD.__new__(FFmpegFD)
    with network_policy.guarded_public_connections(), pytest.raises(OSError, match="External network downloaders"):
        FFmpegFD.real_download(downloader, "out.mp4", {"url": "http://127.0.0.1/stream.m3u8", "protocol": "m3u8"})


def test_media_tool_timeouts_and_kill_do_not_leave_children(tmp_path):
    pid_file = tmp_path / "pid"
    with pytest.raises(subprocess.SubprocessError):
        run_media([sys.executable, "-c",
                   "import os, pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                   "sys.stdout.write('x' * 4096); sys.stdout.flush(); time.sleep(30)", str(pid_file)],
                  timeout=1, max_output=1024)
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)
