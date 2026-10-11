"""Block local and private TCP destinations for the desktop clipping process.

The guard checks the address passed to connect, after DNS resolution. This
also catches redirects and DNS changes made after the UI's initial URL check.
It is installed only in BridgeClip's local worker process.
"""

import ipaddress
import socket
import sys

_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_installed = False


def _internal_socketpair(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0):
    """Build a self-connected pair without exempting ordinary loopback traffic.

    Windows implements socket.socketpair() by connecting to a temporary local
    listener. Use the original connect only for the client socket created here;
    all other connects continue through the destination guard.
    """
    if family not in (socket.AF_INET, socket.AF_INET6):
        raise ValueError("Only AF_INET and AF_INET6 socket pairs are supported")
    if type != socket.SOCK_STREAM or proto != 0:
        raise ValueError("Only stream socket pairs with protocol zero are supported")
    host = "127.0.0.1" if family == socket.AF_INET else "::1"
    listener = socket.socket(family, type, proto)
    try:
        listener.bind((host, 0))
        listener.listen()
        address = listener.getsockname()[:2]
        client = socket.socket(family, type, proto)
        try:
            client.setblocking(False)
            try:
                _original_connect(client, address)
            except (BlockingIOError, InterruptedError):
                pass
            client.setblocking(True)
            server, _ = listener.accept()
        except BaseException:
            client.close()
            raise
    finally:
        listener.close()
    try:
        if server.getsockname() != client.getpeername() or client.getsockname() != server.getpeername():
            raise ConnectionError("Unexpected socket pair peer")
    except BaseException:
        server.close()
        client.close()
        raise
    return server, client


# (address, port) pairs on this computer or its private network that the user
# chose in Settings: a local Whisper server or an OpenAI-compatible endpoint.
# Nothing else on a local or private network is reachable.
_allowed_local_endpoints: set[tuple[str, int]] = set()


def _unmapped(ip):
    mapped = getattr(ip, "ipv4_mapped", None)
    return mapped if mapped is not None else ip


def _is_allowable_local(ip) -> bool:
    """Loopback and private-network addresses only.

    Link-local (including the 169.254.169.254 cloud-metadata address),
    multicast, unspecified and reserved ranges are never allowed, whatever the
    setting says.
    """
    ip = _unmapped(ip)
    if ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved:
        return False
    return ip.is_loopback or ip.is_private


def allow_local_endpoint(host: str, port: int) -> bool:
    """Allow connections to ``host:port`` when it is a loopback or private address.

    Returns whether any address was allowed. A public host needs no exception.
    """
    try:
        port_num = int(port)
        if not host or not 0 < port_num < 65536:
            return False
        addresses = {info[4][0] for info in socket.getaddrinfo(host, port_num, socket.AF_UNSPEC, socket.SOCK_STREAM)}
    except (TypeError, ValueError, OSError):
        return False
    allowed = False
    for text in addresses:
        try:
            ip = ipaddress.ip_address(text.split("%")[0])
        except ValueError:
            continue
        if _is_allowable_local(ip):
            _allowed_local_endpoints.add((str(ip), port_num))
            _allowed_local_endpoints.add((str(_unmapped(ip)), port_num))
            allowed = True
    return allowed


def _endpoint_of(url: str):
    from urllib.parse import urlsplit

    try:
        parts = urlsplit((url or "").strip())
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return None, None
        return parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return None, None


def init_allowed_endpoints_from_env() -> None:
    """Allow only the local services the user configured for this run."""
    import os

    urls = []
    if os.environ.get("TRANSCRIPTION_SOURCE") == "local" and os.environ.get("LOCAL_WHISPER_MODE") == "server":
        urls.append(os.environ.get("LOCAL_WHISPER_URL", ""))
    if os.environ.get("AI_PROVIDER") == "openai-compatible":
        urls.append(os.environ.get("OPENAI_COMPATIBLE_URL", ""))
    for url in urls:
        host, port = _endpoint_of(url)
        if host and port:
            allow_local_endpoint(host, port)


def _is_allowed(ip_text: str, port) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text.split("%")[0])
    except ValueError:
        return False
    return (str(ip), port) in _allowed_local_endpoints or (str(_unmapped(ip)), port) in _allowed_local_endpoints


def _public_address(sock: socket.socket, address):
    if sock.family not in (socket.AF_INET, socket.AF_INET6):
        return address
    if not isinstance(address, tuple) or len(address) < 2:
        raise OSError("Network destination is invalid")
    host, port = address[:2]
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_global or _is_allowed(str(ip), port):
            return address
        raise OSError("Local network destinations are not allowed")
    except ValueError:
        pass

    results = socket.getaddrinfo(host, port, sock.family, sock.type, sock.proto)
    if not results:
        raise OSError("Network destination could not be resolved")
    # Every address the name resolves to must be public or an allowed endpoint,
    # so a name that also points at the local network cannot be used to reach it.
    for result in results:
        if not ipaddress.ip_address(result[4][0].split("%")[0]).is_global and not _is_allowed(result[4][0], port):
            raise OSError("Local network destinations are not allowed")
    return results[0][4]


def _guard_async_connect(original):
    async def guarded_sock_connect(loop, sock, address):
        return await original(loop, sock, _public_address(sock, address))

    return guarded_sock_connect


def install() -> None:
    global _installed
    init_allowed_endpoints_from_env()
    if _installed:
        return

    def guarded_connect(sock, address):
        return _original_connect(sock, _public_address(sock, address))

    def guarded_connect_ex(sock, address):
        return _original_connect_ex(sock, _public_address(sock, address))

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    # Python's Windows socketpair fallback uses socket.connect() to reach its
    # own ephemeral loopback listener. Keep that private IPC working without
    # allowing any arbitrary local destination through guarded_connect().
    fallback_socketpair = getattr(socket, "_fallback_socketpair", None)
    if fallback_socketpair is not None and socket.socketpair is fallback_socketpair:
        socket.socketpair = _internal_socketpair
    # Windows Proactor connects through ConnectEx instead of socket.connect(),
    # so guard the event loop's destination before it reaches that API.
    if sys.platform == "win32":
        from asyncio.proactor_events import BaseProactorEventLoop

        BaseProactorEventLoop.sock_connect = _guard_async_connect(
            BaseProactorEventLoop.sock_connect
        )
    _installed = True
