"""TLS relay to the volume-session gateway (vol-bastion, ENG-6414).

Run by OpenSSH as a ProxyCommand: stdin/stdout <-> TLS to the gateway, with
the session's short hostname as the SNI the gateway routes on. TLS is only
an envelope for that hostname (SSH stays end to end), so the gateway's
self-signed certificate is authenticated by its pinned SHA-256 instead of a CA.
"""

import hashlib
import hmac
import os
import select
import socket
import ssl
import threading


class GatewayError(Exception):
    pass


def _connect(host: str, port: int, sni: str, cert_sha256: str) -> ssl.SSLSocket:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    try:
        sock = socket.create_connection((host, port), timeout=10)
        try:
            tls = context.wrap_socket(sock, server_hostname=sni)
        except BaseException:
            sock.close()
            raise
    except (OSError, ssl.SSLError) as exc:
        raise GatewayError(f"cannot connect to the gateway {host}:{port}: {exc}") from exc
    actual = hashlib.sha256(tls.getpeercert(binary_form=True) or b"").hexdigest()
    if not hmac.compare_digest(actual, cert_sha256.replace(":", "").lower()):
        tls.close()
        raise GatewayError(
            f"the gateway {host}:{port} presented an unexpected certificate "
            f"(SHA-256 {actual}); refusing to connect"
        )
    tls.settimeout(None)
    return tls


def _drain(tls: ssl.SSLSocket, stdout_fd: int) -> bool:
    """Write everything the socket has ready to stdout without blocking (a
    readable socket may hold only a TLS ticket, not data). True once the server
    closed the connection."""
    tls.setblocking(False)
    try:
        while chunk := tls.recv(64 * 1024):
            view = memoryview(chunk)
            while view:
                view = view[os.write(stdout_fd, view) :]
    except ssl.SSLWantReadError:
        return False
    finally:
        tls.setblocking(True)
    return True


def relay(gateway: str, cert_sha256: str, sni: str, stdin_fd: int, stdout_fd: int) -> None:
    """Pipe stdin_fd/stdout_fd through TLS to `gateway` (host:port) until
    either side closes. Raises GatewayError when it cannot connect or the
    certificate does not match."""
    host, _, port = gateway.rpartition(":")
    if not host or not port.isdigit():
        raise GatewayError(f"invalid gateway {gateway!r}; expected host:port")
    tls = _connect(host, int(port), sni, cert_sha256)

    # One thread owns the TLS socket (OpenSSL objects are not thread-safe).
    # A helper thread only moves stdin into a socketpair, so a single select()
    # covers both directions on every OS (Windows cannot select on pipes).
    inbound, outbound = socket.socketpair()

    def read_stdin() -> None:
        try:
            while chunk := os.read(stdin_fd, 64 * 1024):
                outbound.sendall(chunk)
        except OSError:
            pass
        finally:
            outbound.shutdown(socket.SHUT_WR)

    threading.Thread(target=read_stdin, daemon=True).start()
    watching = [inbound, tls]
    try:
        while True:
            for ready in select.select(watching, [], [])[0]:
                if ready is inbound:
                    if chunk := inbound.recv(64 * 1024):
                        tls.sendall(chunk)
                    else:
                        # stdin closed: send only a TCP FIN and keep draining
                        # the server's reply until it closes. (SSLSocket.shutdown
                        # would drop the TLS state.)
                        watching.remove(inbound)
                        socket.socket.shutdown(tls, socket.SHUT_WR)
                elif _drain(tls, stdout_fd):
                    return
    except OSError:
        pass
    finally:
        tls.close()
        inbound.close()
        outbound.close()
