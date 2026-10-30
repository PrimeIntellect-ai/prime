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

_CHUNK = 64 * 1024
_MAX_QUEUED = 1024 * 1024


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


def _drain(tls: ssl.SSLSocket, stdout_fd: int) -> tuple[bool, bool]:
    """Write everything the (non-blocking) socket has ready to stdout (a
    readable socket may hold only a TLS ticket, not data). Returns (closed,
    wants_write): closed once the server closed the connection; wants_write
    when TLS needs the socket writable before it can read on.

    The stdout write may block, but only while ssh is not reading; ssh always
    reads its ProxyCommand's stdout, independently of what it writes to stdin,
    so this cannot wait on us."""
    try:
        while chunk := tls.recv(64 * 1024):
            view = memoryview(chunk)
            while view:
                view = view[os.write(stdout_fd, view) :]
    except ssl.SSLWantReadError:
        return False, False
    except ssl.SSLWantWriteError:
        return False, True
    return True, False


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
    # The TLS socket never blocks: outbound bytes are queued and written when
    # it is writable, so a peer that is not reading cannot stop us from reading
    # its data (a blocking write would deadlock a full-duplex transfer). Stdin
    # is not read while the queue is full, which pushes back on ssh.
    tls.setblocking(False)
    queue = bytearray()
    inflight = b""  # after a Want* error, retry with the same bytes
    stdin_open = True
    fin_sent = False
    write_wants_read = read_wants_write = False
    try:
        while True:
            if not inflight and queue:
                inflight = bytes(queue[:_CHUNK])
                del queue[:_CHUNK]
            readers: list[socket.socket] = [tls]
            writers: list[socket.socket] = []
            if stdin_open and len(queue) < _MAX_QUEUED:
                readers.append(inbound)
            if inflight and not write_wants_read or read_wants_write:
                writers.append(tls)
            ready_r, ready_w, _ = select.select(readers, writers, [])
            if tls in ready_w or (tls in ready_r and write_wants_read):
                write_wants_read = False
                try:
                    inflight = inflight[tls.send(inflight) :]
                except ssl.SSLWantWriteError:
                    pass
                except ssl.SSLWantReadError:
                    write_wants_read = True
            if inbound in ready_r:
                if chunk := inbound.recv(_CHUNK):
                    queue += chunk
                else:
                    stdin_open = False
            if tls in ready_r or tls in ready_w:
                closed, read_wants_write = _drain(tls, stdout_fd)
                if closed:
                    return
            if not stdin_open and not fin_sent and not inflight and not queue:
                # stdin closed and flushed: send only a TCP FIN and keep
                # draining the server's reply until it closes. (SSLSocket.shutdown
                # would drop the TLS state.)
                fin_sent = True
                socket.socket.shutdown(tls, socket.SHUT_WR)
    except OSError:
        pass
    finally:
        tls.close()
        inbound.close()
        outbound.close()
