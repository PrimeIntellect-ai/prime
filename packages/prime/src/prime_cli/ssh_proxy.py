"""OpenSSH ProxyCommand transport for VM sandbox SSH sessions."""

import os
import socket
import sys
import threading


def main() -> None:
    host, port, session_id = sys.argv[1:]
    with socket.create_connection((host, int(port))) as connection:
        connection.sendall(f"PRIME-SSH-SESSION {session_id}\n".encode())

        def upload() -> None:
            try:
                while chunk := os.read(sys.stdin.fileno(), 64 * 1024):
                    connection.sendall(chunk)
                connection.shutdown(socket.SHUT_WR)
            except (BrokenPipeError, OSError):
                pass

        threading.Thread(target=upload, daemon=True).start()
        while chunk := connection.recv(64 * 1024):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
