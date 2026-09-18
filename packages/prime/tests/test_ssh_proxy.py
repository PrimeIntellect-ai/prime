import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from prime_cli.commands import sandbox


def test_ssh_proxy_sends_session_prefix_then_relays_bytes():
    with socket.create_server(("127.0.0.1", 0)) as server:
        host, port = server.getsockname()
        process = subprocess.Popen(
            [sys.executable, "-m", "prime_cli.ssh_proxy", host, str(port), "session-1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        with server.accept()[0] as connection:
            received = b""
            while not received.endswith(b"\n"):
                received += connection.recv(1024)
            assert received == b"PRIME-SSH-SESSION session-1\n"

            assert process.stdin is not None
            process.stdin.write(b"client bytes")
            process.stdin.flush()
            assert connection.recv(12) == b"client bytes"

            connection.sendall(b"server bytes")
            connection.shutdown(socket.SHUT_WR)

        assert process.stdout is not None
        assert process.stdout.read() == b"server bytes"
        assert process.wait(timeout=5) == 0


def test_ssh_command_authorizes_key_and_uses_session_proxy(monkeypatch, tmp_path):
    authorized_keys = []

    def create_ssh_session(_sandbox_id, public_key):
        authorized_keys.append(public_key)
        return SimpleNamespace(session_id="session-1", host="ssh.example.com", port=2222)

    client = SimpleNamespace(
        get=lambda _sandbox_id: SimpleNamespace(status="RUNNING"),
        create_ssh_session=create_ssh_session,
        close_ssh_session=lambda _sandbox_id, _session_id: None,
    )
    commands = []

    def run(command, **_kwargs):
        if command[0] == "ssh-keygen":
            Path(f"{command[-1]}.pub").write_text("ssh-ed25519 key")
        else:
            commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sandbox, "APIClient", lambda: object())
    monkeypatch.setattr(sandbox, "SandboxClient", lambda _api: client)
    monkeypatch.setattr(sandbox.shutil, "which", lambda _command: "/usr/bin/tool")
    monkeypatch.setattr(sandbox.tempfile, "mkdtemp", lambda **_kwargs: str(tmp_path))
    monkeypatch.setattr(sandbox.subprocess, "run", run)

    with pytest.raises(typer.Exit) as exit_info:
        sandbox.ssh_connect("sbx-1", None, None)

    assert exit_info.value.exit_code == 0
    assert authorized_keys == ["ssh-ed25519 key"]
    assert any("prime_cli.ssh_proxy" in arg for arg in commands[0])
