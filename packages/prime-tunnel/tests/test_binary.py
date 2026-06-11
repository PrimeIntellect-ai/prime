import hashlib
import io
import zipfile

from prime_tunnel import binary


class _FakeStream:
    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def raise_for_status(self):
        return None

    def iter_bytes(self, chunk_size: int):
        for offset in range(0, len(self._data), chunk_size):
            yield self._data[offset : offset + chunk_size]


def _windows_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("frp_0.66.0_windows_amd64/frpc.exe", b"fake frpc exe")
    return buffer.getvalue()


def test_get_platform_key_normalizes_windows_arch(monkeypatch):
    monkeypatch.setattr(binary.platform, "system", lambda: "Windows")
    monkeypatch.setattr(binary.platform, "machine", lambda: "AMD64")

    assert binary._get_platform_key() == ("Windows", "x86_64")

    monkeypatch.setattr(binary.platform, "machine", lambda: "ARM64")

    assert binary._get_platform_key() == ("Windows", "arm64")


def test_get_frpc_path_uses_exe_on_windows(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(binary.platform, "system", lambda: "Windows")
    monkeypatch.setattr(binary.platform, "machine", lambda: "AMD64")

    def fake_download(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake frpc exe")

    monkeypatch.setattr(binary, "_download_frpc", fake_download)

    assert binary.get_frpc_path().name == "frpc.exe"


def test_download_frpc_extracts_windows_zip(monkeypatch, tmp_path):
    archive_data = _windows_zip()
    platform_key = ("Windows", "x86_64")
    monkeypatch.setattr(binary, "_get_platform_key", lambda: platform_key)
    monkeypatch.setitem(binary.FRPC_URLS, platform_key, "https://example.invalid/frp.zip")
    monkeypatch.setitem(
        binary.FRPC_CHECKSUMS,
        platform_key,
        hashlib.sha256(archive_data).hexdigest(),
    )
    monkeypatch.setattr(binary.httpx, "stream", lambda *_args, **_kwargs: _FakeStream(archive_data))

    dest = tmp_path / "frpc.exe"

    binary._download_frpc(dest)

    assert dest.read_bytes() == b"fake frpc exe"
