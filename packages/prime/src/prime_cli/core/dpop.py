"""DPoP (RFC 9449) sender-constrained sessions for `prime login --dpop`."""

import base64
import contextlib
import hashlib
import json
import secrets
import time
from pathlib import Path
from typing import Any, Generator, Optional, Protocol

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

from .config import Config, _write_private

DEVICE_START_PATH = "/api/v1/auth/device/start"
DEVICE_TOKEN_PATH = "/api/v1/auth/device/token"
TOKEN_REFRESH_PATH = "/api/v1/auth/token/refresh"
CLIENT_ID = "prime-cli"
DEVICE_CODE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
KEYCHAIN_SERVICE = "prime-cli-dpop"
REFRESH_MARGIN_SECONDS = 60


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


class DeviceKey(Protocol):
    def public_jwk(self) -> dict[str, str]: ...

    def sign(self, message: bytes) -> bytes:
        """Return a raw (r || s) ES256 signature."""
        ...


class KeyProvider(Protocol):
    def create(self, name: str) -> DeviceKey: ...

    def load(self, name: str) -> Optional[DeviceKey]: ...

    def delete(self, name: str) -> None: ...


class SoftwareKey:
    def __init__(self, private_key: ec.EllipticCurvePrivateKey) -> None:
        self._key = private_key

    def public_jwk(self) -> dict[str, str]:
        numbers = self._key.public_key().public_numbers()
        return {
            "kty": "EC",
            "crv": "P-256",
            "x": b64url(numbers.x.to_bytes(32, "big")),
            "y": b64url(numbers.y.to_bytes(32, "big")),
        }

    def sign(self, message: bytes) -> bytes:
        r, s = decode_dss_signature(self._key.sign(message, ec.ECDSA(hashes.SHA256())))
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")


class KeychainKeyProvider:
    """P-256 key stored as PKCS#8 in the OS keychain (macOS Keychain, Windows
    Credential Locker, Linux Secret Service). Exportable by anyone who can read
    the keychain item; hardware-backed providers should replace this per OS."""

    def create(self, name: str) -> DeviceKey:
        import keyring

        private_key = ec.generate_private_key(ec.SECP256R1())
        pem = private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        keyring.set_password(KEYCHAIN_SERVICE, name, pem.decode())
        return SoftwareKey(private_key)

    def load(self, name: str) -> Optional[DeviceKey]:
        import keyring
        from keyring.errors import KeyringError

        try:
            pem = keyring.get_password(KEYCHAIN_SERVICE, name)
        except KeyringError:
            return None
        if not pem:
            return None
        private_key = serialization.load_pem_private_key(pem.encode(), password=None)
        if not isinstance(private_key, ec.EllipticCurvePrivateKey):
            return None
        return SoftwareKey(private_key)

    def delete(self, name: str) -> None:
        import keyring
        from keyring.errors import KeyringError

        with contextlib.suppress(KeyringError):
            keyring.delete_password(KEYCHAIN_SERVICE, name)


def key_provider() -> KeyProvider:
    return KeychainKeyProvider()


def make_proof(
    key: DeviceKey,
    method: str,
    url: str,
    access_token: Optional[str] = None,
    nonce: Optional[str] = None,
) -> str:
    htu = httpx.URL(url).copy_with(query=None, fragment=None)
    header = {"typ": "dpop+jwt", "alg": "ES256", "jwk": key.public_jwk()}
    claims: dict[str, Any] = {
        "htm": method.upper(),
        "htu": str(htu),
        "iat": int(time.time()),
        "jti": b64url(secrets.token_bytes(16)),
    }
    if access_token:
        claims["ath"] = b64url(hashlib.sha256(access_token.encode()).digest())
    if nonce:
        claims["nonce"] = nonce
    signing_input = ".".join(
        b64url(json.dumps(part, separators=(",", ":")).encode()) for part in (header, claims)
    )
    return f"{signing_input}.{b64url(key.sign(signing_input.encode()))}"


def _session_path(config: Config) -> Path:
    name = config._sanitize_environment_name(config.current_environment)
    return config.config_dir / f"dpop-{name}.json"


def load_session(config: Config) -> Optional[dict[str, Any]]:
    path = _session_path(config)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_session(
    config: Config, token_response: dict[str, Any], previous_refresh_token: Optional[str] = None
) -> dict[str, Any]:
    if str(token_response.get("token_type", "")).lower() != "dpop":
        raise ValueError("Server did not issue a DPoP-bound token")
    session = {
        "access_token": token_response["access_token"],
        "refresh_token": token_response.get("refresh_token") or previous_refresh_token,
        "expires_at": time.time() + int(token_response.get("expires_in", 900)),
    }
    _write_private(_session_path(config), json.dumps(session, indent=2))
    return session


def delete_session(config: Config) -> bool:
    """Delete the session file and its device key; return whether a session existed."""
    path = _session_path(config)
    if not path.exists():
        return False
    path.unlink()
    key_provider().delete(config.current_environment)
    return True


class DPoPAuth(httpx.Auth):
    """Attach a fresh DPoP proof to every request, refreshing the bound access
    token shortly before it expires. Without a session (the login token
    endpoint) it sends proofs only."""

    requires_response_body = True

    def __init__(
        self,
        key: DeviceKey,
        config: Optional[Config] = None,
        session: Optional[dict[str, Any]] = None,
    ) -> None:
        self.key = key
        self.config = config
        self.session = session
        self.nonce: Optional[str] = None

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        if self.session and self.session["expires_at"] - time.time() < REFRESH_MARGIN_SECONDS:
            yield from self._refresh(request.url)
        yield from self._send(request)

    def _send(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, Any]:
        token = self.session["access_token"] if self.session else None
        if token:
            request.headers["Authorization"] = f"DPoP {token}"
        request.headers["DPoP"] = make_proof(
            self.key, request.method, str(request.url), token, self.nonce
        )
        response = yield request
        nonce = response.headers.get("DPoP-Nonce")
        if nonce and nonce != self.nonce:
            self.nonce = nonce
            if response.status_code in (400, 401):
                request.headers["DPoP"] = make_proof(
                    self.key, request.method, str(request.url), token, nonce
                )
                response = yield request
        return response

    def _refresh(self, url: httpx.URL) -> Generator[httpx.Request, httpx.Response, None]:
        assert self.session is not None and self.config is not None
        session, self.session = self.session, None
        request = httpx.Request(
            "POST",
            url.copy_with(path=TOKEN_REFRESH_PATH, query=None),
            json={"grant_type": "refresh_token", "refresh_token": session["refresh_token"]},
        )
        response = yield from self._send(request)
        self.session = session
        if response.status_code == 200:
            # On failure the stale token is sent anyway and the API's 401 surfaces.
            with contextlib.suppress(KeyError, ValueError):
                self.session = save_session(self.config, response.json(), session["refresh_token"])


def load_auth(config: Config) -> Optional[DPoPAuth]:
    session = load_session(config)
    if not session:
        return None
    key = key_provider().load(config.current_environment)
    return DPoPAuth(key, config, session) if key else None
