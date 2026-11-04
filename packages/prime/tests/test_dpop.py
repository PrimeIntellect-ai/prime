import base64
import hashlib
import json
import time
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from prime_cli.core import Config, dpop


def _b64decode(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def _verify(proof: str) -> tuple[dict[str, Any], dict[str, Any]]:
    header_b64, claims_b64, sig_b64 = proof.split(".")
    header = json.loads(_b64decode(header_b64))
    jwk = header["jwk"]
    public_key = ec.EllipticCurvePublicNumbers(
        int.from_bytes(_b64decode(jwk["x"]), "big"),
        int.from_bytes(_b64decode(jwk["y"]), "big"),
        ec.SECP256R1(),
    ).public_key()
    sig = _b64decode(sig_b64)
    der = encode_dss_signature(int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big"))
    public_key.verify(der, f"{header_b64}.{claims_b64}".encode(), ec.ECDSA(hashes.SHA256()))
    return header, json.loads(_b64decode(claims_b64))


def _key() -> dpop.SoftwareKey:
    return dpop.SoftwareKey(ec.generate_private_key(ec.SECP256R1()))


def _ath(token: str) -> str:
    return dpop.b64url(hashlib.sha256(token.encode()).digest())


def test_proof_claims_and_signature() -> None:
    key = _key()
    proof = dpop.make_proof(key, "post", "https://api.example.com/api/v1/pods?x=1#f", "tok")

    header, claims = _verify(proof)

    assert header == {"typ": "dpop+jwt", "alg": "ES256", "jwk": key.public_jwk()}
    assert set(header["jwk"]) == {"kty", "crv", "x", "y"}
    assert claims["htm"] == "POST"
    assert claims["htu"] == "https://api.example.com/api/v1/pods"
    assert abs(claims["iat"] - time.time()) < 5
    assert claims["ath"] == _ath("tok")
    assert claims["jti"] != _verify(dpop.make_proof(key, "POST", "https://a.b/"))[1]["jti"]


def test_proof_without_token_omits_ath() -> None:
    _, claims = _verify(dpop.make_proof(_key(), "POST", "https://a.b/token", nonce="n1"))
    assert "ath" not in claims
    assert claims["nonce"] == "n1"


def test_expiring_token_is_refreshed_with_proof_from_same_key(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PRIME_CONTEXT", raising=False)
    config = Config()
    key = _key()
    session = dpop.save_session(
        config,
        {"token_type": "DPoP", "access_token": "old", "refresh_token": "r1", "expires_in": 10},
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == dpop.TOKEN_REFRESH_PATH:
            assert json.loads(request.content)["refresh_token"] == "r1"
            return httpx.Response(
                200,
                json={"token_type": "DPoP", "access_token": "new", "expires_in": 900},
            )
        return httpx.Response(200, json={"ok": True})

    auth = dpop.DPoPAuth(key, config, session)
    with httpx.Client(transport=httpx.MockTransport(handler), auth=auth) as client:
        client.get("https://api.example.com/api/v1/pods")

    refresh, api = seen
    assert "Authorization" not in refresh.headers
    refresh_header, refresh_claims = _verify(refresh.headers["DPoP"])
    assert refresh_header["jwk"] == key.public_jwk()
    assert refresh_claims["htu"] == f"https://api.example.com{dpop.TOKEN_REFRESH_PATH}"
    assert "ath" not in refresh_claims

    assert api.headers["Authorization"] == "DPoP new"
    _, api_claims = _verify(api.headers["DPoP"])
    assert api_claims["ath"] == _ath("new")

    stored = dpop.load_session(config)
    assert stored is not None
    assert stored["access_token"] == "new"
    assert stored["refresh_token"] == "r1"
