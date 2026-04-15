from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Iterator, Optional

import httpx

from prime_cli.core import Config


class InferenceAPIError(Exception):
    pass


class InferencePaymentRequiredError(InferenceAPIError):
    pass


def _extract_error_message(response: httpx.Response) -> str:
    text = response.text.strip()
    return text or response.reason_phrase or "Unknown error"


def _extract_payment_error_message(response: httpx.Response) -> str:
    try:
        return response.json()["error"]["message"].strip()
    except Exception:
        return _extract_error_message(response)


class InferenceClient:
    """
    Minimal client for the OpenAI-compatible Prime Inference API:
      - GET /v1/models
      - GET /v1/models/{model_id}
      - POST /v1/chat/completions
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        team_id: Optional[str] = None,
        inference_url: Optional[str] = None,
        timeout: Optional[float | httpx.Timeout] = None,
    ) -> None:
        # Load config
        self.config = Config()

        self.api_key = api_key or self.config.api_key
        if not self.api_key:
            raise InferenceAPIError(
                "No API key. Run `prime config set-api-key` or set PRIME_API_KEY."
            )

        self.team_id = team_id if team_id is not None else self.config.team_id
        self.inference_url = (inference_url or self.config.inference_url).rstrip("/")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.team_id:
            headers["X-Prime-Team-ID"] = self.team_id

        self._client = httpx.Client(
            headers=headers,
            timeout=timeout or httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=60.0),
        )

    def list_models(self) -> Dict[str, Any]:
        url = f"{self.inference_url}/models"
        resp = self._client.get(url)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            message = _extract_error_message(e.response)
            if status == 402:
                raise InferencePaymentRequiredError(
                    f"Payment required. {_extract_payment_error_message(e.response)}"
                ) from e
            raise InferenceAPIError(f"GET {url} failed: {status} {message}") from e
        return resp.json()

    def retrieve_model(self, model_id: str) -> Dict[str, Any]:
        url = f"{self.inference_url}/models/{model_id}"
        resp = self._client.get(url)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            message = _extract_error_message(e.response)
            # Treat common "unknown model" responses as a dedicated error
            if status in (400, 404, 422):
                raise InferenceAPIError(
                    f"Model '{model_id}' not found or unavailable (GET {url} → {status})."
                ) from e
            if status == 402:
                raise InferencePaymentRequiredError(
                    f"Payment required. {_extract_payment_error_message(e.response)}"
                ) from e
            raise InferenceAPIError(f"GET {url} failed: {status} {message}") from e
        return resp.json()

    def chat_completion(
        self, payload: Dict[str, Any], stream: bool = False
    ) -> Dict[str, Any] | Iterable[Dict[str, Any]]:
        url = f"{self.inference_url}/chat/completions"

        if not stream:
            resp = self._client.post(url, json=payload)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                message = _extract_error_message(e.response)
                if status == 402:
                    raise InferencePaymentRequiredError(
                        f"Payment required. {_extract_payment_error_message(e.response)}"
                    ) from e
                raise InferenceAPIError(f"POST {url} failed: {status} {message}") from e
            return resp.json()

        # Streamed (SSE-style: lines prefixed with 'data: ')
        def _stream() -> Iterator[Dict[str, Any]]:
            with self._client.stream("POST", url, json=payload) as r:
                try:
                    r.raise_for_status()
                except httpx.HTTPStatusError as e:
                    e.response.read()
                    status = e.response.status_code
                    message = _extract_error_message(e.response)
                    if status == 402:
                        raise InferencePaymentRequiredError(
                            f"Payment required. {_extract_payment_error_message(e.response)}"
                        ) from e
                    raise InferenceAPIError(f"POST {url} failed: {status} {message}") from e
                for line in r.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:].strip()
                    else:
                        # Some servers don't prefix; try parsing anyway
                        data = line.strip()

                    if not data or data == "[DONE]":
                        if data == "[DONE]":
                            return
                        continue
                    try:
                        chunk = json.loads(data)
                        yield chunk
                    except json.JSONDecodeError:
                        # Skip unparsable lines quietly
                        continue

        return _stream()
