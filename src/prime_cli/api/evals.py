from typing import Any, Dict, Optional

import httpx

from ..config import Config


class EvalsAPIError(Exception):
    pass


class EvalsClient:
    """
    Client for the Prime Evals API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> None:
        self.config = Config()

        self.api_key = api_key or self.config.api_key
        if not self.api_key:
            raise EvalsAPIError("No API key. Run `prime config set-api-key` or set PRIME_API_KEY.")

        self.team_id = team_id if team_id is not None else self.config.team_id

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.team_id:
            headers["X-Prime-Team-ID"] = self.team_id

        self._client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=60.0),
        )

    def list_evals(self) -> Dict[str, Any]:
        url = f"{self.config.base_url}/api/v1/evals/runs"
        resp = self._client.get(url)

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise EvalsAPIError(
                f"GET {url} failed: {e.response.status_code} {e.response.text}"
            ) from e
        return resp.json()

    def get_eval(self, eval_id: str) -> Dict[str, Any]:
        url = f"{self.config.base_url}/api/v1/evals/runs/{eval_id}"
        resp = self._client.get(url)

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status in (400, 404, 422):
                raise EvalsAPIError(f"Eval '{eval_id}' not found (GET {url} → {status}).") from e
            raise EvalsAPIError(f"GET {url} failed: {status} {e.response.text}") from e
        return resp.json()

    def push_eval(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.config.base_url}/api/v1/evals/push"
        resp = self._client.post(url, json=payload)

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise EvalsAPIError(
                f"POST {url} failed: {e.response.status_code} {e.response.text}"
            ) from e
        return resp.json()

    def delete_eval(self, eval_id: str) -> Dict[str, Any]:
        url = f"{self.config.base_url}/api/v1/evals/runs/{eval_id}"
        resp = self._client.delete(url)

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status in (400, 404, 422):
                raise EvalsAPIError(f"Eval '{eval_id}' not found (DELETE {url} → {status}).") from e
            raise EvalsAPIError(f"DELETE {url} failed: {status} {e.response.text}") from e
        return resp.json()
