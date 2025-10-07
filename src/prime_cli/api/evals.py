from typing import Any, Dict, List, Optional

import httpx

from ..config import Config


class EvalsAPIError(Exception):
    pass


class EnvironmentNotFoundError(EvalsAPIError):
    """Raised when an environment is not found in the hub."""

    pass


class EvalsClient:
    """
    Client for the Prime Evals API
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

    def create_evaluation(
        self,
        name: str,
        environment_ids: Optional[List[str]] = None,
        suite_id: Optional[str] = None,
        run_id: Optional[str] = None,
        version_id: Optional[str] = None,
        model_name: Optional[str] = None,
        dataset: Optional[str] = None,
        framework: Optional[str] = None,
        task_type: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.config.base_url}/api/v1/evaluations/"
        payload = {
            "name": name,
            "environment_ids": environment_ids,
            "suite_id": suite_id,
            "run_id": run_id,
            "version_id": version_id,
            "model_name": model_name,
            "dataset": dataset,
            "framework": framework,
            "task_type": task_type,
            "description": description,
            "tags": tags or [],
            "metadata": metadata,
            "metrics": metrics,
        }
        payload = {k: v for k, v in payload.items() if v is not None or k in ["tags"]}

        resp = self._client.post(url, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise EvalsAPIError(
                f"POST {url} failed: {e.response.status_code} {e.response.text}"
            ) from e
        return resp.json()

    def push_samples(self, evaluation_id: str, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        url = f"{self.config.base_url}/api/v1/evaluations/{evaluation_id}/samples"
        payload = {"samples": samples}

        resp = self._client.post(url, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise EvalsAPIError(
                f"POST {url} failed: {e.response.status_code} {e.response.text}"
            ) from e
        return resp.json()

    def finalize_evaluation(
        self, evaluation_id: str, metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = f"{self.config.base_url}/api/v1/evaluations/{evaluation_id}/finalize"
        payload = {"metrics": metrics} if metrics else {}

        resp = self._client.post(url, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise EvalsAPIError(
                f"POST {url} failed: {e.response.status_code} {e.response.text}"
            ) from e
        return resp.json()

    def list_evaluations(
        self,
        environment_id: Optional[str] = None,
        suite_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Dict[str, Any]:
        url = f"{self.config.base_url}/api/v1/evaluations/"
        params = {"skip": skip, "limit": limit}
        if environment_id:
            params["environment_id"] = environment_id
        if suite_id:
            params["suite_id"] = suite_id

        resp = self._client.get(url, params=params)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise EvalsAPIError(
                f"GET {url} failed: {e.response.status_code} {e.response.text}"
            ) from e
        return resp.json()

    def get_evaluation(self, evaluation_id: str) -> Dict[str, Any]:
        url = f"{self.config.base_url}/api/v1/evaluations/{evaluation_id}"
        resp = self._client.get(url)

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status in (400, 404, 422):
                raise EvalsAPIError(
                    f"Evaluation '{evaluation_id}' not found (GET {url} → {status})."
                ) from e
            raise EvalsAPIError(f"GET {url} failed: {status} {e.response.text}") from e
        return resp.json()

    def get_samples(self, evaluation_id: str, page: int = 1, limit: int = 100) -> Dict[str, Any]:
        url = f"{self.config.base_url}/api/v1/evaluations/{evaluation_id}/samples"
        params = {"page": page, "limit": limit}

        resp = self._client.get(url, params=params)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise EvalsAPIError(
                f"GET {url} failed: {e.response.status_code} {e.response.text}"
            ) from e
        return resp.json()

    def check_environment_exists(self, env_id: str, version: str = "latest") -> bool:
        if "/" in env_id:
            owner, name = env_id.split("/", 1)
            url = f"{self.config.base_url}/api/v1/environmentshub/{owner}/{name}/@{version}"

            try:
                resp = self._client.get(url)
                resp.raise_for_status()
                return True
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status in (404, 422):
                    return False

                raise EvalsAPIError(
                    f"Error checking environment '{env_id}': {status} {e.response.text}"
                ) from e
            except httpx.RequestError as e:
                raise EvalsAPIError(
                    f"Request failed while checking environment '{env_id}': {e}"
                ) from e
        else:
            name = env_id

            url = f"{self.config.base_url}/api/v1/environmentshub/"
            params = {
                "include_teams": True,
                "limit": 100,
            }

            try:
                resp = self._client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

                environments = data.get("data", data.get("environments", []))

                for env in environments:
                    env_name = env.get("name", "")
                    if env_name == name:
                        return True

                return False

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status in (404, 422):
                    return False
                raise EvalsAPIError(
                    f"Error checking environment '{env_id}': {status} {e.response.text}"
                ) from e
            except httpx.RequestError as e:
                raise EvalsAPIError(
                    f"Request failed while checking environment '{env_id}': {e}"
                ) from e
