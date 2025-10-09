from typing import Any, Dict, List, Optional

from prime_core import APIClient, AsyncAPIClient

from .exceptions import EvalsAPIError


class EvalsClient:
    """
    Client for the Prime Evals API
    """

    def __init__(self, api_client: APIClient) -> None:
        self.client = api_client

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
        """Create a new evaluation"""
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

        response = self.client.request("POST", "/evaluations/", json=payload)
        return response

    def push_samples(self, evaluation_id: str, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Push evaluation samples"""
        payload = {"samples": samples}
        response = self.client.request(
            "POST", f"/evaluations/{evaluation_id}/samples", json=payload
        )
        return response

    def finalize_evaluation(
        self, evaluation_id: str, metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Finalize an evaluation with final metrics"""
        payload = {"metrics": metrics} if metrics else {}
        response = self.client.request(
            "POST", f"/evaluations/{evaluation_id}/finalize", json=payload
        )
        return response

    def list_evaluations(
        self,
        environment_id: Optional[str] = None,
        suite_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """List evaluations with optional filters"""
        params: Dict[str, Any] = {"skip": skip, "limit": limit}
        if environment_id:
            params["environment_id"] = environment_id
        if suite_id:
            params["suite_id"] = suite_id

        response = self.client.request("GET", "/evaluations/", params=params)
        return response

    def get_evaluation(self, evaluation_id: str) -> Dict[str, Any]:
        """Get evaluation details by ID"""
        response = self.client.request("GET", f"/evaluations/{evaluation_id}")
        return response

    def get_samples(self, evaluation_id: str, page: int = 1, limit: int = 100) -> Dict[str, Any]:
        """Get samples for an evaluation"""
        params: Dict[str, Any] = {"page": page, "limit": limit}
        response = self.client.request(
            "GET", f"/evaluations/{evaluation_id}/samples", params=params
        )
        return response

    def check_environment_exists(self, env_id: str, version: str = "latest") -> bool:
        """Check if an environment exists in the hub"""
        try:
            if "/" in env_id:
                owner, name = env_id.split("/", 1)
                self.client.request("GET", f"/environmentshub/{owner}/{name}/@{version}")
                return True
            else:
                name = env_id
                params: Dict[str, Any] = {
                    "include_teams": True,
                    "limit": 100,
                }
                data = self.client.request("GET", "/environmentshub/", params=params)
                environments = data.get("data", data.get("environments", []))

                for env in environments:
                    env_name = env.get("name", "")
                    if env_name == name:
                        return True

                return False
        except Exception as e:
            # If 404 or not found, return False
            if "404" in str(e) or "not found" in str(e).lower():
                return False
            # Re-raise other errors
            raise EvalsAPIError(f"Error checking environment '{env_id}': {e}") from e


class AsyncEvalsClient:
    """Async client for Prime Evals API"""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.client = AsyncAPIClient(api_key=api_key)

    async def create_evaluation(
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
        """Create a new evaluation"""
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

        response = await self.client.request("POST", "/evaluations/", json=payload)
        return response

    async def push_samples(
        self, evaluation_id: str, samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Push evaluation samples"""
        payload = {"samples": samples}
        response = await self.client.request(
            "POST", f"/evaluations/{evaluation_id}/samples", json=payload
        )
        return response

    async def finalize_evaluation(
        self, evaluation_id: str, metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Finalize an evaluation with final metrics"""
        payload = {"metrics": metrics} if metrics else {}
        response = await self.client.request(
            "POST", f"/evaluations/{evaluation_id}/finalize", json=payload
        )
        return response

    async def list_evaluations(
        self,
        environment_id: Optional[str] = None,
        suite_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """List evaluations with optional filters"""
        params: Dict[str, Any] = {"skip": skip, "limit": limit}
        if environment_id:
            params["environment_id"] = environment_id
        if suite_id:
            params["suite_id"] = suite_id

        response = await self.client.request("GET", "/evaluations/", params=params)
        return response

    async def get_evaluation(self, evaluation_id: str) -> Dict[str, Any]:
        """Get evaluation details by ID"""
        response = await self.client.request("GET", f"/evaluations/{evaluation_id}")
        return response

    async def get_samples(
        self, evaluation_id: str, page: int = 1, limit: int = 100
    ) -> Dict[str, Any]:
        """Get samples for an evaluation"""
        params: Dict[str, Any] = {"page": page, "limit": limit}
        response = await self.client.request(
            "GET", f"/evaluations/{evaluation_id}/samples", params=params
        )
        return response

    async def check_environment_exists(self, env_id: str, version: str = "latest") -> bool:
        """Check if an environment exists in the hub"""
        try:
            if "/" in env_id:
                owner, name = env_id.split("/", 1)
                await self.client.request("GET", f"/environmentshub/{owner}/{name}/@{version}")
                return True
            else:
                name = env_id
                params: Dict[str, Any] = {
                    "include_teams": True,
                    "limit": 100,
                }
                data = await self.client.request("GET", "/environmentshub/", params=params)
                environments = data.get("data", data.get("environments", []))

                for env in environments:
                    env_name = env.get("name", "")
                    if env_name == name:
                        return True

                return False
        except Exception as e:
            # If 404 or not found, return False
            if "404" in str(e) or "not found" in str(e).lower():
                return False
            # Re-raise other errors
            raise EvalsAPIError(f"Error checking environment '{env_id}': {e}") from e

    async def aclose(self) -> None:
        """Close the async client"""
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncEvalsClient":
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit"""
        await self.aclose()
