import asyncio
import sys
from typing import Any, Dict, List, Optional, Union

from .core import APIError, AsyncAPIClient
from .exceptions import EvalsAPIError, InvalidEvaluationError


def _build_user_agent() -> str:
    """Build User-Agent string for prime-evals"""
    from prime_evals import __version__

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return f"prime-evals/{__version__} python/{python_version}"


class EvalsClient:
    """
    Client for the Prime Evals API
    """

    def __init__(self, api_client: Any) -> None:
        """Initialize with an API client.

        Args:
            api_client: Any API client with get/post/request methods and a config attribute.
                       Can be prime_cli.core.APIClient or prime_evals.core.APIClient.
        """
        self.client = api_client

    def _lookup_environment_id(self, env_id: str) -> str:
        """
        Lookup an environment by ID to verify it exists.

        Raises:
            EvalsAPIError: If the environment does not exist (404)
        """
        try:
            lookup_data: Dict[str, Any] = {"id": env_id}
            response = self.client.post("/environmentshub/lookup", json=lookup_data)
            return response["data"]["id"]
        except APIError as e:
            raise EvalsAPIError(
                f"Environment with ID '{env_id}' does not exist in the hub. "
                f"Please verify the environment ID is correct."
            ) from e

    def _lookup_environment_by_slug(self, owner_slug: str, name: str) -> str:
        """
        Lookup an environment by owner slug and name (lookup only, does not create).

        Raises:
            EvalsAPIError: If the environment does not exist (404)
        """
        try:
            # Try team_slug first (lookup endpoint supports team_slug)
            # If owner is not a team, backend will handle it appropriately
            lookup_data: Dict[str, Any] = {"name": name, "team_slug": owner_slug}
            response = self.client.post("/environmentshub/lookup", json=lookup_data)
            return response["data"]["id"]
        except APIError as e:
            raise EvalsAPIError(
                f"Environment '{owner_slug}/{name}' does not exist in the hub. "
                f"Please ensure the environment exists and you have access to it."
            ) from e

    def _resolve_environment_id(self, env_name: str) -> str:
        """
        Resolve environment ID by name (get-or-create behavior).
        Only used when no owner slug is provided.

        Raises:
            EvalsAPIError: If the environment does not exist (404)
        """
        try:
            resolve_data: Dict[str, Any] = {"name": env_name}

            if self.client.config.team_id:
                resolve_data["team_id"] = self.client.config.team_id

            response = self.client.post("/environmentshub/resolve", json=resolve_data)
            return response["data"]["id"]

        except APIError as e:
            raise EvalsAPIError(
                f"Environment '{env_name}' does not exist in the hub. "
                f"Please push the environment first with: prime env push"
            ) from e

    def _resolve_environments(
        self, environments: List[Union[str, Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """
        Resolve a list of environments from various identifier formats to database IDs.
        """
        resolved_environments = []
        for env in environments:
            # Handle string inputs (convert to dict format)
            if isinstance(env, str):
                env = {"slug": env} if "/" in env else {"name": env}

            resolved_env = env.copy() if isinstance(env, dict) else {}

            # Handle different identifier types explicitly
            # Check for explicit "slug" or "name" keys first
            try:
                if "slug" in resolved_env:
                    # Owner/name format, lookup (does not create)
                    slug = resolved_env.pop("slug")
                    if "/" not in slug:
                        # Invalid slug format - skip this environment
                        continue
                    owner_slug, name = slug.split("/", 1)
                    resolved_env["id"] = self._lookup_environment_by_slug(owner_slug, name)
                elif "name" in resolved_env:
                    # Just a name, resolve to database ID (get-or-create)
                    resolved_env["id"] = self._resolve_environment_id(resolved_env.pop("name"))
                elif "id" in resolved_env:
                    # "id" key exists - validate it exists in the hub via lookup
                    resolved_env["id"] = self._lookup_environment_id(resolved_env["id"])
                else:
                    # Skip environments without valid identifiers
                    continue
                resolved_environments.append(resolved_env)
            except EvalsAPIError:
                # Skip environments that don't exist in the hub
                # Continue processing remaining environments
                continue
        return resolved_environments

    def create_evaluation(
        self,
        name: str,
        environments: Optional[List[Dict[str, str]]] = None,
        suite_id: Optional[str] = None,
        run_id: Optional[str] = None,
        model_name: Optional[str] = None,
        dataset: Optional[str] = None,
        framework: Optional[str] = None,
        task_type: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        is_public: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Create a new evaluation

        Either run_id or environments must be provided.
        Environments should be a list of dicts with 'id' and optional 'version_id'.

        Example: [{"id": "simpleqa", "version_id": "v1"}]

        Raises:
            InvalidEvaluationError: If neither run_id nor environments is provided
        """
        if not run_id and not environments:
            raise InvalidEvaluationError(
                "Either 'run_id' or 'environments' must be provided. "
                "For environment evals, provide environments=[{'id': 'env-id', 'version_id': 'v1'}]"
            )

        resolved_environments = None
        if environments:
            resolved_environments = self._resolve_environments(environments)

            # Validate that we have at least one resolved environment if run_id is not provided
            # This check happens AFTER resolution to catch cases where all environments were invalid
            if not resolved_environments and not run_id:
                raise InvalidEvaluationError(
                    "All provided environments lack valid identifiers (slug, name, or id). "
                    "Either provide valid environment identifiers or provide a 'run_id'. "
                )

        payload = {
            "name": name,
            "environments": resolved_environments,
            "suite_id": suite_id,
            "run_id": run_id,
            "model_name": model_name,
            "dataset": dataset,
            "framework": framework,
            "task_type": task_type,
            "description": description,
            "tags": tags or [],
            "metadata": metadata,
            "metrics": metrics,
        }
        # Include team_id from config if set (for team-owned evaluations)
        if self.client.config.team_id:
            payload["team_id"] = self.client.config.team_id
        # Only include is_public if it's explicitly set (not None)
        if is_public is not None:
            payload["is_public"] = is_public
        payload = {k: v for k, v in payload.items() if v is not None or k in ["tags"]}

        response = self.client.request("POST", "/evaluations/", json=payload)
        return response

    def push_samples(
        self,
        evaluation_id: str,
        samples: List[Dict[str, Any]],
        batch_size: int = 256,
    ) -> Dict[str, Any]:
        """Push evaluation samples in batches to avoid request size limits."""
        if not samples:
            return {}

        total_samples_pushed = 0
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            payload = {"samples": batch}
            self.client.request("POST", f"/evaluations/{evaluation_id}/samples", json=payload)
            total_samples_pushed += len(batch)

        return {"samples_pushed": total_samples_pushed}

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
        env_name: Optional[str] = None,
        suite_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """List evaluations with optional filters"""
        params: Dict[str, Any] = {"skip": skip, "limit": limit}
        if env_name:
            params["environment_name"] = env_name
        if suite_id:
            params["suite_id"] = suite_id

        response = self.client.request("GET", "/evaluations/", params=params)
        return response

    def get_evaluation(self, evaluation_id: str) -> Dict[str, Any]:
        """Get evaluation details by ID"""
        response = self.client.request("GET", f"/evaluations/{evaluation_id}")
        return response

    def update_evaluation(
        self,
        evaluation_id: str,
        name: Optional[str] = None,
        model_name: Optional[str] = None,
        dataset: Optional[str] = None,
        framework: Optional[str] = None,
        task_type: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "name": name,
            "model_name": model_name,
            "dataset": dataset,
            "framework": framework,
            "task_type": task_type,
            "description": description,
            "tags": tags if tags is not None else [],
            "metadata": metadata,
            "metrics": metrics,
        }
        payload = {k: v for k, v in payload.items() if v is not None or k in ["tags"]}

        response = self.client.request("PUT", f"/evaluations/{evaluation_id}", json=payload)
        return response

    def get_samples(self, evaluation_id: str, page: int = 1, limit: int = 100) -> Dict[str, Any]:
        """Get samples for an evaluation"""
        params: Dict[str, Any] = {"page": page, "limit": limit}
        response = self.client.request(
            "GET", f"/evaluations/{evaluation_id}/samples", params=params
        )
        return response


class AsyncEvalsClient:
    """Async client for Prime Evals API"""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.client = AsyncAPIClient(api_key=api_key, user_agent=_build_user_agent())

    async def _lookup_environment_id(self, env_id: str) -> str:
        """
        Lookup an environment by ID to verify it exists.

        Raises:
            EvalsAPIError: If the environment does not exist (404)
        """
        try:
            lookup_data: Dict[str, Any] = {"id": env_id}
            response = await self.client.post("/environmentshub/lookup", json=lookup_data)
            return response["data"]["id"]
        except APIError as e:
            raise EvalsAPIError(
                f"Environment with ID '{env_id}' does not exist in the hub. "
                f"Please verify the environment ID is correct."
            ) from e

    async def _lookup_environment_by_slug(self, owner_slug: str, name: str) -> str:
        """
        Lookup an environment by owner slug and name (lookup only, does not create).

        Raises:
            EvalsAPIError: If the environment does not exist (404)
        """
        try:
            # Try team_slug first (lookup endpoint supports team_slug)
            # If owner is not a team, backend will handle it appropriately
            lookup_data: Dict[str, Any] = {"name": name, "team_slug": owner_slug}
            response = await self.client.post("/environmentshub/lookup", json=lookup_data)
            return response["data"]["id"]
        except APIError as e:
            raise EvalsAPIError(
                f"Environment '{owner_slug}/{name}' does not exist in the hub. "
                f"Please ensure the environment exists and you have access to it."
            ) from e

    async def _resolve_environment_id(self, env_name: str) -> str:
        """
        Resolve environment ID by name (get-or-create behavior).
        Only used when no owner slug is provided.

        Raises:
            EvalsAPIError: If the environment does not exist (404)
        """
        try:
            resolve_data: Dict[str, Any] = {"name": env_name}

            if self.client.config.team_id:
                resolve_data["team_id"] = self.client.config.team_id

            response = await self.client.post("/environmentshub/resolve", json=resolve_data)
            return response["data"]["id"]

        except APIError as e:
            raise EvalsAPIError(
                f"Environment '{env_name}' does not exist in the hub. "
                f"Please push the environment first with: prime env push"
            ) from e

    async def _resolve_environments(
        self, environments: List[Union[str, Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """
        Resolve a list of environments from various identifier formats to database IDs.
        """

        async def resolve_env(env: Union[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
            # Handle string inputs (convert to dict format)
            if isinstance(env, str):
                env = {"slug": env} if "/" in env else {"name": env}

            resolved_env = env.copy() if isinstance(env, dict) else {}
            # Handle different identifier types explicitly
            # Check for explicit "slug" or "name" keys first
            try:
                if "slug" in resolved_env:
                    # Owner/name format, lookup (does not create)
                    slug = resolved_env.pop("slug")
                    if "/" not in slug:
                        # Invalid slug format - skip this environment
                        return None
                    owner_slug, name = slug.split("/", 1)
                    resolved_env["id"] = await self._lookup_environment_by_slug(owner_slug, name)
                elif "name" in resolved_env:
                    # Just a name, resolve to database ID (get-or-create)
                    resolved_env["id"] = await self._resolve_environment_id(
                        resolved_env.pop("name")
                    )
                elif "id" in resolved_env:
                    # "id" key exists - validate it exists in the hub via lookup
                    resolved_env["id"] = await self._lookup_environment_id(resolved_env["id"])
                else:
                    # Skip environments without valid identifiers
                    return None
                return resolved_env
            except EvalsAPIError:
                # Skip environments that don't exist in the hub
                # Return None to filter them out
                return None

        resolved_environments_list = await asyncio.gather(
            *[resolve_env(env) for env in environments]
        )
        # Filter out None values (environments without valid identifiers or resolution failures)
        return [env for env in resolved_environments_list if env is not None]

    async def create_evaluation(
        self,
        name: str,
        environments: Optional[List[Dict[str, str]]] = None,
        suite_id: Optional[str] = None,
        run_id: Optional[str] = None,
        model_name: Optional[str] = None,
        dataset: Optional[str] = None,
        framework: Optional[str] = None,
        task_type: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        is_public: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Create a new evaluation

        Either run_id or environments must be provided.
        Environments should be a list of dicts with 'id' and optional 'version_id'.

        Example: [{"id": "simpleqa", "version_id": "v1"}]

        Raises:
            InvalidEvaluationError: If neither run_id nor environments is provided
        """
        if not run_id and not environments:
            raise InvalidEvaluationError(
                "Either 'run_id' or 'environments' must be provided. "
                "For environment evals, provide environments=[{'id': 'env-id', 'version_id': 'v1'}]"
            )

        resolved_environments = None
        if environments:
            resolved_environments = await self._resolve_environments(environments)

            # Validate that we have at least one resolved environment if run_id is not provided
            # This check happens AFTER resolution to catch cases where all environments were invalid
            if not resolved_environments and not run_id:
                raise InvalidEvaluationError(
                    "All provided environments lack valid identifiers (slug, name, or id). "
                    "Either provide valid environment identifiers or provide a 'run_id'. "
                )

        payload = {
            "name": name,
            "environments": resolved_environments,
            "suite_id": suite_id,
            "run_id": run_id,
            "model_name": model_name,
            "dataset": dataset,
            "framework": framework,
            "task_type": task_type,
            "description": description,
            "tags": tags or [],
            "metadata": metadata,
            "metrics": metrics,
        }
        # Include team_id from config if set (for team-owned evaluations)
        if self.client.config.team_id:
            payload["team_id"] = self.client.config.team_id
        # Only include is_public if it's explicitly set (not None)
        if is_public is not None:
            payload["is_public"] = is_public
        payload = {k: v for k, v in payload.items() if v is not None or k in ["tags"]}

        response = await self.client.request("POST", "/evaluations/", json=payload)
        return response

    async def push_samples(
        self,
        evaluation_id: str,
        samples: List[Dict[str, Any]],
        batch_size: int = 256,
    ) -> Dict[str, Any]:
        """Push evaluation samples in batches."""
        if not samples:
            return {}

        total_samples_pushed = 0
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            payload = {"samples": batch}
            await self.client.request("POST", f"/evaluations/{evaluation_id}/samples", json=payload)
            total_samples_pushed += len(batch)

        return {"samples_pushed": total_samples_pushed}

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
        env_name: Optional[str] = None,
        suite_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """List evaluations with optional filters"""
        params: Dict[str, Any] = {"skip": skip, "limit": limit}
        if env_name:
            params["environment_name"] = env_name
        if suite_id:
            params["suite_id"] = suite_id

        response = await self.client.request("GET", "/evaluations/", params=params)
        return response

    async def get_evaluation(self, evaluation_id: str) -> Dict[str, Any]:
        """Get evaluation details by ID"""
        response = await self.client.request("GET", f"/evaluations/{evaluation_id}")
        return response

    async def update_evaluation(
        self,
        evaluation_id: str,
        name: Optional[str] = None,
        model_name: Optional[str] = None,
        dataset: Optional[str] = None,
        framework: Optional[str] = None,
        task_type: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "name": name,
            "model_name": model_name,
            "dataset": dataset,
            "framework": framework,
            "task_type": task_type,
            "description": description,
            "tags": tags if tags is not None else [],
            "metadata": metadata,
            "metrics": metrics,
        }
        payload = {k: v for k, v in payload.items() if v is not None or k in ["tags"]}

        response = await self.client.request("PUT", f"/evaluations/{evaluation_id}", json=payload)
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

    async def aclose(self) -> None:
        """Close the async client"""
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncEvalsClient":
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit"""
        await self.aclose()
