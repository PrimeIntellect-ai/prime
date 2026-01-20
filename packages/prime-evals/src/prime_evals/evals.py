import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .core import APIError, AsyncAPIClient
from .exceptions import EvalsAPIError, InvalidEvaluationError


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429
    return isinstance(exc, httpx.RequestError)


def _build_user_agent() -> str:
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
        max_payload_bytes: int = 512 * 1024,
        max_workers: int = 4,
    ) -> Dict[str, Any]:
        """Push evaluation samples in adaptive batches with concurrent uploads."""
        if not samples:
            return {"samples_pushed": 0}
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        batches = self._build_batches(samples, max_payload_bytes)
        total_samples_pushed = 0
        errors = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._upload_batch, evaluation_id, b): i
                for i, b in enumerate(batches)
            }
            for future in as_completed(futures):
                try:
                    total_samples_pushed += future.result()
                except Exception as e:
                    errors.append(f"Batch {futures[future] + 1}: {e}")

        if errors:
            raise EvalsAPIError(f"Failed to push samples: {'; '.join(errors)}")

        return {"samples_pushed": total_samples_pushed}

    def _upload_batch(self, evaluation_id: str, batch: List[Dict[str, Any]]) -> int:
        """Upload a single batch of samples with retry on rate limit."""
        url = f"{self.client.base_url}/api/v1/evaluations/{evaluation_id}/samples"
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": _build_user_agent(),
        }
        if self.client.api_key:
            headers["Authorization"] = f"Bearer {self.client.api_key}"

        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=16),
            reraise=True,
        )
        def do_upload() -> int:
            response = httpx.post(url, json={"samples": batch}, headers=headers, timeout=30.0)
            response.raise_for_status()
            return len(batch)

        try:
            return do_upload()
        except httpx.HTTPStatusError as e:
            raise EvalsAPIError(f"HTTP {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise EvalsAPIError(f"Request failed: {e}") from e

    def _build_batches(
        self, samples: List[Dict[str, Any]], max_payload_bytes: int
    ) -> List[List[Dict[str, Any]]]:
        """Build batches that fit within payload size limit."""
        batches: List[List[Dict[str, Any]]] = []
        current_batch: List[Dict[str, Any]] = []
        current_bytes = 20

        for idx, sample in enumerate(samples):
            sample_size = len(json.dumps(sample)) + 1

            if sample_size + 20 > max_payload_bytes:
                raise EvalsAPIError(
                    f"Sample {idx} exceeds maximum payload size "
                    f"({sample_size} bytes > {max_payload_bytes - 20} bytes limit)"
                )

            if current_bytes + sample_size > max_payload_bytes and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_bytes = 20

            current_batch.append(sample)
            current_bytes += sample_size

        if current_batch:
            batches.append(current_batch)

        return batches

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
        max_payload_bytes: int = 512 * 1024,
        max_concurrent: int = 4,
    ) -> Dict[str, Any]:
        """Push evaluation samples in adaptive batches with concurrent uploads."""
        if not samples:
            return {"samples_pushed": 0}
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")

        batches = self._build_batches(samples, max_payload_bytes)
        semaphore = asyncio.Semaphore(max_concurrent)
        errors: List[str] = []

        base_url = self.client.base_url
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": _build_user_agent(),
        }
        if self.client.api_key:
            headers["Authorization"] = f"Bearer {self.client.api_key}"

        async def upload_batch(idx: int, batch: List[Dict[str, Any]]) -> int:
            url = f"{base_url}/api/v1/evaluations/{evaluation_id}/samples"

            @retry(
                retry=retry_if_exception(_is_retryable),
                stop=stop_after_attempt(5),
                wait=wait_exponential(multiplier=1, min=1, max=16),
                reraise=True,
            )
            async def do_upload() -> int:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(url, json={"samples": batch}, headers=headers)
                    response.raise_for_status()
                    return len(batch)

            async with semaphore:
                try:
                    return await do_upload()
                except httpx.HTTPStatusError as e:
                    errors.append(f"Batch {idx + 1}: HTTP {e.response.status_code}")
                    return 0
                except httpx.RequestError as e:
                    errors.append(f"Batch {idx + 1}: {e}")
                    return 0

        results = await asyncio.gather(*[upload_batch(i, b) for i, b in enumerate(batches)])

        if errors:
            raise EvalsAPIError(f"Failed to push samples: {'; '.join(errors)}")

        return {"samples_pushed": sum(results)}

    def _build_batches(
        self, samples: List[Dict[str, Any]], max_payload_bytes: int
    ) -> List[List[Dict[str, Any]]]:
        """Build batches that fit within payload size limit."""
        batches: List[List[Dict[str, Any]]] = []
        current_batch: List[Dict[str, Any]] = []
        current_bytes = 20

        for idx, sample in enumerate(samples):
            sample_size = len(json.dumps(sample)) + 1

            if sample_size + 20 > max_payload_bytes:
                raise EvalsAPIError(
                    f"Sample {idx} exceeds maximum payload size "
                    f"({sample_size} bytes > {max_payload_bytes - 20} bytes limit)"
                )

            if current_bytes + sample_size > max_payload_bytes and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_bytes = 20

            current_batch.append(sample)
            current_bytes += sample_size

        if current_batch:
            batches.append(current_batch)

        return batches

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
