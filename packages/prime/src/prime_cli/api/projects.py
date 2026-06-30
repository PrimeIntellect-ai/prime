"""Lab Projects API client."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from prime_cli.core import APIClient, APIError


class Project(BaseModel):
    id: str
    name: str
    slug: str
    description: Optional[str] = None
    status: str
    user_id: Optional[str] = Field(None, alias="userId")
    team_id: Optional[str] = Field(None, alias="teamId")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    archived_at: Optional[datetime] = Field(None, alias="archivedAt")

    model_config = ConfigDict(populate_by_name=True)


class ProjectsClient:
    def __init__(self, client: APIClient) -> None:
        self.client = client

    def list(
        self,
        team_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Project], int]:
        params: dict[str, object] = {
            "limit": limit,
            "offset": offset,
        }
        if team_id:
            params["teamId"] = team_id

        try:
            response = self.client.get("/projects", params=params)
            data = response.get("data", [])
            total = int(response.get("totalCount", response.get("total_count", len(data))))
            return [Project.model_validate(item) for item in data], total
        except Exception as exc:
            raise APIError(f"Failed to list projects: {exc}") from exc

    def create(
        self,
        name: str,
        slug: Optional[str] = None,
        description: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> Project:
        payload = {
            "name": name,
            "slug": slug,
            "description": description,
            "teamId": team_id,
        }
        payload = {key: value for key, value in payload.items() if value is not None}

        try:
            response = self.client.post("/projects", json=payload)
            return Project.model_validate(response["data"])
        except Exception as exc:
            raise APIError(f"Failed to create project: {exc}") from exc

    def get(self, project_ref: str, team_id: Optional[str] = None) -> Project:
        params = {"teamId": team_id} if team_id else None
        try:
            response = self.client.get(f"/projects/{project_ref}", params=params)
            return Project.model_validate(response["data"])
        except Exception as exc:
            raise APIError(f"Failed to get project: {exc}") from exc

    def update(
        self,
        project_ref: str,
        name: Optional[str] = None,
        slug: Optional[str] = None,
        description: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> Project:
        payload = {
            "name": name,
            "slug": slug,
            "description": description,
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        params = {"teamId": team_id} if team_id else None
        try:
            response = self.client.patch(f"/projects/{project_ref}", json=payload, params=params)
            return Project.model_validate(response["data"])
        except Exception as exc:
            raise APIError(f"Failed to update project: {exc}") from exc
