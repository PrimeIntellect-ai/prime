from typing import Any, Dict, Optional

from prime_cli.core import Config


def optional_team_params(config: Config) -> Optional[Dict[str, Any]]:
    return {"teamId": config.team_id} if config.team_id else None
