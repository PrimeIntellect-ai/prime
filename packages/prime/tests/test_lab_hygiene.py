from __future__ import annotations

from pathlib import Path

from prime_cli.lab_agents import agent_project_skills_dirs, known_agent_names
from prime_cli.lab_hygiene import LAB_GITIGNORE_PATTERNS, LAB_TRACKED_PREFIXES


def test_every_agent_project_skill_dir_is_covered_by_hygiene(tmp_path: Path) -> None:
    base = tmp_path.resolve()
    gitignore = set(LAB_GITIGNORE_PATTERNS)
    tracked = set(LAB_TRACKED_PREFIXES)

    for agent in known_agent_names():
        for skill_dir in agent_project_skills_dirs(agent, base):
            relative = skill_dir.relative_to(base).as_posix()
            assert f"/{relative}/" in gitignore, (
                f"{agent}: {relative} is created by setup but not in LAB_GITIGNORE_PATTERNS"
            )
            assert f"{relative}/" in tracked, (
                f"{agent}: {relative} is created by setup but not in LAB_TRACKED_PREFIXES"
            )
