"""Run configs pulled from the prime-rl repo, so `prime train` can launch a
recipe without the user downloading TOML files first.

`recipe:<model>[/<data>]`, anywhere a config path is accepted, expands to
prime-rl's recipe stack `recipes/<model>/rl.toml`, its compute overlay under
`recipes/<model>/gpu/` and, when given, `recipes/data/<data>.toml`.

Recipes come from prime-rl's main branch, resolved to a commit once; every
file is read at that commit, so one invocation never mixes files from two commits.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import quote

import httpx

from prime_cli.core import default_user_agent

PRIME_RL_REPO = "PrimeIntellect-ai/prime-rl"
DEFAULT_RECIPES_REF = "main"

RECIPE_PREFIX = "recipe:"

_GITHUB_API = "https://api.github.com"
_GITHUB_RAW = "https://raw.githubusercontent.com"
_RECIPES_DIR = "recipes"
_DEFAULT_GPU = "h200"
# Recipe-folder entries that are not models.
_NON_MODEL_DIRS = frozenset({"data"})
_TIMEOUT = 15.0
# `recipe:<model>` or `recipe:<model>/<data>`.
_SPEC_RE = re.compile(r"\A([A-Za-z0-9][A-Za-z0-9._-]*)(?:/([A-Za-z0-9][A-Za-z0-9._-]*))?\Z")
_SHA_RE = re.compile(r"\A[0-9a-f]{40}\Z")


def is_remote_config(arg: str) -> bool:
    return arg.startswith(RECIPE_PREFIX)


def parse_recipe_spec(arg: str) -> tuple[str, Optional[str]]:
    """`recipe:<model>[/<data>]` -> (model, data), checked before any request."""
    match = _SPEC_RE.match(arg[len(RECIPE_PREFIX) :])
    if not match:
        raise ValueError(
            f"{arg!r}: expected recipe:<model> or recipe:<model>/<data>, "
            "e.g. recipe:muse-glimmer-30b/swe. See `prime train recipes`."
        )
    return match.group(1), match.group(2)


@dataclass
class Recipe:
    model: str
    gpus: List[str]
    data: List[str]

    @property
    def gpu(self) -> Optional[str]:
        """The GPU overlay `recipe:` uses: h200, else the only one, else None
        (several overlays and none is h200: not launchable by name)."""
        if _DEFAULT_GPU in self.gpus:
            return _DEFAULT_GPU
        if len(self.gpus) == 1:
            return self.gpus[0]
        return None

    @property
    def launchable(self) -> bool:
        return not self.gpus or self.gpu is not None


def _headers(accept: str) -> Dict[str, str]:
    headers = {"Accept": accept, "User-Agent": default_user_agent()}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get(url: str, accept: str, what: str) -> httpx.Response:
    try:
        resp = httpx.get(url, headers=_headers(accept), timeout=_TIMEOUT, follow_redirects=True)
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        raise ValueError(f"Could not reach GitHub to fetch {what}: {exc}") from exc
    if resp.status_code in (403, 429):
        has_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        hint = "" if has_token else " Set GITHUB_TOKEN to lift GitHub's anonymous rate limit."
        raise ValueError(f"GitHub refused the request for {what} (HTTP {resp.status_code}).{hint}")
    return resp


@dataclass
class PrimeRLSource:
    """prime-rl at one commit. Network calls are lazy and cached."""

    ref: str = DEFAULT_RECIPES_REF
    _sha: Optional[str] = None
    _paths: Optional[List[str]] = None

    @property
    def sha(self) -> str:
        if self._sha is None:
            what = f"prime-rl ref {self.ref!r}"
            resp = _get(
                f"{_GITHUB_API}/repos/{PRIME_RL_REPO}/commits/{self.ref}",
                "application/vnd.github.sha",
                what,
            )
            if resp.status_code in (404, 422):
                raise ValueError(f"{what} not found in {PRIME_RL_REPO}.")
            if resp.status_code != 200:
                raise ValueError(f"GitHub returned HTTP {resp.status_code} for {what}.")
            sha = resp.text.strip()
            if not _SHA_RE.match(sha):
                raise ValueError(f"Unexpected GitHub response resolving {what}.")
            self._sha = sha
        return self._sha

    @property
    def label(self) -> str:
        return f"prime-rl@{self.sha[:7]}"

    @property
    def paths(self) -> List[str]:
        if self._paths is None:
            what = f"the file list of prime-rl at {self.sha[:7]}"
            resp = _get(
                f"{_GITHUB_API}/repos/{PRIME_RL_REPO}/git/trees/{self.sha}?recursive=1",
                "application/vnd.github+json",
                what,
            )
            if resp.status_code != 200:
                raise ValueError(f"GitHub returned HTTP {resp.status_code} for {what}.")
            try:
                tree = resp.json()["tree"]
                self._paths = [e["path"] for e in tree if e.get("type") == "blob"]
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Unexpected GitHub response for {what}.") from exc
        return self._paths

    def read(self, path: str) -> str:
        if path not in self.paths:
            raise ValueError(f"{path} not found in prime-rl at {self.ref} ({self.sha[:7]}).")
        what = f"{path} from prime-rl"
        url = f"{_GITHUB_RAW}/{PRIME_RL_REPO}/{self.sha}/{quote(path, safe='/')}"
        resp = _get(url, "text/plain", what)
        if resp.status_code != 200:
            raise ValueError(f"GitHub returned HTTP {resp.status_code} for {what}.")
        return resp.text

    def recipes(self) -> List[Recipe]:
        prefix = f"{_RECIPES_DIR}/"
        data = sorted(
            p[len(prefix + "data/") : -len(".toml")]
            for p in self.paths
            if p.startswith(prefix + "data/") and p.endswith(".toml") and p.count("/") == 2
        )
        found: List[Recipe] = []
        for path in self.paths:
            parts = path.split("/")
            if len(parts) == 3 and parts[0] == _RECIPES_DIR and parts[2] == "rl.toml":
                model = parts[1]
                if model in _NON_MODEL_DIRS:
                    continue
                gpus = sorted(
                    p.split("/")[-1][: -len(".toml")]
                    for p in self.paths
                    if p.startswith(f"{prefix}{model}/gpu/")
                    and p.endswith(".toml")
                    and p.count("/") == 3
                )
                found.append(Recipe(model=model, gpus=gpus, data=data))
        return sorted(found, key=lambda r: r.model)

    def expand(self, arg: str) -> List[str]:
        """Repo paths a `recipe:` argument stands for, in merge order."""
        model, data = parse_recipe_spec(arg)
        recipes = {r.model: r for r in self.recipes()}
        recipe = recipes.get(model)
        if recipe is None:
            known = ", ".join(recipes) or "none"
            raise ValueError(
                f"Unknown recipe {model!r} in prime-rl at {self.ref} ({self.sha[:7]}). "
                f"Available: {known}. See `prime train recipes`."
            )
        paths = [f"{_RECIPES_DIR}/{model}/rl.toml"]
        if recipe.gpus:
            gpu = recipe.gpu
            if gpu is None:
                raise ValueError(
                    f"Recipe {model!r} has several GPU overlays ({', '.join(recipe.gpus)}) "
                    f"and none is {_DEFAULT_GPU}; download the recipe from prime-rl's "
                    f"{_RECIPES_DIR}/{model}/ and pass the files directly."
                )
            paths.append(f"{_RECIPES_DIR}/{model}/gpu/{gpu}.toml")
        if data:
            if data not in recipe.data:
                raise ValueError(
                    f"Unknown data overlay {data!r} for recipe {model!r}. "
                    f"Available: {', '.join(recipe.data) or 'none'}."
                )
            paths.append(f"{_RECIPES_DIR}/data/{data}.toml")
        return paths
