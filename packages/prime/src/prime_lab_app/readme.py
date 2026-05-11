"""README cleaning and link extraction for environment source views."""

from __future__ import annotations

import re
from html import unescape
from urllib.parse import urlparse


def readme_links(readme: str, env_slug: str = "") -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    normalized = readme_markdown(readme, env_slug)
    for label, url in re.findall(r"(?<!!)\[([^\]]+)\]\((https?://[^)\s]+)\)", normalized):
        url = _clean_url(url)
        if _skip_readme_url(url) or url in seen:
            continue
        seen.add(url)
        links.append((_friendly_link_label(label, url, env_slug), url))
    for match in re.finditer(r"(?<!\]\()(https?://[^\s<>'\")]+)", normalized):
        url = _clean_url(match.group(1))
        if _skip_readme_url(url) or url in seen:
            continue
        seen.add(url)
        links.append((_friendly_link_label("", url, env_slug), url))
    return links


def readme_markdown(readme: str, env_slug: str = "") -> str:
    text = _convert_html_anchors(readme, env_slug)
    text = _strip_markdown_images(text)
    text = _strip_html_images(text)
    text = re.sub(r"</?(?:br|p|div)\b[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = _linkify_raw_urls(text, env_slug)
    if len(text) > 12_000:
        text = text[:12_000].rstrip() + "\n..."
    return text.rstrip() or " "


def _convert_html_anchors(readme: str, env_slug: str) -> str:
    pattern = re.compile(r"<a\b(?P<attrs>[^>]*)>(?P<body>.*?)</a>", re.IGNORECASE | re.DOTALL)

    def replace(match: re.Match[str]) -> str:
        url = _html_attr(match.group("attrs"), "href")
        if not url or _skip_readme_url(url):
            return ""
        label = _friendly_link_label(_html_body_label(match.group("body")), url, env_slug)
        return f"[{label}]({_clean_url(url)})"

    return pattern.sub(replace, readme)


def _strip_markdown_images(readme: str) -> str:
    return re.sub(r"!\[([^\]]*)\]\((https?://[^)\s]+)\)", r"\1", readme)


def _strip_html_images(readme: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return _html_attr(match.group(1), "alt")

    return re.sub(r"<img\b([^>]*)>", replace, readme, flags=re.IGNORECASE | re.DOTALL)


def _linkify_raw_urls(readme: str, env_slug: str) -> str:
    def replace(match: re.Match[str]) -> str:
        url = _clean_url(match.group(1))
        if _skip_readme_url(url):
            return ""
        return f"[{_friendly_link_label('', url, env_slug)}]({url})"

    return re.sub(r"(?<!\]\()(https?://[^\s<>'\")]+)", replace, readme)


def _html_attr(attrs: str, name: str) -> str:
    pattern = re.compile(rf"""{name}\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
    match = pattern.search(attrs)
    return unescape(match.group(1)).strip() if match else ""


def _html_body_label(body: str) -> str:
    image_alt = _html_attr(body, "alt")
    if image_alt:
        return image_alt
    body = re.sub(r"<[^>]+>", "", body)
    return unescape(" ".join(body.split())).strip()


def _friendly_link_label(label: str, url: str, env_slug: str) -> str:
    cleaned = unescape(re.sub(r"<[^>]+>", "", label)).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned and not cleaned.startswith("http"):
        return cleaned

    parsed = urlparse(url)
    host = parsed.netloc.removeprefix("www.")
    env_name = env_slug.split("/", 1)[-1] if env_slug else ""
    path_parts = [part for part in parsed.path.split("/") if part]
    if host == "github.com":
        if "environments" in path_parts:
            env_index = path_parts.index("environments")
            if len(path_parts) > env_index + 1:
                return f"{path_parts[env_index + 1]} source"
        return "GitHub"
    if host.endswith("primeintellect.ai"):
        return f"{env_name} on platform" if env_name else "View on platform"
    if env_name and env_name in parsed.path:
        return env_name
    return host or url


def _clean_url(url: str) -> str:
    return url.strip().rstrip(".,;")


def _skip_readme_url(url: str) -> bool:
    host = urlparse(url).netloc.removeprefix("www.")
    return host in {"img.shields.io", "shields.io"}


def truncate_label(label: str, max_width: int) -> str:
    if len(label) <= max_width:
        return label
    return label[: max_width - 1].rstrip() + "…"
