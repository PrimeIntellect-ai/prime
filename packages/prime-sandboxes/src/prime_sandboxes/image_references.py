"""Shared image-reference classification helpers."""

_DOCKER_HUB_HOSTS = {"docker.io", "index.docker.io", "registry-1.docker.io"}


def is_docker_hub_reference(reference: str) -> bool:
    """Return whether an image reference resolves through Docker Hub."""
    reference = reference.strip().split("@", 1)[0]
    first_segment, separator, _ = reference.partition("/")
    if not separator:
        return True
    first_segment = first_segment.lower()
    if first_segment in _DOCKER_HUB_HOSTS:
        return True
    return not ("." in first_segment or ":" in first_segment or first_segment == "localhost")
