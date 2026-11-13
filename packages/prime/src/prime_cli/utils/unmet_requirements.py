"""Prints, one per line, the requirements of the distributions in a `--target` directory
that nothing importable satisfies: `unmet_requirements.py <target>`. Runs in the workspace
interpreter, not the CLI's, with the target first on its path."""

import sys
from collections import defaultdict
from importlib import metadata

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def unmet_requirements(target: str) -> list[str]:
    distributions = {
        canonicalize_name(distribution.metadata["Name"]): [
            Requirement(line) for line in distribution.requires or []
        ]
        for distribution in metadata.distributions(path=[target])
    }
    # A distribution's extra-gated requirements apply when something in the target asks for
    # that extra, e.g. `foo[bar]`.
    extras = defaultdict(lambda: {""})
    for requirements in distributions.values():
        for requirement in requirements:
            extras[canonicalize_name(requirement.name)].update(requirement.extras)

    unmet = set()
    for name, requirements in distributions.items():
        for requirement in requirements:
            marker = requirement.marker
            if marker and not any(marker.evaluate({"extra": extra}) for extra in extras[name]):
                continue
            # The marker holds here; printed with it, an installer would re-evaluate it
            # without the extra and skip the requirement.
            requirement.marker = None
            try:
                version = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                unmet.add(str(requirement))
                continue
            if not requirement.specifier.contains(version, prereleases=True):
                unmet.add(str(requirement))
    return sorted(unmet)


def main() -> None:
    target = sys.argv[1]
    sys.path.insert(0, target)
    sys.stdout.writelines(f"{requirement}\n" for requirement in unmet_requirements(target))


if __name__ == "__main__":
    main()
