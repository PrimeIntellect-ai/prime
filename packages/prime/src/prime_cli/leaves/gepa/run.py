"""Raw Verifiers passthrough for ``prime gepa run``."""

Config = None
POSITIONALS = ()


def run(argv: list[str]) -> None:
    from prime_cli.commands.gepa import run_gepa_cmd

    run_gepa_cmd(argv)
