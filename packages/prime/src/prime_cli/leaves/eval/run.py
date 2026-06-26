"""Raw Verifiers passthrough for ``prime eval run``."""

Config = None
POSITIONALS = ()


def run(argv: list[str]) -> None:
    from prime_cli.commands.evals import run_eval_cmd

    run_eval_cmd(argv)
