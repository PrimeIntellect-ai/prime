"""Prime CLI entrypoint."""

import sys

from prime_cli.command_router import app


def run() -> None:
    try:
        app.main(sys.argv[1:])
    except KeyboardInterrupt:
        print("\nOperation cancelled")
        raise SystemExit(130) from None


if __name__ == "__main__":
    run()
