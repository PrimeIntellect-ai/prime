from __future__ import annotations

import datetime as _dt
import sys
from typing import Any, Dict, Iterable, List, Optional, cast

import typer
from rich.table import Table

from ..api.inference import InferenceAPIError, InferenceClient
from ..utils import (
    PlainTyper,
    get_console,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)

app = PlainTyper(
    help="Run and manage Prime Inference\n\n"
    "Use `prime eval run` for environment evals with Prime Inference.",
    no_args_is_help=True,
)
console = get_console()

MODELS_JSON_HELP = json_output_help(
    "Typical OpenAI schema: .object?, .data[] = {id, created, pricing?}",
    "Compatibility fallback: .models[] may be present instead of .data[]",
)


def _fmt_created(val: str) -> str:
    """Format created timestamp for display"""
    try:
        # if epoch seconds
        ts = int(val)
        return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return val or ""


def _fmt_price(x) -> str:
    """Format USD per 1M tokens (mtok) for display."""
    if x is None:
        return ""
    try:
        f = float(x)
        return f"${f:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)


@app.command("models", epilog=MODELS_JSON_HELP)
def list_models(
    output: str = typer.Option("table", "--output", "-o", help="table|json"),
) -> None:
    """List available models from Prime Inference (/v1/models)."""
    validate_output_format(output, console)
    try:
        client = InferenceClient()
        data = client.list_models()

        # Expect OpenAI-style: {"object":"list","data":[{"id":..., ...}, ...]}
        models = []
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                models = data["data"]
            elif "models" in data and isinstance(data["models"], list):
                models = data["models"]  # be liberal in what we accept
        elif isinstance(data, list):
            models = data

        if output == "json":
            output_data_as_json(data, console)
            return

        if not models:
            console.print("[yellow]No models returned.[/yellow]")
            return

        table = Table(title="Prime Inference — Models")
        table.add_column("id", style="cyan")
        table.add_column("created", style="magenta")
        table.add_column("input $/1M tok", style="green", justify="right")
        table.add_column("output $/1M tok", style="green", justify="right")

        for m in models:
            mid = str(m.get("id", ""))
            created = _fmt_created(str(m.get("created", "")))
            pricing = m.get("pricing") or {}
            pin = pricing.get("input_usd_per_mtok")
            pout = pricing.get("output_usd_per_mtok")

            table.add_row(
                mid,
                created,
                _fmt_price(pin),
                _fmt_price(pout),
            )

        console.print(table)

    except InferenceAPIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        raise typer.Exit(1)


CHAT_JSON_HELP = json_output_help(
    "Full chat completion response: {id, model, choices[], usage?, ...}",
    "Each choice has .message.content with the assistant reply",
)


def _build_messages(message: str, system: Optional[str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})
    return messages


def _print_stream(chunks: Iterable[Dict[str, Any]]) -> None:
    for chunk in chunks:
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        piece = delta.get("content")
        if piece:
            sys.stdout.write(piece)
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()


@app.command("chat", epilog=CHAT_JSON_HELP)
def chat(
    model: str = typer.Argument(..., help="Model id (see `prime inference models`)"),
    message: Optional[str] = typer.Argument(
        None, help="User message. If omitted, reads from stdin."
    ),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    stream: bool = typer.Option(False, "--stream", help="Stream tokens as they arrive"),
    temperature: Optional[float] = typer.Option(
        None, "--temperature", "-t", help="Sampling temperature"
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", help="Maximum tokens to generate"
    ),
    output: str = typer.Option("text", "--output", "-o", help="text|json"),
) -> None:
    """Send a one-shot chat message to a Prime Inference model.

    Examples:
      prime inference chat <model-id> "say hi"
      echo "explain RL in one line" | prime inference chat <model-id>
      prime inference chat <model-id> "hi" --stream
    """
    if output not in ("text", "json"):
        console.print(f"[red]Error:[/red] invalid output format '{output}'. Supported: text, json")
        raise typer.Exit(1)

    if stream and output == "json":
        console.print("[red]Error:[/red] --stream is not supported with --output json.")
        raise typer.Exit(1)

    if message is None:
        if sys.stdin.isatty():
            console.print("[red]Error:[/red] no message provided (pass as arg or via stdin).")
            raise typer.Exit(1)
        message = sys.stdin.read().strip()
        if not message:
            console.print("[red]Error:[/red] empty message from stdin.")
            raise typer.Exit(1)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": _build_messages(message, system),
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if stream:
        payload["stream"] = True

    try:
        client = InferenceClient()
        if stream:
            stream_result = client.chat_completion(payload, stream=True)
            _print_stream(stream_result)  # type: ignore[arg-type]
            return

        with console.status(f"[bold blue]Waiting for {model}...", spinner="dots"):
            raw = client.chat_completion(payload)
        if not isinstance(raw, dict):
            console.print("[red]Error:[/red] unexpected non-JSON response from inference.")
            raise typer.Exit(1)
        result = cast(Dict[str, Any], raw)

        if output == "json":
            output_data_as_json(result, console)
            return

        choices = result.get("choices") or []
        if not choices:
            console.print("[yellow]No choices returned.[/yellow]")
            return
        content = (choices[0].get("message") or {}).get("content") or ""
        sys.stdout.write(content)
        if not content.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()

    except typer.Exit:
        raise
    except InferenceAPIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        raise typer.Exit(1)
