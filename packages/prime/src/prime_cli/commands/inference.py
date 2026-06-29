from __future__ import annotations

import sys
from typing import Any, Dict, Iterable, List, Optional, cast

from rich.table import Table

from ..api.inference import InferenceAPIError, InferenceClient
from ..utils import (
    get_console,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)
from ..utils.formatters import format_price_per_mtok
from .inference_configs import InferenceChatConfig, InferenceModelsConfig

console = get_console()

MODELS_JSON_HELP = json_output_help(
    "Typical OpenAI schema: .object?, .data[] = {id, created, pricing?}",
    "Compatibility fallback: .models[] may be present instead of .data[]",
)

_SORT_KEYS = ("id", "input", "output")
_ORDER_KEYS = ("asc", "desc")


def _price(m: Dict[str, Any], key: str) -> Optional[float]:
    pricing = m.get("pricing") or {}
    val = pricing.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _sort_models(models: List[Dict[str, Any]], sort: str, order: str) -> List[Dict[str, Any]]:
    reverse = order == "desc"
    if sort == "id":
        return sorted(models, key=lambda m: str(m.get("id", "")).lower(), reverse=reverse)

    price_key = "input_usd_per_mtok" if sort == "input" else "output_usd_per_mtok"

    # Null pricing always sorts last regardless of order.
    def key(m: Dict[str, Any]) -> tuple:
        p = _price(m, price_key)
        if p is None:
            return (1, 0.0)
        return (0, -p if reverse else p)

    return sorted(models, key=key)


def list_models(config: InferenceModelsConfig) -> None:
    """List available models from Prime Inference (/v1/models)."""
    output = config.output
    search = config.search
    sort = config.sort
    order = config.order

    validate_output_format(output, console)
    if sort not in _SORT_KEYS:
        console.print(f"[red]Error:[/red] --sort must be one of: {', '.join(_SORT_KEYS)}")
        raise SystemExit(1)
    if order not in _ORDER_KEYS:
        console.print(f"[red]Error:[/red] --order must be one of: {', '.join(_ORDER_KEYS)}")
        raise SystemExit(1)

    try:
        client = InferenceClient(require_auth=False)
        data = client.list_models()

        # Expect OpenAI-style: {"object":"list","data":[{"id":..., ...}, ...]}
        models: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                models = data["data"]
            elif "models" in data and isinstance(data["models"], list):
                models = data["models"]  # be liberal in what we accept
        elif isinstance(data, list):
            models = data

        if search:
            needle = search.lower()
            models = [m for m in models if needle in str(m.get("id", "")).lower()]

        models = _sort_models(models, sort, order)

        if output == "json":
            if isinstance(data, dict):
                payload = dict(data)
                if "data" in payload:
                    payload["data"] = models
                elif "models" in payload:
                    payload["models"] = models
                else:
                    payload = {"data": models}
                output_data_as_json(payload, console)
            else:
                output_data_as_json(models, console)
            return

        if not models:
            console.print("[yellow]No models returned.[/yellow]")
            return

        table = Table(title="Prime Inference — Models")
        table.add_column("id", style="cyan")
        table.add_column("input $/1M tok", style="green", justify="right")
        table.add_column("output $/1M tok", style="green", justify="right")

        for m in models:
            mid = str(m.get("id", ""))
            pricing = m.get("pricing") or {}
            pin = pricing.get("input_usd_per_mtok")
            pout = pricing.get("output_usd_per_mtok")

            table.add_row(
                mid,
                format_price_per_mtok(pin),
                format_price_per_mtok(pout),
            )

        console.print(table)

    except InferenceAPIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        raise SystemExit(1)


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


def chat(config: InferenceChatConfig) -> None:
    """Send a one-shot chat message to a Prime Inference model.

    Examples:
      prime inference chat <model-id> "say hi"
      echo "explain RL in one line" | prime inference chat <model-id>
      prime inference chat <model-id> "hi" --stream
    """
    model = config.model
    message = config.message
    system = config.system
    stream = config.stream
    temperature = config.temperature
    max_tokens = config.max_tokens
    output = config.output

    if output not in ("text", "json"):
        console.print(f"[red]Error:[/red] invalid output format '{output}'. Supported: text, json")
        raise SystemExit(1)

    if stream and output == "json":
        console.print("[red]Error:[/red] --stream is not supported with --output json.")
        raise SystemExit(1)

    if message is None:
        if sys.stdin.isatty():
            console.print("[red]Error:[/red] no message provided (pass as arg or via stdin).")
            raise SystemExit(1)
        message = sys.stdin.read().strip()
        if not message:
            console.print("[red]Error:[/red] empty message from stdin.")
            raise SystemExit(1)

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
            raise SystemExit(1)
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

    except SystemExit:
        raise
    except InferenceAPIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        raise SystemExit(1)
