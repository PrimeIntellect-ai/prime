from __future__ import annotations

import sys
from typing import Any, Dict, Iterable, List, Optional, cast

import typer
from rich.table import Table
from rich.text import Text

from ..api.inference import InferenceAPIError, InferenceClient
from ..utils import (
    PlainTyper,
    get_console,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)
from ..utils.formatters import format_price_per_mtok

app = PlainTyper(
    help="Run and manage Prime Inference\n\n"
    "Use `prime eval run` for environment evals with Prime Inference.",
    no_args_is_help=True,
)
console = get_console()

MODELS_JSON_HELP = json_output_help(
    "Typical OpenAI schema: .object?, .data[] = {id, display_name?, created, pricing?, specs?}",
    "Compatibility fallback: .models[] may be present instead of .data[]",
)

_SORT_KEYS = ("id", "input", "output")
_ORDER_KEYS = ("asc", "desc")


def _format_cache_price(value: Any) -> str:
    """Render one side of the cache read/write pair. None means "no data"
    (renders as an em-dash); 0 is a real price (free) and renders normally."""
    return format_price_per_mtok(value) if value is not None else "—"


def _format_token_count(value: Any) -> str:
    """Compact token counts for table cells: 200000 -> '200k', 1048576 -> '1.05M'."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return "—"
    if v <= 0:
        return "—"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}".rstrip("0").rstrip(".") + "M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}".rstrip("0").rstrip(".") + "k"
    return str(v)


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


@app.command("models", epilog=MODELS_JSON_HELP)
def list_models(
    output: str = typer.Option("table", "--output", "-o", help="table|json"),
    search: Optional[str] = typer.Option(
        None, "--search", "-q", help="Case-insensitive substring match on model id"
    ),
    sort: str = typer.Option("id", "--sort", "-s", help="Sort by: id, input, output"),
    order: str = typer.Option("asc", "--order", "-d", help="Sort order (direction): asc, desc"),
) -> None:
    """List available models from Prime Inference (/v1/models)."""
    validate_output_format(output, console)
    if sort not in _SORT_KEYS:
        console.print(f"[red]Error:[/red] --sort must be one of: {', '.join(_SORT_KEYS)}")
        raise typer.Exit(1)
    if order not in _ORDER_KEYS:
        console.print(f"[red]Error:[/red] --order must be one of: {', '.join(_ORDER_KEYS)}")
        raise typer.Exit(1)

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
        # Catalog columns appear only when the endpoint serves the data, so
        # the table stays slim against older /models responses.
        show_name = any(m.get("display_name") for m in models)
        show_cache = any(
            (m.get("pricing") or {}).get("cache_read_usd_per_mtok") is not None
            or (m.get("pricing") or {}).get("cache_write_usd_per_mtok") is not None
            for m in models
        )
        show_specs = any(m.get("specs") for m in models)

        table.add_column("id", style="cyan", overflow="fold")
        if show_name:
            table.add_column("name")
        table.add_column("input $/1M tok", style="green", justify="right")
        table.add_column("output $/1M tok", style="green", justify="right")
        if show_cache:
            table.add_column("cache r/w $/1M tok", style="green", justify="right")
        if show_specs:
            table.add_column("context", justify="right")
            table.add_column("max out", justify="right")
            table.add_column("reasoning")

        for m in models:
            mid = str(m.get("id", ""))
            pricing = m.get("pricing") or {}
            specs = m.get("specs") or {}

            # Catalog values (id, display_name) are untrusted upstream data —
            # Text cells render them verbatim, so Rich markup in a name can
            # neither restyle the table nor raise MarkupError.
            row: List[Any] = [Text(mid)]
            if show_name:
                name = m.get("display_name")
                row.append(Text(str(name)) if name else "—")
            row.append(format_price_per_mtok(pricing.get("input_usd_per_mtok")))
            row.append(format_price_per_mtok(pricing.get("output_usd_per_mtok")))
            if show_cache:
                cache_read = pricing.get("cache_read_usd_per_mtok")
                cache_write = pricing.get("cache_write_usd_per_mtok")
                if cache_read is None and cache_write is None:
                    row.append("—")
                else:
                    cache_cell = (
                        f"{_format_cache_price(cache_read)} / {_format_cache_price(cache_write)}"
                    )
                    row.append(cache_cell)
            if show_specs:
                row.append(_format_token_count(specs.get("context_window")))
                row.append(_format_token_count(specs.get("max_output_tokens")))
                row.append("✓" if specs.get("supports_reasoning") else "—")
            table.add_row(*row)

        console.print(table)
        if show_specs:
            console.print("[dim]reasoning ✓ = supports reasoning effort[/dim]")

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


EVALUATE_JSON_HELP = json_output_help(
    "Full evaluation response: {answers{}, rounding?, usage?, warnings?}",
    "Each answer is {type, ...}: boolean -> {probability}, "
    "choice -> {choice, probabilities?}, score -> {score, probabilities?}",
)

_EVAL_QUESTION_TYPES = ("boolean", "choice", "score")


def _parse_question(spec: str) -> Dict[str, Any]:
    """Parse 'id:type:instructions[:extras]' into the AI SDK v4 wire shape.

    boolean: refunded:boolean:Was a refund issued?
    choice:  tone:choice:Classify the tone:professional=Formal,casual
    score:   quality:score:Rate it:Bad,Okay,Good,Excellent
    """
    parts = spec.split(":", 3)
    if len(parts) < 3:
        raise ValueError(
            f"Invalid --question '{spec}'. "
            "Expected 'id:type:instructions[:extras]' with type in {boolean, choice, score}."
        )
    qid, qtype, rest = parts[0].strip(), parts[1].strip().lower(), parts[2].lstrip()
    if not qid:
        raise ValueError(f"Invalid --question '{spec}': empty question id.")
    if qtype not in _EVAL_QUESTION_TYPES:
        raise ValueError(
            f"Invalid --question '{spec}': type must be one of {', '.join(_EVAL_QUESTION_TYPES)}."
        )
    instructions: str = rest
    extras: Optional[str] = None
    if qtype in ("choice", "score") and len(parts) == 4:
        extras = parts[3].strip()
    question: Dict[str, Any] = {"type": qtype, "instructions": instructions}
    if qtype == "choice":
        if not extras:
            raise ValueError(
                f"Invalid --question '{spec}': choice questions need criteria, "
                "'id:choice:instructions:opt1=Description,opt2' (1-255 options)."
            )
        criteria: Dict[str, Any] = {}
        for opt in extras.split(","):
            opt = opt.strip()
            if not opt:
                continue
            name, eq, desc = opt.partition("=")
            criteria[name.strip()] = desc.strip() if eq else None
        if not criteria:
            raise ValueError(f"Invalid --question '{spec}': no choice criteria parsed.")
        question["criteria"] = criteria
    elif qtype == "score":
        if not extras:
            raise ValueError(
                f"Invalid --question '{spec}': score questions need ordered levels, "
                "'id:score:instructions:Low,Mid,High' (at least two)."
            )
        levels = [level.strip() for level in extras.split(",") if level.strip()]
        if len(levels) < 2:
            raise ValueError(f"Invalid --question '{spec}': score needs at least two levels.")
        question["criteria"] = levels
    return question


def _format_answer(answer: Dict[str, Any]) -> str:
    """Compact one-line rendering of an answer for the text table."""
    atype = answer.get("type")
    if atype == "boolean":
        return f"{answer.get('probability')} (true probability)"
    if atype == "choice":
        probs = answer.get("probabilities")
        extra = f" probs={probs}" if probs else ""
        return f"{answer.get('choice')}{extra}"
    if atype == "score":
        return f"{answer.get('score')}"
    return str(answer)


@app.command("evaluate", epilog=EVALUATE_JSON_HELP)
def evaluate(
    model: str = typer.Argument(..., help="Evaluation model id (e.g. typesafe-ai/jev)"),
    state: Optional[str] = typer.Argument(
        None, help="Shared state to evaluate. If omitted, reads from stdin."
    ),
    question: List[str] = typer.Option(
        None,
        "--question",
        "-q",
        help="Typed question 'id:type:instructions[:extras]'. Repeatable. "
        "boolean: '-q ok:boolean:Was a refund issued?' "
        "choice: '-q tone:choice:Classify:formal=Formal,casual' "
        "score: '-q quality:score:Rate it:Bad,Okay,Good'",
    ),
    state_file: Optional[str] = typer.Option(
        None, "--state-file", help="Read the shared state from a file instead of an argument"
    ),
    output: str = typer.Option("text", "--output", "-o", help="text|json"),
) -> None:
    """Evaluate shared state against typed questions with an evaluation model.

    Evaluation models (e.g. typesafe-ai/jev) return structured answers
    (boolean probabilities, choices, scores) instead of chat text.

    Examples:
      prime inference evaluate typesafe-ai/jev "Refund issued." \
        -q "ok:boolean:Was a refund issued?"
      prime inference evaluate typesafe-ai/jev \
        -q "tone:choice:Classify:professional=Formal,casual=Rude" < transcript.txt
    """
    if output not in ("text", "json"):
        console.print(f"[red]Error:[/red] invalid output format '{output}'. Supported: text, json")
        raise typer.Exit(1)

    if not question:
        console.print("[red]Error:[/red] at least one --question is required.")
        raise typer.Exit(1)

    if state_file:
        try:
            state = open(state_file).read()
        except OSError as e:
            console.print(f"[red]Error:[/red] cannot read --state-file: {e}")
            raise typer.Exit(1)
    if state is None:
        if sys.stdin.isatty():
            console.print(
                "[red]Error:[/red] no state provided (pass as arg, via stdin, or --state-file)."
            )
            raise typer.Exit(1)
        state = sys.stdin.read()
    if not state.strip():
        console.print("[red]Error:[/red] state is empty.")
        raise typer.Exit(1)

    questions: Dict[str, Any] = {}
    try:
        for spec in question:
            q = _parse_question(spec)
            questions[spec.split(":", 1)[0].strip()] = q
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    payload: Dict[str, Any] = {
        "model": model,
        "state": state,
        "questions": questions,
    }

    try:
        client = InferenceClient()
        with console.status(f"[bold blue]Evaluating with {model}...", spinner="dots"):
            result = client.evaluation(payload)

        if output == "json":
            output_data_as_json(result, console)
            return

        answers = result.get("answers") or {}
        if not answers:
            console.print("[yellow]No answers returned.[/yellow]")
            return

        table = Table(title=f"Evaluation — {model}")
        table.add_column("Question")
        table.add_column("Type")
        table.add_column("Answer")
        for qid, answer in answers.items():
            if not isinstance(answer, dict):
                table.add_row(qid, "—", str(answer))
                continue
            table.add_row(qid, str(answer.get("type", "—")), _format_answer(answer))
        console.print(table)

        usage = result.get("usage") or {}
        warnings = result.get("warnings") or []
        if warnings:
            console.print(f"[yellow]Warnings:[/yellow] {warnings}")
        if usage:
            console.print(
                f"[dim]tokens: {usage.get('inputTokens', '?')} in / "
                f"{usage.get('outputTokens', '?')} out[/dim]"
            )

    except typer.Exit:
        raise
    except InferenceAPIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        raise typer.Exit(1)
