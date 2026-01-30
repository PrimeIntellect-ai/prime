import asyncio
import re
from dataclasses import dataclass
from typing import Any, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from prime_cli.core import APIError, AsyncAPIClient

console = Console()

ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
PROGRESS_BAR = re.compile(r".*\|[█▏▎▍▌▋▊▉ ]{10,}\|.*")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def filter_progress_bars(text: str) -> str:
    lines = text.splitlines()
    filtered = []
    for line in lines:
        if PROGRESS_BAR.search(line) or re.search(r"\d+%\|", line):
            if "100%" in line:
                match = re.search(r"([^|]*100%\|[█▏▎▍▌▋▊▉ ]+\|[^\n]*?)(?=\d+%\||$)", line)
                if match:
                    filtered.append(match.group(1).strip())
                else:
                    filtered.append(line)
            continue
        if line.strip():
            filtered.append(line)
    return "\n".join(filtered)


STATUS_MESSAGES = {
    "Waiting for container to start...",
    "No logs available",
    "Unable to retrieve logs",
    "Failed to fetch logs from sandbox",
}


def is_status_message(text: str) -> bool:
    stripped = text.strip()
    return any(stripped.startswith(msg) for msg in STATUS_MESSAGES)


def clean_logs(text: str) -> str:
    cleaned = filter_progress_bars(strip_ansi(text))
    if is_status_message(cleaned):
        return ""
    return cleaned


@dataclass
class HostedEvalConfig:
    environment_id: str
    inference_model: str
    num_examples: int
    rollouts_per_example: int
    env_args: Optional[dict[str, str]] = None
    name: Optional[str] = None
    timeout_minutes: Optional[int] = None
    allow_sandbox_access: bool = False
    allow_instances_access: bool = False
    custom_secrets: Optional[dict[str, str]] = None


@dataclass
class HostedEvalResult:
    evaluation_id: str
    status: str
    viewer_url: Optional[str]
    total_samples: int
    avg_score: Optional[float]
    min_score: Optional[float]
    max_score: Optional[float]
    error_message: Optional[str] = None
    logs: Optional[str] = None


async def create_hosted_evaluation(
    client: AsyncAPIClient,
    config: HostedEvalConfig,
) -> dict[str, Any]:
    eval_config: dict[str, Any] = {
        "num_examples": config.num_examples,
        "rollouts_per_example": config.rollouts_per_example,
        "allow_sandbox_access": config.allow_sandbox_access,
        "allow_instances_access": config.allow_instances_access,
    }

    if config.env_args:
        eval_config["env_args"] = config.env_args

    if config.timeout_minutes:
        eval_config["timeout_minutes"] = config.timeout_minutes

    if config.custom_secrets:
        eval_config["custom_secrets"] = config.custom_secrets

    payload: dict[str, Any] = {
        "environment_ids": [config.environment_id],
        "inference_model": config.inference_model,
        "eval_config": eval_config,
    }

    if config.name:
        payload["name"] = config.name

    return await client.post("/hosted-evaluations", json=payload)


async def get_evaluation(client: AsyncAPIClient, evaluation_id: str) -> dict[str, Any]:
    return await client.get(f"/evaluations/{evaluation_id}")


async def get_evaluation_logs(client: AsyncAPIClient, evaluation_id: str) -> str:
    try:
        response = await client.get(f"/hosted-evaluations/{evaluation_id}/logs")
        return response.get("logs") or ""
    except APIError:
        return ""


async def run_hosted_evaluation(
    config: HostedEvalConfig,
    poll_interval: float = 10.0,
    stream_logs: bool = True,
) -> HostedEvalResult:
    async with AsyncAPIClient() as client:
        console.print(
            f"[cyan]Creating hosted evaluation for environment {config.environment_id}[/cyan]"
        )
        console.print(f"[dim]Model: {config.inference_model}[/dim]")
        console.print(
            f"[dim]Configuration: num_examples={config.num_examples}, "
            f"rollouts_per_example={config.rollouts_per_example}[/dim]"
        )
        console.print()

        result = await create_hosted_evaluation(client, config)
        evaluation_id = result.get("evaluation_id")

        if not evaluation_id:
            raise APIError(f"Failed to get evaluation ID from response. Response: {result}")

        console.print(f"[green]✓ Created hosted evaluation:[/green] {evaluation_id}")
        console.print()

        last_logs = ""
        terminal_statuses = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"}
        consecutive_errors = 0

        with Live(
            Panel(
                Text.assemble(
                    "[cyan]⠋[/cyan]",
                    " Waiting for evaluation to start...",
                ),
                title="[bold]Hosted Evaluation[/bold]",
                border_style="blue",
            ),
            refresh_per_second=4,
            console=console,
        ) as live:
            first_poll = True
            while True:
                if not first_poll:
                    await asyncio.sleep(poll_interval)
                first_poll = False

                try:
                    eval_data = await get_evaluation(client, evaluation_id)
                    status = eval_data.get("status", "UNKNOWN")
                    consecutive_errors = 0

                    status_color = {
                        "PENDING": "yellow",
                        "RUNNING": "cyan",
                        "COMPLETED": "green",
                        "FAILED": "red",
                        "TIMEOUT": "red",
                        "CANCELLED": "yellow",
                    }.get(status, "white")

                    total_samples = eval_data.get("total_samples", 0)
                    status_text = Text.assemble(
                        "Status: ",
                        (status, status_color),
                        f" | Samples: {total_samples}",
                    )
                    live.update(
                        Panel(
                            status_text,
                            title="[bold]Hosted Evaluation[/bold]",
                            border_style="blue",
                        )
                    )

                    if stream_logs and status in ("RUNNING", "COMPLETED", "FAILED"):
                        raw_logs = await get_evaluation_logs(client, evaluation_id)
                        logs = clean_logs(raw_logs) if raw_logs else ""

                        if logs and logs != last_logs:
                            old_lines = last_logs.splitlines() if last_logs else []
                            new_lines = logs.splitlines()

                            # Calculate new lines to print (avoid duplicates)
                            if not last_logs:
                                lines_to_print = new_lines
                            else:
                                overlap = 0
                                max_overlap = min(len(old_lines), len(new_lines))
                                for i in range(1, max_overlap + 1):
                                    if old_lines[-i:] == new_lines[:i]:
                                        overlap = i
                                lines_to_print = new_lines[overlap:]

                            # Print new lines using console.out which doesn't interfere with Live
                            if lines_to_print:
                                for line in lines_to_print:
                                    live.console.print(line)

                            last_logs = logs

                    if status in terminal_statuses:
                        live.stop()
                        break

                except APIError as e:
                    consecutive_errors += 1
                    if "429" in str(e):
                        if consecutive_errors >= 3:
                            live.console.print("[yellow]Rate limited. Waiting 30s...[/yellow]")
                            await asyncio.sleep(30)
                        else:
                            await asyncio.sleep(10)
                        continue
                    raise

        eval_data = await get_evaluation(client, evaluation_id)
        final_logs = await get_evaluation_logs(client, evaluation_id)

        return HostedEvalResult(
            evaluation_id=evaluation_id,
            status=eval_data.get("status", "UNKNOWN"),
            viewer_url=eval_data.get("viewer_url"),
            total_samples=eval_data.get("total_samples", 0),
            avg_score=eval_data.get("avg_score"),
            min_score=eval_data.get("min_score"),
            max_score=eval_data.get("max_score"),
            error_message=eval_data.get("error_message"),
            logs=final_logs,
        )


def print_hosted_result(result: HostedEvalResult) -> None:
    console.print()
    console.rule("[bold]Hosted Evaluation Results[/bold]")
    console.print()
    console.print(f"[cyan]Evaluation ID:[/cyan] {result.evaluation_id}")

    status_color = {
        "COMPLETED": "green",
        "FAILED": "red",
        "TIMEOUT": "red",
        "CANCELLED": "yellow",
    }.get(result.status, "white")
    console.print(f"[cyan]Status:[/cyan] [{status_color}]{result.status}[/{status_color}]")
    console.print(f"[cyan]Total samples:[/cyan] {result.total_samples}")

    if result.avg_score is not None:
        console.print(f"[cyan]Average score:[/cyan] {result.avg_score:.4f}")
    if result.min_score is not None:
        console.print(f"[cyan]Min score:[/cyan] {result.min_score:.4f}")
    if result.max_score is not None:
        console.print(f"[cyan]Max score:[/cyan] {result.max_score:.4f}")

    console.print()

    if result.viewer_url:
        console.print(f"[bold green]View results:[/bold green] {result.viewer_url}")

    if result.error_message:
        console.print(f"\n[red]Error:[/red] {result.error_message}")

    console.print()
