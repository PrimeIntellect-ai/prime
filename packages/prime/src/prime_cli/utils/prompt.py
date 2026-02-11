import re
from typing import Any, Callable, Dict, List, Optional

import typer
from rich.console import Console

console = Console()

_ENV_VAR_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


def validate_env_var_name(name: str, item_type: str = "secret") -> bool:
    if _ENV_VAR_NAME_PATTERN.match(name):
        return True

    console.print(f"[red]Invalid {item_type} name: '{name}'[/red]")
    console.print(
        f"[dim]{item_type.capitalize()} names must be uppercase letters, digits, "
        "and underscores, starting with a letter (e.g., MY_SECRET, API_KEY_2).[/dim]"
    )
    return False


def confirm_or_skip(message: str, yes_flag: bool, default: bool = False) -> bool:
    """Show confirmation prompt or skip if --yes flag is provided."""
    if yes_flag:
        return True
    return bool(typer.confirm(message, default=default))


def any_provided(*values: Any) -> bool:
    return any(v is not None for v in values)


def _default_display_fn(item: Dict[str, Any]) -> str:
    """Default display function for interactive selection."""
    name = item.get("name", "")
    desc = item.get("description") or ""
    return f"{name} - {desc}" if desc else name


def require_selection(
    items: List[Dict[str, Any]],
    action: str,
    empty_message: str,
    item_type: str = "item",
    display_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> Dict[str, Any]:
    if not items:
        console.print(f"[yellow]{empty_message}[/yellow]")
        raise typer.Exit()

    selected = select_item_interactive(items, action, item_type=item_type, display_fn=display_fn)
    if not selected:
        console.print("\n[dim]Cancelled.[/dim]")
        raise typer.Exit()

    return selected


def select_item_interactive(
    items: List[Dict[str, Any]],
    action: str = "select",
    item_type: str = "item",
    display_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> Optional[Dict[str, Any]]:
    """Display items and let user select one interactively.

    Args:
        items: List of items to select from
        action: Action verb for the prompt (e.g., "delete", "update")
        item_type: Type of item being selected (e.g., "secret", "variable")
        display_fn: Function to format each item for display.
                   Defaults to showing 'name' and 'description' fields.

    Returns:
        Selected item or None if cancelled
    """
    if not items:
        return None

    formatter = display_fn or _default_display_fn

    console.print(f"\n[bold]Select a {item_type} to {action}:[/bold]\n")
    for i, item in enumerate(items, 1):
        console.print(f"  {i}. {formatter(item)}")
    console.print()

    while True:
        try:
            selection_str = typer.prompt("Select (empty to cancel)", default="")
            if not selection_str:
                return None
            selection = int(selection_str)
            if 1 <= selection <= len(items):
                return items[selection - 1]
            console.print(f"[red]Please enter a number between 1 and {len(items)}[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")
        except KeyboardInterrupt:
            return None


def prompt_for_value(
    prompt_text: str,
    required: bool = True,
    hide_input: bool = False,
) -> Optional[str]:
    """Prompt for a value with optional cancellation.

    Args:
        prompt_text: Text to display for the prompt
        required: If True, empty input cancels. If False, empty input is allowed.
        hide_input: If True, hide the input (for passwords/secrets)

    Returns:
        The entered value, or None if cancelled
    """
    try:
        suffix = " (empty to cancel)" if required else ""
        value = typer.prompt(f"{prompt_text}{suffix}", default="", hide_input=hide_input)
        if required and not value:
            return None
        return value
    except KeyboardInterrupt:
        return None
