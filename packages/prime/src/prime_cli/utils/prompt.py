import re
from typing import Any, Callable, Dict, List, Optional

import questionary
import typer

from .plain import get_console

console = get_console()

# Shared look for every interactive prompt: a green accent and no reverse-video
# bar on the current row (prompt_toolkit's default "selected" style is reverse).
PROMPT_STYLE = questionary.Style(
    [
        ("qmark", "fg:green bold"),
        ("pointer", "fg:green bold"),
        ("highlighted", "fg:green bold noreverse"),
        ("selected", "fg:green noreverse"),
        ("answer", "fg:green bold"),
    ]
)


def ask_select(message: str, choices: List[Any], **kwargs: Any) -> Any:
    return questionary.select(message, choices=choices, style=PROMPT_STYLE, **kwargs).ask()


def ask_checkbox(message: str, choices: List[Any], **kwargs: Any) -> Any:
    return questionary.checkbox(message, choices=choices, style=PROMPT_STYLE, **kwargs).ask()


def ask_text(message: str, **kwargs: Any) -> Any:
    return questionary.text(message, style=PROMPT_STYLE, **kwargs).ask()


def ask_password(message: str, **kwargs: Any) -> Any:
    return questionary.password(message, style=PROMPT_STYLE, **kwargs).ask()


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


def confirm(message: str, default: bool = False) -> bool:
    """Ask a yes/no question inline. Returns ``default`` behaviour on cancel (False)."""
    return bool(questionary.confirm(message, default=default, style=PROMPT_STYLE).ask())


def confirm_or_skip(message: str, yes_flag: bool, default: bool = False) -> bool:
    """Show confirmation prompt or skip if --yes flag is provided."""
    if yes_flag:
        return True
    return confirm(message, default=default)


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
    page_size: Optional[int] = None,
) -> Dict[str, Any]:
    if not items:
        console.print(f"[yellow]{empty_message}[/yellow]")
        raise typer.Exit()

    selected = select_item_interactive(
        items, action, item_type=item_type, display_fn=display_fn, page_size=page_size
    )
    if not selected:
        console.print("\n[dim]Cancelled.[/dim]")
        raise typer.Exit()

    return selected


def select_item_interactive(
    items: List[Dict[str, Any]],
    action: str = "select",
    item_type: str = "item",
    display_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    page_size: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Display items and let the user pick one with arrow keys.

    Args:
        items: List of items to select from
        action: Action verb for the prompt (e.g., "delete", "update")
        item_type: Type of item being selected (e.g., "secret", "variable")
        display_fn: Function to format each item for display.
                   Defaults to showing 'name' and 'description' fields.
        page_size: Rows to show at once before scrolling.

    Returns:
        Selected item or None if cancelled
    """
    if not items:
        return None

    formatter = display_fn or _default_display_fn
    choices = [questionary.Choice(title=formatter(item), value=item) for item in items]
    return ask_select(f"Select a {item_type} to {action}", choices)


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
    ask = ask_password if hide_input else ask_text
    value = ask(prompt_text)
    if value is None:
        return None
    if required and not value:
        return None
    return value
