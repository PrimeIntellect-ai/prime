import base64
import time
import webbrowser
from typing import Optional

import httpx
import typer
from click.exceptions import Abort
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives.asymmetric import rsa

from prime_cli.core import Config, dpop

from ..client import APIClient, APIError
from ..utils import PlainTyper, get_console, require_persistent_context
from .teams import fetch_teams

app = PlainTyper(help="Login to Prime Intellect")
console = get_console()


def fetch_and_select_team(client: APIClient, config: Config) -> None:
    """Fetch user's teams and prompt for selection."""
    try:
        teams = fetch_teams(client)

        if not teams:
            config.set_team(None)
            config.update_current_environment_file()
            return

        console.print("\n[bold]Select:[/bold]\n")
        console.print("  [cyan](1)[/cyan] Personal")
        for idx, team in enumerate(teams, start=2):
            role = team.get("role", "member")
            role_display = role.lower()
            role_badge = (
                f"[yellow](role: {role_display})[/yellow]"
                if role == "admin"
                else f"[dim](role: {role_display})[/dim]"
            )
            console.print(f"  [cyan]({idx})[/cyan] {team.get('name', 'Unknown')} {role_badge}")

        console.print("\n[dim]You can always change this by running 'prime login' again.[/dim]")

        while True:
            try:
                selection = typer.prompt("Select", type=int, default=1)

                if selection == 1:
                    config.set_team(None)
                    config.update_current_environment_file()
                    console.print("[green]Using personal account.[/green]")
                    return

                if 2 <= selection <= len(teams) + 1:
                    selected_team = teams[selection - 2]
                    team_id = selected_team.get("teamId")
                    team_name = selected_team.get("name", "Unknown")
                    team_role = selected_team.get("role", "member")

                    if not team_id:
                        console.print("[yellow]Invalid team. Using personal account.[/yellow]")
                        config.set_team(None)
                        config.update_current_environment_file()
                        return

                    config.set_team(team_id, team_name=team_name, team_role=team_role)
                    config.update_current_environment_file()
                    console.print(f"[green]Using team '{team_name}'.[/green]")
                    return

                console.print(f"[red]Invalid selection. Enter 1-{len(teams) + 1}.[/red]")
            except Abort:
                config.set_team(None)
                config.update_current_environment_file()
                return

    except Abort:
        config.set_team(None)
        config.update_current_environment_file()
    except (APIError, Exception):
        config.set_team(None)
        config.update_current_environment_file()


# Flag value -> field on the challenge request. The minted key carries these
# as per-key caps, enforced on top of the account's own limits.
KEY_LIMIT_FIELDS = {
    "max_concurrent_sandboxes": "maxConcurrentSandboxes",
    "max_sandbox_creations_per_hour": "maxSandboxCreationsPerHour",
    "max_sandbox_cpu_cores": "maxSandboxCpuCores",
    "max_sandbox_gpu_count": "maxSandboxGpuCount",
    "max_concurrent_tunnels": "maxConcurrentTunnels",
    "max_tunnel_creations_per_hour": "maxTunnelCreationsPerHour",
    "max_tunnel_ttl_hours": "maxTunnelTtlHours",
}
KEY_LIMITS_PANEL = "API Key Limits"


def generate_ephemeral_keypair() -> tuple[rsa.RSAPrivateKey, str]:
    """Generate a temporary RSA key pair for secure communication"""
    try:
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()

        # Serialize public key to PEM format
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

        return private_key, public_pem
    except Exception as e:
        console.print(f"[red]Error generating keypair: {str(e)}[/red]")
        raise typer.Exit(1)


def decrypt_challenge_response(
    private_key: rsa.RSAPrivateKey, encrypted_response: bytes
) -> Optional[bytes]:
    """Decrypt the challenge response using the private key"""
    try:
        decrypted: bytes = private_key.decrypt(
            encrypted_response,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return decrypted
    except Exception as e:
        console.print(f"[red]Error decrypting response: {str(e)}[/red]")
        return None


def _finish_login(client: APIClient, config: Config) -> None:
    """Record the logged-in user's id, then confirm."""
    try:
        whoami_resp = client.get("/user/whoami")
        data = whoami_resp.get("data") if isinstance(whoami_resp, dict) else None
        if isinstance(data, dict):
            user_id = data.get("id")
            if user_id:
                config.set_user_id(user_id, user_name=data.get("name"))
                config.update_current_environment_file()
    except (APIError, Exception):
        console.print("[yellow]Logged in, but failed to fetch user id[/yellow]")

    console.print("[green]Successfully logged in![/green]")


def dpop_login(config: Config, headless: bool) -> None:
    """RFC 8628 device-code login yielding a DPoP-bound (RFC 9449) session."""
    provider = dpop.key_provider()
    try:
        key = provider.create(config.current_environment)
    except Exception as e:
        console.print(f"[red]Could not store a device key in the OS keychain:[/red] {e}")
        console.print("Use 'prime login' without --dpop to log in with an API key.")
        raise typer.Exit(1)

    base_url = config.base_url
    start_response = httpx.post(
        f"{base_url}{dpop.DEVICE_START_PATH}", json={"client_id": dpop.CLIENT_ID}
    )
    if start_response.status_code != 200:
        provider.delete(config.current_environment)
        console.print(f"[red]Failed to start login:[/red] {start_response.text}")
        raise typer.Exit(1)
    start = start_response.json()
    verification_url = start.get("verification_uri_complete") or start["verification_uri"]

    console.print("\n[bold blue]🔐 Login Required[/bold blue]")
    console.print(
        f"\nOpen [link={verification_url}]{verification_url}[/link] and confirm the code:\n\n"
        f"[bold green]{start['user_code']}[/bold green]\n"
    )
    console.print("[dim]Waiting for authentication...[/dim]")
    if not headless:
        try:
            webbrowser.open(verification_url, new=2)
        except Exception:
            pass

    interval = int(start.get("interval", 5))
    deadline = time.time() + int(start.get("expires_in", 900))
    with httpx.Client(auth=dpop.DPoPAuth(key), timeout=30.0) as http:
        while True:
            if time.time() > deadline:
                error = "expired_token"
                break
            time.sleep(interval)
            response = http.post(
                f"{base_url}{dpop.DEVICE_TOKEN_PATH}",
                json={
                    "grant_type": dpop.DEVICE_CODE_GRANT,
                    "device_code": start["device_code"],
                    "client_id": dpop.CLIENT_ID,
                },
            )
            body = response.json()
            if response.status_code == 200:
                dpop.save_session(config, body)
                error = None
                break
            error = body.get("error") or body.get("detail")
            if error == "slow_down":
                interval += 5
            elif error != "authorization_pending":
                break
    if error:
        provider.delete(config.current_environment)
        console.print(f"[red]Login failed:[/red] {error}")
        raise typer.Exit(1)

    # A stored API key would take precedence over the new session.
    config.set_api_key("")
    config.update_current_environment_file()
    client = APIClient()
    _finish_login(client, config)
    if config.api_key:
        console.print(
            "[yellow]Note:[/yellow] PRIME_API_KEY is set and takes precedence over this session."
        )
    fetch_and_select_team(client, config)


@app.callback(invoke_without_command=True)
def login(
    headless: bool = typer.Option(False, "--headless", help="Don't attempt to open browser"),
    use_dpop: bool = typer.Option(
        False,
        "--dpop",
        help="Log in with a short-lived session bound to a key on this device "
        "instead of minting an API key",
    ),
    max_concurrent_sandboxes: Optional[int] = typer.Option(
        None,
        "--max-concurrent-sandboxes",
        min=0,
        help="Cap sandboxes running at once on the new key",
        rich_help_panel=KEY_LIMITS_PANEL,
    ),
    max_sandbox_creations_per_hour: Optional[int] = typer.Option(
        None,
        "--max-sandbox-creations-per-hour",
        min=0,
        help="Cap sandbox creations per hour on the new key",
        rich_help_panel=KEY_LIMITS_PANEL,
    ),
    max_sandbox_cpu_cores: Optional[int] = typer.Option(
        None,
        "--max-sandbox-cpu-cores",
        min=0,
        help="Cap total sandbox CPU cores on the new key",
        rich_help_panel=KEY_LIMITS_PANEL,
    ),
    max_sandbox_gpu_count: Optional[int] = typer.Option(
        None,
        "--max-sandbox-gpu-count",
        min=0,
        help="Cap total sandbox GPUs on the new key",
        rich_help_panel=KEY_LIMITS_PANEL,
    ),
    max_concurrent_tunnels: Optional[int] = typer.Option(
        None,
        "--max-concurrent-tunnels",
        min=0,
        help="Cap tunnels open at once on the new key",
        rich_help_panel=KEY_LIMITS_PANEL,
    ),
    max_tunnel_creations_per_hour: Optional[int] = typer.Option(
        None,
        "--max-tunnel-creations-per-hour",
        min=0,
        help="Cap tunnel creations per hour on the new key",
        rich_help_panel=KEY_LIMITS_PANEL,
    ),
    max_tunnel_ttl_hours: Optional[int] = typer.Option(
        None,
        "--max-tunnel-ttl-hours",
        min=1,
        help="Cap tunnel lifetime in hours on the new key",
        rich_help_panel=KEY_LIMITS_PANEL,
    ),
) -> None:
    """Login to Prime Intellect"""
    require_persistent_context()
    flag_values = locals()
    key_limits = {
        field: flag_values[flag]
        for flag, field in KEY_LIMIT_FIELDS.items()
        if flag_values[flag] is not None
    }
    config = Config()
    settings = config.view()

    if not settings["base_url"]:
        console.print(
            "Base URL not configured.",
            "Please run 'prime config set-base-url' first.",
        )
        raise typer.Exit(1)

    if use_dpop:
        if key_limits:
            console.print("[red]API key limits cannot be combined with --dpop.[/red]")
            raise typer.Exit(1)
        dpop_login(config, headless)
        return

    private_key = None
    try:
        # Generate secure keypair
        private_key, public_pem = generate_ephemeral_keypair()

        response = httpx.post(
            f"{settings['base_url']}/api/v1/auth_challenge/generate",
            json={
                "encryptionPublicKey": public_pem,
                **({"limits": key_limits} if key_limits else {}),
            },
        )

        if response.status_code != 200:
            console.print(
                "[red]Failed to generate challenge:[/red]",
                f"{response.json().get('detail', 'Unknown error')}",
            )
            raise typer.Exit(1)

        challenge_response = response.json()

        challenge_code = challenge_response["challenge"]
        challenge_url = (
            f"{settings['frontend_url']}/dashboard/tokens/challenge?code={challenge_code}"
        )

        console.print("\n[bold blue]🔐 Login Required[/bold blue]")
        console.print("\n[bold]Follow these steps to authenticate:[/bold]\n")

        console.print(
            f"[bold yellow]1.[/bold yellow] Open the following link in your browser:\n"
            f"[link={challenge_url}]{challenge_url}[/link]"
        )
        console.print(
            f"[bold yellow]2.[/bold yellow] Your code should be pre-filled. Code:\n\n"
            f"[bold green]{challenge_code}[/bold green]\n"
        )
        console.print("[dim]Waiting for authentication...[/dim]")

        # Try to open the browser automatically (after all prints to avoid errors polluting output)
        if not headless:
            try:
                webbrowser.open(challenge_url, new=2)
            except Exception:
                pass

        challenge_auth_header = f"Bearer {challenge_response['status_auth_token']}"
        while True:
            try:
                status_response = httpx.get(
                    f"{settings['base_url']}/api/v1/auth_challenge/status",
                    params={"challenge": challenge_response["challenge"]},
                    headers={"Authorization": challenge_auth_header},
                )

                if status_response.status_code == 404:
                    console.print("[red]Challenge expired[/red]")
                    break

                status_data = status_response.json()
                if status_data.get("result"):
                    # Decrypt the result
                    encrypted_result = base64.b64decode(status_data["result"])
                    decrypted_result = decrypt_challenge_response(private_key, encrypted_result)
                    if decrypted_result:
                        # Update config with decrypted token
                        api_key = decrypted_result.decode()
                        config.set_api_key(api_key)
                        # Also update the current environment's saved file
                        config.update_current_environment_file()

                        client = APIClient(api_key=api_key)
                        _finish_login(client, config)
                        if key_limits:
                            summary = ", ".join(f"{k}={v}" for k, v in key_limits.items())
                            console.print(f"[dim]API key limits: {summary}[/dim]")
                        fetch_and_select_team(client, config)
                    else:
                        console.print("[red]Failed to decrypt authentication token[/red]")
                    break

                time.sleep(5)
            except httpx.RequestError:
                console.print("[red]Failed to connect to server. Retrying...[/red]")
                time.sleep(5)
                continue

    except KeyboardInterrupt:
        console.print("\n[yellow]Login cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception:
        raise typer.Exit(1)
    finally:
        # Ensure private key is securely wiped
        if private_key:
            del private_key
