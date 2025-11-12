import base64
import time
import webbrowser
from typing import Optional

import httpx
import typer
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives.asymmetric import rsa
from prime_core import Config
from rich.console import Console

from ..client import APIClient, APIError

app = typer.Typer(help="Login to Prime Intellect")
console = Console()


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


@app.callback(invoke_without_command=True)
def login() -> None:
    """Login to Prime Intellect"""
    config = Config()
    settings = config.view()

    if not settings["base_url"]:
        console.print(
            "Base URL not configured.",
            "Please run 'prime config set-base-url' first.",
        )
        raise typer.Exit(1)

    private_key = None
    try:
        # Generate secure keypair
        private_key, public_pem = generate_ephemeral_keypair()

        response = httpx.post(
            f"{settings['base_url']}/api/v1/auth_challenge/generate",
            json={
                "encryptionPublicKey": public_pem,
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

        # Try to open the browser automatically
        try:
            webbrowser.open(challenge_url, new=2)
            console.print(
                "[bold yellow]1.[/bold yellow] We've opened the login page in your browser."
            )
        except Exception:
            pass

        console.print(
            f"[bold yellow]1.[/bold yellow] Open the following link in your browser:\n"
            f"[link={challenge_url}]{challenge_url}[/link]"
        )

        console.print(
            f"[bold yellow]2.[/bold yellow] Your code should be pre-filled. Code:\n\n"
            f"[bold green]{challenge_code}[/bold green]\n"
        )
        console.print("[dim]Waiting for authentication...[/dim]")

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

                        # Attempt to fetch the current user id
                        try:
                            client = APIClient(api_key=api_key)
                            whoami_resp = client.get("/user/whoami")
                            data = (
                                whoami_resp.get("data") if isinstance(whoami_resp, dict) else None
                            )
                            if isinstance(data, dict):
                                user_id = data.get("id")
                                if user_id:
                                    config.set_user_id(user_id)
                                    config.update_current_environment_file()
                            console.print("[green]Successfully logged in![/green]")
                        except (APIError, Exception):
                            console.print("[yellow]Logged in, but failed to fetch user id[/yellow]")
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
