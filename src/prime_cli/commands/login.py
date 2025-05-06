import base64
import time

import requests
import typer
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives.asymmetric import rsa
from rich.console import Console

from ..config import Config

app = typer.Typer(help="Login to Prime Intellect")
console = Console()


def generate_ephemeral_keypair():
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


def decrypt_challenge_response(private_key, encrypted_response):
    """Decrypt the challenge response using the private key"""
    try:
        decrypted = private_key.decrypt(
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
            "[red]Base URL not configured. Please run 'prime config set-base-url' first.[/red]"
        )
        raise typer.Exit(1)

    private_key = None
    try:
        # Generate secure keypair
        private_key, public_pem = generate_ephemeral_keypair()

        print(f"{settings['base_url']}/api/v1/auth_challenge/generate")

        response = requests.post(
            f"{settings['base_url']}/api/v1/auth_challenge/generate",
            json={
                "encryptionPublicKey": public_pem,
            },
        )

        print(response.json())
        if response.status_code != 200:
            console.print("[red]Failed to generate challenge[/red]")
            raise typer.Exit(1)

        challenge_response = response.json()

        console.print("\n[bold blue]To login, please follow these steps:[/bold blue]")
        console.print(
            "1. Open [link]https://app.primeintellect.ai/dashboard/tokens/challenge[/link]"
        )
        console.print(
            f"2. Enter this code: [bold green]{challenge_response['challenge']}[/bold green]"
        )
        console.print("\nWaiting for authentication...")

        while True:
            status_response = requests.get(
                f"{settings['base_url']}/api/v1/auth_challenge/status",
                params={"challenge": challenge_response["challenge"]},
            )

            if status_response.status_code == 404:
                console.print("[red]Challenge expired[/red]")
                break

            status_data = status_response.json()
            if status_data.get("result"):
                # Decrypt the result
                encrypted_result = base64.b64decode(status_data["result"])
                decrypted_result = decrypt_challenge_response(
                    private_key, encrypted_result
                )
                if decrypted_result:
                    # Update config with decrypted token
                    config.set_api_key(decrypted_result.decode())
                    console.print("[green]Successfully logged in![/green]")
                else:
                    console.print("[red]Failed to decrypt authentication token[/red]")
                break

            time.sleep(5)

    except KeyboardInterrupt:
        console.print("\n[yellow]Login cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]An error occurred: {str(e)}[/red]")
        raise typer.Exit(1)
    finally:
        # Ensure private key is securely wiped
        if private_key:
            del private_key
