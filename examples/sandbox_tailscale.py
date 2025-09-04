import os
import sys

from prime_cli.api.client import APIClient
from prime_cli.api.sandbox import CreateSandboxRequest, SandboxClient


def main() -> None:
    # Check if TS_AUTHKEY is set
    ts_authkey = os.getenv("TS_AUTHKEY")
    if not ts_authkey:
        print("Error: TS_AUTHKEY environment variable is not set")
        print("Please set your Tailscale auth key: export TS_AUTHKEY=your_auth_key")
        sys.exit(1)

    client = APIClient()  # Automatically loads API key from ~/.prime/config.json
    sandbox_client = SandboxClient(client)

    # create a sandbox
    print("Creating sandbox...")
    sandbox = sandbox_client.create(
        CreateSandboxRequest(
            name="sandbox-ssh",
            docker_image="ubuntu:22.04",
            start_command="sleep infinity",
            cpu_cores=1,
            memory_gb=2,
            timeout_minutes=120,
        )
    )

    print("Waiting for sandbox to start...")
    sandbox_client.wait_for_creation(sandbox.id, max_attempts=120)

    # Update package list and install curl
    print("Installing curl...")
    sandbox_client.execute_command(
        sandbox_id=sandbox.id,
        command="apt-get update && apt-get install -y curl",
    )

    # Install tailscale
    print("Installing tailscale...")
    install_response = sandbox_client.execute_command(
        sandbox_id=sandbox.id,
        command="curl -fsSL https://tailscale.com/install.sh | sh",
    )
    print(install_response)

    # Start tailscale daemon in background with userspace networking
    print("Starting tailscale daemon...")
    try:
        start_daemon = sandbox_client.execute_command(
            sandbox_id=sandbox.id,
            command=(
                "nohup tailscaled --tun=userspace-networking "
                "--socks5-server=localhost:1055 > /tmp/tailscaled.log 2>&1 & "
                "echo 'Started tailscaled daemon'"
            ),
            timeout=10,
        )
    except Exception as e:
        print(f"Error starting tailscale daemon: {e}")
        sys.exit(1)
    print(start_daemon)

    # Wait for daemon to start
    print("Waiting for daemon to start...")
    sandbox_client.execute_command(
        sandbox_id=sandbox.id,
        command="sleep 3",
    )

    # Authenticate with tailscale using environment variable
    print("Authenticating with tailscale...")
    auth_response = sandbox_client.execute_command(
        sandbox_id=sandbox.id,
        command="tailscale up --ssh --authkey=$TS_AUTHKEY",
        timeout=10,
        env={"TS_AUTHKEY": ts_authkey},
    )
    print(auth_response)

    whoami_response = sandbox_client.execute_command(
        sandbox_id=sandbox.id,
        command="whoami",
    )
    print(whoami_response)

    ip_response = sandbox_client.execute_command(
        sandbox_id=sandbox.id,
        command="tailscale ip -4 | head -n1",
    )
    print(ip_response)

    print(
        f"Connect: ssh -o StrictHostKeyChecking=accept-new "
        f"{whoami_response.stdout.strip()}@{ip_response.stdout.strip()}"
    )

    print("Done")


if __name__ == "__main__":
    main()
