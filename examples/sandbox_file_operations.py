#!/usr/bin/env python3
"""
Sandbox File Operations Demo

This example demonstrates how to upload and download files to/from sandboxes using the Prime CLI.

Usage:
    python examples/sandbox_file_operations.py
"""

import asyncio
import os
import tempfile

from prime_cli.api.sandbox import (
    AsyncSandboxClient,
    CreateSandboxRequest,
)
from prime_cli.config import Config


async def main() -> None:
    # Initialize the client with configuration
    client = AsyncSandboxClient()

    try:
        print("Creating a sandbox...")

        # Create a sandbox
        request = CreateSandboxRequest(
            name="file-operations-demo",
            docker_image="python:3.11-slim",
            start_command="tail -f /dev/null",  # Keep container running
            cpu_cores=1,
            memory_gb=2,
            disk_size_gb=10,
            timeout_minutes=30,
        )

        sandbox = await client.create(request)
        print(f"Sandbox created: {sandbox.id}")

        # Wait for sandbox to be running
        print("⏳ Waiting for sandbox to be running...")
        await client.wait_for_creation(sandbox.id)
        print("Sandbox is running!")

        # Create a temporary file to upload
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
            temp_file.write(
                """#!/usr/bin/env python3
# This is a test Python script uploaded to the sandbox

def hello_from_sandbox():
    print("Hello from the sandbox!")
    print("This file was uploaded from the host machine.")
    
    # Create some output
    with open('/tmp/sandbox_output.txt', 'w') as f:
        f.write("This file was created inside the sandbox!\\n")
        f.write("Current working directory: /sandbox-workspace\\n")
    
    return "File operations successful!"

if __name__ == "__main__":
    result = hello_from_sandbox()
    print(result)
"""
            )
            temp_file_path = temp_file.name

        try:
            print("📤 Uploading file to sandbox...")

            # Upload the file to the sandbox
            upload_response = await client.upload_file(
                sandbox_id=sandbox.id,
                file_path="/sandbox-workspace/test_script.py",
                local_file_path=temp_file_path,
            )
            print("File uploaded successfully!")
            print(f"   Path: {upload_response.path}")
            print(f"   Size: {upload_response.size} bytes")
            print(f"   Timestamp: {upload_response.timestamp}")

            # Make the script executable and run it
            print("🔧 Making script executable and running it...")

            # Execute commands to make it executable and run
            chmod_result = await client.execute_command(
                sandbox.id, "chmod +x /sandbox-workspace/test_script.py"
            )
            print(f"chmod result: {chmod_result.exit_code}")

            # Run the script
            run_result = await client.execute_command(
                sandbox.id, "cd /sandbox-workspace && python test_script.py"
            )
            print("📋 Script output:")
            print(f"   stdout: {run_result.stdout}")
            if run_result.stderr:
                print(f"   stderr: {run_result.stderr}")
            print(f"   exit_code: {run_result.exit_code}")

            # Download the file that was created inside the sandbox
            print("📥 Downloading file created in sandbox...")

            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".txt", delete=False
            ) as download_file:
                download_path = download_file.name

            try:
                await client.download_file(
                    sandbox_id=sandbox.id,
                    file_path="/tmp/sandbox_output.txt",
                    local_file_path=download_path,
                )

                print("File downloaded successfully!")
                print(f"Downloaded to: {download_path}")

                # Read and display the downloaded content
                with open(download_path, "r") as f:
                    content = f.read()
                print("Downloaded file content:")
                print(content)

            except Exception as e:
                print(f"Failed to download file: {e}")
            finally:
                # Clean up downloaded file
                if os.path.exists(download_path):
                    os.unlink(download_path)

            # List files in the workspace to verify upload
            print("Listing files in sandbox workspace...")
            ls_result = await client.execute_command(sandbox.id, "ls -la /sandbox-workspace/")
            print("📋 Workspace contents:")
            print(ls_result.stdout)

        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)

        # Clean up the sandbox
        print("🧹 Cleaning up sandbox...")
        await client.delete(sandbox.id)
        print("Sandbox cleaned up!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await client.aclose()


def sync_example() -> None:
    """Example using synchronous client"""
    from prime_cli.api.client import APIClient
    from prime_cli.api.sandbox import SandboxClient

    print("\n🔄 Running synchronous example...")

    # Initialize sync client
    api_client = APIClient()
    client = SandboxClient(api_client)

    try:
        # Create sandbox
        request = CreateSandboxRequest(
            name="sync-file-demo",
            docker_image="python:3.11-slim",
            start_command="tail -f /dev/null",
            cpu_cores=1,
            memory_gb=2,
            timeout_minutes=15,
        )

        sandbox = client.create(request)
        print(f"Sync sandbox created: {sandbox.id}")

        # Wait for running
        client.wait_for_creation(sandbox.id)
        print("Sync sandbox is running!")

        # Create a simple text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
            temp_file.write(
                "Hello from sync client!\nThis demonstrates synchronous file operations."
            )
            temp_file_path = temp_file.name

        try:
            # Upload file
            upload_response = client.upload_file(
                sandbox_id=sandbox.id,
                file_path="/sandbox-workspace/sync_test.txt",
                local_file_path=temp_file_path,
            )
            print(f"Sync upload successful: {upload_response.path}")

            # Verify file exists
            cat_result = client.execute_command(sandbox.id, "cat /sandbox-workspace/sync_test.txt")
            print(f"File content in sandbox: {cat_result.stdout}")

        finally:
            os.unlink(temp_file_path)

        # Cleanup
        client.delete(sandbox.id)
        print("Sync sandbox cleaned up!")

    except Exception as e:
        print(f"Sync error: {e}")


if __name__ == "__main__":
    print("Sandbox File Operations Demo")
    print("=" * 50)

    # Check configuration
    config = Config()
    if not config.api_key:
        print("Error: No API key found. Please run 'prime auth login' first.")
        exit(1)

    print(f"Using API endpoint: {config.base_url}")
    if config.team_id:
        print(f"Team: {config.team_id}")

    # Run async example
    asyncio.run(main())

    # Run sync example
    sync_example()

    print("\n✨ Demo completed!")
