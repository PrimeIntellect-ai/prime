#!/usr/bin/env python3
"""
Sandbox CP Command Demo

This example demonstrates the new 'prime sandbox cp' command for copying files
between local and sandbox environments using Unix-style syntax.

Usage:
    python examples/sandbox_cp_demo.py
"""

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

from prime_cli.api.sandbox import AsyncSandboxClient, CreateSandboxRequest
from prime_cli.config import Config


async def main() -> None:
    # Initialize the client
    client = AsyncSandboxClient()

    try:
        print("Creating a sandbox for cp command demo...")

        # Create a sandbox
        request = CreateSandboxRequest(
            name="cp-command-demo",
            docker_image="python:3.11-slim",
            start_command="tail -f /dev/null",
            cpu_cores=1,
            memory_gb=2,
            disk_size_gb=10,
            timeout_minutes=30,
        )

        sandbox = await client.create(request)
        sandbox_id = sandbox.id
        print(f"Sandbox created: {sandbox_id}")

        # Wait for sandbox to be running
        print("Waiting for sandbox to be running...")
        await client.wait_for_creation(sandbox_id)
        print("Sandbox is running!")

        # Create test files and directories
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a single test file
            test_file = temp_path / "test.txt"
            test_file.write_text("Hello from the cp command demo!\n")

            # Create a directory with multiple files
            test_dir = temp_path / "test_directory"
            test_dir.mkdir()
            (test_dir / "file1.txt").write_text("File 1 content\n")
            (test_dir / "file2.txt").write_text("File 2 content\n")
            (test_dir / "subdir").mkdir()
            (test_dir / "subdir" / "nested.txt").write_text("Nested file content\n")

            print("\n" + "=" * 50)
            print("DEMO: Single File Upload")
            print("=" * 50)

            # Upload single file using cp command
            cmd = ["prime", "sandbox", "cp", str(test_file), f"{sandbox_id}:/tmp/uploaded.txt"]
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"Error: {result.stderr}")

            # Verify file was uploaded
            verify_result = await client.execute_command(sandbox_id, "cat /tmp/uploaded.txt")
            print(f"Verification - File content in sandbox:\n{verify_result.stdout}")

            print("\n" + "=" * 50)
            print("DEMO: Directory Upload (Recursive)")
            print("=" * 50)

            # Upload directory using cp command with -r flag
            cmd = [
                "prime",
                "sandbox",
                "cp",
                "-r",
                str(test_dir),
                f"{sandbox_id}:/tmp/uploaded_dir",
            ]
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"Error: {result.stderr}")

            # Verify directory was uploaded
            verify_result = await client.execute_command(
                sandbox_id, "find /tmp/uploaded_dir -type f -exec echo {} \\; -exec cat {} \\;"
            )
            print(f"Verification - Directory contents in sandbox:\n{verify_result.stdout}")

            print("\n" + "=" * 50)
            print("DEMO: Single File Download")
            print("=" * 50)

            # Create a file in the sandbox
            await client.execute_command(
                sandbox_id, "echo 'File created in sandbox' > /tmp/sandbox_file.txt"
            )

            # Download file using cp command
            download_path = temp_path / "downloaded.txt"
            cmd = [
                "prime",
                "sandbox",
                "cp",
                f"{sandbox_id}:/tmp/sandbox_file.txt",
                str(download_path),
            ]
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"Error: {result.stderr}")

            # Verify file was downloaded
            if download_path.exists():
                content = download_path.read_text()
                print(f"Verification - Downloaded file content:\n{content}")

            print("\n" + "=" * 50)
            print("DEMO: Directory Download (Recursive)")
            print("=" * 50)

            # Create a directory structure in the sandbox
            await client.execute_command(
                sandbox_id,
                "mkdir -p /tmp/sandbox_dir/subdir && "
                "echo 'File A' > /tmp/sandbox_dir/fileA.txt && "
                "echo 'File B' > /tmp/sandbox_dir/fileB.txt && "
                "echo 'Nested' > /tmp/sandbox_dir/subdir/nested.txt",
            )

            # Download directory using cp command with -r flag
            download_dir = temp_path / "downloaded_dir"
            cmd = [
                "prime",
                "sandbox",
                "cp",
                "-r",
                f"{sandbox_id}:/tmp/sandbox_dir",
                str(download_dir),
            ]
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"Error: {result.stderr}")

            # Verify directory was downloaded
            if download_dir.exists():
                print("Verification - Downloaded directory structure:")
                for root, dirs, files in os.walk(download_dir):
                    level = root.replace(str(download_dir), "").count(os.sep)
                    indent = " " * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = " " * 2 * (level + 1)
                    for file in files:
                        file_path = Path(root) / file
                        content = file_path.read_text().strip()
                        print(f"{subindent}{file}: {content}")

        # Clean up the sandbox
        print("\nCleaning up sandbox...")
        await client.delete(sandbox_id)
        print("Sandbox cleaned up!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await client.aclose()


if __name__ == "__main__":
    print("Sandbox CP Command Demo")
    print("=" * 50)

    # Check configuration
    config = Config()
    if not config.api_key:
        print("Error: No API key found. Please run 'prime auth login' first.")
        exit(1)

    print(f"Using API endpoint: {config.base_url}")
    if config.team_id:
        print(f"Team: {config.team_id}")

    # Run the demo
    asyncio.run(main())

    print("\nDemo completed!")
