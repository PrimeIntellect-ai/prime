#!/usr/bin/env python3
"""
Sandbox Folder Operations Demo

This example demonstrates how to upload and download entire folders to/from sandboxes
using tar archives, similar to what the 'prime sandbox cp -r' command does internally.

Usage:
    python examples/sandbox_folder_operations.py
"""

import asyncio
import os
import tarfile
import tempfile
from pathlib import Path

from prime_cli.api.sandbox import AsyncSandboxClient, CreateSandboxRequest
from prime_cli.config import Config


async def upload_folder(
    client: AsyncSandboxClient, sandbox_id: str, local_folder: Path, remote_path: str
) -> None:
    """Upload an entire folder to a sandbox using tar archiving."""

    if not local_folder.exists() or not local_folder.is_dir():
        raise ValueError(f"Local folder does not exist: {local_folder}")

    print(f"Preparing to upload folder: {local_folder}")

    # Create a temporary tar archive
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp_tar:
        tmp_tar_path = tmp_tar.name

    try:
        # Create tar archive of the folder
        print("  Creating archive...")
        with tarfile.open(tmp_tar_path, "w") as tar:
            tar.add(local_folder, arcname=local_folder.name)

        # Upload tar file to sandbox
        tar_remote = f"/tmp/upload_{os.urandom(4).hex()}.tar"
        print("  Uploading archive to sandbox...")
        await client.upload_file(
            sandbox_id=sandbox_id, file_path=tar_remote, local_file_path=tmp_tar_path
        )

        # Extract tar in sandbox
        print(f"  Extracting to {remote_path}...")
        extract_cmd = f"mkdir -p {remote_path} && tar -xf {tar_remote} -C {remote_path}"
        result = await client.execute_command(sandbox_id, extract_cmd)

        if result.exit_code != 0:
            raise RuntimeError(f"Failed to extract archive: {result.stderr}")

        # Clean up tar file in sandbox
        await client.execute_command(sandbox_id, f"rm -f {tar_remote}")

        print(f"Folder uploaded successfully to {remote_path}")

    finally:
        # Clean up local tar file
        if os.path.exists(tmp_tar_path):
            os.unlink(tmp_tar_path)


async def download_folder(
    client: AsyncSandboxClient, sandbox_id: str, remote_folder: str, local_path: Path
) -> None:
    """Download an entire folder from a sandbox using tar archiving."""

    print(f"Preparing to download folder: {remote_folder}")

    # Check if remote path exists and is a directory
    result = await client.execute_command(
        sandbox_id, f"test -d {remote_folder} && echo 'exists' || echo 'not found'"
    )

    if result.stdout.strip() != "exists":
        raise ValueError(f"Remote folder does not exist: {remote_folder}")

    # Create tar archive in sandbox
    tar_path = f"/tmp/download_{os.urandom(4).hex()}.tar"
    tar_cmd = f"tar -cf {tar_path} -C $(dirname {remote_folder}) $(basename {remote_folder})"

    print("  Creating archive in sandbox...")
    result = await client.execute_command(sandbox_id, tar_cmd)

    if result.exit_code != 0:
        raise RuntimeError(f"Failed to create archive: {result.stderr}")

    # Download tar file
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp_tar:
        tmp_tar_path = tmp_tar.name

    try:
        print("  Downloading archive...")
        await client.download_file(
            sandbox_id=sandbox_id, file_path=tar_path, local_file_path=tmp_tar_path
        )

        # Extract locally
        print(f"  Extracting to {local_path}...")
        local_path.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tmp_tar_path, "r") as tar:
            tar.extractall(local_path)

        print(f"Folder downloaded successfully to {local_path}")

        # Clean up tar file in sandbox
        await client.execute_command(sandbox_id, f"rm -f {tar_path}")

    finally:
        # Clean up local tar file
        if os.path.exists(tmp_tar_path):
            os.unlink(tmp_tar_path)


async def main() -> None:
    # Initialize the client
    client = AsyncSandboxClient()

    try:
        print("Creating a sandbox for folder operations demo...")

        # Create a sandbox
        request = CreateSandboxRequest(
            name="folder-operations-demo",
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

        # Create a test folder structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a sample project structure
            project_dir = temp_path / "my_project"
            project_dir.mkdir()

            # Create some files and subdirectories
            (project_dir / "README.md").write_text("# My Project\n\nThis is a test project.")
            (project_dir / "main.py").write_text(
                "def main():\n    print('Hello from my project!')\n"
            )

            src_dir = project_dir / "src"
            src_dir.mkdir()
            (src_dir / "__init__.py").write_text("")
            (src_dir / "utils.py").write_text("def helper():\n    return 'Helper function'\n")

            data_dir = project_dir / "data"
            data_dir.mkdir()
            (data_dir / "config.json").write_text('{"version": "1.0", "debug": true}')

            print("\n" + "=" * 50)
            print("DEMO: Upload Folder to Sandbox")
            print("=" * 50)

            # Upload the folder
            await upload_folder(
                client=client,
                sandbox_id=sandbox_id,
                local_folder=project_dir,
                remote_path="/workspace",
            )

            # Verify the upload
            print("\nVerifying uploaded folder structure:")
            result = await client.execute_command(sandbox_id, "find /workspace -type f | sort")
            print("Files in sandbox:")
            print(result.stdout)

            # Show content of a file
            result = await client.execute_command(sandbox_id, "cat /workspace/my_project/main.py")
            print("\nContent of main.py:")
            print(result.stdout)

            print("\n" + "=" * 50)
            print("DEMO: Download Folder from Sandbox")
            print("=" * 50)

            # Create another folder in the sandbox
            await client.execute_command(
                sandbox_id,
                """
                mkdir -p /workspace/sandbox_project/lib &&
                echo '# Sandbox Project' > /workspace/sandbox_project/README.md &&
                echo 'print(\"Created in sandbox\")' > /workspace/sandbox_project/run.py &&
                echo 'def sandbox_func(): pass' > /workspace/sandbox_project/lib/module.py
                """,
            )

            # Download the folder
            download_path = temp_path / "downloaded_project"
            await download_folder(
                client=client,
                sandbox_id=sandbox_id,
                remote_folder="/workspace/sandbox_project",
                local_path=download_path,
            )

            # Verify the download
            print("\nVerifying downloaded folder structure:")
            for root, dirs, files in os.walk(download_path):
                level = root.replace(str(download_path), "").count(os.sep)
                indent = "  " * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = "  " * (level + 1)
                for file in files:
                    print(f"{subindent}{file}")

            # Show content of a downloaded file
            run_py = download_path / "sandbox_project" / "run.py"
            if run_py.exists():
                print("\nContent of downloaded run.py:")
                print(run_py.read_text())

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
    print("Sandbox Folder Operations Demo")
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
