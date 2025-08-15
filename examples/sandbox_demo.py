#!/usr/bin/env python3
"""
Simple Sandbox API Demo - shows auth, basic usage, and file operations
"""

import os
import tempfile
from prime_cli.api.client import APIClient, APIError
from prime_cli.api.sandbox import CreateSandboxRequest, SandboxClient


def create_test_file(content: str, filename: str) -> str:
    """Create a temporary test file with given content"""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)

    with open(file_path, 'w') as f:
        f.write(content)

    return file_path


def main() -> None:
    """Simple sandbox demo with file operations"""
    try:
        # 1. Authentication - uses API key from config or environment
        # Run 'prime login' first to set up your API key
        client = APIClient()  # Automatically loads API key from ~/.prime/config.json
        sandbox_client = SandboxClient(client)

        # 2. Create a sandbox
        request = CreateSandboxRequest(
            name="demo-sandbox",
            docker_image="python:3.11-slim",
            start_command="tail -f /dev/null",  # Keep container running indefinitely
            cpu_cores=1,
            memory_gb=2,
            timeout_minutes=120,  # 2 hours to avoid timeout during demo
        )

        print("Creating sandbox...")
        sandbox = sandbox_client.create(request)
        print(f"✅ Created: {sandbox.name} ({sandbox.id})")

        # 3. Wait for sandbox to be running
        print("\nWaiting for sandbox to be running...")
        sandbox_client.wait_for_creation(sandbox.id, max_attempts=60)
        print("✅ Sandbox is running!")

        # 4. Execute commands in the sandbox
        print("\nExecuting commands...")

        # Test basic commands that definitely work
        result = sandbox_client.execute_command(sandbox.id, "whoami")
        print(f"Current user: {result.stdout.strip()}")

        result = sandbox_client.execute_command(sandbox.id, "pwd")
        print(f"Working directory: {result.stdout.strip()}")

        result = sandbox_client.execute_command(sandbox.id, "python --version")
        print(f"Python version: {result.stdout.strip()}")

        # List files in working directory
        result = sandbox_client.execute_command(sandbox.id, "ls -la")
        print(f"Files in working directory:\n{result.stdout}")

        # Test inline Python execution (no file creation needed)
        result = sandbox_client.execute_command(
            sandbox.id, "python -c 'print(\"Hello from sandbox!\")'"
        )
        print(f"Python hello: {result.stdout.strip()}")

        result = sandbox_client.execute_command(
            sandbox.id, "python -c 'print(f\"2 + 2 = {2 + 2}\")'"
        )
        print(f"Math result: {result.stdout.strip()}")

        # Check environment
        result = sandbox_client.execute_command(sandbox.id, "env | grep SANDBOX")
        print(f"Sandbox environment variables:\n{result.stdout}")

        # 5. File Operations Demo
        print("\n" + "="*50)
        print("FILE OPERATIONS DEMO")
        print("="*50)

        # Create a test file locally
        test_content = "Hello from local machine!\nThis is a test file for upload/download demo.\n"
        local_file_path = create_test_file(test_content, "test_upload.txt")
        print(f"\n📁 Created local test file: {local_file_path}")
        print(f"📄 Content: {repr(test_content)}")

        # Upload file to sandbox
        print("\n📤 Uploading file to sandbox...")
        upload_response = sandbox_client.upload_path(
            sandbox_id=sandbox.id,
            local_path=local_file_path,
            sandbox_path="/tmp/test_upload.txt"
        )
        print(f"✅ Upload successful: {upload_response.message}")
        print(f"   Files uploaded: {upload_response.files_uploaded}")
        print(f"   Bytes uploaded: {upload_response.bytes_uploaded}")

        # Verify file exists in sandbox
        print("\n🔍 Verifying file in sandbox...")
        result = sandbox_client.execute_command(sandbox.id, "ls -la /tmp/test_upload.txt")
        print(f"File listing: {result.stdout.strip()}")

        result = sandbox_client.execute_command(sandbox.id, "cat /tmp/test_upload.txt")
        print(f"File content: {result.stdout.strip()}")

        # Create a file in the sandbox
        print("\n📝 Creating file in sandbox...")
        sandbox_content = "Hello from sandbox!\nThis file was created inside the sandbox.\n"
        result = sandbox_client.execute_command(
            sandbox.id,
            f"echo '{sandbox_content}' > /tmp/sandbox_created.txt"
        )
        print(f"File creation result: {result.stdout.strip()}")

        # Download file from sandbox
        print("\n📥 Downloading file from sandbox...")
        download_path = "/tmp/downloaded_sandbox_file.txt"
        sandbox_client.download_path(
            sandbox_id=sandbox.id,
            sandbox_path="/tmp/sandbox_created.txt",
            local_path=download_path
        )
        print(f"✅ Downloaded to: {download_path}")

        # Verify downloaded content
        with open(download_path, 'r') as f:
            downloaded_content = f.read()
        print(f"📄 Downloaded content: {repr(downloaded_content)}")

        # Clean up local files
        print("\n🧹 Cleaning up local files...")
        os.unlink(local_file_path)
        os.unlink(download_path)
        os.rmdir(os.path.dirname(local_file_path))
        print("✅ Local files cleaned up")

        # 6. List all sandboxes
        print("\nYour sandboxes:")
        sandbox_list = sandbox_client.list()
        for sb in sandbox_list.sandboxes:
            print(f"  {sb.name}: {sb.status}")

        # 7. Get logs
        print(f"\nLogs for {sandbox.name}:")
        logs = sandbox_client.get_logs(sandbox.id)
        print(logs)

        # 8. Clean up
        print(f"\n🗑️  Deleting {sandbox.name}...")
        sandbox_client.delete(sandbox.id)
        print("✅ Deleted")

        print("\n🎉 Sandbox demo with file operations completed successfully!")

    except APIError as e:
        print(f"❌ API Error: {e}")
        print("💡 Make sure you're logged in: run 'prime login' first")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
