#!/usr/bin/env python3
"""
Simple Sandbox API Demo - shows auth, basic usage, and file operations using async client
"""

import asyncio
import glob
import os
import shutil
import tempfile
import traceback

from prime_cli.api.client import APIClient, APIError
from prime_cli.api.sandbox import CreateSandboxRequest, SandboxClient, AsyncSandboxClient


def create_test_file(content: str, filename: str) -> str:
    """Create a temporary test file with given content"""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)

    with open(file_path, "w") as f:
        f.write(content)

    return file_path


async def main() -> None:
    """Simple sandbox demo with file operations using async client"""
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

        # 5. File Operations Demo - Showcasing Async Concurrency
        print("\n" + "=" * 50)
        print("CONCURRENT FILE OPERATIONS DEMO (ASYNC)")
        print("=" * 50)

        # Create multiple test files locally
        test_files = []
        for i in range(5):
            content = f"Test file {i+1} content\nThis demonstrates concurrent async operations.\n"
            file_path = create_test_file(content, f"test_file_{i+1}.txt")
            test_files.append((file_path, f"/tmp/uploaded_file_{i+1}.txt"))
            print(f"📁 Created local test file {i+1}: {file_path}")

        # Use async client for concurrent operations
        async with AsyncSandboxClient() as async_client:
            # CONCURRENT UPLOADS - Upload all files at once
            print("\n📤 Uploading 5 files concurrently...")
            start_time = asyncio.get_event_loop().time()
            
            upload_tasks = [
                async_client.upload_path(
                    sandbox_id=sandbox.id,
                    local_path=local_path,
                    sandbox_path=sandbox_path
                )
                for local_path, sandbox_path in test_files
            ]
            
            # Run all uploads concurrently
            upload_results = await asyncio.gather(*upload_tasks)
            
            upload_time = asyncio.get_event_loop().time() - start_time
            print(f"✅ All uploads completed in {upload_time:.2f} seconds")
            for i, result in enumerate(upload_results):
                print(f"   File {i+1}: {result.bytes_uploaded} bytes uploaded")

            # Create multiple files in sandbox for download demo
            print("\n📝 Creating files in sandbox for download demo...")
            for i in range(5):
                content = f"Sandbox file {i+1}\nCreated for concurrent download demo.\n"
                sandbox_client.execute_command(
                    sandbox.id, 
                    f"echo '{content}' > /tmp/sandbox_file_{i+1}.txt"
                )

            # CONCURRENT DOWNLOADS - Download all files at once
            print("\n📥 Downloading 5 files concurrently...")
            start_time = asyncio.get_event_loop().time()
            
            download_tasks = [
                async_client.download_path(
                    sandbox_id=sandbox.id,
                    sandbox_path=f"/tmp/sandbox_file_{i+1}.txt",
                    local_path=f"/tmp/downloaded_file_{i+1}.txt"
                )
                for i in range(5)
            ]
            
            # Run all downloads concurrently
            await asyncio.gather(*download_tasks)
            
            download_time = asyncio.get_event_loop().time() - start_time
            print(f"✅ All downloads completed in {download_time:.2f} seconds")

            # MIXED OPERATIONS - Upload and download different files concurrently
            print("\n🔄 Running mixed upload/download operations concurrently...")
            start_time = asyncio.get_event_loop().time()
            
            mixed_tasks = []
            
            # Add some uploads
            for i in range(3):
                content = f"Mixed operation upload {i+1}\n"
                file_path = create_test_file(content, f"mixed_upload_{i+1}.txt")
                mixed_tasks.append(
                    async_client.upload_path(
                        sandbox_id=sandbox.id,
                        local_path=file_path,
                        sandbox_path=f"/tmp/mixed_upload_{i+1}.txt"
                    )
                )
            
            # Add some downloads
            for i in range(3):
                mixed_tasks.append(
                    async_client.download_path(
                        sandbox_id=sandbox.id,
                        sandbox_path=f"/tmp/sandbox_file_{i+1}.txt",
                        local_path=f"/tmp/mixed_download_{i+1}.txt"
                    )
                )
            
            # Run all mixed operations concurrently
            mixed_results = await asyncio.gather(*mixed_tasks, return_exceptions=True)
            
            mixed_time = asyncio.get_event_loop().time() - start_time
            print(f"✅ Mixed operations completed in {mixed_time:.2f} seconds")
            
            # Check for any errors
            for i, result in enumerate(mixed_results):
                if isinstance(result, Exception):
                    print(f"   Operation {i+1} failed: {result}")
                elif hasattr(result, 'bytes_uploaded'):
                    print(f"   Upload {i+1}: Success")
                else:
                    print(f"   Download {i-2}: Success")

        # Verify one of the downloaded files
        with open("/tmp/downloaded_file_1.txt", "r") as f:
            downloaded_content = f.read()
        print(f"\n📄 Sample downloaded content: {repr(downloaded_content)}")

        # Performance comparison note
        print("\n💡 Performance Note:")
        print("   Sequential operations would take ~5x longer than concurrent operations.")
        print("   This demo showcases the power of async I/O for parallel file transfers.")

        # Clean up local files
        print("\n🧹 Cleaning up local test files...")
        
        # Clean up test files
        for pattern in ["/tmp/test_file_*.txt", "/tmp/downloaded_file_*.txt", 
                       "/tmp/mixed_*.txt", "/tmp/uploaded_file_*.txt"]:
            for file in glob.glob(pattern):
                try:
                    os.unlink(file)
                except:
                    pass
        
        # Clean up temp directories
        for local_path, _ in test_files:
            try:
                parent_dir = os.path.dirname(local_path)
                if os.path.exists(parent_dir) and parent_dir.startswith("/tmp/"):
                    shutil.rmtree(parent_dir)
            except:
                pass
        
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
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
