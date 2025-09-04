#!/usr/bin/env python3
"""
Prime CLI Async Sandbox Demo - Showcases high-performance concurrent file operations
Demonstrates the async improvements implemented in Prime CLI v0.3.23+

Key Features Demonstrated:
- AsyncSandboxClient with proper async/await patterns
- Concurrent file uploads using asyncio.gather()
- Concurrent file downloads using asyncio.gather()
- Mixed concurrent operations (upload + download simultaneously)
- Proper resource management with async context managers
- Error handling in concurrent operations
- Performance comparison: sequential vs concurrent operations
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path

from prime_cli.api.client import APIClient, APIError
from prime_cli.api.sandbox import CreateSandboxRequest, SandboxClient, AsyncSandboxClient


def create_test_files(count: int = 5) -> list[str]:
    """Create multiple test files for concurrent operations demo"""
    temp_dir = tempfile.mkdtemp()
    files = []
    
    for i in range(count):
        file_path = os.path.join(temp_dir, f"test_file_{i}.txt")
        with open(file_path, "w") as f:
            f.write(f"🚀 Prime CLI Async Demo - File {i}\n")
            f.write(f"Demonstrating concurrent file operations\n")
            f.write(f"File size: {200 + i * 50} characters\n")
            f.write(f"Content: {'=' * (200 + i * 50)}\n")
        files.append(file_path)
    
    print(f"📁 Created {count} test files in {temp_dir}")
    return files


async def demo_sequential_operations(async_client: AsyncSandboxClient, sandbox_id: str, test_files: list[str]) -> float:
    """Demonstrate sequential file operations (old way)"""
    print("\n📊 Testing Sequential Operations (Old Way)...")
    
    start_time = time.time()
    
    # Sequential uploads
    for i, file_path in enumerate(test_files[:3]):
        await async_client.upload_path(
            sandbox_id, 
            file_path, 
            f"/workspace/sequential_{i}.txt"
        )
        print(f"  ✅ Uploaded file {i}")
    
    # Sequential downloads
    for i in range(3):
        await async_client.download_path(
            sandbox_id,
            f"/workspace/sequential_{i}.txt",
            f"/tmp/downloaded_sequential_{i}.txt"
        )
        print(f"  ✅ Downloaded file {i}")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"⏱️  Sequential operations completed in {duration:.2f} seconds")
    
    return duration


async def demo_concurrent_operations(async_client: AsyncSandboxClient, sandbox_id: str, test_files: list[str]) -> float:
    """Demonstrate concurrent file operations (NEW async way)"""
    print("\n🚀 Testing Concurrent Operations (NEW Async Way)...")
    
    start_time = time.time()
    
    # Concurrent uploads using asyncio.gather()
    print("  🔄 Starting concurrent uploads...")
    upload_tasks = [
        async_client.upload_path(
            sandbox_id, 
            file_path, 
            f"/workspace/concurrent_{i}.txt"
        )
        for i, file_path in enumerate(test_files[:3])
    ]
    await asyncio.gather(*upload_tasks)
    print("  ✅ All uploads completed concurrently!")
    
    # Concurrent downloads using asyncio.gather()
    print("  🔄 Starting concurrent downloads...")
    download_tasks = [
        async_client.download_path(
            sandbox_id,
            f"/workspace/concurrent_{i}.txt",
            f"/tmp/downloaded_concurrent_{i}.txt"
        )
        for i in range(3)
    ]
    await asyncio.gather(*download_tasks)
    print("  ✅ All downloads completed concurrently!")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"⚡ Concurrent operations completed in {duration:.2f} seconds")
    
    return duration


async def demo_mixed_concurrent_operations(async_client: AsyncSandboxClient, sandbox_id: str, test_files: list[str]) -> None:
    """Demonstrate mixed upload/download operations running simultaneously"""
    print("\n🔀 Testing Mixed Concurrent Operations...")
    
    start_time = time.time()
    
    # Mix of uploads and downloads running at the same time
    mixed_tasks = []
    
    # Add upload tasks
    for i in range(2):
        if i < len(test_files):
            mixed_tasks.append(
                async_client.upload_path(
                    sandbox_id,
                    test_files[i],
                    f"/workspace/mixed_upload_{i}.txt"
                )
            )
    
    # Add download tasks (from previously uploaded files)
    for i in range(2):
        mixed_tasks.append(
            async_client.download_path(
                sandbox_id,
                f"/workspace/concurrent_{i}.txt",
                f"/tmp/mixed_download_{i}.txt"
            )
        )
    
    # Run all operations concurrently
    print(f"  🔄 Running {len(mixed_tasks)} mixed operations concurrently...")
    await asyncio.gather(*mixed_tasks)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"  ✅ Mixed operations completed in {duration:.2f} seconds")


async def demo_error_handling(async_client: AsyncSandboxClient, sandbox_id: str) -> None:
    """Demonstrate error handling in concurrent operations"""
    print("\n🛡️  Testing Error Handling in Concurrent Operations...")
    
    # Mix of valid and invalid operations
    mixed_tasks = [
        # Valid upload
        async_client.upload_path(sandbox_id, __file__, "/workspace/valid_demo.py"),
        # Invalid download (file doesn't exist)
        async_client.download_path(sandbox_id, "/workspace/nonexistent.txt", "/tmp/invalid.txt"),
        # Another valid upload
        async_client.upload_path(sandbox_id, __file__, "/workspace/valid_demo2.py"),
    ]
    
    try:
        results = await asyncio.gather(*mixed_tasks, return_exceptions=True)
        
        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = sum(1 for r in results if isinstance(r, Exception))
        
        print(f"  ✅ Successful operations: {successes}")
        print(f"  ❌ Failed operations: {failures}")
        print("  🎯 Error handling working correctly - failures don't crash other operations!")
        
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")


async def main() -> None:
    """Prime CLI Async Demo - showcases the power of concurrent file operations"""
    print("🚀 Prime CLI Async Sandbox Demo")
    print("=" * 50)
    
    try:
        # 1. Setup - Authentication and sandbox creation
        print("🔑 Setting up authentication...")
        print("   Run 'prime login' first to configure your API key")
        client = APIClient()  # Loads API key from config
        sandbox_client = SandboxClient(client)  # For sync operations
        
        print("📦 Creating demo sandbox...")
        request = CreateSandboxRequest(
            name=f"async-demo-{int(time.time())}",
            docker_image="python:3.11-slim",  # Use known working image
            cpu_cores=1,
            memory_gb=2,  # Standard memory
            timeout_minutes=60,  # Standard timeout
        )
        
        sandbox = sandbox_client.create(request)
        print(f"✅ Created sandbox: {sandbox.name} ({sandbox.id})")
        
        print("⏳ Waiting for sandbox to be ready...")
        sandbox_client.wait_for_creation(sandbox.id, max_attempts=60)
        print("✅ Sandbox is ready!")
        
        # 2. Prepare test data
        test_files = create_test_files(5)
        
        # 3. Demonstrate async operations with AsyncSandboxClient
        async with AsyncSandboxClient() as async_client:
            print(f"\n💫 Using AsyncSandboxClient for high-performance operations")
            
            # Sequential vs Concurrent comparison
            sequential_time = await demo_sequential_operations(async_client, sandbox.id, test_files)
            concurrent_time = await demo_concurrent_operations(async_client, sandbox.id, test_files)
            
            # Calculate performance improvement
            improvement = (sequential_time - concurrent_time) / sequential_time * 100
            print(f"\n📈 Performance Improvement: {improvement:.1f}% faster with async!")
            print(f"   Sequential: {sequential_time:.2f}s")
            print(f"   Concurrent: {concurrent_time:.2f}s")
            
            # Advanced async patterns
            await demo_mixed_concurrent_operations(async_client, sandbox.id, test_files)
            await demo_error_handling(async_client, sandbox.id)
        
        # 4. Cleanup
        print("\n🧹 Cleaning up...")
        
        # Clean up local test files
        for file_path in test_files:
            try:
                os.unlink(file_path)
            except:
                pass
        
        # Clean up downloaded files
        import glob
        for pattern in ["/tmp/downloaded_*.txt", "/tmp/mixed_*.txt", "/tmp/invalid.txt"]:
            for file_path in glob.glob(pattern):
                try:
                    os.unlink(file_path)
                except:
                    pass
        
        # Delete the sandbox we created
        sandbox_client.delete(sandbox.id)
        print(f"✅ Deleted sandbox: {sandbox.id}")
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Takeaways:")
        print("• AsyncSandboxClient provides significant performance improvements")
        print("• asyncio.gather() enables true concurrent file operations")
        print("• Proper error handling prevents failures from affecting other operations")
        print("• Async context managers ensure proper resource cleanup")
        print("• Prime CLI now supports high-performance file transfer workflows!")
        
    except APIError as e:
        print(f"❌ API Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async demo
    asyncio.run(main())