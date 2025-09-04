#!/usr/bin/env python3
"""
Advanced async concurrency test for Prime CLI
Tests the asyncio.gather() concurrent operations we implemented
"""

import asyncio
import os
import tempfile
import time
import sys
from pathlib import Path

# Add the src directory to Python path so we can import prime_cli
sys.path.insert(0, str(Path(__file__).parent / "src"))

from prime_cli.api.sandbox import AsyncSandboxClient
from prime_cli.config import Config


class AsyncTestSuite:
    def __init__(self, sandbox_id: str):
        self.sandbox_id = sandbox_id
        self.test_files = []
        self.tests_passed = 0
        self.tests_failed = 0

    def log_info(self, msg: str):
        print(f"[INFO] {msg}")

    def log_success(self, msg: str):
        print(f"\033[0;32m[SUCCESS]\033[0m {msg}")
        self.tests_passed += 1

    def log_error(self, msg: str):
        print(f"\033[0;31m[ERROR]\033[0m {msg}")
        self.tests_failed += 1

    def create_test_files(self, count: int = 10):
        """Create test files for concurrent operations"""
        self.log_info(f"Creating {count} test files...")
        
        for i in range(count):
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'_async_test_{i}.txt') as f:
                f.write(f"Async test file {i}\nCreated at: {time.time()}\nContent: {'x' * (100 + i * 10)}")
                self.test_files.append(f.name)
        
        self.log_success(f"Created {len(self.test_files)} test files")

    async def test_concurrent_uploads(self):
        """Test uploading multiple files concurrently"""
        self.log_info("Testing concurrent uploads with asyncio.gather()...")
        
        async with AsyncSandboxClient() as client:
            start_time = time.time()
            
            # Create upload tasks for all files
            upload_tasks = [
                client.upload_path(
                    self.sandbox_id,
                    file_path,
                    f"/workspace/async_upload_{i}.txt"
                )
                for i, file_path in enumerate(self.test_files[:5])  # Upload first 5 files
            ]
            
            # Run all uploads concurrently
            try:
                results = await asyncio.gather(*upload_tasks)
                end_time = time.time()
                
                duration = end_time - start_time
                self.log_success(f"Concurrent uploads completed in {duration:.2f}s")
                
                # Verify all uploads succeeded
                success_count = sum(1 for r in results if r.success)
                if success_count == len(upload_tasks):
                    self.log_success(f"All {success_count} uploads successful")
                else:
                    self.log_error(f"Only {success_count}/{len(upload_tasks)} uploads successful")
                    return False
                
            except Exception as e:
                self.log_error(f"Concurrent upload failed: {e}")
                return False
        
        return True

    async def test_concurrent_downloads(self):
        """Test downloading multiple files concurrently"""
        self.log_info("Testing concurrent downloads with asyncio.gather()...")
        
        async with AsyncSandboxClient() as client:
            start_time = time.time()
            
            # Create download tasks
            download_tasks = [
                client.download_path(
                    self.sandbox_id,
                    f"/workspace/async_upload_{i}.txt",
                    f"/tmp/async_download_{i}.txt"
                )
                for i in range(5)  # Download the 5 files we uploaded
            ]
            
            try:
                # Run all downloads concurrently
                await asyncio.gather(*download_tasks)
                end_time = time.time()
                
                duration = end_time - start_time
                self.log_success(f"Concurrent downloads completed in {duration:.2f}s")
                
                # Verify downloaded files exist
                downloaded_count = 0
                for i in range(5):
                    download_path = f"/tmp/async_download_{i}.txt"
                    if os.path.exists(download_path):
                        downloaded_count += 1
                
                if downloaded_count == 5:
                    self.log_success(f"All {downloaded_count} downloads successful")
                else:
                    self.log_error(f"Only {downloaded_count}/5 downloads successful")
                    return False
                    
            except Exception as e:
                self.log_error(f"Concurrent download failed: {e}")
                return False
        
        return True

    async def test_mixed_concurrent_operations(self):
        """Test mixed upload/download operations running concurrently"""
        self.log_info("Testing mixed concurrent operations...")
        
        async with AsyncSandboxClient() as client:
            start_time = time.time()
            
            # Mix of upload and download tasks
            mixed_tasks = []
            
            # Add some new uploads
            for i in range(5, 8):  # Upload files 5-7
                if i < len(self.test_files):
                    mixed_tasks.append(
                        client.upload_path(
                            self.sandbox_id,
                            self.test_files[i],
                            f"/workspace/mixed_upload_{i}.txt"
                        )
                    )
            
            # Add some downloads of existing files
            for i in range(3):  # Download files 0-2
                mixed_tasks.append(
                    client.download_path(
                        self.sandbox_id,
                        f"/workspace/async_upload_{i}.txt",
                        f"/tmp/mixed_download_{i}.txt"
                    )
                )
            
            try:
                # Run mixed operations concurrently
                results = await asyncio.gather(*mixed_tasks, return_exceptions=True)
                end_time = time.time()
                
                duration = end_time - start_time
                
                # Count successes and failures
                successes = sum(1 for r in results if not isinstance(r, Exception))
                failures = len(results) - successes
                
                if failures == 0:
                    self.log_success(f"Mixed operations completed in {duration:.2f}s ({successes} operations)")
                else:
                    self.log_error(f"Mixed operations had {failures} failures out of {len(results)} operations")
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            self.log_error(f"  Task {i}: {result}")
                    return False
                    
            except Exception as e:
                self.log_error(f"Mixed concurrent operations failed: {e}")
                return False
        
        return True

    async def test_error_handling_concurrency(self):
        """Test concurrent operations with some expected failures"""
        self.log_info("Testing concurrent error handling...")
        
        async with AsyncSandboxClient() as client:
            # Mix of valid and invalid operations
            mixed_tasks = [
                # Valid upload
                client.upload_path(self.sandbox_id, self.test_files[0], "/workspace/valid_upload.txt"),
                # Invalid download (file doesn't exist)
                client.download_path(self.sandbox_id, "/workspace/nonexistent.txt", "/tmp/invalid_download.txt"),
                # Another valid upload
                client.upload_path(self.sandbox_id, self.test_files[1], "/workspace/valid_upload_2.txt"),
            ]
            
            try:
                results = await asyncio.gather(*mixed_tasks, return_exceptions=True)
                
                # Should have 2 successes and 1 failure
                successes = sum(1 for r in results if not isinstance(r, Exception))
                failures = sum(1 for r in results if isinstance(r, Exception))
                
                if successes == 2 and failures == 1:
                    self.log_success("Error handling in concurrent operations working correctly")
                else:
                    self.log_error(f"Unexpected results: {successes} successes, {failures} failures")
                    return False
                    
            except Exception as e:
                self.log_error(f"Concurrent error handling test failed: {e}")
                return False
        
        return True

    def cleanup(self):
        """Clean up test files"""
        self.log_info("Cleaning up test files...")
        
        for file_path in self.test_files:
            try:
                os.unlink(file_path)
            except:
                pass
        
        # Clean up downloaded files
        import glob
        for pattern in ["/tmp/async_download_*.txt", "/tmp/mixed_download_*.txt", "/tmp/invalid_download.txt"]:
            for file_path in glob.glob(pattern):
                try:
                    os.unlink(file_path)
                except:
                    pass

    async def run_all_tests(self):
        """Run all async concurrency tests"""
        print("=" * 60)
        print("Prime CLI Async Concurrency Test Suite")
        print("=" * 60)
        
        try:
            # Setup
            self.create_test_files(10)
            
            # Run tests
            await self.test_concurrent_uploads()
            await self.test_concurrent_downloads()  
            await self.test_mixed_concurrent_operations()
            await self.test_error_handling_concurrency()
            
        finally:
            self.cleanup()
        
        # Results
        print("\n" + "=" * 40)
        print("TEST RESULTS")
        print("=" * 40)
        print(f"Tests Passed: \033[0;32m{self.tests_passed}\033[0m")
        print(f"Tests Failed: \033[0;31m{self.tests_failed}\033[0m") 
        print(f"Total Tests: {self.tests_passed + self.tests_failed}")
        
        if self.tests_failed == 0:
            print("\033[0;32m🎉 All async concurrency tests passed!\033[0m")
            return True
        else:
            print("\033[0;31m❌ Some async concurrency tests failed.\033[0m")
            return False


async def main():
    if len(sys.argv) != 2:
        print("Usage: python test_async_concurrency.py <sandbox-id>")
        sys.exit(1)
    
    sandbox_id = sys.argv[1]
    
    # Verify we can connect
    try:
        config = Config()
        if not config.api_key:
            print("Error: No API key configured. Run 'prime config use local' first.")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Run tests
    test_suite = AsyncTestSuite(sandbox_id)
    success = await test_suite.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())