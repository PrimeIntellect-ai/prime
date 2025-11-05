"""Tests for sandbox file operations (upload/download)"""

import tempfile
from pathlib import Path

import pytest

from prime_sandboxes import CreateSandboxRequest


@pytest.fixture(scope="module")
def shared_sandbox(sandbox_client):
    """Create a sandbox for this test module"""
    sandbox = None
    try:
        print("\n[SETUP] Creating sandbox for file operations tests...")
        sandbox = sandbox_client.create(
            CreateSandboxRequest(
                name="test-file-ops",
                docker_image="python:3.11-slim",
                cpu_cores=1,
                memory_gb=2,
                timeout_minutes=60,
            )
        )
        print(f"[SETUP] Created: {sandbox.id}")
        sandbox_client.wait_for_creation(sandbox.id)
        print("[SETUP] Sandbox ready!")
        yield sandbox
    finally:
        if sandbox and sandbox.id:
            try:
                sandbox_client.delete(sandbox.id)
                print(f"[TEARDOWN] Deleted {sandbox.id}")
            except Exception as e:
                print(f"[TEARDOWN] Warning: {e}")


def test_upload_text_file(sandbox_client, shared_sandbox):
    """Test uploading a text file to sandbox"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        test_content = "Hello from test!\nLine 2\nLine 3"
        f.write(test_content)
        local_path = f.name

    try:
        # Upload file
        remote_path = "/tmp/test_upload.txt"
        print(f"\nUploading file to {remote_path}...")
        result = sandbox_client.upload_file(shared_sandbox.id, remote_path, local_path)
        print(f"✓ Upload result: {result}")

        assert result.success is True
        assert result.path == remote_path

        # Verify file exists and has correct content
        cmd_result = sandbox_client.execute_command(shared_sandbox.id, f"cat {remote_path}")
        assert cmd_result.exit_code == 0
        assert cmd_result.stdout.strip() == test_content
    finally:
        Path(local_path).unlink(missing_ok=True)


def test_upload_and_download_file(sandbox_client, shared_sandbox):
    """Test uploading and downloading a file"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        test_content = "Upload and download test\nMultiple lines\n123456"
        f.write(test_content)
        upload_path = f.name

    download_path = None
    try:
        # Upload file
        remote_path = "/tmp/test_roundtrip.txt"
        print(f"\nUploading file to {remote_path}...")
        upload_result = sandbox_client.upload_file(shared_sandbox.id, remote_path, upload_path)
        assert upload_result.success is True

        # Download file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            download_path = f.name

        print(f"Downloading file from {remote_path}...")
        sandbox_client.download_file(shared_sandbox.id, remote_path, download_path)
        print("✓ Download complete")

        # Verify downloaded content matches uploaded content
        with open(download_path, "r") as f:
            downloaded_content = f.read()

        assert downloaded_content == test_content
    finally:
        Path(upload_path).unlink(missing_ok=True)
        if download_path:
            Path(download_path).unlink(missing_ok=True)


def test_upload_binary_file(sandbox_client, shared_sandbox):
    """Test uploading a binary file"""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        binary_content = bytes(range(256))
        f.write(binary_content)
        local_path = f.name

    try:
        # Upload binary file
        remote_path = "/tmp/test_binary.bin"
        print(f"\nUploading binary file to {remote_path}...")
        result = sandbox_client.upload_file(shared_sandbox.id, remote_path, local_path)
        assert result.success is True

        # Verify file exists and size is correct
        cmd_result = sandbox_client.execute_command(shared_sandbox.id, f"wc -c < {remote_path}")
        assert cmd_result.exit_code == 0
        assert int(cmd_result.stdout.strip()) == len(binary_content)
    finally:
        Path(local_path).unlink(missing_ok=True)


def test_download_nonexistent_file(sandbox_client, shared_sandbox):
    """Test downloading a file that doesn't exist"""
    from prime_core import APIError

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        download_path = f.name

    try:
        remote_path = "/tmp/nonexistent_file.txt"
        print(f"\nAttempting to download nonexistent file {remote_path}...")

        with pytest.raises(APIError):
            sandbox_client.download_file(shared_sandbox.id, remote_path, download_path)
        print("✓ Correctly raised APIError for nonexistent file")
    finally:
        Path(download_path).unlink(missing_ok=True)


def test_upload_python_script_and_execute(sandbox_client, shared_sandbox):
    """Test uploading a Python script and executing it"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        script_content = """#!/usr/bin/env python3
import sys
print("Hello from uploaded script!")
print(f"Python version: {sys.version}")
"""
        f.write(script_content)
        local_path = f.name

    try:
        remote_path = "/tmp/test_script.py"
        print(f"\nUploading Python script to {remote_path}...")
        result = sandbox_client.upload_file(shared_sandbox.id, remote_path, local_path)
        assert result.success is True

        # Make script executable and run it
        sandbox_client.execute_command(shared_sandbox.id, f"chmod +x {remote_path}")

        cmd_result = sandbox_client.execute_command(shared_sandbox.id, f"python3 {remote_path}")
        assert cmd_result.exit_code == 0
        assert "Hello from uploaded script!" in cmd_result.stdout
        assert "Python version:" in cmd_result.stdout
        print(f"✓ Script executed successfully:\n{cmd_result.stdout}")
    finally:
        Path(local_path).unlink(missing_ok=True)
