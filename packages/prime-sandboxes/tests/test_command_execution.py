"""Tests for advanced sandbox command execution"""

import time

import pytest

from prime_sandboxes import CreateSandboxRequest
from prime_sandboxes.exceptions import CommandTimeoutError


@pytest.fixture(scope="module")
def shared_sandbox(sandbox_client):
    """Create a sandbox for this test module"""
    sandbox = None
    try:
        print("\n[SETUP] Creating sandbox for command execution tests...")
        sandbox = sandbox_client.create(
            CreateSandboxRequest(
                name="test-cmd-exec",
                docker_image="python:3.11-slim",
                cpu_cores=1,
                memory_gb=2,
                timeout_minutes=60,
            )
        )
        print(f"[SETUP] Created: {sandbox.id}")
        sandbox_client.wait_for_creation(sandbox.id, max_attempts=120)
        print("[SETUP] Sandbox ready!")
        yield sandbox
    finally:
        if sandbox and sandbox.id:
            try:
                sandbox_client.delete(sandbox.id)
                print(f"[TEARDOWN] Deleted {sandbox.id}")
            except Exception as e:
                print(f"[TEARDOWN] Warning: {e}")


def test_command_with_working_directory(sandbox_client, shared_sandbox):
    """Test executing command with working directory"""
    # Create a test directory
    sandbox_client.execute_command(
        shared_sandbox.id, "mkdir -p /tmp/workdir && echo 'test' > /tmp/workdir/file.txt"
    )

    # Execute command in working directory
    print("\nExecuting command with working_dir=/tmp/workdir...")
    result = sandbox_client.execute_command(
        shared_sandbox.id, "pwd && ls -la", working_dir="/tmp/workdir"
    )

    assert result.exit_code == 0
    assert "/tmp/workdir" in result.stdout
    assert "file.txt" in result.stdout
    print(f"✓ Command executed in working directory:\n{result.stdout}")


def test_command_with_environment_variables(sandbox_client, shared_sandbox):
    """Test executing command with custom environment variables"""
    env_vars = {
        "TEST_VAR": "hello_world",
        "CUSTOM_VALUE": "12345",
        "PATH_APPEND": "/custom/path",
    }

    print("\nExecuting command with custom environment variables...")
    result = sandbox_client.execute_command(
        shared_sandbox.id,
        "echo $TEST_VAR && echo $CUSTOM_VALUE && echo $PATH_APPEND",
        env=env_vars,
    )

    assert result.exit_code == 0
    assert "hello_world" in result.stdout
    assert "12345" in result.stdout
    assert "/custom/path" in result.stdout
    print(f"✓ Environment variables set correctly:\n{result.stdout}")


def test_command_with_different_exit_codes(sandbox_client, shared_sandbox):
    """Test commands that exit with different codes"""
    # Success (exit 0)
    result = sandbox_client.execute_command(shared_sandbox.id, "exit 0")
    assert result.exit_code == 0

    # Various non-zero exit codes
    for exit_code in [1, 2, 127]:
        print(f"\nTesting exit code {exit_code}...")
        result = sandbox_client.execute_command(shared_sandbox.id, f"exit {exit_code}")
        assert result.exit_code == exit_code
        print(f"✓ Command exited with code {exit_code}")


def test_command_stdout_stderr_separation(sandbox_client, shared_sandbox):
    """Test that stdout and stderr are properly separated"""
    command = "echo 'stdout message' && echo 'stderr message' >&2"

    print(f"\nExecuting command: {command}")
    result = sandbox_client.execute_command(shared_sandbox.id, command)

    assert result.exit_code == 0
    assert "stdout message" in result.stdout
    assert "stderr message" in result.stderr
    print(f"✓ stdout: {result.stdout.strip()}")
    print(f"✓ stderr: {result.stderr.strip()}")


def test_command_with_multiline_output(sandbox_client, shared_sandbox):
    """Test command that produces multi-line output"""
    command = "for i in 1 2 3 4 5; do echo Line $i; done"

    print(f"\nExecuting command: {command}")
    result = sandbox_client.execute_command(shared_sandbox.id, command)

    assert result.exit_code == 0
    lines = result.stdout.strip().split("\n")
    assert len(lines) == 5
    for i in range(1, 6):
        assert f"Line {i}" in result.stdout
    print(f"✓ Multi-line output received:\n{result.stdout}")


def test_command_with_pipes_and_redirection(sandbox_client, shared_sandbox):
    """Test command with pipes and redirection"""
    command = "echo 'hello world' | tr 'a-z' 'A-Z' | tee /tmp/output.txt"

    print(f"\nExecuting command: {command}")
    result = sandbox_client.execute_command(shared_sandbox.id, command)

    assert result.exit_code == 0
    assert "HELLO WORLD" in result.stdout

    # Verify file was created
    verify_result = sandbox_client.execute_command(shared_sandbox.id, "cat /tmp/output.txt")
    assert "HELLO WORLD" in verify_result.stdout
    print("✓ Pipes and redirection work correctly")


def test_command_with_long_running_process(sandbox_client, shared_sandbox):
    """Test command that runs for a few seconds"""
    command = "sleep 3 && echo 'completed'"

    print(f"\nExecuting long-running command: {command}")
    result = sandbox_client.execute_command(shared_sandbox.id, command, timeout=10)

    assert result.exit_code == 0
    assert "completed" in result.stdout
    print("✓ Long-running command completed successfully")


def test_command_timeout_short(sandbox_client, shared_sandbox):
    """Test that command timeout works correctly"""
    command = "sleep 30"

    print(f"\nExecuting command with short timeout: {command}")
    with pytest.raises(CommandTimeoutError) as exc_info:
        sandbox_client.execute_command(shared_sandbox.id, command, timeout=2)

    assert shared_sandbox.id in str(exc_info.value)
    print(f"✓ Command correctly timed out: {exc_info.value}")


def test_command_with_combined_working_dir_and_env(sandbox_client, shared_sandbox):
    """Test command with both working_dir and env parameters"""
    # Setup test directory
    sandbox_client.execute_command(shared_sandbox.id, "mkdir -p /tmp/env_test")

    env_vars = {"MY_VAR": "test_value", "ANOTHER_VAR": "another_value"}

    print("\nExecuting command with both working_dir and env...")
    result = sandbox_client.execute_command(
        shared_sandbox.id,
        "pwd && echo $MY_VAR && echo $ANOTHER_VAR",
        working_dir="/tmp/env_test",
        env=env_vars,
    )

    assert result.exit_code == 0
    assert "/tmp/env_test" in result.stdout
    assert "test_value" in result.stdout
    assert "another_value" in result.stdout
    print(f"✓ Both working_dir and env work together:\n{result.stdout}")


def test_command_creates_and_reads_file(sandbox_client, shared_sandbox):
    """Test command that creates a file and another reads it"""
    content = "This is test content for file operations"

    # Create file
    print("\nCreating file...")
    create_result = sandbox_client.execute_command(
        shared_sandbox.id, f"echo '{content}' > /tmp/test_file.txt"
    )
    assert create_result.exit_code == 0

    # Read file
    print("Reading file...")
    read_result = sandbox_client.execute_command(shared_sandbox.id, "cat /tmp/test_file.txt")
    assert read_result.exit_code == 0
    assert content in read_result.stdout
    print(f"✓ File operations successful: {read_result.stdout.strip()}")


def test_start_background_job(sandbox_client, shared_sandbox):
    """Test start_background_job and get_background_job for long-running commands"""
    print("\nTesting start_background_job with 5s sleep...")
    job = sandbox_client.start_background_job(
        shared_sandbox.id,
        'sleep 5 && echo "background\'s done"',
    )

    assert job.job_id is not None
    assert job.sandbox_id == shared_sandbox.id
    print(f"✓ Job started: {job.job_id}")

    # Poll until complete
    for _ in range(20):
        status = sandbox_client.get_background_job(shared_sandbox.id, job)
        if status.completed:
            break
        time.sleep(1)

    assert status.completed
    assert status.exit_code == 0
    assert "background's done" in status.stdout
    print(f"✓ Background execution completed: {status.stdout.strip()}")


def test_start_background_job_with_working_dir(sandbox_client, shared_sandbox):
    """Test start_background_job with working directory"""
    sandbox_client.execute_command(shared_sandbox.id, "mkdir -p /tmp/bgtest")

    print("\nTesting start_background_job with working_dir...")
    job = sandbox_client.start_background_job(
        shared_sandbox.id,
        "pwd > output.txt && sleep 5 && cat output.txt",
        working_dir="/tmp/bgtest",
    )

    # Poll until complete
    for _ in range(20):
        status = sandbox_client.get_background_job(shared_sandbox.id, job)
        if status.completed:
            break
        time.sleep(1)

    assert status.completed
    assert status.exit_code == 0
    assert "/tmp/bgtest" in status.stdout
    print(f"✓ Background with working_dir: {status.stdout.strip()}")


def test_start_background_job_with_exit_command(sandbox_client, shared_sandbox):
    """Test start_background_job handles commands ending with exit"""
    print("\nTesting start_background_job with exit command...")
    job = sandbox_client.start_background_job(
        shared_sandbox.id,
        "echo 'before exit' && exit 5",
    )

    # Poll until complete
    for _ in range(20):
        status = sandbox_client.get_background_job(shared_sandbox.id, job)
        if status.completed:
            break
        time.sleep(1)

    assert status.completed
    assert status.exit_code == 5
    assert "before exit" in status.stdout
    print(f"✓ Exit command handled correctly, exit_code={status.exit_code}")
