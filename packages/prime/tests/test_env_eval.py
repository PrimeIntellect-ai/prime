"""Tests for prime env eval command with Prime Inference"""

import os
import subprocess
import tempfile

import pytest

# Use a small/fast model for testing
TEST_MODEL = "deepseek/deepseek-chat"


@pytest.fixture(scope="module")
def install_math_env():
    """Install the single-turn-math environment for testing"""
    result = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "single_turn_math",
            "--extra-index-url",
            "https://hub.primeintellect.ai/primeintellect/simple/",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"Failed to install single_turn_math: {result.stderr}")
    yield
    # Cleanup: uninstall after tests
    subprocess.run(
        ["uv", "pip", "uninstall", "single_turn_math", "-y"],
        capture_output=True,
    )


def test_env_eval_single_turn_math(install_math_env):
    """Test running prime env eval with single_turn_math environment

    This test runs a minimal evaluation (1 example, 1 rollout) against
    Prime Inference to verify the end-to-end eval pipeline works.
    """
    # Run from a temp directory to avoid polluting the source tree with results
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "uv",
                "run",
                "prime",
                "env",
                "eval",
                "single_turn_math",
                "-m",
                TEST_MODEL,
                "-n",
                "1",  # Only 1 example
                "-r",
                "1",  # Only 1 rollout
                "-c",
                "1",  # Max 1 concurrent request
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=tmpdir,
            env={**os.environ, "PRIME_API_KEY": os.environ.get("PRIME_API_KEY", "")},
        )

        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")

        assert result.returncode == 0, f"Eval failed: {result.stderr}\n{result.stdout}"


def test_env_eval_invalid_model(install_math_env):
    """Test that prime env eval fails gracefully with invalid model"""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "uv",
                "run",
                "prime",
                "env",
                "eval",
                "single_turn_math",
                "-m",
                "nonexistent/fake-model-12345",
                "-n",
                "1",
                "-r",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=tmpdir,
            env={**os.environ, "PRIME_API_KEY": os.environ.get("PRIME_API_KEY", "")},
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0, f"Expected failure but got: {result.stdout}"
        # Should show error about invalid model
        output = result.stdout.lower() + result.stderr.lower()
        assert "invalid model" in output or "not found" in output


def test_env_eval_missing_environment():
    """Test that prime env eval fails gracefully with missing environment"""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "uv",
                "run",
                "prime",
                "env",
                "eval",
                "nonexistent_env_xyz_12345",
                "-m",
                TEST_MODEL,
                "-n",
                "1",
                "-r",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=tmpdir,
            env={**os.environ, "PRIME_API_KEY": os.environ.get("PRIME_API_KEY", "")},
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0, f"Expected failure but got: {result.stdout}\n{result.stderr}"
