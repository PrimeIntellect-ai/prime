"""Tests for private environment installation and caching."""

import os
import subprocess
from pathlib import Path

import pytest

# Test with a known private environment
ENV_OWNER = "prime-cli-test"
ENV_NAME = "private-reverse-text"


@pytest.fixture
def temp_home(tmp_path: Path):
    """Temporarily set HOME to a temp directory for cache isolation."""
    original_home = os.environ.get("HOME")
    os.environ["HOME"] = str(tmp_path)

    yield tmp_path

    # Cleanup: uninstall after tests
    subprocess.run(
        ["uv", "pip", "uninstall", ENV_NAME.replace("-", "_"), "-y"],
        capture_output=True,
    )

    # Restore HOME
    if original_home:
        os.environ["HOME"] = original_home


class TestPrivateEnvInstall:
    """Tests for private environment installation."""

    @pytest.mark.skipif(
        not os.environ.get("PRIME_API_KEY"),
        reason="PRIME_API_KEY not set - required for private env access",
    )
    def test_install_private_env_creates_cache(self, temp_home: Path):
        """Test that installing a private env creates the correct cache structure."""
        # Install the private environment
        result = subprocess.run(
            ["uv", "run", "prime", "env", "install", f"{ENV_OWNER}/{ENV_NAME}"],
            capture_output=True,
            text=True,
            timeout=300,
            env={
                **os.environ,
                "HOME": str(temp_home),
                "PRIME_API_KEY": os.environ.get("PRIME_API_KEY", ""),
            },
        )

        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")

        assert result.returncode == 0, f"Install failed: {result.stderr}\n{result.stdout}"

        # Verify cache structure: ~/.prime/wheel_cache/{owner}/{name}/{version}/
        envs_cache = temp_home / ".prime" / "wheel_cache"
        assert envs_cache.exists(), "Cache directory ~/.prime/wheel_cache/ not created"

        owner_dir = envs_cache / ENV_OWNER
        assert owner_dir.exists(), f"Owner directory not created: {owner_dir}"

        name_dir = owner_dir / ENV_NAME
        assert name_dir.exists(), f"Environment directory not created: {name_dir}"

        # Should have at least one version directory
        version_dirs = [d for d in name_dir.iterdir() if d.is_dir()]
        assert len(version_dirs) > 0, "No version directories found"

        version_dir = version_dirs[0]
        # Note: version may be "latest" if API doesn't return semantic_version
        # The important thing is the cache exists and wheel was built

        # Verify wheel was built
        dist_dir = version_dir / "dist"
        assert dist_dir.exists(), f"dist/ directory not created: {dist_dir}"

        wheels = list(dist_dir.glob("*.whl"))
        assert len(wheels) > 0, "No wheel file found in dist/"

        # Verify metadata was saved
        metadata_path = version_dir / ".prime" / ".env-metadata.json"
        assert metadata_path.exists(), f"Metadata file not created: {metadata_path}"

    @pytest.mark.skipif(
        not os.environ.get("PRIME_API_KEY"),
        reason="PRIME_API_KEY not set - required for private env access",
    )
    def test_installed_private_env_can_be_loaded(self, temp_home: Path):
        """Test that an installed private env can be loaded by verifiers."""
        # First install the environment
        install_result = subprocess.run(
            ["uv", "run", "prime", "env", "install", f"{ENV_OWNER}/{ENV_NAME}"],
            capture_output=True,
            text=True,
            timeout=300,
            env={
                **os.environ,
                "HOME": str(temp_home),
                "PRIME_API_KEY": os.environ.get("PRIME_API_KEY", ""),
            },
        )

        assert install_result.returncode == 0, (
            f"Install failed: {install_result.stderr}\n{install_result.stdout}"
        )

        # Try to load the environment using verifiers, both with and without the owner/name
        load_script = f"""
import sys
try:
    from verifiers import load_environment
    env = load_environment('{ENV_NAME.replace("-", "_")}')
    env = load_environment('{ENV_OWNER}/{ENV_NAME}')
    print(f"Successfully loaded: {{type(env).__name__}}")
    sys.exit(0)
except ImportError as e:
    print(f"Import error: {{e}}")
    sys.exit(1)
except Exception as e:
    print(f"Load error: {{e}}")
    sys.exit(1)
"""

        load_result = subprocess.run(
            ["uv", "run", "python", "-c", load_script],
            capture_output=True,
            text=True,
            timeout=60,
            env={
                **os.environ,
                "HOME": str(temp_home),
                "PRIME_API_KEY": os.environ.get("PRIME_API_KEY", ""),
            },
        )

        print(f"Load stdout: {load_result.stdout}")
        print(f"Load stderr: {load_result.stderr}")

        assert load_result.returncode == 0, (
            f"Failed to load environment: {load_result.stderr}\n{load_result.stdout}"
        )
        assert "Successfully loaded" in load_result.stdout

    @pytest.mark.skipif(
        not os.environ.get("PRIME_API_KEY"),
        reason="PRIME_API_KEY not set - required for private env access",
    )
    def test_cached_wheel_is_reused(self, temp_home: Path):
        """Test that subsequent installs reuse the cached wheel."""
        # First install
        result1 = subprocess.run(
            ["uv", "run", "prime", "env", "install", f"{ENV_OWNER}/{ENV_NAME}"],
            capture_output=True,
            text=True,
            timeout=300,
            env={
                **os.environ,
                "HOME": str(temp_home),
                "PRIME_API_KEY": os.environ.get("PRIME_API_KEY", ""),
            },
        )
        assert result1.returncode == 0, f"First install failed: {result1.stderr}"

        # Get the wheel modification time
        envs_cache = temp_home / ".prime" / "wheel_cache" / ENV_OWNER / ENV_NAME
        version_dirs = list(envs_cache.iterdir())
        assert len(version_dirs) > 0, f"No version dirs in {envs_cache}"
        wheel_files = list((version_dirs[0] / "dist").glob("*.whl"))
        assert len(wheel_files) > 0, f"No wheels in {version_dirs[0] / 'dist'}"
        wheel_mtime_1 = wheel_files[0].stat().st_mtime

        # Second install (should reuse cache)
        result2 = subprocess.run(
            ["uv", "run", "prime", "env", "install", f"{ENV_OWNER}/{ENV_NAME}"],
            capture_output=True,
            text=True,
            timeout=300,
            env={
                **os.environ,
                "HOME": str(temp_home),
                "PRIME_API_KEY": os.environ.get("PRIME_API_KEY", ""),
            },
        )
        assert result2.returncode == 0, f"Second install failed: {result2.stderr}"

        # Verify cache message appears
        assert "Using cached wheel" in result2.stdout or "Using cached" in result2.stdout, (
            f"Expected cache reuse message in output: {result2.stdout}"
        )

        # Verify wheel wasn't rebuilt (same mtime)
        wheel_mtime_2 = wheel_files[0].stat().st_mtime
        assert wheel_mtime_1 == wheel_mtime_2, "Wheel was rebuilt instead of reusing cache"


class TestPathComponentValidation:
    """Tests for path component validation (prevents cache directory escape)."""

    def test_validate_blocks_double_dot(self):
        """Test that '..' in path components is rejected."""
        from prime_cli.commands.env import _validate_path_component

        with pytest.raises(ValueError, match="cannot contain '\\.\\.'"):
            _validate_path_component("..", "owner")

        with pytest.raises(ValueError, match="cannot contain '\\.\\.'"):
            _validate_path_component("foo/../bar", "name")

    def test_validate_blocks_path_separators(self):
        """Test that path separators in components are rejected."""
        from prime_cli.commands.env import _validate_path_component

        with pytest.raises(ValueError, match="path separators"):
            _validate_path_component("foo/bar", "version")

        with pytest.raises(ValueError, match="path separators"):
            _validate_path_component("foo\\bar", "version")

    def test_validate_blocks_empty(self):
        """Test that empty components are rejected."""
        from prime_cli.commands.env import _validate_path_component

        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_path_component("", "owner")

    def test_validate_blocks_null_bytes(self):
        """Test that null bytes in components are rejected."""
        from prime_cli.commands.env import _validate_path_component

        with pytest.raises(ValueError, match="null bytes"):
            _validate_path_component("foo\x00bar", "name")

    def test_validate_allows_normal_components(self):
        """Test that normal path components are allowed."""
        from prime_cli.commands.env import _validate_path_component

        # These should not raise
        _validate_path_component("primeintellect", "owner")
        _validate_path_component("my-environment", "name")
        _validate_path_component("1.0.0", "version")
        _validate_path_component("latest", "version")
        _validate_path_component("v2.3.4-beta.1", "version")


class TestSafeTarExtract:
    """Tests for tar extraction security (tar-slip prevention)."""

    def test_safe_extract_blocks_absolute_path(self, tmp_path: Path):
        """Test that absolute paths in tarball are rejected."""
        import tarfile

        from prime_cli.commands.env import _safe_tar_extract

        # Create a tarball with absolute path
        tar_path = tmp_path / "malicious.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add a file with absolute path
            info = tarfile.TarInfo(name="/etc/malicious")
            info.size = 0
            tar.addfile(info)

        dest = tmp_path / "dest"
        dest.mkdir()

        with tarfile.open(tar_path, "r:gz") as tar:
            with pytest.raises(ValueError, match="absolute path"):
                _safe_tar_extract(tar, dest)

    def test_safe_extract_blocks_path_traversal(self, tmp_path: Path):
        """Test that path traversal (..) in tarball is rejected."""
        import tarfile

        from prime_cli.commands.env import _safe_tar_extract

        # Create a tarball with path traversal
        tar_path = tmp_path / "malicious.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            info = tarfile.TarInfo(name="../../../etc/malicious")
            info.size = 0
            tar.addfile(info)

        dest = tmp_path / "dest"
        dest.mkdir()

        with tarfile.open(tar_path, "r:gz") as tar:
            with pytest.raises(ValueError, match="'\\.\\.'"):
                _safe_tar_extract(tar, dest)

    def test_safe_extract_allows_normal_paths(self, tmp_path: Path):
        """Test that normal paths in tarball are allowed."""
        import tarfile

        from prime_cli.commands.env import _safe_tar_extract

        # Create a tarball with normal paths
        tar_path = tmp_path / "normal.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add normal files
            info1 = tarfile.TarInfo(name="file.txt")
            info1.size = 0
            tar.addfile(info1)

            info2 = tarfile.TarInfo(name="subdir/nested.txt")
            info2.size = 0
            tar.addfile(info2)

        dest = tmp_path / "dest"
        dest.mkdir()

        with tarfile.open(tar_path, "r:gz") as tar:
            _safe_tar_extract(tar, dest)  # Should not raise

        assert (dest / "file.txt").exists()
        assert (dest / "subdir" / "nested.txt").exists()

    def test_safe_extract_blocks_symlinks(self, tmp_path: Path):
        """Test that symlinks in tarball are rejected (prevents symlink attacks)."""
        import tarfile

        from prime_cli.commands.env import _safe_tar_extract

        # Create a tarball with a symlink
        tar_path = tmp_path / "malicious.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add a symlink pointing outside
            info = tarfile.TarInfo(name="evil_link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/tmp"
            tar.addfile(info)

        dest = tmp_path / "dest"
        dest.mkdir()

        with tarfile.open(tar_path, "r:gz") as tar:
            with pytest.raises(ValueError, match="symlink"):
                _safe_tar_extract(tar, dest)

    def test_safe_extract_blocks_hardlinks(self, tmp_path: Path):
        """Test that hardlinks in tarball are rejected."""
        import tarfile

        from prime_cli.commands.env import _safe_tar_extract

        # Create a tarball with a hardlink
        tar_path = tmp_path / "malicious.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add a hardlink
            info = tarfile.TarInfo(name="evil_hardlink")
            info.type = tarfile.LNKTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)

        dest = tmp_path / "dest"
        dest.mkdir()

        with tarfile.open(tar_path, "r:gz") as tar:
            with pytest.raises(ValueError, match="hardlink"):
                _safe_tar_extract(tar, dest)
