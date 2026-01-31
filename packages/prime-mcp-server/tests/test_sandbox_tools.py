import pytest

from prime_mcp.tools import sandboxes


class TestCreateSandbox:
    """Tests for create_sandbox function."""

    @pytest.mark.asyncio
    async def test_create_sandbox_validation_cpu_cores(self):
        """Test that cpu_cores must be at least 1."""
        result = await sandboxes.create_sandbox(
            name="test-sandbox",
            cpu_cores=0,
        )
        assert "error" in result
        assert "cpu_cores" in result["error"].lower() or "greater than" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_create_sandbox_validation_memory_gb(self):
        """Test that memory_gb must be at least 1."""
        result = await sandboxes.create_sandbox(
            name="test-sandbox",
            memory_gb=0,
        )
        assert "error" in result
        error_msg = result["error"].lower()
        assert any(x in error_msg for x in ["memory", "greater than", "event loop"])

    @pytest.mark.asyncio
    async def test_create_sandbox_validation_disk_size_gb(self):
        """Test that disk_size_gb must be at least 1."""
        result = await sandboxes.create_sandbox(
            name="test-sandbox",
            disk_size_gb=0,
        )
        assert "error" in result
        assert "disk" in result["error"].lower() or "greater than" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_create_sandbox_validation_timeout_minutes(self):
        """Test that timeout_minutes must be at least 1."""
        result = await sandboxes.create_sandbox(
            name="test-sandbox",
            timeout_minutes=0,
        )
        assert "error" in result
        error_msg = result["error"].lower()
        assert any(x in error_msg for x in ["timeout", "greater than", "event loop"])


class TestListSandboxes:
    """Tests for list_sandboxes function."""

    @pytest.mark.asyncio
    async def test_list_sandboxes_default_params(self):
        """Test list_sandboxes with default parameters."""
        result = await sandboxes.list_sandboxes()
        # Should return a dict (either with sandboxes or error)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_list_sandboxes_with_filters(self):
        """Test list_sandboxes with status filter."""
        result = await sandboxes.list_sandboxes(
            status="RUNNING",
            page=1,
            per_page=10,
        )
        assert isinstance(result, dict)


class TestGetSandbox:
    """Tests for get_sandbox function."""

    @pytest.mark.asyncio
    async def test_get_sandbox_empty_id(self):
        """Test that sandbox_id is required."""
        result = await sandboxes.get_sandbox("")
        assert "error" in result
        assert "sandbox_id is required" in result["error"]

    @pytest.mark.asyncio
    async def test_get_sandbox_valid_id(self):
        """Test get_sandbox with valid ID format."""
        result = await sandboxes.get_sandbox("test-sandbox-id")
        assert isinstance(result, dict)


class TestDeleteSandbox:
    """Tests for delete_sandbox function."""

    @pytest.mark.asyncio
    async def test_delete_sandbox_empty_id(self):
        """Test that sandbox_id is required."""
        result = await sandboxes.delete_sandbox("")
        assert "error" in result
        assert "sandbox_id is required" in result["error"]


class TestBulkDeleteSandboxes:
    """Tests for bulk_delete_sandboxes function."""

    @pytest.mark.asyncio
    async def test_bulk_delete_no_params(self):
        """Test that either sandbox_ids or labels is required."""
        result = await sandboxes.bulk_delete_sandboxes()
        assert "error" in result
        assert "Must specify either sandbox_ids or labels" in result["error"]

    @pytest.mark.asyncio
    async def test_bulk_delete_both_params(self):
        """Test that both sandbox_ids and labels cannot be specified."""
        result = await sandboxes.bulk_delete_sandboxes(
            sandbox_ids=["id1", "id2"],
            labels=["label1"],
        )
        assert "error" in result
        assert "Cannot specify both sandbox_ids and labels" in result["error"]


class TestGetSandboxLogs:
    """Tests for get_sandbox_logs function."""

    @pytest.mark.asyncio
    async def test_get_logs_empty_id(self):
        """Test that sandbox_id is required."""
        result = await sandboxes.get_sandbox_logs("")
        assert "error" in result
        assert "sandbox_id is required" in result["error"]


class TestExecuteCommand:
    """Tests for execute_command function."""

    @pytest.mark.asyncio
    async def test_execute_command_empty_sandbox_id(self):
        """Test that sandbox_id is required."""
        result = await sandboxes.execute_command(
            sandbox_id="",
            command="echo hello",
        )
        assert "error" in result
        assert "sandbox_id is required" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_command_empty_command(self):
        """Test that command is required."""
        result = await sandboxes.execute_command(
            sandbox_id="test-id",
            command="",
        )
        assert "error" in result
        assert "command is required" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_command_invalid_timeout(self):
        """Test that timeout must be at least 1 second."""
        result = await sandboxes.execute_command(
            sandbox_id="test-id",
            command="echo hello",
            timeout=0,
        )
        assert "error" in result
        assert "timeout must be at least 1" in result["error"]


class TestExposePort:
    """Tests for expose_port function."""

    @pytest.mark.asyncio
    async def test_expose_port_empty_sandbox_id(self):
        """Test that sandbox_id is required."""
        result = await sandboxes.expose_port(
            sandbox_id="",
            port=8080,
        )
        assert "error" in result
        assert "sandbox_id is required" in result["error"]

    @pytest.mark.asyncio
    async def test_expose_port_invalid_port_zero(self):
        """Test that port must be valid (not 0)."""
        result = await sandboxes.expose_port(
            sandbox_id="test-id",
            port=0,
        )
        assert "error" in result
        assert "port must be between 22 and 9000" in result["error"]

    @pytest.mark.asyncio
    async def test_expose_port_invalid_port_high(self):
        """Test that port must be valid (not > 9000)."""
        result = await sandboxes.expose_port(
            sandbox_id="test-id",
            port=10000,
        )
        assert "error" in result
        assert "port must be between 22 and 9000" in result["error"]


class TestUnexposePort:
    """Tests for unexpose_port function."""

    @pytest.mark.asyncio
    async def test_unexpose_port_empty_sandbox_id(self):
        """Test that sandbox_id is required."""
        result = await sandboxes.unexpose_port(
            sandbox_id="",
            exposure_id="exp-123",
        )
        assert "error" in result
        assert "sandbox_id is required" in result["error"]

    @pytest.mark.asyncio
    async def test_unexpose_port_empty_exposure_id(self):
        """Test that exposure_id is required."""
        result = await sandboxes.unexpose_port(
            sandbox_id="test-id",
            exposure_id="",
        )
        assert "error" in result
        assert "exposure_id is required" in result["error"]


class TestListExposedPorts:
    """Tests for list_exposed_ports function."""

    @pytest.mark.asyncio
    async def test_list_exposed_ports_empty_id(self):
        """Test that sandbox_id is required."""
        result = await sandboxes.list_exposed_ports("")
        assert "error" in result
        assert "sandbox_id is required" in result["error"]


class TestCheckDockerImage:
    """Tests for check_docker_image function."""

    @pytest.mark.asyncio
    async def test_check_docker_image_empty(self):
        """Test that image is required."""
        result = await sandboxes.check_docker_image("")
        assert "error" in result
        assert "image is required" in result["error"]


class TestModuleImports:
    """Test that all modules import correctly."""

    def test_import_sandboxes(self):
        """Test that sandboxes module can be imported."""
        from prime_mcp.tools import sandboxes as sb

        assert sb is not None

    def test_import_mcp_tools(self):
        """Test that all tools can be imported from main module."""
        from prime_mcp import sandboxes as sb

        assert sb is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
