import pytest

from prime_mcp.tools import availability, pods, ssh


@pytest.mark.asyncio
async def test_check_gpu_availability():
    """Test check_gpu_availability function signature and structure"""
    result = await availability.check_gpu_availability(
        gpu_type="A100_80GB",
        regions="us-east",
        socket="PCIe",
        security="secure_cloud",
    )

    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_check_cluster_availability():
    """Test check_cluster_availability function signature"""
    result = await availability.check_cluster_availability(
        regions=["us-east", "us-west"],
        gpu_count=4,
        gpu_type="H100_80GB",
        socket="SXM5",
        security="secure_cloud",
    )

    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_list_pods():
    """Test list_pods function"""
    result = await pods.list_pods(offset=0, limit=10)

    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_get_pods_history():
    """Test get_pods_history function"""
    result = await pods.get_pods_history(
        limit=10, offset=0, sort_by="terminatedAt", sort_order="desc"
    )

    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_get_pods_status():
    """Test get_pods_status function"""
    result = await pods.get_pods_status(pod_ids=None)

    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_create_pod_validation():
    """Test create_pod parameter validation"""
    # Test gpu_count validation
    result = await pods.create_pod(
        cloud_id="test-cloud-id",
        gpu_type="A100_80GB",
        provider_type="runpod",
        gpu_count=0,  # Invalid
    )

    assert "error" in result
    assert "gpu_count must be greater than 0" in result["error"]


@pytest.mark.asyncio
async def test_create_pod_disk_size_validation():
    """Test create_pod disk_size validation"""
    result = await pods.create_pod(
        cloud_id="test-cloud-id",
        gpu_type="A100_80GB",
        provider_type="runpod",
        disk_size=0,  # Invalid
    )

    assert "error" in result
    assert "disk_size must be greater than 0" in result["error"]


@pytest.mark.asyncio
async def test_create_pod_vcpus_validation():
    """Test create_pod vcpus validation"""
    result = await pods.create_pod(
        cloud_id="test-cloud-id",
        gpu_type="A100_80GB",
        provider_type="runpod",
        vcpus=0,  # Invalid
    )

    assert "error" in result
    assert "vcpus must be greater than 0" in result["error"]


@pytest.mark.asyncio
async def test_create_pod_memory_validation():
    """Test create_pod memory validation"""
    result = await pods.create_pod(
        cloud_id="test-cloud-id",
        gpu_type="A100_80GB",
        provider_type="runpod",
        memory=0,  # Invalid
    )

    assert "error" in result
    assert "memory must be greater than 0" in result["error"]


@pytest.mark.asyncio
async def test_manage_ssh_keys_list():
    """Test SSH key listing"""
    result = await ssh.manage_ssh_keys(action="list", offset=0, limit=10)

    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_manage_ssh_keys_add_validation():
    """Test SSH key add validation"""
    # Missing required parameters
    result = await ssh.manage_ssh_keys(action="add")

    assert "error" in result
    assert "key_name and public_key are required" in result["error"]


@pytest.mark.asyncio
async def test_manage_ssh_keys_delete_validation():
    """Test SSH key delete validation"""
    # Missing required parameter
    result = await ssh.manage_ssh_keys(action="delete")

    assert "error" in result
    assert "key_id is required" in result["error"]


@pytest.mark.asyncio
async def test_manage_ssh_keys_set_primary_validation():
    """Test SSH key set_primary validation"""
    # Missing required parameter
    result = await ssh.manage_ssh_keys(action="set_primary")

    assert "error" in result
    assert "key_id is required" in result["error"]


@pytest.mark.asyncio
async def test_manage_ssh_keys_invalid_action():
    """Test SSH key management with invalid action"""
    result = await ssh.manage_ssh_keys(action="invalid_action")

    assert "error" in result
    assert "Invalid action" in result["error"]


@pytest.mark.asyncio
async def test_get_pods_history_sort_validation():
    """Test get_pods_history with valid sort parameters"""
    # Test with valid sort_by options
    result1 = await pods.get_pods_history(sort_by="terminatedAt")
    assert isinstance(result1, dict)

    result2 = await pods.get_pods_history(sort_by="createdAt")
    assert isinstance(result2, dict)

    # Test with valid sort_order options
    result3 = await pods.get_pods_history(sort_order="asc")
    assert isinstance(result3, dict)

    result4 = await pods.get_pods_history(sort_order="desc")
    assert isinstance(result4, dict)


def test_imports():
    """Test that all main modules can be imported"""
    from prime_mcp import make_prime_request, mcp

    assert mcp is not None
    assert make_prime_request is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
