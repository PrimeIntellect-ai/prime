"""Tests for sandbox CRUD operations, listing, and bulk operations"""

from prime_sandboxes import CreateSandboxRequest


def test_create_sandbox_with_custom_config(sandbox_client):
    """Test creating a sandbox with custom configuration"""
    sandbox = None
    try:
        print("\nCreating sandbox with custom configuration...")
        sandbox = sandbox_client.create(
            CreateSandboxRequest(
                name="test-custom-config",
                docker_image="python:3.11-slim",
                cpu_cores=2,
                memory_gb=4,
                disk_size_gb=10,
                timeout_minutes=120,
                labels=["test", "custom-config"],
            )
        )
        print(f"✓ Created sandbox: {sandbox.id}")

        assert sandbox.id is not None
        assert sandbox.name == "test-custom-config"
        assert sandbox.cpu_cores == 2
        assert sandbox.memory_gb == 4
        assert sandbox.disk_size_gb == 10
        assert sandbox.timeout_minutes == 120
        assert "test" in sandbox.labels
        assert "custom-config" in sandbox.labels
        print("✓ Sandbox configuration verified")
    finally:
        if sandbox and sandbox.id:
            print(f"\nCleaning up sandbox {sandbox.id}...")
            try:
                sandbox_client.delete(sandbox.id)
                print(f"✓ Deleted sandbox {sandbox.id}")
            except Exception as e:
                print(f"Warning: Failed to delete sandbox {sandbox.id}: {e}")


def test_get_sandbox(sandbox_client):
    """Test getting sandbox details"""
    sandbox = None
    try:
        print("\nCreating sandbox...")
        sandbox = sandbox_client.create(
            CreateSandboxRequest(
                name="test-get-sandbox",
                docker_image="python:3.11-slim",
            )
        )
        print(f"✓ Created sandbox: {sandbox.id}")

        # Get sandbox details
        print("Fetching sandbox details...")
        retrieved = sandbox_client.get(sandbox.id)

        assert retrieved.id == sandbox.id
        assert retrieved.name == "test-get-sandbox"
        assert retrieved.docker_image == "python:3.11-slim"
        assert retrieved.status in ["PENDING", "PROVISIONING", "RUNNING"]
        print(f"✓ Retrieved sandbox details: status={retrieved.status}")
    finally:
        if sandbox and sandbox.id:
            print(f"\nCleaning up sandbox {sandbox.id}...")
            try:
                sandbox_client.delete(sandbox.id)
                print(f"✓ Deleted sandbox {sandbox.id}")
            except Exception as e:
                print(f"Warning: Failed to delete sandbox {sandbox.id}: {e}")


def test_list_sandboxes(sandbox_client):
    """Test listing sandboxes"""
    sandbox = None
    try:
        print("\nCreating sandbox...")
        sandbox = sandbox_client.create(
            CreateSandboxRequest(
                name="test-list-sandbox",
                docker_image="python:3.11-slim",
                labels=["test-list"],
            )
        )
        print(f"✓ Created sandbox: {sandbox.id}")

        # List all sandboxes
        print("Listing sandboxes...")
        list_response = sandbox_client.list(per_page=50)

        assert list_response.sandboxes is not None
        assert len(list_response.sandboxes) > 0
        assert list_response.total >= 1

        # Check our sandbox is in the list
        found = False
        for s in list_response.sandboxes:
            if s.id == sandbox.id:
                found = True
                break

        assert found, f"Created sandbox {sandbox.id} not found in list"
        print(f"✓ Listed {len(list_response.sandboxes)} sandboxes")
    finally:
        if sandbox and sandbox.id:
            print(f"\nCleaning up sandbox {sandbox.id}...")
            try:
                sandbox_client.delete(sandbox.id)
                print(f"✓ Deleted sandbox {sandbox.id}")
            except Exception as e:
                print(f"Warning: Failed to delete sandbox {sandbox.id}: {e}")


def test_list_sandboxes_with_label_filter(sandbox_client, unique_id):
    """Test listing sandboxes filtered by label"""
    sandbox = None
    test_label = f"test-label-filter-{unique_id}"

    try:
        print(f"\nCreating sandbox with label '{test_label}'...")
        sandbox = sandbox_client.create(
            CreateSandboxRequest(
                name=f"test-label-filter-{unique_id}",
                docker_image="python:3.11-slim",
                labels=[test_label],
            )
        )
        print(f"✓ Created sandbox: {sandbox.id}")

        # List sandboxes with this label
        print(f"Listing sandboxes with label '{test_label}'...")
        list_response = sandbox_client.list(labels=[test_label])

        # Should find at least our sandbox
        assert len(list_response.sandboxes) >= 1

        # All sandboxes should have our label
        found_our_sandbox = False
        for s in list_response.sandboxes:
            assert test_label in s.labels
            if s.id == sandbox.id:
                found_our_sandbox = True

        assert found_our_sandbox
        print(f"✓ Found {len(list_response.sandboxes)} sandbox(es) with label")
    finally:
        if sandbox and sandbox.id:
            print(f"\nCleaning up sandbox {sandbox.id}...")
            try:
                sandbox_client.delete(sandbox.id)
                print(f"✓ Deleted sandbox {sandbox.id}")
            except Exception as e:
                print(f"Warning: Failed to delete sandbox {sandbox.id}: {e}")


def test_delete_sandbox(sandbox_client):
    """Test deleting a sandbox"""
    print("\nCreating sandbox...")
    sandbox = sandbox_client.create(
        CreateSandboxRequest(
            name="test-delete",
            docker_image="python:3.11-slim",
        )
    )
    sandbox_id = sandbox.id
    print(f"✓ Created sandbox: {sandbox_id}")

    # Delete sandbox
    print(f"Deleting sandbox {sandbox_id}...")
    result = sandbox_client.delete(sandbox_id)
    print(f"✓ Delete result: {result}")

    # Try to get the deleted sandbox - it should be gone or marked as TERMINATED
    try:
        deleted = sandbox_client.get(sandbox_id)
        # If we can still get it, it should be terminated
        assert deleted.status == "TERMINATED"
        print("✓ Sandbox status is TERMINATED")
    except Exception:
        # Or it might be completely gone, which is also fine
        print("✓ Sandbox completely removed")


def test_bulk_delete_by_ids(sandbox_client):
    """Test bulk deleting sandboxes by IDs"""
    sandboxes = []
    try:
        # Create multiple sandboxes
        print("\nCreating 3 sandboxes for bulk delete test...")
        for i in range(3):
            sandbox = sandbox_client.create(
                CreateSandboxRequest(
                    name=f"test-bulk-delete-{i}",
                    docker_image="python:3.11-slim",
                )
            )
            sandboxes.append(sandbox)
            print(f"✓ Created sandbox {i + 1}: {sandbox.id}")

        sandbox_ids = [s.id for s in sandboxes]

        # Bulk delete
        print(f"\nBulk deleting {len(sandbox_ids)} sandboxes...")
        result = sandbox_client.bulk_delete(sandbox_ids=sandbox_ids)

        assert len(result.succeeded) == len(sandbox_ids)
        print(f"✓ Bulk deleted {len(result.succeeded)} sandboxes")

        # Clear the list so we don't try to delete again in finally
        sandboxes = []
    finally:
        # Clean up any remaining sandboxes
        for sandbox in sandboxes:
            try:
                print(f"Cleaning up sandbox {sandbox.id}...")
                sandbox_client.delete(sandbox.id)
            except Exception as e:
                print(f"Warning: Failed to delete {sandbox.id}: {e}")


def test_bulk_delete_by_labels(sandbox_client, unique_id):
    """Test bulk deleting sandboxes by labels"""
    test_label = f"test-bulk-delete-{unique_id}"
    sandboxes = []

    try:
        # Create sandboxes with specific label
        print(f"\nCreating 2 sandboxes with label '{test_label}'...")
        for i in range(2):
            sandbox = sandbox_client.create(
                CreateSandboxRequest(
                    name=f"test-bulk-delete-label-{unique_id}-{i}",
                    docker_image="python:3.11-slim",
                    labels=[test_label],
                )
            )
            sandboxes.append(sandbox)
            print(f"✓ Created sandbox {i + 1}: {sandbox.id}")

        # Bulk delete by label
        print(f"\nBulk deleting sandboxes with label '{test_label}'...")
        result = sandbox_client.bulk_delete(labels=[test_label])

        assert len(result.succeeded) >= 2
        print(f"✓ Bulk deleted {len(result.succeeded)} sandboxes by label")

        # Clear the list so we don't try to delete again in finally
        sandboxes = []
    finally:
        # Clean up any remaining sandboxes
        for sandbox in sandboxes:
            try:
                print(f"Cleaning up sandbox {sandbox.id}...")
                sandbox_client.delete(sandbox.id)
            except Exception as e:
                print(f"Warning: Failed to delete {sandbox.id}: {e}")


def test_get_logs(sandbox_client):
    """Test getting sandbox logs"""
    sandbox = None
    try:
        print("\nCreating sandbox...")
        sandbox = sandbox_client.create(
            CreateSandboxRequest(
                name="test-logs",
                docker_image="python:3.11-slim",
            )
        )
        print(f"✓ Created sandbox: {sandbox.id}")

        print("Waiting for sandbox to be ready...")
        sandbox_client.wait_for_creation(sandbox.id, max_attempts=120)

        # Execute a command to generate some output
        sandbox_client.execute_command(sandbox.id, "echo 'test log message'")

        # Get logs
        print("Fetching sandbox logs...")
        logs = sandbox_client.get_logs(sandbox.id)

        assert logs is not None
        assert isinstance(logs, str)
        print(f"✓ Retrieved logs ({len(logs)} chars)")
    finally:
        if sandbox and sandbox.id:
            print(f"\nCleaning up sandbox {sandbox.id}...")
            try:
                sandbox_client.delete(sandbox.id)
                print(f"✓ Deleted sandbox {sandbox.id}")
            except Exception as e:
                print(f"Warning: Failed to delete sandbox {sandbox.id}: {e}")


def test_wait_for_creation(sandbox_client):
    """Test waiting for sandbox to be ready"""
    sandbox = None
    try:
        print("\nCreating sandbox...")
        sandbox = sandbox_client.create(
            CreateSandboxRequest(
                name="test-wait",
                docker_image="python:3.11-slim",
            )
        )
        print(f"✓ Created sandbox: {sandbox.id}")

        # Wait for it to be ready
        print("Waiting for sandbox to be ready...")
        sandbox_client.wait_for_creation(sandbox.id, max_attempts=120)
        print("✓ Sandbox is ready!")

        # Verify it's actually running and reachable
        retrieved = sandbox_client.get(sandbox.id)
        assert retrieved.status == "RUNNING"

        # Try executing a command to confirm it's reachable
        result = sandbox_client.execute_command(sandbox.id, "echo 'ready'")
        assert result.exit_code == 0
        assert "ready" in result.stdout
        print("✓ Sandbox is reachable and functional")
    finally:
        if sandbox and sandbox.id:
            print(f"\nCleaning up sandbox {sandbox.id}...")
            try:
                sandbox_client.delete(sandbox.id)
                print(f"✓ Deleted sandbox {sandbox.id}")
            except Exception as e:
                print(f"Warning: Failed to delete sandbox {sandbox.id}: {e}")


def test_sandbox_lifecycle(sandbox_client):
    """Test complete sandbox lifecycle: create -> use -> delete"""
    sandbox = None
    try:
        # Create
        print("\n[LIFECYCLE] Creating sandbox...")
        sandbox = sandbox_client.create(
            CreateSandboxRequest(
                name="test-lifecycle",
                docker_image="python:3.11-slim",
                labels=["lifecycle-test"],
            )
        )
        print(f"✓ Created: {sandbox.id}")

        # Wait for ready
        print("[LIFECYCLE] Waiting for sandbox to be ready...")
        sandbox_client.wait_for_creation(sandbox.id, max_attempts=120)
        print("✓ Sandbox is ready")

        # Use it
        print("[LIFECYCLE] Using sandbox...")
        result = sandbox_client.execute_command(sandbox.id, "echo 'lifecycle test'")
        assert result.exit_code == 0
        print("✓ Executed command successfully")

        # Get details
        print("[LIFECYCLE] Getting sandbox details...")
        details = sandbox_client.get(sandbox.id)
        assert details.status == "RUNNING"
        print(f"✓ Status: {details.status}")

        # Delete
        print("[LIFECYCLE] Deleting sandbox...")
        sandbox_client.delete(sandbox.id)
        print("✓ Deleted successfully")

        sandbox = None  # Don't try to delete again in finally
    finally:
        if sandbox and sandbox.id:
            print(f"\nCleaning up sandbox {sandbox.id}...")
            try:
                sandbox_client.delete(sandbox.id)
                print(f"✓ Deleted sandbox {sandbox.id}")
            except Exception as e:
                print(f"Warning: Failed to delete sandbox {sandbox.id}: {e}")
