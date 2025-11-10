"""
Basic usage example for Prime MCP.

This example demonstrates:
- Checking GPU availability
- Creating a pod
- Monitoring pod status
- Managing SSH keys
- Cleaning up resources

Before running:
1. Set PRIME_API_KEY environment variable, or
2. Run `prime login` to configure your API key
"""

import asyncio

from prime_mcp import availability, pods, ssh


async def main():
    """Run the basic MCP example."""
    print("=== Prime MCP Basic Usage Example ===\n")

    print("1. Checking GPU availability...")
    available = await availability.check_gpu_availability(
        gpu_type="RTX4090_24GB", security="community_cloud"
    )

    if "error" in available:
        print(f"   Error: {available['error']}\n")
        return

    print(f"   ✓ Found {len(available.get('data', []))} available options\n")

    if not available.get("data"):
        print("   No GPUs available with these criteria")
        return

    gpu_opts = available.get("RTX4090_24GB", [])
    spot_opts = [o for o in gpu_opts if o.get("isSpot")]
    on_demand_opts = [o for o in gpu_opts if not o.get("isSpot")]

    print(f"   Spot instances: {len(spot_opts)} (⚠️ cheaper, can be terminated)")
    print(f"   On-demand instances: {len(on_demand_opts)} (✓ guaranteed)")

    for i, option in enumerate(gpu_opts[:3]):
        print(f"\n   Option {i + 1}:")
        print(f"     Provider: {option.get('provider')}")
        print(f"     GPU: {option.get('gpuType')} x{option.get('gpuCount')}")
        print(f"     Price: ${option.get('prices', {}).get('onDemand')}/hr")
        print(f"     Location: {option.get('dataCenter')}")
        print(f"     Spot: {'Yes ⚠️' if option.get('isSpot') else 'No ✓'}")
        print()

    print("2. Listing existing pods...")
    pods_list = await pods.list_pods(limit=5)

    if "error" in pods_list:
        print(f"   Error: {pods_list['error']}\n")
    else:
        existing_pods = pods_list.get("data", [])
        print(f"   ✓ Found {len(existing_pods)} active pods\n")

        for pod in existing_pods:
            print(f"   Pod: {pod.get('name', 'Unnamed')}")
            print(f"     ID: {pod.get('id')}")
            print(f"     Status: {pod.get('status')}")
            print(f"     GPU: {pod.get('gpuName')} x{pod.get('gpuCount')}")
            print()

    print("3. Checking cluster availability...")
    clusters = await availability.check_cluster_availability(
        gpu_type="A100_80GB", gpu_count=4, security="secure_cloud"
    )

    if "error" in clusters:
        print(f"   Error: {clusters['error']}\n")
    else:
        cluster_options = clusters.get("data", [])
        print(f"   ✓ Found {len(cluster_options)} cluster configurations\n")

    print("4. Managing SSH keys...")
    ssh_keys = await ssh.manage_ssh_keys(action="list")

    if "error" in ssh_keys:
        print(f"   Error: {ssh_keys['error']}\n")
    else:
        keys = ssh_keys.get("data", [])
        print(f"   ✓ Found {len(keys)} SSH keys\n")

        for key in keys:
            print(f"   Key: {key.get('name')}")
            print(f"     ID: {key.get('id')}")
            print(f"     Primary: {key.get('isPrimary', False)}")
            print()

    print("5. Pod creation example (commented out):")
    print("   To create a pod, uncomment the following code:\n")
    print("""
    # IMPORTANT CHOICES:
    # 1. Choose spot vs on-demand
    gpu_opts = available.get("RTX4090_24GB", [])
    on_demand = [o for o in gpu_opts if not o.get("isSpot")]  # Guaranteed
    # spot = [o for o in gpu_opts if o.get("isSpot")]  # Cheaper but can be terminated
    
    # 2. Select instance
    selected = on_demand[0] if on_demand else gpu_opts[0]
    
    # 3. Choose your image!
    # Options: ubuntu_22_cuda_12, cuda_12_4_pytorch_2_5, prime_rl, etc.
    # Check selected.get('images') for available images
    
    # 4. Ensure SSH key is added (see step 4 above)
    
    new_pod = await pods.create_pod(
        cloud_id=selected.get("cloudId"),
        gpu_type="RTX4090_24GB",
        provider_type=selected.get("provider"),
        data_center_id=selected.get("dataCenter"),
        name="example-pod",
        gpu_count=1,
        disk_size=50,
        image="cuda_12_4_pytorch_2_5"  # YOUR IMAGE CHOICE
    )
    
    if "error" not in new_pod:
        print(f"   ✓ Created pod: {new_pod.get('id')}")
        print(f"     Status: {new_pod.get('status')}")
        print(f"     Spot: {new_pod.get('isSpot')}")
        print(f"     SSH: {new_pod.get('sshConnection')}")
        
        # Monitor status
        status = await pods.get_pods_status(pod_ids=[new_pod.get('id')])
        print(f"     Installation: {status}")
        
        # When done, delete the pod
        # await pods.delete_pod(new_pod.get('id'))
    """)

    print("\n6. Getting pods history...")
    history = await pods.get_pods_history(limit=5, sort_by="terminatedAt", sort_order="desc")

    if "error" in history:
        print(f"   Error: {history['error']}\n")
    else:
        historical_pods = history.get("data", [])
        print(f"   ✓ Found {len(historical_pods)} historical pods\n")

        for pod in historical_pods[:3]:
            print(f"   Pod: {pod.get('name', 'Unnamed')}")
            print(f"     Status: {pod.get('status')}")
            print(f"     Terminated: {pod.get('terminatedAt', 'N/A')}")
            print()

    print("=== Example completed successfully! ===")
    print("\nNext steps:")
    print("1. Uncomment the pod creation code to create a real pod")
    print("2. Use the MCP server with Claude Desktop or other MCP clients")
    print("3. Explore the API reference in the README")


if __name__ == "__main__":
    asyncio.run(main())
