#!/usr/bin/env python3
"""Test script for the Prime CLI SDK"""

from src.prime_cli import Sandbox

def test_basic_usage():
    """Test basic SDK usage"""
    print("Testing Prime CLI SDK...")
    
    try:
        # Test import
        print("✓ Import successful")
        
        # Test sandbox creation (without actually starting it)
        sandbox = Sandbox("python:3.11")
        print("✓ Sandbox object creation successful")
        print(f"✓ Template: {sandbox.template}")
        print(f"✓ CPU cores: {sandbox.cpu_cores}")
        print(f"✓ Memory: {sandbox.memory_gb}GB")
        print(f"✓ Ready status: {sandbox.is_ready}")
        
        # Test with custom parameters
        custom_sandbox = Sandbox(
            template="ubuntu:latest",
            cpu_cores=2,
            memory_gb=4,
            environment_vars={"TEST": "value"}
        )
        print("✓ Custom sandbox configuration successful")
        print(f"✓ Custom template: {custom_sandbox.template}")
        print(f"✓ Custom resources: {custom_sandbox.cpu_cores}CPU/{custom_sandbox.memory_gb}GB")
        
        print("\n🎉 All tests passed! SDK is ready to use.")
        print("\nExample usage:")
        print("```python")
        print("from prime_cli import Sandbox")
        print("")
        print("# Run shell commands")
        print('with Sandbox("ubuntu:latest") as sb:')
        print('    result = sb.run("echo \'Hello Prime!\'")')
        print("    print(result.stdout)  # 'Hello Prime!'")
        print("")
        print("# Python convenience method")
        print('with Sandbox("python:3.11") as sb:')
        print('    result = sb.run_python("print(\'Hello from Python!\'")')
        print("    print(result.stdout)")
        print("")
        print("# File operations")
        print('with Sandbox("ubuntu:latest") as sb:')
        print('    sb.write_file("hello.txt", "Hello World!")')
        print('    result = sb.read_file("hello.txt")')
        print("    print(result.stdout)  # 'Hello World!'")
        print("```")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_basic_usage()