#!/usr/bin/env python3
"""
Prime CLI SDK Demo - High-level sandbox interface
"""

from prime_cli import Sandbox
from prime_cli.sdk import SandboxError

def main() -> None:
    """SDK demo showing the high-level interface"""
    
    print("🚀 Prime CLI SDK Demo")
    print("=" * 50)
    
    try:
        # Basic shell command execution
        print("\n1. Basic shell commands")
        with Sandbox("ubuntu:latest") as sb:
            print(f"Sandbox ID: {sb.sandbox_id}")
            
            result = sb.run("whoami")
            print(f"User: {result.stdout.strip()}")
            
            result = sb.run("pwd")
            print(f"Working directory: {result.stdout.strip()}")
            
            result = sb.run("echo 'Hello from Prime!'")
            print(f"Echo: {result.stdout.strip()}")
        
        # File operations
        print("\n2. File operations")
        with Sandbox("alpine:latest") as sb:
            # Create a file
            sb.write_file("hello.txt", "Hello World!\nFrom the Prime sandbox!")
            print("✅ Created hello.txt")
            
            # Check if file exists
            exists = sb.file_exists("hello.txt")
            print(f"File exists: {exists}")
            
            # Read the file
            result = sb.read_file("hello.txt")
            print(f"File content:\n{result.stdout}")
            
            # List files
            result = sb.list_files()
            print(f"Directory listing:\n{result.stdout}")
            
            # Create a directory
            sb.create_directory("mydir")
            result = sb.list_files()
            print(f"After creating directory:\n{result.stdout}")
        
        # Python-specific container
        print("\n3. Python container")
        with Sandbox("python:3.11-slim") as sb:
            # Use the Python convenience method
            result = sb.run_python("print('Hello from Python!')")
            print(f"Python output: {result.stdout.strip()}")
            
            # Math calculation
            result = sb.run_python("import math; print(f'π = {math.pi:.4f}')")
            print(f"Math result: {result.stdout.strip()}")
            
            # Create and run a Python script
            script_content = """
            import json
            import sys

            data = {
                "message": "Hello from Python script!",
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "platform": sys.platform
            }

            print(json.dumps(data, indent=2))
            """
            sb.write_file("script.py", script_content)
            result = sb.run("python script.py")
            print(f"Script output:\n{result.stdout}")
        
        # Shell script execution
        print("\n4. Shell script execution")
        with Sandbox("ubuntu:latest") as sb:
            script = """#!/bin/bash
            echo "=== System Information ==="
            echo "Hostname: $(hostname)"
            echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"')"
            echo "CPU cores: $(nproc)"
            echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
            echo "Date: $(date)"
            """
            result = sb.run_script(script, "sysinfo.sh")
            print(f"System info:\n{result.stdout}")
        
        # Environment variables and working directory
        print("\n5. Environment and working directory")
        with Sandbox("ubuntu:latest") as sb:
            # Set environment variables
            env = {"MY_VAR": "Hello", "SANDBOX_ID": sb.sandbox_id or "unknown"}
            result = sb.run("echo $MY_VAR $SANDBOX_ID", env=env)
            print(f"Environment variables: {result.stdout.strip()}")
            
            # Working directory operations
            sb.create_directory("/tmp/myworkspace")
            result = sb.run("pwd", working_dir="/tmp/myworkspace")
            print(f"Working directory: {result.stdout.strip()}")
        
        print("\n✅ All demos completed successfully!")
        
    except SandboxError as e:
        print(f"❌ Sandbox Error: {e}")
        print("💡 Make sure you're logged in: run 'prime login' first")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")


if __name__ == "__main__":
    main()
