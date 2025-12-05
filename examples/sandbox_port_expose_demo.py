import socket
import time
import urllib.request

from prime_sandboxes import APIClient, APIError, CreateSandboxRequest, SandboxClient


def verify_http(url: str) -> bool:
    """Verify HTTP endpoint is accessible and returns expected response."""
    try:
        # Add User-Agent header to avoid 403 from bot protection
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.getcode()
            body = response.read().decode("utf-8")
            # Python's http.server returns a directory listing HTML
            if status == 200 and "Directory listing" in body:
                return True
            return False
    except Exception as e:
        print(f"    HTTP verification error: {e}")
        return False


def verify_tcp(tls_socket: str, test_message: bytes = b"Hello") -> bool:
    """Verify TCP endpoint is accessible and echoes back data."""
    try:
        # Parse host:port from socket address
        host, port_str = tls_socket.rsplit(":", 1)
        port = int(port_str)

        # Connect with raw TCP
        with socket.create_connection((host, port), timeout=10) as sock:
            sock.sendall(test_message)
            response = sock.recv(1024)
            expected = b"Echo: " + test_message
            return response == expected
    except Exception as e:
        print(f"    TCP verification error: {e}")
        return False


def main() -> None:
    """Demonstrate HTTP and TCP port exposure"""
    try:
        client = APIClient()
        sandbox_client = SandboxClient(client)

        request = CreateSandboxRequest(
            name="port-expose-demo",
            docker_image="python:3.11-slim",
            start_command="tail -f /dev/null",
            cpu_cores=1,
            memory_gb=2,
            timeout_minutes=30,
        )

        print("Creating sandbox...")
        sandbox = sandbox_client.create(request)
        print(f"Created: {sandbox.name} ({sandbox.id})")

        print("\nWaiting for sandbox to be running...")
        sandbox_client.wait_for_creation(sandbox.id, max_attempts=60)
        print("Sandbox is running!")

        print("\n--- HTTP Port Exposure ---")
        print("Starting HTTP server on port 8000...")
        sandbox_client.execute_command(
            sandbox.id,
            "nohup python -m http.server 8000 > /tmp/http.log 2>&1 &",
        )
        time.sleep(2)  # Give server time to start

        # Expose the HTTP port
        http_exposure = sandbox_client.expose(
            sandbox_id=sandbox.id,
            port=8000,
            name="web-server",
            protocol="HTTP",
        )
        print("HTTP port exposed!")
        print(f"  Exposure ID: {http_exposure.exposure_id}")
        print(f"  URL: {http_exposure.url}")
        print(f"  TLS Socket: {http_exposure.tls_socket}")
        time.sleep(10)

        # Verify HTTP endpoint is accessible
        print("  Verifying HTTP endpoint...")
        if verify_http(http_exposure.url):
            print("  HTTP verification: SUCCESS")
        else:
            print("  HTTP verification: FAILED")

        # Start a TCP echo server in the sandbox
        print("\n--- TCP Port Exposure ---")
        print("Starting TCP echo server on port 9000...")

        # Create a simple TCP echo server
        tcp_server_code = """
import socket
import threading

def handle_client(conn, addr):
    print(f"Connection from {addr}")
    while True:
        data = conn.recv(1024)
        if not data:
            break
        conn.sendall(b"Echo: " + data)
    conn.close()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", 9000))
server.listen(5)
print("TCP server listening on port 9000")

while True:
    conn, addr = server.accept()
    thread = threading.Thread(target=handle_client, args=(conn, addr))
    thread.daemon = True
    thread.start()
"""
        # Write and run the TCP server
        sandbox_client.execute_command(
            sandbox.id,
            f"cat > /tmp/tcp_server.py << 'SCRIPT'\n{tcp_server_code}\nSCRIPT",
        )
        sandbox_client.execute_command(
            sandbox.id,
            "nohup python /tmp/tcp_server.py > /tmp/tcp.log 2>&1 &",
        )
        time.sleep(2)  # Give server time to start

        # Expose the TCP port
        tcp_exposure = sandbox_client.expose(
            sandbox_id=sandbox.id,
            port=9000,
            name="echo-server",
            protocol="TCP",
        )
        print("TCP port exposed!")
        print(f"  Exposure ID: {tcp_exposure.exposure_id}")
        print(f"  TLS Socket: {tcp_exposure.tls_socket}")
        if tcp_exposure.external_port:
            print(f"  External Port: {tcp_exposure.external_port}")
        time.sleep(120)

        # Verify TCP endpoint is accessible
        print("  Verifying TCP endpoint...")
        if verify_tcp(tcp_exposure.tls_socket):
            print("  TCP verification: SUCCESS (echo server responded correctly)")
        else:
            print("  TCP verification: FAILED")

        # List all exposed ports
        print("\n--- All Exposed Ports ---")
        ports_response = sandbox_client.list_exposed_ports(sandbox.id)
        for port in ports_response.exposures:
            print(f"  {port.name} (port {port.port}):")
            print(f"    Protocol: {port.protocol}")
            print(f"    Exposure ID: {port.exposure_id}")
            if port.protocol == "HTTP":
                print(f"    URL: {port.url}")
            else:
                print(f"    TLS Socket: {port.tls_socket}")

        # Usage instructions
        print("\n--- How to Connect ---")
        print(f"HTTP: curl {http_exposure.url}")
        print("TCP:  Use the TLS socket address with a TCP client")

        # Clean up exposures
        # print("\n--- Cleanup ---")
        # print("Removing port exposures...")
        # sandbox_client.unexpose(sandbox.id, http_exposure.exposure_id)
        # print(f"  Removed HTTP exposure: {http_exposure.exposure_id}")
        # sandbox_client.unexpose(sandbox.id, tcp_exposure.exposure_id)
        # print(f"  Removed TCP exposure: {tcp_exposure.exposure_id}")

        # # Delete sandbox
        # print(f"\nDeleting sandbox {sandbox.name}...")
        # sandbox_client.delete(sandbox.id)
        print("Done!")

    except APIError as e:
        print(f"API Error: {e}")
        print("Make sure you're logged in: run 'prime login' first")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
