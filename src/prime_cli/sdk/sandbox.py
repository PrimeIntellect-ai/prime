import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from ..api.client import APIClient
from ..api.sandbox import (
    CommandResponse,
    CreateSandboxRequest,
    SandboxClient,
    SandboxStatus,
)


@dataclass
class Result:
    """Result of code execution in a sandbox"""

    stdout: str
    stderr: str
    exit_code: int

    @property
    def success(self) -> bool:
        """Whether the command executed successfully (exit code 0)"""
        return self.exit_code == 0


class SandboxError(Exception):
    """Exception raised for sandbox-related errors"""

    pass


class Sandbox:
    """High-level sandbox interface for running code"""

    def __init__(
        self,
        template: str = "python:3.11",
        cpu_cores: int = 1,
        memory_gb: int = 2,
        disk_size_gb: int = 10,
        gpu_count: int = 0,
        timeout_minutes: int = 60,
        working_dir: str = "/workspace",
        environment_vars: Optional[Dict[str, str]] = None,
        team_id: Optional[str] = None,
    ):
        """Initialize sandbox configuration

        Args:
            template: Docker image template (default: python:3.11)
            cpu_cores: Number of CPU cores (default: 1)
            memory_gb: Memory in GB (default: 2)
            disk_size_gb: Disk size in GB (default: 10)
            gpu_count: Number of GPUs (default: 0)
            timeout_minutes: Timeout in minutes (default: 60)
            working_dir: Working directory (default: /workspace)
            environment_vars: Environment variables dict
            team_id: Team ID for the sandbox
        """
        self.template = template
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.disk_size_gb = disk_size_gb
        self.gpu_count = gpu_count
        self.timeout_minutes = timeout_minutes
        self.working_dir = working_dir
        self.environment_vars = environment_vars
        self.team_id = team_id

        self._client: Optional[APIClient] = None
        self._sandbox_client: Optional[SandboxClient] = None
        self._sandbox_id: Optional[str] = None
        self._is_ready = False

    def __enter__(self) -> "Sandbox":
        """Enter context manager - create and start sandbox"""
        self._start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - cleanup sandbox"""
        self._cleanup()

    def _start(self) -> None:
        """Start the sandbox"""
        try:
            # Initialize clients
            self._client = APIClient()
            self._sandbox_client = SandboxClient(self._client)

            # Generate unique sandbox name
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            sandbox_name = f"sdk-{timestamp}"

            # Create sandbox request
            request = CreateSandboxRequest(
                name=sandbox_name,
                docker_image=self.template,
                cpu_cores=self.cpu_cores,
                memory_gb=self.memory_gb,
                disk_size_gb=self.disk_size_gb,
                gpu_count=self.gpu_count,
                timeout_minutes=self.timeout_minutes,
                working_dir=self.working_dir,
                environment_vars=self.environment_vars,
                team_id=self.team_id,
            )

            # Create sandbox
            sandbox = self._sandbox_client.create(request)
            self._sandbox_id = sandbox.id

            # Wait for sandbox to be ready
            self._wait_for_ready()

        except Exception as e:
            raise SandboxError(f"Failed to start sandbox: {e}")

    def _wait_for_ready(self, max_wait_seconds: int = 300) -> None:
        """Wait for sandbox to reach RUNNING status"""
        if not self._sandbox_client or not self._sandbox_id:
            raise SandboxError("Sandbox not initialized")

        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            try:
                sandbox = self._sandbox_client.get(self._sandbox_id)

                if sandbox.status == SandboxStatus.RUNNING:
                    self._is_ready = True
                    return
                elif sandbox.status == SandboxStatus.ERROR:
                    raise SandboxError("Sandbox failed to start with error status")
                elif sandbox.status == SandboxStatus.TERMINATED:
                    raise SandboxError("Sandbox was terminated during startup")

                # Wait before checking again
                time.sleep(2)

            except Exception as e:
                raise SandboxError(f"Error checking sandbox status: {e}")

        raise SandboxError(f"Sandbox failed to become ready within {max_wait_seconds} seconds")

    def _cleanup(self) -> None:
        """Clean up sandbox resources"""
        if self._sandbox_client and self._sandbox_id:
            try:
                self._sandbox_client.delete(self._sandbox_id)
            except Exception:
                # Ignore cleanup errors to avoid masking original exceptions
                pass
            finally:
                self._sandbox_id = None
                self._is_ready = False

    def run(
        self, command: str, working_dir: Optional[str] = None, env: Optional[Dict[str, str]] = None
    ) -> Result:
        """Run a command in the sandbox (primary interface)

        Args:
            command: Shell command to execute
            working_dir: Working directory for execution (optional)
            env: Environment variables for the command (optional)

        Returns:
            Result object with stdout, stderr, exit_code, and success properties

        Raises:
            SandboxError: If sandbox is not ready or execution fails
        """
        if not self._is_ready or not self._sandbox_client or not self._sandbox_id:
            raise SandboxError("Sandbox is not ready. Use within a context manager.")

        try:
            response: CommandResponse = self._sandbox_client.execute_command(
                sandbox_id=self._sandbox_id,
                command=command,
                working_dir=working_dir,
                env=env,
            )

            return Result(
                stdout=response.stdout,
                stderr=response.stderr,
                exit_code=response.exit_code,
            )

        except Exception as e:
            raise SandboxError(f"Failed to execute command: {e}")

    def run_python(self, code: str, working_dir: Optional[str] = None) -> Result:
        """Convenience method to run Python code (if python is available in the container)

        Args:
            code: Python code to execute
            working_dir: Working directory for execution (optional)

        Returns:
            Result object with stdout, stderr, exit_code, and success properties
        """
        command = f"python -c {repr(code)}"
        return self.run(command, working_dir=working_dir)

    def run_script(
        self, script_content: str, filename: str = "script.sh", working_dir: Optional[str] = None
    ) -> Result:
        """Write and execute a script file

        Args:
            script_content: Content of the script
            filename: Name for the script file (default: script.sh)
            working_dir: Working directory for execution (optional)

        Returns:
            Result object with stdout, stderr, exit_code, and success properties
        """
        # Write the script
        self.write_file(filename, script_content, working_dir=working_dir)
        # Make it executable and run it
        self.run(f"chmod +x {filename}", working_dir=working_dir)
        return self.run(f"./{filename}", working_dir=working_dir)

    def write_file(self, file_path: str, content: str, working_dir: Optional[str] = None) -> Result:
        """Write content to a file in the sandbox

        Args:
            file_path: Path to the file to create/write
            content: Content to write to the file
            working_dir: Working directory for the operation (optional)

        Returns:
            Result object with stdout, stderr, exit_code, and success properties

        Raises:
            SandboxError: If sandbox is not ready or operation fails
        """
        # Use cat with heredoc for reliable file creation
        escaped_content = content.replace("'", "'\"'\"'")
        command = f"cat > '{file_path}' << 'EOF'\n{content}\nEOF"
        return self.run(command, working_dir=working_dir)

    def read_file(self, file_path: str, working_dir: Optional[str] = None) -> Result:
        """Read content from a file in the sandbox

        Args:
            file_path: Path to the file to read
            working_dir: Working directory for the operation (optional)

        Returns:
            Result object with file content in stdout

        Raises:
            SandboxError: If sandbox is not ready or operation fails
        """
        command = f"cat '{file_path}'"
        return self.run(command, working_dir=working_dir)

    def list_files(self, directory: str = ".", working_dir: Optional[str] = None) -> Result:
        """List files in a directory

        Args:
            directory: Directory to list (default: current directory)
            working_dir: Working directory for the operation (optional)

        Returns:
            Result object with directory listing in stdout

        Raises:
            SandboxError: If sandbox is not ready or operation fails
        """
        command = f"ls -la '{directory}'"
        return self.run(command, working_dir=working_dir)

    def create_directory(self, directory: str, working_dir: Optional[str] = None) -> Result:
        """Create a directory in the sandbox

        Args:
            directory: Directory path to create
            working_dir: Working directory for the operation (optional)

        Returns:
            Result object with stdout, stderr, exit_code, and success properties

        Raises:
            SandboxError: If sandbox is not ready or operation fails
        """
        command = f"mkdir -p '{directory}'"
        return self.run(command, working_dir=working_dir)

    def delete_file(self, file_path: str, working_dir: Optional[str] = None) -> Result:
        """Delete a file or directory in the sandbox

        Args:
            file_path: Path to the file or directory to delete
            working_dir: Working directory for the operation (optional)

        Returns:
            Result object with stdout, stderr, exit_code, and success properties

        Raises:
            SandboxError: If sandbox is not ready or operation fails
        """
        command = f"rm -rf '{file_path}'"
        return self.run(command, working_dir=working_dir)

    def file_exists(self, file_path: str, working_dir: Optional[str] = None) -> bool:
        """Check if a file exists in the sandbox

        Args:
            file_path: Path to check
            working_dir: Working directory for the operation (optional)

        Returns:
            True if file exists, False otherwise

        Raises:
            SandboxError: If sandbox is not ready or operation fails
        """
        result = self.run(f"test -e '{file_path}'", working_dir=working_dir)
        return result.success

    @property
    def is_ready(self) -> bool:
        """Whether the sandbox is ready for code execution"""
        return self._is_ready

    @property
    def sandbox_id(self) -> Optional[str]:
        """The sandbox ID (None if not started)"""
        return self._sandbox_id
