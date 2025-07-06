"""
Sandbox service for secure code execution using Docker containers.
"""

import asyncio
import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

try:
    import docker
    from docker.errors import DockerException
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None
    
    class DockerException(Exception):
        pass

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from app.core.config import settings
from app.core.telemetry import get_telemetry
from app.models.schemas import SandboxMetrics


class SandboxError(Exception):
    """Base exception for sandbox operations."""
    pass


class SandboxService:
    """Service for managing secure Docker sandboxes."""
    
    def __init__(self):
        self.telemetry = get_telemetry()
        self.active_containers: Dict[str, Dict[str, Any]] = {}
        self.client = None
        self.docker_available = False
        self._initialize_docker()
    
    def _initialize_docker(self) -> None:
        """Initialize Docker client."""
        if not DOCKER_AVAILABLE:
            self.telemetry.log_event("Docker not available in environment", level="warning")
            return
            
        try:
            self.client = docker.from_env()
            self.client.ping()
            self.docker_available = True
            self.telemetry.log_event("Docker client initialized successfully")
        except DockerException as e:
            self.telemetry.log_event(f"Docker initialization failed: {e}", level="warning")
            self.docker_available = False
        except Exception as e:
            self.telemetry.log_event(f"Docker initialization failed: {e}", level="warning")
            self.docker_available = False
    
    async def create_sandbox(
        self,
        correlation_id: str,
        image: str = "alpine:latest"
    ) -> str:
        """
        Create a new sandbox container.
        
        Args:
            correlation_id: Unique identifier for the sandbox
            image: Docker image to use
            
        Returns:
            Container ID
            
        Raises:
            SandboxError: If sandbox creation fails
        """
        with self.telemetry.trace_operation(
            "create_sandbox",
            correlation_id=correlation_id,
            image=image
        ):
            try:
                temp_dir = tempfile.mkdtemp(prefix="backspace_sandbox_")
                
                container_config = self._prepare_container_config(
                    correlation_id=correlation_id,
                    temp_dir=temp_dir,
                    image=image
                )
                
                container = self.client.containers.run(
                    **container_config,
                    detach=True
                )
                
                self.active_containers[correlation_id] = {
                    "container": container,
                    "temp_dir": temp_dir,
                    "created_at": time.time()
                }
                
                await asyncio.sleep(1)
                
                try:
                    stdout, stderr, exit_code = await self.execute_command(
                        correlation_id=correlation_id,
                        command="apk add --no-cache git python3 py3-pip"
                    )
                    if exit_code != 0:
                        self.telemetry.log_event(
                            "apk install failed",
                            correlation_id=correlation_id,
                            exit_code=exit_code,
                            stdout=stdout,
                            stderr=stderr
                        )
                    else:
                        stdout, stderr, exit_code = await self.execute_command(
                            correlation_id=correlation_id,
                            command="git --version"
                        )
                        if exit_code == 0:
                            self.telemetry.log_event(
                                "Git installed successfully",
                                correlation_id=correlation_id,
                                git_version=stdout.strip()
                            )
                        else:
                            self.telemetry.log_event(
                                "Git verification failed",
                                correlation_id=correlation_id,
                                exit_code=exit_code,
                                stderr=stderr
                            )
                except Exception as e:
                    self.telemetry.log_error(
                        e,
                        context={"correlation_id": correlation_id, "step": "install_git"},
                        correlation_id=correlation_id
                    )
                
                self.telemetry.log_event(
                    "Sandbox created",
                    correlation_id=correlation_id,
                    container_id=container.id,
                    image=image
                )
                
                return container.id
                
            except Exception as e:
                self.telemetry.log_error(
                    e,
                    context={"correlation_id": correlation_id, "image": image},
                    correlation_id=correlation_id
                )
                raise SandboxError(f"Failed to create sandbox: {e}")
    
    def _prepare_container_config(
        self,
        correlation_id: str,
        temp_dir: str,
        image: str = "alpine:latest"
    ) -> Dict[str, Any]:
        """
        Prepare container configuration with security settings.
        
        Args:
            correlation_id: Unique identifier
            temp_dir: Temporary directory path
            image: Docker image to use
            
        Returns:
            Container configuration dictionary
        """
        volumes = {
            temp_dir: {"bind": "/workspace", "mode": "rw"}
        }
        
        # Make container name unique by adding timestamp
        unique_id = f"{correlation_id}_{int(time.time() * 1000)}"
        
        return {
            "image": image,
            "name": f"sandbox_{unique_id}",
            "volumes": volumes,
            "working_dir": "/workspace",
            "user": "root",
            "network_mode": "bridge",
            "mem_limit": "512m",
            "cpu_quota": 50000,
            "cpu_period": 100000,
            "pids_limit": 100,
            "read_only": False,
            "tmpfs": {
                "/tmp": "rw,noexec,nosuid,size=100m"
            },
            "environment": {
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HOME": "/workspace"
            },
            "command": ["sleep", "infinity"]  # Keep container running
        }
    
    async def execute_command(
        self,
        correlation_id: str,
        command: str,
        timeout: Optional[int] = None,
        working_dir: Optional[str] = None,
        user: Optional[str] = None
    ) -> Tuple[str, str, int]:
        """
        Execute a command in the sandbox.
        
        Args:
            correlation_id: Sandbox identifier
            command: Command to execute
            timeout: Timeout in seconds
            working_dir: Working directory for the command
            user: User to run command as (default: root for git install, 1000:1000 for others)
            
        Returns:
            Tuple of (stdout, stderr, exit_code)
            
        Raises:
            SandboxError: If command execution fails
        """
        if not self.docker_available or not self.client:
            raise SandboxError("Docker is not available - sandbox functionality is disabled")
            
        with self.telemetry.trace_operation(
            "execute_command",
            correlation_id=correlation_id,
            command=command[:100]  # Truncate for logging
        ):
            if correlation_id not in self.active_containers:
                raise SandboxError(f"Sandbox {correlation_id} not found")
            
            container = self.active_containers[correlation_id]["container"]
            timeout = timeout or settings.sandbox_timeout
            
            if user is None:
                user = "root" if "apk" in command or "apt-get" in command else "1000:1000"
            
            try:    
                exec_result = container.exec_run(
                    command,
                    stdout=True,
                    stderr=True,
                    tty=False,
                    workdir=working_dir or "/workspace",
                    user=user
                )
                
                stdout = exec_result.output.decode("utf-8", errors="replace")
                stderr = ""
                exit_code = exec_result.exit_code
                
                self.telemetry.log_event(
                    "Command executed",
                    correlation_id=correlation_id,
                    command=command[:100],
                    exit_code=exit_code,
                    stdout_length=len(stdout),
                    user=user
                )
                
                return stdout, stderr, exit_code
                
            except Exception as e:
                self.telemetry.log_error(
                    e,
                    context={
                        "correlation_id": correlation_id,
                        "command": command[:100],
                        "user": user
                    },
                    correlation_id=correlation_id
                )
                raise SandboxError(f"Command execution failed: {e}")
    
    async def read_file(
        self,
        correlation_id: str,
        file_path: str
    ) -> str:
        """
        Read a file from the sandbox.
        
        Args:
            correlation_id: Sandbox identifier
            file_path: Path to the file to read
            
        Returns:
            File contents as string
            
        Raises:
            SandboxError: If file reading fails
        """
        with self.telemetry.trace_operation(
            "read_file",
            correlation_id=correlation_id,
            file_path=file_path
        ):
            stdout, stderr, exit_code = await self.execute_command(
                correlation_id=correlation_id,
                command=f"cat '{file_path}'"
            )
            
            if exit_code != 0:
                raise SandboxError(f"Failed to read file {file_path}: {stderr}")
            
            return stdout
    
    async def write_file(
        self,
        correlation_id: str,
        file_path: str,
        content: str
    ) -> None:
        """
        Write content to a file in the sandbox.
        
        Args:
            correlation_id: Sandbox identifier
            file_path: Path to the file to write
            content: Content to write
            
        Raises:
            SandboxError: If file writing fails
        """
        with self.telemetry.trace_operation(
            "write_file",
            correlation_id=correlation_id,
            file_path=file_path,
            content_length=len(content)
        ):
            dir_path = os.path.dirname(file_path)
            if dir_path and dir_path != "/workspace":
                mkdir_cmd = f"mkdir -p '{dir_path}'"
                stdout, stderr, exit_code = await self.execute_command(
                    correlation_id=correlation_id,
                    command=mkdir_cmd
                )
                if exit_code != 0:
                    raise SandboxError(f"Failed to create directory {dir_path}: {stderr}")
            
            if correlation_id not in self.active_containers:
                raise SandboxError(f"Sandbox {correlation_id} not found")
            
            container = self.active_containers[correlation_id]["container"]
            
            import base64
            content_b64 = base64.b64encode(content.encode('utf-8')).decode('utf-8')
            
            python_cmd = f"python3 -c \"import base64; open('{file_path}', 'w', encoding='utf-8').write(base64.b64decode('{content_b64}').decode('utf-8'))\""
            
            try:
                exec_result = container.exec_run(
                    python_cmd,
                    stdout=True,
                    stderr=True,
                    tty=False,
                    workdir="/workspace",
                    user="1000:1000"
                )
                
                if exec_result.exit_code != 0:
                    raise SandboxError(f"Failed to write file: {exec_result.output.decode('utf-8', errors='replace')}")
                
                verify_result = container.exec_run(
                    f"test -f '{file_path}' && echo 'File exists' || echo 'File does not exist'",
                    stdout=True,
                    stderr=True,
                    tty=False,
                    workdir="/workspace",
                    user="1000:1000"
                )
                
                if "File does not exist" in verify_result.output.decode('utf-8', errors='replace'):
                    raise SandboxError(f"File verification failed for {file_path}")
                
            except Exception as e:
                raise SandboxError(f"Failed to write file {file_path}: {e}")
    
    async def list_files(
        self,
        correlation_id: str,
        directory: str = "/workspace"
    ) -> List[str]:
        """
        List files in a directory.
        
        Args:
            correlation_id: Sandbox identifier
            directory: Directory to list
            
        Returns:
            List of file paths
            
        Raises:
            SandboxError: If listing fails
        """
        with self.telemetry.trace_operation(
            "list_files",
            correlation_id=correlation_id,
            directory=directory
        ):
            stdout, stderr, exit_code = await self.execute_command(
                correlation_id=correlation_id,
                command=f"find '{directory}' -type f | head -1000"
            )
            
            if exit_code != 0:
                raise SandboxError(f"Failed to list files in {directory}: {stderr}")
            
            return [line.strip() for line in stdout.split("\n") if line.strip()]
    
    async def get_metrics(self, correlation_id: str) -> SandboxMetrics:
        """
        Get metrics for a sandbox container.
        
        Args:
            correlation_id: Sandbox identifier
            
        Returns:
            Sandbox metrics
            
        Raises:
            SandboxError: If metrics collection fails
        """
        if not self.docker_available or not self.client:
            raise SandboxError("Docker is not available - sandbox functionality is disabled")
            
        with self.telemetry.trace_operation(
            "get_metrics",
            correlation_id=correlation_id
        ):
            if correlation_id not in self.active_containers:
                raise SandboxError(f"Sandbox {correlation_id} not found")
            
            container = self.active_containers[correlation_id]["container"]
            created_at = self.active_containers[correlation_id]["created_at"]
            
            try:
                stats = container.stats(stream=False)
                
                cpu_usage = 0.0
                if "cpu_stats" in stats and "precpu_stats" in stats:
                    cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                               stats["precpu_stats"]["cpu_usage"]["total_usage"]
                    system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                                  stats["precpu_stats"]["system_cpu_usage"]
                    
                    cpu_usage = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
                
                memory_usage = 0.0
                if "memory_stats" in stats and "usage" in stats["memory_stats"]:
                    memory_usage = stats["memory_stats"]["usage"] / (1024 * 1024)  # MB
                
                network_io = {}
                if "networks" in stats:
                    for network, data in stats["networks"].items():
                        network_io[network] = {
                            "rx_bytes": data.get("rx_bytes", 0),
                            "tx_bytes": data.get("tx_bytes", 0)
                        }
                
                return SandboxMetrics(
                    container_id=container.id,
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    disk_usage=0.0,
                    network_io=network_io,
                    uptime=time.time() - created_at
                )
                
            except Exception as e:
                self.telemetry.log_error(
                    e,
                    context={"correlation_id": correlation_id},
                    correlation_id=correlation_id
                )
                raise SandboxError(f"Failed to get metrics: {e}")
    
    async def cleanup_sandbox(self, correlation_id: str) -> None:
        """
        Clean up a sandbox container and its resources.
        
        Args:
            correlation_id: Sandbox identifier
        """
        with self.telemetry.trace_operation(
            "cleanup_sandbox",
            correlation_id=correlation_id
        ):
            if correlation_id not in self.active_containers:
                return
            
            container_info = self.active_containers[correlation_id]
            container = container_info["container"]
            temp_dir = container_info["temp_dir"]
            
            try:
                # Stop and remove container
                container.stop(timeout=5)
                container.remove()
                
                # Remove temporary directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                
                # Remove from active containers
                del self.active_containers[correlation_id]
                
                self.telemetry.log_event(
                    "Sandbox cleaned up",
                    correlation_id=correlation_id,
                    container_id=container.id
                )
                
            except Exception as e:
                self.telemetry.log_error(
                    e,
                    context={"correlation_id": correlation_id},
                    correlation_id=correlation_id
                )   
    
    async def cleanup_all_sandboxes(self) -> None:
        """Clean up all active sandboxes."""
        with self.telemetry.trace_operation("cleanup_all_sandboxes"):
            correlation_ids = list(self.active_containers.keys())
            
            for correlation_id in correlation_ids:
                try:
                    await self.cleanup_sandbox(correlation_id)
                except Exception as e:
                    self.telemetry.log_error(
                        e,
                        context={"correlation_id": correlation_id},
                        correlation_id=correlation_id
                    )
    
    def get_active_sandboxes(self) -> List[str]:
        """
        Get list of active sandbox correlation IDs.
        
        Returns:
            List of correlation IDs
        """
        return list(self.active_containers.keys())
    
    def is_sandbox_active(self, correlation_id: str) -> bool:
        """
        Check if a sandbox is active.
        
        Args:
            correlation_id: Sandbox identifier
            
        Returns:
            True if sandbox is active, False otherwise
        """
        return correlation_id in self.active_containers
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the sandbox service.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # If Docker is not available, still consider service healthy for API purposes
            if not self.docker_available:
                self.telemetry.log_event("Health check: Docker not available but service is healthy", level="info")
                return True
                
            # If Docker is available, check if it's responsive
            if self.client:
                self.client.ping()
                return True
            else:
                return False
                
        except Exception as e:
            self.telemetry.log_error(
                e,
                context={"service": "sandbox", "operation": "health_check"}
            )
            return False 