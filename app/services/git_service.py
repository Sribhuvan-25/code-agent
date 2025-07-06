"""
Git service for repository operations and GitHub API interactions.
"""

import os
import re
from typing import Dict, Any, Optional
from urllib.parse import urlparse

import httpx

try:
    from github import Github
    from github.GithubException import GithubException
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    Github = None
    
    class GithubException(Exception):
        pass

from app.core.config import settings
from app.core.telemetry import get_telemetry
from app.services.sandbox import SandboxService, SandboxError


class GitError(Exception):
    """Base exception for Git operations."""
    pass


class GitService:
    """Service for Git operations and GitHub API interactions."""
    
    def __init__(self):
        self.telemetry = get_telemetry()
        self.github_client = None
        if GITHUB_AVAILABLE:
            self._initialize_github_client()
        else:
            self.telemetry.log_event(
                "PyGithub not available - GitHub API functionality disabled",
                level="warning"
            )
    
    def _initialize_github_client(self) -> None:
        """Initialize GitHub client."""
        try:
            if not GITHUB_AVAILABLE:
                raise GitError("PyGithub is not available")
                
            if settings.github_token:
                self.github_client = Github(settings.github_token)
                # Test the connection
                user = self.github_client.get_user()
                self.telemetry.log_event(
                    "GitHub client initialized",
                    username=user.login
                )
            else:
                self.telemetry.log_event(
                    "GitHub token not configured",
                    level="warning"
                )
        except Exception as e:
            self.telemetry.log_error(e, context={"service": "git"})
            raise GitError(f"Failed to initialize GitHub client: {e}")
    
    async def clone_repository(
        self,
        correlation_id: str,
        repo_url: str,
        sandbox_service: SandboxService,
        branch: Optional[str] = None,
        github_token: Optional[str] = None
    ) -> str:
        """
        Clone a repository into the sandbox.
        
        Args:
            correlation_id: Sandbox identifier
            repo_url: Repository URL to clone
            sandbox_service: Sandbox service instance
            branch: Specific branch to checkout (optional)
            github_token: GitHub token for authentication (optional)
            
        Returns:
            Path to the cloned repository in the sandbox
            
        Raises:
            GitError: If cloning fails
        """
        with self.telemetry.trace_operation(
            "clone_repository",
            correlation_id=correlation_id,
            repo_url=repo_url
        ):
            try:
                repo_name = self._extract_repo_name(repo_url)
                sandbox_repo_path = f"/workspace/{repo_name}"
                token = github_token or settings.github_token
                
                if token:
                    clone_url = repo_url.replace("https://", f"https://x-access-token:{token}@")
                    stdout, stderr, exit_code = await sandbox_service.execute_command(
                        correlation_id=correlation_id,
                        command=f"git clone {clone_url} {sandbox_repo_path}"
                    )
                else:
                    stdout, stderr, exit_code = await sandbox_service.execute_command(
                        correlation_id=correlation_id,
                        command=f"git clone {repo_url} {sandbox_repo_path}"
                    )
                
                if exit_code != 0:
                    raise GitError(f"Failed to clone repository: {stderr}")
                
                # Try to set git config, but don't fail if it doesn't work
                try:
                    await self._ensure_git_user_config(correlation_id, sandbox_repo_path, sandbox_service)
                except Exception as config_error:
                    self.telemetry.log_event(
                        "Git config setup failed but continuing",
                        correlation_id=correlation_id,
                        error=str(config_error),
                        level="warning"
                    )
                
                if branch:
                    stdout, stderr, exit_code = await sandbox_service.execute_command(
                        correlation_id=correlation_id,
                        command=f"git checkout {branch}",
                        working_dir=sandbox_repo_path
                    )
                    
                    if exit_code != 0:
                        stdout, stderr, exit_code = await sandbox_service.execute_command(
                            correlation_id=correlation_id,
                            command=f"git checkout -b {branch}",
                            working_dir=sandbox_repo_path
                        )
                        
                        if exit_code != 0:
                            raise GitError(f"Failed to checkout branch {branch}: {stderr}")
                
                self.telemetry.log_event(
                    "Repository cloned successfully",
                    correlation_id=correlation_id,
                    repo_url=repo_url,
                    repo_path=sandbox_repo_path,
                    branch=branch
                )
                
                return sandbox_repo_path
                
            except SandboxError as e:
                raise GitError(f"Sandbox error during repository clone: {e}")
            except Exception as e:
                self.telemetry.log_error(
                    e,
                    context={"correlation_id": correlation_id, "repo_url": repo_url},
                    correlation_id=correlation_id
                )
                raise GitError(f"Failed to clone repository: {e}")
    
    def _extract_repo_name(self, repo_url: str) -> str:
        """
        Extract repository name from URL.
        
        Args:
            repo_url: Repository URL
            
        Returns:
            Repository name
        """
        parsed = urlparse(repo_url)
        path = parsed.path.strip("/")
        
        if path.endswith(".git"):
            path = path[:-4]
        
        repo_name = path.split("/")[-1]
        
        return repo_name
    
    async def _ensure_git_user_config(self, correlation_id: str, repo_path: str, sandbox_service: SandboxService) -> None:
        """Ensure git user.name and user.email are set in the repo."""
        try:
            stdout, stderr, exit_code = await sandbox_service.execute_command(
                correlation_id=correlation_id,
                command="git config user.email 'backspace-agent@example.com'",
                working_dir=repo_path
            )
            
            if exit_code != 0:
                self.telemetry.log_event(
                    "Git config user.email failed, trying alternative",
                    correlation_id=correlation_id,
                    stderr=stderr,
                    exit_code=exit_code,
                    level="warning"
                )
                
                # Try without cd command
                await sandbox_service.execute_command(
                    correlation_id=correlation_id,
                    command=f"cd {repo_path} && git config user.email 'backspace-agent@example.com'",
                    working_dir="/"
                )
            
            stdout, stderr, exit_code = await sandbox_service.execute_command(
                correlation_id=correlation_id,
                command="git config user.name 'Backspace Agent'",
                working_dir=repo_path
            )
            
            if exit_code != 0:
                self.telemetry.log_event(
                    "Git config user.name failed, trying alternative",
                    correlation_id=correlation_id,
                    stderr=stderr,
                    exit_code=exit_code,
                    level="warning"
                )
                
                # Try without cd command
                await sandbox_service.execute_command(
                    correlation_id=correlation_id,
                    command=f"cd {repo_path} && git config user.name 'Backspace Agent'",
                    working_dir="/"
                )
        except Exception as e:
            self.telemetry.log_event(
                "Git config setup failed, continuing without it",
                correlation_id=correlation_id,
                error=str(e),
                level="warning"
            )

    async def create_branch(
        self,
        correlation_id: str,
        repo_path: str,
        branch_name: str,
        sandbox_service: SandboxService
    ) -> None:
        """
        Create a new branch in the repository.
        
        Args:
            correlation_id: Sandbox identifier
            repo_path: Path to the repository
            branch_name: Name of the new branch
            sandbox_service: Sandbox service instance
            
        Raises:
            GitError: If branch creation fails
        """
        with self.telemetry.trace_operation(
            "create_branch",
            correlation_id=correlation_id,
            branch_name=branch_name
        ):
            try:
                await self._ensure_git_user_config(correlation_id, repo_path, sandbox_service)
                stdout, stderr, exit_code = await sandbox_service.execute_command(
                    correlation_id=correlation_id,
                    command=f"git checkout -b {branch_name}",
                    working_dir=repo_path
                )
                
                if exit_code != 0:
                    raise GitError(f"Failed to create branch {branch_name}: {stderr}")
                
                self.telemetry.log_event(
                    "Branch created successfully",
                    correlation_id=correlation_id,
                    branch_name=branch_name,
                    repo_path=repo_path
                )
                
            except SandboxError as e:
                raise GitError(f"Sandbox error during branch creation: {e}")
            except Exception as e:
                self.telemetry.log_error(
                    e,
                    context={"correlation_id": correlation_id, "branch_name": branch_name},
                    correlation_id=correlation_id
                )
                raise GitError(f"Failed to create branch: {e}")
    
    async def commit_changes(
        self,
        correlation_id: str,
        repo_path: str,
        message: str,
        sandbox_service: SandboxService
    ) -> str:
        """
        Commit changes to the repository.
        
        Args:
            correlation_id: Sandbox identifier
            repo_path: Path to the repository
            message: Commit message
            sandbox_service: Sandbox service instance
            
        Returns:
            Commit hash
            
        Raises:
            GitError: If commit fails
        """
        with self.telemetry.trace_operation(
            "commit_changes",
            correlation_id=correlation_id,
            message=message[:50]
        ):
            try:
                await self._ensure_git_user_config(correlation_id, repo_path, sandbox_service)
                stdout, stderr, exit_code = await sandbox_service.execute_command(
                    correlation_id=correlation_id,
                    command="git add .",
                    working_dir=repo_path
                )
                
                if exit_code != 0:
                    raise GitError(f"Failed to add changes: {stderr}")

                stdout, stderr, exit_code = await sandbox_service.execute_command(
                    correlation_id=correlation_id,
                    command="git diff --cached --quiet",
                    working_dir=repo_path
                )
                
                if exit_code == 0:
                    self.telemetry.log_event(
                        "No changes to commit",
                        correlation_id=correlation_id,
                        repo_path=repo_path
                    )
                    return "no_changes"
                
                stdout, stderr, exit_code = await sandbox_service.execute_command(
                    correlation_id=correlation_id,
                    command=f'git commit -m "{message}"',
                    working_dir=repo_path
                )
                
                if exit_code != 0:
                    raise GitError(f"Failed to commit changes: {stderr}")
                
                stdout, stderr, exit_code = await sandbox_service.execute_command(
                    correlation_id=correlation_id,
                    command="git rev-parse HEAD",
                    working_dir=repo_path
                )
                
                if exit_code != 0:
                    raise GitError(f"Failed to get commit hash: {stderr}")
                
                commit_hash = stdout.strip()
                
                self.telemetry.log_event(
                    "Changes committed successfully",
                    correlation_id=correlation_id,
                    commit_hash=commit_hash,
                    message=message
                )
                
                return commit_hash
                
            except SandboxError as e:
                raise GitError(f"Sandbox error during commit: {e}")
            except Exception as e:
                self.telemetry.log_error(
                    e,
                    context={"correlation_id": correlation_id, "message": message},
                    correlation_id=correlation_id
                )
                raise GitError(f"Failed to commit changes: {e}")
    
    async def push_changes(
        self,
        correlation_id: str,
        repo_path: str,
        branch_name: str,
        sandbox_service: SandboxService,
        github_token: Optional[str] = None
    ) -> None:
        """
        Push changes to the remote repository.
        
        Args:
            correlation_id: Sandbox identifier
            repo_path: Path to the repository
            branch_name: Name of the branch to push
            sandbox_service: Sandbox service instance
            github_token: GitHub token for authentication (optional)
            
        Raises:
            GitError: If push fails
        """
        with self.telemetry.trace_operation(
            "push_changes",
            correlation_id=correlation_id,
            branch_name=branch_name
        ):
            try:
                token = github_token or settings.github_token
                if not token:
                    self.telemetry.log_event(
                        "No GitHub token available - push may fail",
                        correlation_id=correlation_id,
                        level="warning"
                    )
                
                # Set up authentication if token is available
                if token:
                    await sandbox_service.execute_command(
                        correlation_id=correlation_id,
                        command='git config credential.helper "store --file=/tmp/git-credentials"',
                        working_dir=repo_path
                    )
                    
                    await sandbox_service.execute_command(
                        correlation_id=correlation_id,
                        command=f'echo "https://x-access-token:{token}@github.com" > /tmp/git-credentials',
                        working_dir=repo_path
                    )
                
                # First check if we have anything to push
                stdout, stderr, exit_code = await sandbox_service.execute_command(
                    correlation_id=correlation_id,
                    command="git status --porcelain",
                    working_dir=repo_path
                )
                
                if exit_code == 0 and not stdout.strip():
                    # Check if branch exists on remote
                    stdout, stderr, exit_code = await sandbox_service.execute_command(
                        correlation_id=correlation_id,
                        command=f"git ls-remote --heads origin {branch_name}",
                        working_dir=repo_path
                    )
                    
                    if exit_code == 0 and stdout.strip():
                        self.telemetry.log_event(
                            "Branch already exists on remote and no changes to push",
                            correlation_id=correlation_id,
                            branch_name=branch_name
                        )
                        return
                
                # Try to push with detailed error reporting
                stdout, stderr, exit_code = await sandbox_service.execute_command(
                    correlation_id=correlation_id,
                    command=f"git push -u origin {branch_name}",
                    working_dir=repo_path
                )
                
                if exit_code != 0:
                    # Get more detailed error information
                    error_details = []
                    
                    if stderr:
                        error_details.append(f"Push stderr: {stderr}")
                    if stdout:
                        error_details.append(f"Push stdout: {stdout}")
                    
                    # Check git remote configuration
                    remote_stdout, remote_stderr, remote_exit = await sandbox_service.execute_command(
                        correlation_id=correlation_id,
                        command="git remote -v",
                        working_dir=repo_path
                    )
                    
                    if remote_exit == 0:
                        error_details.append(f"Git remotes: {remote_stdout}")
                    
                    # Check current branch
                    branch_stdout, branch_stderr, branch_exit = await sandbox_service.execute_command(
                        correlation_id=correlation_id,
                        command="git branch -a",
                        working_dir=repo_path
                    )
                    
                    if branch_exit == 0:
                        error_details.append(f"Git branches: {branch_stdout}")
                    
                    # Check if it's an authentication issue
                    if "authentication" in stderr.lower() or "permission" in stderr.lower():
                        error_details.append("This appears to be an authentication issue. Make sure GitHub token is valid and has push permissions.")
                    
                    # Check if it's a remote branch issue
                    if "rejected" in stderr.lower() or "non-fast-forward" in stderr.lower():
                        error_details.append("This appears to be a merge conflict or branch protection issue.")
                    
                    full_error = " | ".join(error_details) if error_details else f"Unknown push error (exit code: {exit_code})"
                    raise GitError(f"Failed to push changes: {full_error}")
                
                self.telemetry.log_event(
                    "Changes pushed successfully",
                    correlation_id=correlation_id,
                    branch_name=branch_name,
                    repo_path=repo_path
                )
                
            except SandboxError as e:
                raise GitError(f"Sandbox error during push: {e}")
            except Exception as e:
                self.telemetry.log_error(
                    e,
                    context={"correlation_id": correlation_id, "branch_name": branch_name},
                    correlation_id=correlation_id
                )
                raise GitError(f"Failed to push changes: {e}")
    
    async def create_pull_request(
        self,
        correlation_id: str,
        repo_url: str,
        branch_name: str,
        title: str,
        body: str
    ) -> str:
        """
        Create a pull request using the GitHub API.
        
        Args:
            correlation_id: Request identifier
            repo_url: Repository URL
            branch_name: Branch with changes
            title: Pull request title
            body: Pull request body
            
        Returns:
            Pull request URL
            
        Raises:
            GitError: If PR creation fails
        """
        if not GITHUB_AVAILABLE:
            raise GitError("PyGithub is not available - cannot create pull request")
            
        with self.telemetry.trace_operation(
            "create_pull_request",
            correlation_id=correlation_id,
            repo_url=repo_url,
            branch_name=branch_name
        ):
            try:
                if not self.github_client:
                    raise GitError("GitHub client not initialized")
                
                owner, repo_name = self._parse_repo_url(repo_url)
                
                repo = self.github_client.get_repo(f"{owner}/{repo_name}")
                
                pr = repo.create_pull(
                    title=title,
                    body=body,
                    head=branch_name,
                    base="main"
                )
                
                pr_url = pr.html_url
                
                self.telemetry.log_event(
                    "Pull request created successfully",
                    correlation_id=correlation_id,
                    pr_url=pr_url,
                    pr_number=pr.number,
                    title=title
                )
                
                return pr_url
                
            except GithubException as e:
                error_msg = f"GitHub API error: {e.data.get('message', str(e)) if hasattr(e, 'data') else str(e)}"
                self.telemetry.log_error(
                    Exception(error_msg),
                    context={"correlation_id": correlation_id, "repo_url": repo_url},
                    correlation_id=correlation_id
                )
                raise GitError(error_msg)
            except Exception as e:
                self.telemetry.log_error(
                    e,
                    context={"correlation_id": correlation_id, "repo_url": repo_url},
                    correlation_id=correlation_id
                )
                raise GitError(f"Failed to create pull request: {e}")
    
    def _parse_repo_url(self, repo_url: str) -> tuple[str, str]:
        """
        Parse repository URL to extract owner and repo name.
        
        Args:
            repo_url: Repository URL
            
        Returns:
            Tuple of (owner, repo_name)
        """
        parsed = urlparse(repo_url)
        path = parsed.path.strip("/")
        
        if path.endswith(".git"):
            path = path[:-4]
        
        parts = path.split("/")
        if len(parts) >= 2:
            return parts[0], parts[1]
        else:
            raise GitError(f"Invalid repository URL format: {repo_url}")
    
    async def get_file_content(
        self,
        correlation_id: str,
        repo_path: str,
        file_path: str,
        sandbox_service: SandboxService
    ) -> str:
        """
        Get the content of a file from the repository.
        
        Args:
            correlation_id: Sandbox identifier
            repo_path: Path to the repository
            file_path: Path to the file
            sandbox_service: Sandbox service instance
            
        Returns:
            File content
            
        Raises:
            GitError: If file reading fails
        """
        with self.telemetry.trace_operation(
            "get_file_content",
            correlation_id=correlation_id,
            file_path=file_path
        ):
            try:
                full_path = f"{repo_path}/{file_path}"
                content = await sandbox_service.read_file(
                    correlation_id=correlation_id,
                    file_path=full_path
                )
                
                self.telemetry.log_event(
                    "File content retrieved",
                    correlation_id=correlation_id,
                    file_path=file_path,
                    content_length=len(content)
                )
                
                return content
                
            except SandboxError as e:
                raise GitError(f"Failed to read file {file_path}: {e}")
            except Exception as e:
                self.telemetry.log_error(
                    e,
                    context={"correlation_id": correlation_id, "file_path": file_path},
                    correlation_id=correlation_id
                )
                raise GitError(f"Failed to get file content: {e}")
    
    async def write_file_content(
        self,
        correlation_id: str,
        repo_path: str,
        file_path: str,
        content: str,
        sandbox_service: SandboxService
    ) -> None:
        """
        Write content to a file in the repository.
        
        Args:
            correlation_id: Sandbox identifier
            repo_path: Path to the repository
            file_path: Path to the file
            content: Content to write
            sandbox_service: Sandbox service instance
            
        Raises:
            GitError: If file writing fails
        """
        with self.telemetry.trace_operation(
            "write_file_content",
            correlation_id=correlation_id,
            file_path=file_path,
            content_length=len(content)
        ):
            try:
                full_path = f"{repo_path}/{file_path}"

                dir_path = os.path.dirname(full_path)
                if dir_path and dir_path != repo_path:
                    await sandbox_service.execute_command(
                        correlation_id=correlation_id,
                        command=f"mkdir -p {dir_path}",
                        working_dir=repo_path
                    )
                
                await sandbox_service.write_file(
                    correlation_id=correlation_id,
                    file_path=full_path,
                    content=content
                )
                
                self.telemetry.log_event(
                    "File content written",
                    correlation_id=correlation_id,
                    file_path=file_path,
                    content_length=len(content)
                )
                
            except SandboxError as e:
                raise GitError(f"Failed to write file {file_path}: {e}")
            except Exception as e:
                self.telemetry.log_error(
                    e,
                    context={"correlation_id": correlation_id, "file_path": file_path},
                    correlation_id=correlation_id
                )
                raise GitError(f"Failed to write file content: {e}")
    
    async def list_repository_files(
        self,
        correlation_id: str,
        repo_path: str,
        sandbox_service: SandboxService,
        max_files: int = 1000
    ) -> list[str]:
        """
        List all files in the repository.
        
        Args:
            correlation_id: Sandbox identifier
            repo_path: Path to the repository
            sandbox_service: Sandbox service instance
            max_files: Maximum number of files to return
            
        Returns:
            List of file paths
            
        Raises:
            GitError: If listing fails
        """
        with self.telemetry.trace_operation(
            "list_repository_files",
            correlation_id=correlation_id,
            repo_path=repo_path
        ):
            try:
                # Use a simple find command without pipes to avoid shell issues
                stdout, stderr, exit_code = await sandbox_service.execute_command(
                    correlation_id=correlation_id,
                    command='find . -type f -not -path "*/.git/*"',
                    working_dir=repo_path
                )
                
                if exit_code != 0:
                    # Try alternative approach if find fails
                    self.telemetry.log_event(
                        f"Find command failed (exit {exit_code}), trying ls approach",
                        correlation_id=correlation_id,
                        stderr=stderr,
                        level="warning"
                    )
                    
                    # Fallback to ls approach
                    stdout, stderr, exit_code = await sandbox_service.execute_command(
                        correlation_id=correlation_id,
                        command='ls -la',
                        working_dir=repo_path
                    )
                    
                    if exit_code != 0:
                        raise GitError(f"Failed to list files with both find and ls. Find stderr: '{stderr}'. Working dir: {repo_path}")
                    
                    # If ls works, try a different find approach
                    stdout, stderr, exit_code = await sandbox_service.execute_command(
                        correlation_id=correlation_id,
                        command='find . -name "*"',
                        working_dir=repo_path
                    )
                    
                    if exit_code != 0:
                        # If even basic find fails, manually build file list
                        stdout, stderr, exit_code = await sandbox_service.execute_command(
                            correlation_id=correlation_id,
                            command='ls -R',
                            working_dir=repo_path
                        )
                        
                        if exit_code != 0:
                            raise GitError(f"All file listing methods failed. Last stderr: '{stderr}'. Working dir: {repo_path}")
                
                # Process the output to get clean relative paths
                files = []
                lines = stdout.split("\n")
                
                for line in lines:
                    if line.strip():
                        # Remove leading './' from find output
                        relative_path = line.strip()
                        if relative_path.startswith("./"):
                            relative_path = relative_path[2:]
                        
                        # Skip git files, directories, and empty paths
                        if (relative_path and 
                            not relative_path.startswith(".git/") and
                            not relative_path.endswith("/") and
                            not relative_path.startswith(".git") and
                            relative_path != "." and
                            relative_path != ".." and
                            ":" not in relative_path):  # Skip ls -R directory headers
                            files.append(relative_path)
                
                # Remove duplicates and limit
                files = list(set(files))
                if len(files) > max_files:
                    files = files[:max_files]
                
                # If no files found, try to debug the issue
                if not files:
                    # Check if directory exists and has content
                    debug_stdout, debug_stderr, debug_exit = await sandbox_service.execute_command(
                        correlation_id=correlation_id,
                        command='pwd && ls -la',
                        working_dir=repo_path
                    )
                    
                    self.telemetry.log_event(
                        "No files found, debugging",
                        correlation_id=correlation_id,
                        debug_stdout=debug_stdout,
                        debug_stderr=debug_stderr,
                        debug_exit=debug_exit,
                        working_dir=repo_path,
                        level="warning"
                    )
                
                self.telemetry.log_event(
                    "Repository files listed",
                    correlation_id=correlation_id,
                    file_count=len(files),
                    repo_path=repo_path,
                    sample_files=files[:5] if files else []
                )
                
                return files
                
            except SandboxError as e:
                raise GitError(f"Failed to list repository files: {e}")
            except Exception as e:
                self.telemetry.log_error(
                    e,
                    context={"correlation_id": correlation_id, "repo_path": repo_path},
                    correlation_id=correlation_id
                )
                raise GitError(f"Failed to list repository files: {e}")
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the Git service.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if not GITHUB_AVAILABLE:
                return False
                
            if not self.github_client:
                return False
                            
            user = self.github_client.get_user()
            return user is not None
            
        except Exception as e:
            self.telemetry.log_error(
                e,
                context={"service": "git", "operation": "health_check"}
            )
            return False 