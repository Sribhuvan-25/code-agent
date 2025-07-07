"""
LangChain tools for the Backspace Coding Agent.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from app.services.sandbox import SandboxService
from app.services.git_service import GitService
from app.core.telemetry import get_telemetry

logger = logging.getLogger(__name__)


class AnalyzeRepositoryInput(BaseModel):
    """Input for repository analysis."""
    correlation_id: str = Field(description="Correlation ID for tracking")
    repo_url: str = Field(description="Repository URL to analyze")


class AnalyzeRepositoryTool(BaseTool):
    """Tool for analyzing repository structure."""
    
    name: str = "analyze_repository"
    description: str = "Analyze the structure and content of a repository"
    args_schema: type[AnalyzeRepositoryInput] = AnalyzeRepositoryInput
    _sandbox_service: SandboxService = PrivateAttr()
    _git_service: GitService = PrivateAttr()
    _telemetry: Any = PrivateAttr()
    
    def __init__(self, sandbox_service: SandboxService, git_service: GitService):
        super().__init__()
        self._sandbox_service = sandbox_service
        self._git_service = git_service
        self._telemetry = get_telemetry()
    
    async def _arun(self, correlation_id: str, repo_url: str) -> Dict[str, Any]:
        if not correlation_id:
            raise ValueError("correlation_id missing in AnalyzeRepositoryTool._arun")
        try:
            # Step 1: Clone the repository
            repo_path = await self._git_service.clone_repository(
                correlation_id=correlation_id,
                repo_url=repo_url,
                sandbox_service=self._sandbox_service
            )
            
            # Step 2: List files
            files = await self._git_service.list_repository_files(
                correlation_id=correlation_id,
                repo_path=repo_path,
                sandbox_service=self._sandbox_service
            )
            
            # Step 3: Analyze file types, key files, and languages
            file_types = set()
            key_files = []
            languages = set()
            dependencies = {}
            
            for f in files:
                ext = os.path.splitext(f)[1]
                if ext:
                    file_types.add(ext)
                if os.path.basename(f).lower() in {
                    "readme.md", "main.py", "app.js", "index.js", "index.html", "package.json",
                    "requirements.txt", "pipfile", "pyproject.toml", "go.mod", "cargo.toml",
                    "main.go", "lib.rs", "main.rs", "composer.json", "gemfile", "pom.xml",
                    "build.gradle", "dockerfile", "docker-compose.yml", "makefile"
                }:
                    key_files.append(f)
                if ext in {".py"}:
                    languages.add("Python")
                elif ext in {".js", ".jsx", ".ts", ".tsx", ".mjs"}:
                    languages.add("JavaScript/TypeScript")
                elif ext in {".go"}:
                    languages.add("Go")
                elif ext in {".rs"}:
                    languages.add("Rust")
                elif ext in {".java", ".kotlin", ".scala"}:
                    languages.add("JVM Languages")
                elif ext in {".php"}:
                    languages.add("PHP")
                elif ext in {".rb"}:
                    languages.add("Ruby")
                elif ext in {".cs", ".vb"}:
                    languages.add(".NET")
                elif ext in {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"}:
                    languages.add("C/C++")
                elif ext in {".swift"}:
                    languages.add("Swift")
                elif ext in {".html", ".htm"}:
                    languages.add("HTML")
                elif ext in {".css", ".scss", ".sass", ".less"}:
                    languages.add("CSS/Styling")
            
            # Step 4: Check for dependency files and extract dependencies
            dependency_files = []
            dependencies = {}
            
            # Look for various dependency file types
            dependency_patterns = {
                "package.json": "javascript",
                "requirements.txt": "python", 
                "Pipfile": "python",
                "pyproject.toml": "python",
                "go.mod": "go",
                "Cargo.toml": "rust", 
                "composer.json": "php",
                "Gemfile": "ruby",
                "pom.xml": "java",
                "build.gradle": "java",
                "packages.config": "csharp",
                "*.csproj": "csharp"
            }
            
            for f in files:
                filename = os.path.basename(f).lower()
                for pattern, lang in dependency_patterns.items():
                    if pattern.startswith("*"):
                        # Handle wildcard patterns like *.csproj
                        if filename.endswith(pattern[1:]):
                            dependency_files.append({"file": f, "type": lang, "format": pattern})
                    elif filename == pattern:
                        dependency_files.append({"file": f, "type": lang, "format": pattern})
            
            # Parse dependency files
            for dep_file in dependency_files:
                try:
                    content = await self._git_service.get_file_content(
                        correlation_id=correlation_id,
                        repo_path=repo_path,
                        file_path=dep_file["file"],
                        sandbox_service=self._sandbox_service
                    )
                    
                    # Parse based on file type
                    if dep_file["format"] == "package.json":
                        import json
                        package_data = json.loads(content)
                        dependencies[dep_file["format"]] = {
                            "dependencies": package_data.get("dependencies", {}),
                            "devDependencies": package_data.get("devDependencies", {}),
                            "peerDependencies": package_data.get("peerDependencies", {}),
                            "optionalDependencies": package_data.get("optionalDependencies", {})
                        }
                    elif dep_file["format"] == "requirements.txt":
                        # Parse requirements.txt format
                        lines = content.strip().split('\n')
                        deps = {}
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                # Extract package name (before ==, >=, etc.)
                                pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].strip()
                                deps[pkg_name] = line
                        dependencies[dep_file["format"]] = {"dependencies": deps}
                    elif dep_file["format"] in ["Pipfile", "pyproject.toml", "Cargo.toml"]:
                        # These are TOML format files
                        try:
                            import toml
                            toml_data = toml.loads(content)
                            if dep_file["format"] == "Pipfile":
                                dependencies[dep_file["format"]] = {
                                    "packages": toml_data.get("packages", {}),
                                    "dev-packages": toml_data.get("dev-packages", {})
                                }
                            elif dep_file["format"] == "Cargo.toml":
                                dependencies[dep_file["format"]] = {
                                    "dependencies": toml_data.get("dependencies", {}),
                                    "dev-dependencies": toml_data.get("dev-dependencies", {})
                                }
                            elif dep_file["format"] == "pyproject.toml":
                                dependencies[dep_file["format"]] = {
                                    "dependencies": toml_data.get("project", {}).get("dependencies", []),
                                    "optional-dependencies": toml_data.get("project", {}).get("optional-dependencies", {})
                                }
                        except ImportError:
                            dependencies[dep_file["format"]] = {"error": "TOML parser not available"}
                    else:
                        # For other formats, just store the raw content for now
                        dependencies[dep_file["format"]] = {"raw_content": content[:500] + "..." if len(content) > 500 else content}
                    
                    # Log analysis for common dependencies
                    self._telemetry.log_event(
                        f"Dependency file analyzed: {dep_file['format']}",
                        correlation_id=correlation_id,
                        language=dep_file["type"],
                        file_path=dep_file["file"]
                    )
                    
                except Exception as e:
                    self._telemetry.log_event(
                        f"Failed to parse {dep_file['format']}",
                        correlation_id=correlation_id,
                        error=str(e),
                        level="warning"
                    )
                    dependencies[dep_file["format"]] = {"error": f"Failed to parse: {str(e)}"}
            
            analysis = {
                "repo_path": repo_path,
                "files": files,
                "file_types": list(file_types),
                "key_files": key_files,
                "languages": list(languages),
                "dependencies": dependencies,
                "dependency_files": dependency_files,
                "has_dependencies": len(dependency_files) > 0
            }
            
            self._telemetry.log_event(
                "Repository analysis completed",
                correlation_id=correlation_id,
                file_count=len(files),
                languages=list(languages)
            )
            
            return analysis
            
        except Exception as e:
            self._telemetry.log_error(
                e,
                context={"step": "analyze_repository", "correlation_id": correlation_id, "repo_url": repo_url},
                correlation_id=correlation_id
            )
            raise

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Synchronous run is not supported. Use async.")


class ReadFileInput(BaseModel):
    """Input for reading files."""
    correlation_id: str = Field(description="Correlation ID for tracking")
    repo_path: str = Field(description="Repository path")
    file_path: str = Field(description="Path to the file to read")


class ReadFileTool(BaseTool):
    """Tool for reading file contents."""
    
    name: str = "read_file"
    description: str = "Read the contents of a file in the repository"
    args_schema: type[ReadFileInput] = ReadFileInput
    _git_service: GitService = PrivateAttr()
    _sandbox_service: SandboxService = PrivateAttr()
    _telemetry: Any = PrivateAttr()
    
    def __init__(self, git_service: GitService, sandbox_service: SandboxService):
        super().__init__()
        self._git_service = git_service
        self._sandbox_service = sandbox_service
        self._telemetry = get_telemetry()
    
    async def _arun(self, correlation_id: str, repo_path: str, file_path: str) -> str:
        """Read file asynchronously."""
        try:
            content = await self._git_service.get_file_content(
                correlation_id=correlation_id,
                repo_path=repo_path,
                file_path=file_path,
                sandbox_service=self._sandbox_service
            )
            
            self._telemetry.log_event(
                "File read successfully",
                correlation_id=correlation_id,
                file_path=file_path,
                content_length=len(content)
            )
            
            return content
            
        except Exception as e:
            self._telemetry.log_error(
                e,
                context={"correlation_id": correlation_id, "file_path": file_path},
                correlation_id=correlation_id
            )
            raise

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Synchronous run is not supported. Use async.")


class WriteFileInput(BaseModel):
    """Input for writing files."""
    correlation_id: str = Field(description="Correlation ID for tracking")
    repo_path: str = Field(description="Repository path")
    file_path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write to the file")


class WriteFileTool(BaseTool):
    """Tool for writing file contents."""
    
    name: str = "write_file"
    description: str = "Write content to a file in the repository"
    args_schema: type[WriteFileInput] = WriteFileInput
    _git_service: GitService = PrivateAttr()
    _sandbox_service: SandboxService = PrivateAttr()
    _telemetry: Any = PrivateAttr()
    
    def __init__(self, git_service: GitService, sandbox_service: SandboxService):
        super().__init__()
        self._git_service = git_service
        self._sandbox_service = sandbox_service
        self._telemetry = get_telemetry()
    
    async def _arun(self, correlation_id: str, repo_path: str, file_path: str, content: str) -> Dict[str, Any]:
        """Write file asynchronously."""
        try:
            await self._git_service.write_file_content(
                correlation_id=correlation_id,
                repo_path=repo_path,
                file_path=file_path,
                content=content,
                sandbox_service=self._sandbox_service
            )
            
            result = {
                "success": True,
                "file_path": file_path,
                "content_length": len(content)
            }
            
            self._telemetry.log_event(
                "File written successfully",
                correlation_id=correlation_id,
                file_path=file_path,
                content_length=len(content)
            )
            
            return result
            
        except Exception as e:
            self._telemetry.log_error(
                e,
                context={"correlation_id": correlation_id, "file_path": file_path},
                correlation_id=correlation_id
            )
            raise

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Synchronous run is not supported. Use async.")


class CreateBranchInput(BaseModel):
    """Input for creating branches."""
    correlation_id: str = Field(description="Correlation ID for tracking")
    repo_path: str = Field(description="Repository path")
    branch_name: str = Field(description="Name of the branch to create")


class CreateBranchTool(BaseTool):
    """Tool for creating git branches."""
    
    name: str = "create_branch"
    description: str = "Create a new branch in the repository"
    args_schema: type[CreateBranchInput] = CreateBranchInput
    _git_service: GitService = PrivateAttr()
    _sandbox_service: SandboxService = PrivateAttr()
    _telemetry: Any = PrivateAttr()
    
    def __init__(self, git_service: GitService, sandbox_service: SandboxService):
        super().__init__()
        self._git_service = git_service
        self._sandbox_service = sandbox_service
        self._telemetry = get_telemetry()
    
    async def _arun(self, correlation_id: str, repo_path: str, branch_name: str) -> Dict[str, Any]:
        """Create branch asynchronously."""
        try:
            await self._git_service.create_branch(
                correlation_id=correlation_id,
                repo_path=repo_path,
                branch_name=branch_name,
                sandbox_service=self._sandbox_service
            )
            
            result = {
                "success": True,
                "branch_name": branch_name
            }
            
            self._telemetry.log_event(
                "Branch created successfully",
                correlation_id=correlation_id,
                branch_name=branch_name
            )
            
            return result
            
        except Exception as e:
            self._telemetry.log_error(
                e,
                context={"correlation_id": correlation_id, "branch_name": branch_name},
                correlation_id=correlation_id
            )
            raise

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Synchronous run is not supported. Use async.")


class CommitChangesInput(BaseModel):
    """Input for committing changes."""
    correlation_id: str = Field(description="Correlation ID for tracking")
    repo_path: str = Field(description="Repository path")
    message: str = Field(description="Commit message")


class CommitChangesTool(BaseTool):
    """Tool for committing git changes."""
    
    name: str = "commit_changes"
    description: str = "Commit changes to the repository"
    args_schema: type[CommitChangesInput] = CommitChangesInput
    _git_service: GitService = PrivateAttr()
    _sandbox_service: SandboxService = PrivateAttr()
    _telemetry: Any = PrivateAttr()
    
    def __init__(self, git_service: GitService, sandbox_service: SandboxService):
        super().__init__()
        self._git_service = git_service
        self._sandbox_service = sandbox_service
        self._telemetry = get_telemetry()
    
    async def _arun(self, correlation_id: str, repo_path: str, message: str) -> Dict[str, Any]:
        """Commit changes asynchronously."""
        try:
            commit_hash = await self._git_service.commit_changes(
                correlation_id=correlation_id,
                repo_path=repo_path,
                message=message,
                sandbox_service=self._sandbox_service
            )
            
            result = {
                "success": True,
                "commit_hash": commit_hash,
                "message": message
            }
            
            self._telemetry.log_event(
                "Changes committed successfully",
                correlation_id=correlation_id,
                commit_hash=commit_hash,
                message=message
            )
            
            return result
            
        except Exception as e:
            self._telemetry.log_error(
                e,
                context={"correlation_id": correlation_id, "message": message},
                correlation_id=correlation_id
            )
            raise

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Synchronous run is not supported. Use async.")


class PushChangesInput(BaseModel):
    """Input for pushing changes."""
    correlation_id: str = Field(description="Correlation ID for tracking")
    repo_path: str = Field(description="Repository path")
    branch_name: str = Field(description="Name of the branch to push")


class PushChangesTool(BaseTool):
    """Tool for pushing git changes."""
    
    name: str = "push_changes"
    description: str = "Push changes to the remote repository"
    args_schema: type[PushChangesInput] = PushChangesInput
    _git_service: GitService = PrivateAttr()
    _sandbox_service: SandboxService = PrivateAttr()
    _telemetry: Any = PrivateAttr()
    
    def __init__(self, git_service: GitService, sandbox_service: SandboxService):
        super().__init__()
        self._git_service = git_service
        self._sandbox_service = sandbox_service
        self._telemetry = get_telemetry()
    
    async def _arun(self, correlation_id: str, repo_path: str, branch_name: str) -> Dict[str, Any]:
        """Push changes asynchronously."""
        try:
            await self._git_service.push_changes(
                correlation_id=correlation_id,
                repo_path=repo_path,
                branch_name=branch_name,
                sandbox_service=self._sandbox_service
            )
            
            result = {
                "success": True,
                "branch_name": branch_name
            }
            
            self._telemetry.log_event(
                "Changes pushed successfully",
                correlation_id=correlation_id,
                branch_name=branch_name
            )
            
            return result
            
        except Exception as e:
            self._telemetry.log_error(
                e,
                context={"correlation_id": correlation_id, "branch_name": branch_name},
                correlation_id=correlation_id
            )
            raise

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Synchronous run is not supported. Use async.")


class ExecuteCommandInput(BaseModel):
    """Input for executing commands."""
    correlation_id: str = Field(description="Correlation ID for tracking")
    command: str = Field(description="Command to execute")
    working_dir: str = Field(description="Working directory for the command")


class ExecuteCommandTool(BaseTool):
    """Tool for executing shell commands in the sandbox."""
    
    name: str = "execute_command"
    description: str = "Execute a shell command in the sandbox environment"
    args_schema: type[ExecuteCommandInput] = ExecuteCommandInput
    _sandbox_service: SandboxService = PrivateAttr()

    def __init__(self, sandbox_service: SandboxService):
        super().__init__()
        self._sandbox_service = sandbox_service
        self._telemetry = get_telemetry()
    
    async def _arun(self, correlation_id: str, command: str, working_dir: str) -> Dict[str, Any]:
        """Execute command asynchronously."""
        try:
            stdout, stderr, exit_code = await self._sandbox_service.execute_command(
                correlation_id=correlation_id,
                command=command,
                working_dir=working_dir
            )
            
            result = {
                "success": exit_code == 0,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "command": command
            }
            
            self._telemetry.log_event(
                "Command executed",
                correlation_id=correlation_id,
                command=command,
                exit_code=exit_code,
                success=exit_code == 0
            )
            
            return result
            
        except Exception as e:
            self._telemetry.log_error(
                e,
                context={"correlation_id": correlation_id, "command": command},
                correlation_id=correlation_id
            )
            raise

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Synchronous run is not supported. Use async.")


def create_toolkit(sandbox_service: SandboxService, git_service: GitService) -> List[BaseTool]:
    """Create a toolkit with all available tools."""
    return [
        AnalyzeRepositoryTool(sandbox_service, git_service),
        ReadFileTool(git_service, sandbox_service),
        WriteFileTool(git_service, sandbox_service),
        CreateBranchTool(git_service, sandbox_service),
        CommitChangesTool(git_service, sandbox_service),
        PushChangesTool(git_service, sandbox_service),
        ExecuteCommandTool(sandbox_service)
    ] 