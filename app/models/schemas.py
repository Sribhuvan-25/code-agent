"""
Pydantic schemas for the Backspace Coding Agent API.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, HttpUrl, validator


class StreamEventType(str, Enum):
    """Types of streaming events."""
    AI_MESSAGE = "AI Message"
    TOOL_BASH = "Tool: Bash"
    TOOL_READ = "Tool: Read"
    TOOL_EDIT = "Tool: Edit"
    TOOL_GIT = "Tool: Git"
    ERROR = "Error"
    SUCCESS = "Success"
    PROGRESS = "Progress"
    METRICS = "Metrics"


class CodeRequest(BaseModel):
    """Request schema for the /code endpoint."""
    
    repo_url: str = Field(
        ...,
        description="URL of the GitHub repository to clone and modify",
        example="https://github.com/example/simple-api"
    )
    
    prompt: str = Field(
        ...,
        description="Natural language prompt describing the code changes to make",
        min_length=10,
        max_length=2000,
        example="Add input validation to all POST endpoints and return proper error messages"
    )
    
    branch_name: Optional[str] = Field(
        default=None,
        description="Custom branch name for the changes (optional)",
        example="feature/add-input-validation"
    )
    
    ai_provider: Optional[str] = Field(
        default=None,
        description="AI provider to use (openai or anthropic)",
        example="openai"
    )
    
    @validator("repo_url")
    def validate_repo_url(cls, v):
        """Validate the repository URL."""
        if not v.startswith("https://github.com/"):
            raise ValueError("Only GitHub repositories are supported")
        return v
    
    @validator("ai_provider")
    def validate_ai_provider(cls, v):
        """Validate the AI provider."""
        if v is not None and v not in ["openai", "anthropic"]:
            raise ValueError("AI provider must be 'openai' or 'anthropic'")
        return v


class StreamEvent(BaseModel):
    """Schema for streaming events."""
    
    type: StreamEventType = Field(
        ...,
        description="Type of the streaming event"
    )
    
    message: Optional[str] = Field(
        default=None,
        description="Human-readable message for the event"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the event"
    )
    
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for request tracing"
    )
    
    # Tool-specific fields
    command: Optional[str] = Field(
        default=None,
        description="Command executed (for Tool: Bash events)"
    )
    
    output: Optional[str] = Field(
        default=None,
        description="Output from command or tool execution"
    )
    
    filepath: Optional[str] = Field(
        default=None,
        description="File path (for Tool: Read/Edit events)"
    )
    
    old_str: Optional[str] = Field(
        default=None,
        description="Old string content (for Tool: Edit events)"
    )
    
    new_str: Optional[str] = Field(
        default=None,
        description="New string content (for Tool: Edit events)"
    )
    
    # Progress fields
    progress: Optional[float] = Field(
        default=None,
        description="Progress percentage (0-100)",
        ge=0,
        le=100
    )
    
    step: Optional[str] = Field(
        default=None,
        description="Current step description"
    )
    
    # Metrics fields
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metrics data"
    )
    
    # Error fields
    error: Optional[str] = Field(
        default=None,
        description="Error message"
    )
    
    error_type: Optional[str] = Field(
        default=None,
        description="Error type"
    )
    
    # Additional context
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context data"
    )


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    
    success: bool = Field(
        default=False,
        description="Always false for error responses"
    )
    
    error: str = Field(
        ...,
        description="Error message"
    )
    
    error_type: str = Field(
        ...,
        description="Type of error"
    )
    
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for request tracing"
    )
    
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error context"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the error"
    )


class HealthCheck(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(
        default="healthy",
        description="Health status"
    )
    
    version: str = Field(
        ...,
        description="Application version"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the health check"
    )
    
    services: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of individual services"
    )


class SandboxMetrics(BaseModel):
    """Metrics for sandbox operations."""
    
    container_id: str = Field(
        ...,
        description="Docker container ID"
    )
    
    cpu_usage: float = Field(
        ...,
        description="CPU usage percentage"
    )
    
    memory_usage: float = Field(
        ...,
        description="Memory usage in MB"
    )
    
    disk_usage: float = Field(
        ...,
        description="Disk usage in MB"
    )
    
    network_io: Dict[str, int] = Field(
        default_factory=dict,
        description="Network I/O stats"
    )
    
    uptime: float = Field(
        ...,
        description="Container uptime in seconds"
    )
    
    process_count: int = Field(
        default=0,
        description="Number of processes in container"
    ) 