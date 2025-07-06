"""
Security utilities for the Backspace Coding Agent.
"""

import re
import urllib.parse
from typing import Optional
from urllib.parse import urlparse

from fastapi import HTTPException, status


class SecurityError(Exception):
    """Base exception for security-related errors."""
    pass


class InputValidationError(SecurityError):
    """Exception raised when input validation fails."""
    pass


def validate_github_url(url: str) -> str:
    """
    Validate and sanitize a GitHub repository URL.
    
    Args:
        url: The GitHub repository URL to validate
        
    Returns:
        The sanitized URL
        
    Raises:
        InputValidationError: If the URL is invalid or not a GitHub repository
    """
    if not url or not isinstance(url, str):
        raise InputValidationError("Repository URL is required")
    
    try:
        parsed = urlparse(url.strip())
    except Exception as e:
        raise InputValidationError(f"Invalid URL format: {e}")
    
    # Check if it's a GitHub URL
    if parsed.netloc.lower() not in ["github.com", "www.github.com"]:
        raise InputValidationError("Only GitHub repositories are supported")
    
    # Check if it's HTTPS
    if parsed.scheme != "https":
        raise InputValidationError("Only HTTPS URLs are supported")
    
    # Validate path format (should be /owner/repo or /owner/repo.git)
    path_parts = parsed.path.strip("/").split("/")
    if len(path_parts) < 2:
        raise InputValidationError("Invalid GitHub repository URL format")
    
    owner, repo = path_parts[0], path_parts[1]
    
    # Remove .git suffix if present
    if repo.endswith(".git"):
        repo = repo[:-4]
    
    # Validate owner and repo names
    if not re.match(r"^[a-zA-Z0-9._-]+$", owner):
        raise InputValidationError("Invalid repository owner name")
    
    if not re.match(r"^[a-zA-Z0-9._-]+$", repo):
        raise InputValidationError("Invalid repository name")
    
    # Return sanitized URL
    return f"https://github.com/{owner}/{repo}.git"


def validate_prompt(prompt: str) -> str:
    """
    Validate and sanitize a coding prompt.
    
    Args:
        prompt: The coding prompt to validate
        
    Returns:
        The sanitized prompt
        
    Raises:
        InputValidationError: If the prompt is invalid
    """
    if not prompt or not isinstance(prompt, str):
        raise InputValidationError("Prompt is required")
    
    prompt = prompt.strip()
    
    if len(prompt) < 10:
        raise InputValidationError("Prompt must be at least 10 characters long")
    
    if len(prompt) > 2000:
        raise InputValidationError("Prompt must be less than 2000 characters")
    
    # Check for potentially dangerous commands
    dangerous_patterns = [
        r"\brm\s+-rf\b",
        r"\bsudo\b",
        r"\bchmod\s+777\b",
        r"\bcurl\s+.*\|\s*sh\b",
        r"\bwget\s+.*\|\s*sh\b",
        r"\bdd\s+if=",
        r"\b/dev/",
        r"\bfork\s*\(\s*\)",
        r"while\s*\(\s*1\s*\)",
        r":\(\)\{\s*:\|:&\s*\};:",  # Fork bomb
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            raise InputValidationError("Prompt contains potentially dangerous commands")
    
    return prompt


def validate_file_path(file_path: str) -> str:
    """
    Validate a file path to prevent directory traversal attacks.
    
    Args:
        file_path: The file path to validate
        
    Returns:
        The sanitized file path
        
    Raises:
        InputValidationError: If the file path is invalid or dangerous
    """
    if not file_path or not isinstance(file_path, str):
        raise InputValidationError("File path is required")
    
    # Normalize the path
    normalized = urllib.parse.unquote(file_path.strip())
    
    # Check for directory traversal attempts
    if ".." in normalized or normalized.startswith("/"):
        raise InputValidationError("Invalid file path: directory traversal detected")
    
    # Check for dangerous paths
    dangerous_paths = [
        "/etc/",
        "/var/",
        "/sys/",
        "/proc/",
        "/dev/",
        "/root/",
        "/home/",
        ".ssh/",
        ".git/",
    ]
    
    for dangerous_path in dangerous_paths:
        if dangerous_path in normalized.lower():
            raise InputValidationError(f"Access to {dangerous_path} is not allowed")
    
    return normalized


def sanitize_branch_name(branch_name: str) -> str:
    """
    Sanitize a branch name to ensure it's valid for Git.
    
    Args:
        branch_name: The branch name to sanitize
        
    Returns:
        The sanitized branch name
    """
    if not branch_name or not isinstance(branch_name, str):
        return "feature/auto-generated"
    
    sanitized = re.sub(r"[^a-zA-Z0-9._/-]", "-", branch_name.strip())
    
    sanitized = sanitized.strip("-/")
    
    if sanitized.startswith("."):
        sanitized = "branch-" + sanitized
    
    if not sanitized:
        sanitized = "feature/auto-generated"
    
    if len(sanitized) > 100:
        sanitized = sanitized[:100].rstrip("-/")
    
    return sanitized


def create_correlation_id() -> str:
    """
    Create a unique correlation ID for request tracing.
    
    Returns:
        A unique correlation ID
    """
    import uuid
    return str(uuid.uuid4())


def validate_environment_security() -> None:
    """
    Validate that required security environment variables are set.
    
    Raises:
        SecurityError: If required security settings are missing
    """
    from app.core.config import settings
    
    if not settings.github_token:
        raise SecurityError("GitHub token is required but not configured")
    
    if not settings.openai_api_key and not settings.anthropic_api_key:
        raise SecurityError("At least one AI provider API key must be configured")


def check_rate_limit(client_id: str, limit: int = 10, window: int = 60) -> bool:
    """
    Check if a client has exceeded the rate limit.
    
    Args:
        client_id: Unique identifier for the client
        limit: Maximum number of requests allowed
        window: Time window in seconds
        
    Returns:
        True if the request is within the rate limit, False otherwise
    """
    import time
    from collections import defaultdict
    
    if not hasattr(check_rate_limit, "requests"):
        check_rate_limit.requests = defaultdict(list)
    
    now = time.time()
    client_requests = check_rate_limit.requests[client_id]
    
    client_requests[:] = [req_time for req_time in client_requests if now - req_time < window]
    
    if len(client_requests) >= limit:
        return False
    
    client_requests.append(now)
    return True 