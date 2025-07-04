"""
Services for the Backspace Coding Agent.
"""

# Core services (always available)
from .streaming import StreamingService
from .sandbox import SandboxService

# Lazy imports for optional services
def get_git_service():
    """Get GitService instance with lazy import."""
    try:
        from .git_service import GitService
        return GitService
    except ImportError:
        return None

def get_agent_service():
    """Get AgentService instance with lazy import."""
    try:
        from .agent_service import AgentService
        return AgentService
    except ImportError:
        return None

# For backwards compatibility
GitService = get_git_service()
AgentService = get_agent_service()

__all__ = [
    "StreamingService",
    "SandboxService",
    "GitService",
    "AgentService",
    "get_git_service",
    "get_agent_service",
] 