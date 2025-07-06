"""
Main /code endpoint for the Backspace Coding Agent.
"""

import asyncio
from typing import Optional
import time

from fastapi import APIRouter, BackgroundTasks, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.security import (
    validate_github_url,
    validate_prompt,
    sanitize_branch_name,
    create_correlation_id,
    check_rate_limit,
    validate_environment_security,
    InputValidationError,
    SecurityError
)
from app.core.telemetry import get_telemetry
from app.models.schemas import CodeRequest, CodeResponse, StreamEvent, StreamEventType
from app.services.streaming import get_streaming_service
from app.services.sandbox import SandboxService
from app.services import get_git_service, get_agent_service

router = APIRouter()
telemetry = get_telemetry()

streaming_service = get_streaming_service()

def get_git_service_instance():
    """Get git service instance with lazy initialization."""
    GitService = get_git_service()
    if GitService is None:
        raise HTTPException(
            status_code=503,
            detail="Git service not available - missing dependencies"
        )
    return GitService()

def get_agent_service_instance(streaming_service=None):
    """Get LangGraph agent service instance with lazy initialization."""
    AgentService = get_agent_service()
    if AgentService is None:
        raise HTTPException(
            status_code=503,
            detail="Agent service not available - missing dependencies"
        )
    return AgentService(streaming_service=streaming_service)

def get_sandbox_service_instance():
    """Get sandbox service instance from main app."""
    from app.main import get_sandbox_service
    sandbox_service = get_sandbox_service()
    if sandbox_service is None:
        raise HTTPException(
            status_code=503,
            detail="Sandbox service not available"
        )
    return sandbox_service

async def get_client_ip(request: Request) -> str:
    """Get client IP address for rate limiting."""
    return request.client.host if request.client else "unknown"


@router.post("/code")
async def create_code_changes(
    request: Request,
    code_request: CodeRequest,
    background_tasks: BackgroundTasks,
    client_ip: str = Depends(get_client_ip)
):
    """
    Create code changes based on a natural language prompt using LangGraph.
    
    This endpoint immediately starts streaming the coding process via Server-Sent Events.
    It uses the sophisticated LangGraph workflow with proper state management.
    
    Args:
        request: FastAPI request object
        code_request: Request payload
        background_tasks: Background tasks manager (unused now)
        client_ip: Client IP address
        
    Returns:
        StreamingResponse with Server-Sent Events
    """
    correlation_id = getattr(request.state, "correlation_id", create_correlation_id())
    
    sandbox_service_instance = get_sandbox_service_instance()
    
    telemetry.log_event(
        "LangGraph code request received",
        correlation_id=correlation_id,
        repo_url=code_request.repo_url,
        prompt_length=len(code_request.prompt),
        client_ip=client_ip
    )
    
    if not check_rate_limit(
        client_ip, 
        settings.rate_limit_requests, 
        settings.rate_limit_window
    ):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    try:
        validate_environment_security()
    except SecurityError as e:
        telemetry.log_error(e, correlation_id=correlation_id)
        raise HTTPException(
            status_code=500,
            detail="Service configuration error. Please contact support."
        )
    
    try:
        sanitized_repo_url = validate_github_url(code_request.repo_url)
        sanitized_prompt = validate_prompt(code_request.prompt)
        sanitized_branch_name = sanitize_branch_name(
            code_request.branch_name or f"backspace-agent-{correlation_id[:8]}"
        )
    except InputValidationError as e:
        telemetry.log_error(e, correlation_id=correlation_id)
        raise HTTPException(status_code=422, detail=str(e))
    
    try:
        response = await streaming_service.create_event_stream(
            correlation_id=correlation_id,
            request=request
        )
        
        active_connections = streaming_service.get_active_connections()
        telemetry.log_event(
            "Event stream created for LangGraph workflow",
            correlation_id=correlation_id,
            active_connections=list(active_connections.keys()),
            connection_count=len(active_connections)
        )
        
        # Start the LangGraph workflow task
        asyncio.create_task(
            process_langgraph_request(
            correlation_id=correlation_id,
            repo_url=sanitized_repo_url,
            prompt=sanitized_prompt,
            branch_name=sanitized_branch_name,
                ai_provider=code_request.ai_provider or "openai"
            )
        )
        
        telemetry.log_event(
            "LangGraph workflow task started",
            correlation_id=correlation_id
        )
        
        return response
        
    except Exception as e:
        telemetry.log_error(e, correlation_id=correlation_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to create streaming response"
        )


async def process_langgraph_request(
    correlation_id: str,
    repo_url: str,
    prompt: str,
    branch_name: str,
    ai_provider: str
):
    """
    Process the code request using LangGraph workflow.
    
    Args:
        correlation_id: Unique request identifier
        repo_url: Repository URL
        prompt: Coding prompt
        branch_name: Branch name for changes
        ai_provider: AI provider to use
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        telemetry.log_event(
            "LangGraph workflow started",
            correlation_id=correlation_id,
            repo_url=repo_url,
            prompt_length=len(prompt),
            ai_provider=ai_provider
        )
        
        # Initialize the LangGraph agent with streaming support
        agent = get_agent_service_instance(streaming_service)
        
        # Run the LangGraph agent workflow with streaming
        result = await agent.run(
            correlation_id=correlation_id,
            repo_url=repo_url,
            prompt=prompt,
            ai_provider=ai_provider
        )
        
        # Extract data from final_state (following CLI pattern)
        final_state = result.get("final_state", {})
        pr_url = final_state.get("pull_request_url")
        branch_name = final_state.get("branch_name")
        commit_hash = final_state.get("commit_hash") 
        changes_made = final_state.get("changes_made", [])
        files_changed = len(changes_made)
        push_success = final_state.get("push_success", False)
        
        # Send final result based on success/failure
        if result.get("success", False):
            await streaming_service.send_success(
                correlation_id=correlation_id,
                message=f"🎉 LangGraph workflow completed successfully!",
                context={
                    "pr_url": pr_url,
                    "branch_name": branch_name,
                    "commit_hash": commit_hash,
                    "files_changed": files_changed,
                    "changes_made": changes_made,
                    "push_success": push_success,
                    "workflow": "langgraph",
                    "duration": asyncio.get_event_loop().time() - start_time
                }
            )
        else:
            error_msg = result.get("error", "Unknown error occurred")
            await streaming_service.send_error(
            correlation_id=correlation_id,
                error=error_msg,
                error_type="LangGraphWorkflowError",
                context={"workflow": "langgraph"}
        )
        
        duration = asyncio.get_event_loop().time() - start_time
        telemetry.log_event(
            "LangGraph workflow completed",
            correlation_id=correlation_id,
            duration=duration,
            success=result.get("success", False),
            pr_url=pr_url,
            branch_name=branch_name,
            commit_hash=commit_hash,
            files_changed=files_changed
        )
        
    except Exception as e:
        telemetry.log_error(e, correlation_id=correlation_id)
        await streaming_service.send_error(
            correlation_id=correlation_id,
            error=str(e),
            error_type=type(e).__name__,
            context={"phase": "langgraph_execution"}
        )
    finally:
        await streaming_service.close_stream(correlation_id)


@router.get("/status/{correlation_id}")
async def get_status(correlation_id: str):
    """
    Get the status of a code request.
    
    Args:
        correlation_id: Request correlation ID
        
    Returns:
        Status information
    """
    try:
        connections = streaming_service.get_active_connections()
        
        if correlation_id in connections:
            return {
                "status": "active",
                "correlation_id": correlation_id,
                "connection_info": connections[correlation_id]
            }
        else:
            return {
                "status": "completed_or_not_found",
                "correlation_id": correlation_id
            }
            
    except Exception as e:
        telemetry.log_error(e, context={"endpoint": "status"})
        raise HTTPException(
            status_code=500,
            detail="Failed to get status"
        )


@router.get("/metrics")
async def get_metrics():
    """
    Get service metrics.
    
    Returns:
        Service metrics
    """
    try:
        metrics = telemetry.get_metrics()
        connections = streaming_service.get_active_connections()
        
        return {
            "active_connections": len(connections),
            "connections": connections,
            "metrics": metrics
        }
        
    except Exception as e:
        telemetry.log_error(e, context={"endpoint": "metrics"})
        raise HTTPException(
            status_code=500,
            detail="Failed to get metrics"
        ) 