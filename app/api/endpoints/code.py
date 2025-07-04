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

def get_agent_service_instance():
    """Get agent service instance with lazy initialization."""
    AgentService = get_agent_service()
    if AgentService is None:
        raise HTTPException(
            status_code=503,
            detail="Agent service not available - missing dependencies"
        )
    return AgentService()

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
    Create code changes based on a natural language prompt.
    
    This endpoint immediately starts streaming the coding process via Server-Sent Events.
    It performs the following steps:
    1. Validates input and performs security checks
    2. Creates a secure sandbox environment
    3. Clones the repository
    4. Runs the AI agent to analyze and modify code
    5. Creates a pull request with the changes
    6. Returns the PR URL and summary
    
    Args:
        request: FastAPI request object
        code_request: Request payload
        background_tasks: Background tasks manager
        client_ip: Client IP address
        
    Returns:
        StreamingResponse with Server-Sent Events
    """
    correlation_id = getattr(request.state, "correlation_id", create_correlation_id())
    
    sandbox_service_instance = get_sandbox_service_instance()
    
    telemetry.log_event(
        "Code request received",
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
            "Event stream created",
            correlation_id=correlation_id,
            active_connections=list(active_connections.keys()),
            connection_count=len(active_connections)
        )
        
        telemetry.log_event(
            "About to start background task",
            correlation_id=correlation_id,
            repo_url=sanitized_repo_url,
            prompt_length=len(sanitized_prompt)
        )
        
        background_tasks.add_task(
            run_background_task,
            correlation_id=correlation_id,
            repo_url=sanitized_repo_url,
            prompt=sanitized_prompt,
            branch_name=sanitized_branch_name,
            ai_provider=code_request.ai_provider
        )
        
        telemetry.log_event(
            "Background task added successfully",
            correlation_id=correlation_id
        )
        
        return response
        
    except Exception as e:
        telemetry.log_error(e, correlation_id=correlation_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to create streaming response"
        )


def run_background_task(
    correlation_id: str,
    repo_url: str,
    prompt: str,
    branch_name: str,
    ai_provider: Optional[str]
):
    """
    Synchronous wrapper function for background task execution.
    This is required because FastAPI background_tasks.add_task() only works with sync functions.
    """
    try:
        print(f"DEBUG: run_background_task called with correlation_id={correlation_id}")
        
        with open(f"/tmp/debug_bg_{correlation_id[:8]}.txt", "w") as f:
            f.write(f"run_background_task called at {time.time()}\n")
            f.write(f"correlation_id: {correlation_id}\n")
            f.write(f"repo_url: {repo_url}\n")
            f.write(f"prompt: {prompt}\n")
        
        telemetry.log_event(
            "run_background_task called",
            correlation_id=correlation_id,
            repo_url=repo_url,
            prompt_length=len(prompt)
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        print(f"DEBUG: Event loop created for correlation_id={correlation_id}")
        telemetry.log_event(
            "Event loop created",
            correlation_id=correlation_id
        )
        
        loop.run_until_complete(
            process_code_request(
                correlation_id=correlation_id,
                repo_url=repo_url,
                prompt=prompt,
                branch_name=branch_name,
                ai_provider=ai_provider
            )
        )
        
        print(f"DEBUG: Async function completed for correlation_id={correlation_id}")
        telemetry.log_event(
            "Async function completed",
            correlation_id=correlation_id
        )
        
    except Exception as e:
        print(f"ERROR: run_background_task failed with error: {e}")
        
        with open(f"/tmp/debug_bg_{correlation_id[:8]}_error.txt", "w") as f:
            f.write(f"run_background_task failed at {time.time()}\n")
            f.write(f"error: {str(e)}\n")
            f.write(f"error_type: {type(e).__name__}\n")
        
        telemetry.log_error(e, correlation_id=correlation_id)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                streaming_service.send_error(
                    correlation_id=correlation_id,
                    error=f"Background task failed: {str(e)}",
                    error_type="BackgroundTaskError"
                )
            )
        except Exception as stream_error:
            print(f"ERROR: Failed to send error via streaming: {stream_error}")
            telemetry.log_error(stream_error, correlation_id=correlation_id)
    finally:
        try:
            loop.close()
            print(f"DEBUG: Event loop closed for correlation_id={correlation_id}")
        except:
            pass


async def process_code_request(
    correlation_id: str,
    repo_url: str,
    prompt: str,
    branch_name: str,
    ai_provider: Optional[str]
):
    """
    Process the code request in the background.
    
    Args:
        correlation_id: Unique request identifier
        repo_url: Repository URL
        prompt: Coding prompt
        branch_name: Branch name for changes
        ai_provider: AI provider to use
    """
    try:
        with open(f"/tmp/process_debug_{correlation_id[:8]}.txt", "w") as f:
            f.write(f"process_code_request called at {time.time()}\n")
            f.write(f"correlation_id: {correlation_id}\n")
    except Exception as debug_error:
        print(f"DEBUG: Failed to write debug file: {debug_error}")
    
    telemetry.log_event(
        "Background task function called",
        correlation_id=correlation_id,
        repo_url=repo_url,
        prompt_length=len(prompt)
    )
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        telemetry.log_event(
            "Background task started",
            correlation_id=correlation_id,
            repo_url=repo_url
        )
        
        sandbox_service_instance = get_sandbox_service_instance()
        if not sandbox_service_instance:
            error_msg = "Sandbox service not available"
            telemetry.log_event(
                "Background task failed - sandbox service unavailable",
                correlation_id=correlation_id,
                level="error"
            )
            await streaming_service.send_error(
                correlation_id=correlation_id,
                error=error_msg,
                error_type="ServiceError"
            )
            return
            
        telemetry.log_event(
            "Sandbox service obtained successfully",
            correlation_id=correlation_id,
            sandbox_service_available=True
        )
        
    except Exception as e:
        telemetry.log_error(e, correlation_id=correlation_id)
        await streaming_service.send_error(
            correlation_id=correlation_id,
            error=f"Failed to initialize background task: {str(e)}",
            error_type="InitializationError"
        )
        return
    
    try:
        telemetry.log_event(
            "Starting sandbox creation",
            correlation_id=correlation_id
        )
        
        await streaming_service.send_ai_message(
            correlation_id=correlation_id,
            message="Creating secure sandbox environment..."
        )
        
        telemetry.log_event(
            "Sent first AI message via streaming",
            correlation_id=correlation_id
        )
        
        await streaming_service.send_progress(
            correlation_id=correlation_id,
            progress=10,
            step="Creating sandbox"
        )
        
        container_id = await sandbox_service_instance.create_sandbox(
            correlation_id=correlation_id
        )
        
        await streaming_service.send_tool_event(
            correlation_id=correlation_id,
            tool_type="bash",
            command=f"docker run sandbox_{correlation_id}",
            output=f"Container {container_id} created successfully"
        )
        
        await streaming_service.send_ai_message(
            correlation_id=correlation_id,
            message="Cloning repository..."
        )
        
        await streaming_service.send_progress(
            correlation_id=correlation_id,
            progress=20,
            step="Cloning repository"
        )
        
        repo_path = await get_git_service_instance().clone_repository(
            correlation_id=correlation_id,
            repo_url=repo_url,
            sandbox_service=sandbox_service_instance
        )
        
        await streaming_service.send_tool_event(
            correlation_id=correlation_id,
            tool_type="git",
            command=f"git clone {repo_url}",
            output=f"Repository cloned to {repo_path}"
        )
        
        await streaming_service.send_ai_message(
            correlation_id=correlation_id,
            message="Analyzing repository structure..."
        )
        
        await streaming_service.send_progress(
            correlation_id=correlation_id,
            progress=30,
            step="Analyzing repository"
        )
        
        repo_analysis = await get_agent_service_instance().analyze_repository(
            correlation_id=correlation_id,
            repo_path=repo_path,
            sandbox_service=sandbox_service_instance
        )
        
        await streaming_service.send_ai_message(
            correlation_id=correlation_id,
            message=f"Repository analysis complete. Found {len(repo_analysis.get('files', []))} files.",
            context={"analysis": repo_analysis}
        )
        
        await streaming_service.send_ai_message(
            correlation_id=correlation_id,
            message="Generating implementation plan..."
        )
        
        await streaming_service.send_progress(
            correlation_id=correlation_id,
            progress=40,
            step="Planning implementation"
        )
        
        implementation_plan = await get_agent_service_instance().create_implementation_plan(
            correlation_id=correlation_id,
            prompt=prompt,
            repo_analysis=repo_analysis,
            ai_provider=ai_provider
        )
        
        await streaming_service.send_ai_message(
            correlation_id=correlation_id,
            message=f"Implementation plan created: {implementation_plan['summary']}",
            context={"plan": implementation_plan}
        )
        
        await streaming_service.send_ai_message(
            correlation_id=correlation_id,
            message=f"Creating branch: {branch_name}"
        )
        
        await streaming_service.send_progress(
            correlation_id=correlation_id,
            progress=50,
            step="Creating branch"
        )
        
        await get_git_service_instance().create_branch(
            correlation_id=correlation_id,
            repo_path=repo_path,
            branch_name=branch_name,
            sandbox_service=sandbox_service_instance
        )
        
        await streaming_service.send_tool_event(
            correlation_id=correlation_id,
            tool_type="git",
            command=f"git checkout -b {branch_name}",
            output=f"Branch {branch_name} created"
        )
        
        await streaming_service.send_ai_message(
            correlation_id=correlation_id,
            message="Implementing code changes..."
        )
        
        await streaming_service.send_progress(
            correlation_id=correlation_id,
            progress=60,
            step="Implementing changes"
        )
        
        changes_made = await get_agent_service_instance().implement_changes(
            correlation_id=correlation_id,
            repo_path=repo_path,
            implementation_plan=implementation_plan,
            sandbox_service=sandbox_service_instance,
            repo_analysis=repo_analysis
        )
        
        for change in changes_made:
            await streaming_service.send_tool_event(
                correlation_id=correlation_id,
                tool_type="edit",
                filepath=change['filepath'],
                output=f"Modified {change['filepath']} - {change.get('description', 'File updated')}",
                context={
                    "action": change.get('action', 'modified'),
                    "old_content": change.get('old_content', ''),
                    "new_content": change.get('new_content', '')
                }
            )
        
        await streaming_service.send_ai_message(
            correlation_id=correlation_id,
            message=f"Code changes implemented. Modified {len(changes_made)} files.",
            context={"changes": changes_made}
        )
        
        await streaming_service.send_ai_message(
            correlation_id=correlation_id,
            message="Committing changes..."
        )
        
        await streaming_service.send_progress(
            correlation_id=correlation_id,
            progress=80,
            step="Committing changes"
        )
        
        commit_hash = await get_git_service_instance().commit_changes(
            correlation_id=correlation_id,
            repo_path=repo_path,
            message=f"feat: {prompt[:50]}...",
            sandbox_service=sandbox_service_instance
        )
        
        await streaming_service.send_tool_event(
            correlation_id=correlation_id,
            tool_type="git",
            command=f"git commit -m 'feat: {prompt[:50]}...'",
            output=f"Changes committed with hash: {commit_hash}"
        )
        
        await streaming_service.send_ai_message(
            correlation_id=correlation_id,
            message="Pushing changes to remote..."
        )
        
        await streaming_service.send_progress(
            correlation_id=correlation_id,
            progress=90,
            step="Pushing changes"
        )
        
        await get_git_service_instance().push_changes(
            correlation_id=correlation_id,
            repo_path=repo_path,
            branch_name=branch_name,
            sandbox_service=sandbox_service_instance
        )
        
        await streaming_service.send_tool_event(
            correlation_id=correlation_id,
            tool_type="git",
            command=f"git push origin {branch_name}",
            output=f"Changes pushed to remote branch {branch_name}"
        )
        
        await streaming_service.send_ai_message(
            correlation_id=correlation_id,
            message="Creating pull request..."
        )
        
        await streaming_service.send_progress(
            correlation_id=correlation_id,
            progress=95,
            step="Creating pull request"
        )
        
        pr_url = await get_git_service_instance().create_pull_request(
            correlation_id=correlation_id,
            repo_url=repo_url,
            branch_name=branch_name,
            title=f"feat: {prompt[:50]}...",
            body=f"## Changes Made\n\n{implementation_plan['summary']}\n\n## Files Modified\n\n" +
                 "\n".join([f"- {change['filepath']}" for change in changes_made]) +
                 f"\n\n## Prompt\n\n{prompt}"
        )
        
        await streaming_service.send_tool_event(
            correlation_id=correlation_id,
            tool_type="git",
            command=f"gh pr create --title 'feat: {prompt[:50]}...' --body '...'",
            output=f"Pull request created: {pr_url}"
        )
        
        duration = asyncio.get_event_loop().time() - start_time
        
        await streaming_service.send_progress(
            correlation_id=correlation_id,
            progress=100,
            step="Complete"
        )
        
        await streaming_service.send_success(
            correlation_id=correlation_id,
            message=f"Code changes completed successfully! Pull request created: {pr_url}",
            context={
                "pr_url": pr_url,
                "branch_name": branch_name,
                "commit_hash": commit_hash,
                "files_changed": [change['filepath'] for change in changes_made],
                "duration": duration
            }
        )
        
        telemetry.log_event(
            "Code request completed",
            correlation_id=correlation_id,
            duration=duration,
            pr_url=pr_url,
            files_changed=len(changes_made)
        )
        
        await streaming_service.close_stream(correlation_id)
        
    except Exception as e:
        telemetry.log_error(e, correlation_id=correlation_id)
        
        await streaming_service.send_error(
            correlation_id=correlation_id,
            error=str(e),
            error_type=type(e).__name__,
            context={"step": "processing"}
        )
        
        await streaming_service.close_stream(correlation_id)
        
    finally:
        try:
            await sandbox_service_instance.cleanup_sandbox(correlation_id)
        except Exception as cleanup_error:
            telemetry.log_error(
                cleanup_error,
                context={"phase": "cleanup"},
                correlation_id=correlation_id
            )


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