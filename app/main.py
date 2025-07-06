"""
Main FastAPI application for the Backspace Coding Agent.
"""

import asyncio
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.core.config import settings
from app.core.security import SecurityError, InputValidationError
from app.core.telemetry import get_telemetry
from app.models.schemas import ErrorResponse, HealthCheck
from app.services.sandbox import SandboxService
from app.api.endpoints import code


# Global service instances
sandbox_service = None
telemetry = get_telemetry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown logic.
    """
    # Startup
    telemetry.log_event("Application starting up")
    
    try:
        # Initialize services
        global sandbox_service
        sandbox_service = SandboxService()
        
        # Perform health checks
        if not await sandbox_service.health_check():
            telemetry.log_event("Sandbox service health check failed", level="warning")
        
        telemetry.log_event("Application started successfully")
        yield
        
    except Exception as e:
        telemetry.log_error(e, context={"phase": "startup"})
        raise
    
    finally:
        # Shutdown
        telemetry.log_event("Application shutting down")
        
        try:
            # Cleanup resources
            if sandbox_service:
                await sandbox_service.cleanup_all_sandboxes()
            
            telemetry.log_event("Application shutdown complete")
            
        except Exception as e:
            telemetry.log_error(e, context={"phase": "shutdown"})


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A sandboxed coding agent that creates pull requests from natural language prompts",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for request logging and correlation IDs
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests with correlation IDs."""
    from app.core.security import create_correlation_id
    
    correlation_id = create_correlation_id()
    
    # Add correlation ID to request state
    request.state.correlation_id = correlation_id
    
    # Log request
    telemetry.log_event(
        "Request received",
        correlation_id=correlation_id,
        method=request.method,
        url=str(request.url),
        user_agent=request.headers.get("user-agent", ""),
        client_ip=request.client.host if request.client else ""
    )
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        response = await call_next(request)
        
        # Log response
        duration = asyncio.get_event_loop().time() - start_time
        telemetry.log_performance(
            "Request completed",
            duration=duration,
            correlation_id=correlation_id,
            status_code=response.status_code,
            method=request.method,
            url=str(request.url)
        )
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response
        
    except Exception as e:
        # Log error
        duration = asyncio.get_event_loop().time() - start_time
        telemetry.log_error(
            e,
            context={
                "correlation_id": correlation_id,
                "method": request.method,
                "url": str(request.url),
                "duration": duration
            },
            correlation_id=correlation_id
        )
        raise


# Global exception handlers
@app.exception_handler(SecurityError)
async def security_error_handler(request: Request, exc: SecurityError):
    """Handle security errors."""
    correlation_id = getattr(request.state, "correlation_id", None)
    
    telemetry.log_error(
        exc,
        context={"url": str(request.url), "method": request.method},
        correlation_id=correlation_id
    )
    
    error_response = ErrorResponse(
        error=str(exc),
        error_type="SecurityError",
        correlation_id=correlation_id
    )
    
    return JSONResponse(
        status_code=403,
        content=error_response.model_dump(mode='json'),
        headers={"X-Correlation-ID": correlation_id} if correlation_id else {}
    )


@app.exception_handler(InputValidationError)
async def validation_error_handler(request: Request, exc: InputValidationError):
    """Handle input validation errors."""
    correlation_id = getattr(request.state, "correlation_id", None)
    
    telemetry.log_error(
        exc,
        context={"url": str(request.url), "method": request.method},
        correlation_id=correlation_id
    )
    
    error_response = ErrorResponse(
        error=str(exc),
        error_type="InputValidationError",
        correlation_id=correlation_id
    )
    
    return JSONResponse(
        status_code=422,
        content=error_response.model_dump(mode='json'),
        headers={"X-Correlation-ID": correlation_id} if correlation_id else {}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    correlation_id = getattr(request.state, "correlation_id", None)
    
    telemetry.log_event(
        "HTTP exception",
        level="warning",
        correlation_id=correlation_id,
        status_code=exc.status_code,
        detail=exc.detail,
        url=str(request.url),
        method=request.method
    )
    
    error_response = ErrorResponse(
        error=exc.detail,
        error_type="HTTPException",
        correlation_id=correlation_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(mode='json'),
        headers={"X-Correlation-ID": correlation_id} if correlation_id else {}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    correlation_id = getattr(request.state, "correlation_id", None)
    
    telemetry.log_error(
        exc,
        context={"url": str(request.url), "method": request.method},
        correlation_id=correlation_id
    )
    
    error_response = ErrorResponse(
        error="Internal server error",
        error_type="InternalError",
        correlation_id=correlation_id,
        context={"message": str(exc)} if settings.debug else None
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(mode='json'),
        headers={"X-Correlation-ID": correlation_id} if correlation_id else {}
    )


# Health check endpoint
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    services = {}
    
    try:
        # Check sandbox service
        if sandbox_service:
            services["sandbox"] = "healthy" if await sandbox_service.health_check() else "unhealthy"
        else:
            services["sandbox"] = "not_initialized"
        
        # Check Docker
        try:
            import docker
            client = docker.from_env()
            client.ping()
            services["docker"] = "healthy"
        except Exception:
            services["docker"] = "unhealthy"
        
        # Overall status
        overall_status = "healthy" if all(
            status in ["healthy", "not_initialized"] for status in services.values()
        ) else "unhealthy"
        
        return HealthCheck(
            status=overall_status,
            version=settings.app_version,
            services=services
        )
        
    except Exception as e:
        telemetry.log_error(e, context={"endpoint": "health_check"})
        return HealthCheck(
            status="unhealthy",
            version=settings.app_version,
            services={"error": str(e)}
        )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - serve the HTML client."""
    return FileResponse("static/index.html")


# Include routers after application is created
def include_routers():
    """Include API routers."""
    try:
        app.include_router(
            code.router,
            prefix="/api/v1",
            tags=["coding"]
        )
    except ImportError as e:
        telemetry.log_error(e, context={"phase": "router_inclusion"})


# Include routers
include_routers()


# Dependency injection
def get_sandbox_service():
    """Get the sandbox service instance."""
    global sandbox_service
    if sandbox_service is None:
        # Lazy initialization if not already initialized
        from app.services.sandbox import SandboxService
        sandbox_service = SandboxService()
        telemetry.log_event("Sandbox service lazy initialized", level="info")
    return sandbox_service


# Application factory
def create_app():
    """Create and configure the FastAPI application."""
    return app


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    ) 