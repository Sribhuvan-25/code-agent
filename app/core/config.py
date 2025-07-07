"""
Configuration settings for the Backspace Coding Agent.
"""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    app_name: str = Field(default="Backspace Coding Agent", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    
    # GitHub Configuration
    github_token: Optional[str] = Field(default=None, description="GitHub Personal Access Token")
    github_api_url: str = Field(default="https://api.github.com", description="GitHub API URL")
    
    # AI Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model to use")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API Key")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", description="Anthropic model to use")
    default_ai_provider: str = Field(default="openai", description="Default AI provider")
    
    # Sandbox Configuration
    sandbox_timeout: int = Field(default=300, description="Sandbox timeout in seconds")
    max_concurrent_jobs: int = Field(default=5, description="Maximum concurrent sandbox jobs")
    docker_image: str = Field(default="python:3.11", description="Default Docker image")
    max_repo_size_mb: int = Field(default=100, description="Maximum repository size in MB")
    
    # Security Configuration
    allowed_hosts: list[str] = Field(default=["*"], description="Allowed hosts")
    cors_origins: list[str] = Field(default=["*"], description="CORS origins")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format")
    
    # Observability Configuration
    jaeger_endpoint: Optional[str] = Field(default=None, description="Jaeger endpoint")
    enable_tracing: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=10, description="Rate limit requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # LangSmith Configuration
    langsmith_api_key: Optional[str] = Field(default=None, description="LangSmith API Key")
    langsmith_project: str = Field(default="backspace-coding-agent", description="LangSmith project name")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", description="LangSmith API endpoint")
    
    # AI Provider Configuration
    ai_provider: str = "openai"  # "openai" or "anthropic"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings 