#!/usr/bin/env python3
"""
Backspace Coding Agent - LangGraph CLI Interface

A command-line tool for AI-powered code generation using the LangGraph agent.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Load .env file if it exists
def load_env():
    """Load environment variables from .env file."""
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Loaded environment from .env file")

# Load environment at startup
load_env()

# Import our core services
from app.core.config import settings
from app.core.telemetry import get_telemetry
from app.agents.coding_agent import create_coding_agent


class BackspaceLangGraphCLI:
    """CLI interface for the Backspace Coding Agent using LangGraph."""
    
    def __init__(self):
        self.telemetry = get_telemetry()
        self.agent = create_coding_agent()
        
    def print_status(self, message: str, level: str = "info"):
        """Print status message with appropriate formatting."""
        timestamp = time.strftime("%H:%M:%S")
        
        if level == "error":
            print(f"[{timestamp}] ❌ {message}")
        elif level == "warning":
            print(f"[{timestamp}] ⚠️  {message}")
        elif level == "success":
            print(f"[{timestamp}] ✅ {message}")
        elif level == "progress":
            print(f"[{timestamp}] 🔄 {message}")
        else:
            print(f"[{timestamp}] ℹ️  {message}")
    
    def print_progress(self, current: int, total: int, step: str):
        """Print progress bar."""
        percentage = int((current / total) * 100)
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f"[{bar}] {percentage}% - {step}")
    
    async def process_repository(
        self,
        repo_url: str,
        prompt: str,
        ai_provider: Optional[str] = None,
        output_format: str = "json",
        github_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a repository with the given prompt using LangGraph agent.
        
        Args:
            repo_url: GitHub repository URL
            prompt: Coding prompt/request
            ai_provider: AI provider to use (openai/anthropic)
            output_format: Output format (json/text)
            github_token: GitHub Personal Access Token (or set GITHUB_TOKEN env var)
            
        Returns:
            Dictionary with results
        """
        correlation_id = f"langgraph-cli-{int(time.time())}"
        
        try:
            self.print_status("Starting Backspace LangGraph Coding Agent...")
            self.print_progress(0, 5, "Initializing")
            
            # Step 1: Initialize agent
            self.print_status("Initializing agent...", "progress")
            self.print_progress(1, 5, "Agent initialized")
            
            # Step 2: Run the agent workflow
            self.print_status("Running agent workflow...", "progress")
            result = await self.agent.run(
                correlation_id=correlation_id,
                repo_url=repo_url,
                prompt=prompt,
                ai_provider=ai_provider or settings.ai_provider
            )
            self.print_progress(2, 5, "Workflow completed")
            
            # Step 3: Extract results
            self.print_status("Extracting results...", "progress")
            final_state = result.get("final_state", {})
            results = {
                "success": result.get("success", False),
                "correlation_id": correlation_id,
                "repo_url": repo_url,
                "prompt": prompt,
                "branch_name": final_state.get("branch_name", "unknown"),
                "commit_hash": final_state.get("commit_hash", "no_changes"),
                "push_success": final_state.get("push_success", False),
                "changes_made": final_state.get("changes_made", []),
                "implementation_result": final_state.get("implementation_result", {}),
                "plan": final_state.get("plan", {}),
                "repo_analysis": final_state.get("repo_analysis", {}),
                "steps_completed": final_state.get("steps_completed", []),
                "error": result.get("error"),
                "errors": result.get("errors", [])
            }
            self.print_progress(3, 5, "Results extracted")
            
            # Step 4: Cleanup
            self.print_status("Cleaning up...", "progress")
            await self.agent.cleanup()
            self.print_progress(4, 5, "Cleanup complete")
            
            # Step 5: Final status
            if results.get("push_success"):
                self.print_status("✅ Agent completed successfully!", "success")
            elif results.get("error"):
                self.print_status(f"❌ Agent failed: {results['error']}", "error")
            else:
                self.print_status("⚠️ Agent completed with warnings", "warning")
            
            self.print_progress(5, 5, "Complete")
            
            return results
            
        except Exception as e:
            self.print_status(f"Error: {str(e)}", "error")
            return {
                "success": False,
                "error": str(e),
                "correlation_id": correlation_id
            }
    
    def print_results(self, results: Dict[str, Any], output_format: str):
        """Print results in the specified format."""
        if output_format == "json":
            print(json.dumps(results, indent=2))
        else:
            print("\n" + "="*50)
            print("🎉 BACKSPACE LANGGRAPH CODING AGENT - COMPLETED")
            print("="*50)
            print(f"Repository: {results.get('repo_url', 'N/A')}")
            print(f"Prompt: {results.get('prompt', 'N/A')}")
            print(f"Branch: {results.get('branch_name', 'N/A')}")
            print(f"Commit: {results.get('commit_hash', 'N/A')}")
            print(f"Push Success: {results.get('push_success', False)}")
            
            if results.get("error"):
                print(f"\n❌ Error: {results['error']}")
            
            errors = results.get("errors", [])
            if errors:
                print(f"\n⚠️ Errors ({len(errors)}):")
                for error in errors[:3]:  # Show first 3 errors
                    print(f"  • {error.get('error', 'Unknown error')}")
            
            changes_made = results.get("changes_made", [])
            if changes_made:
                print(f"\n📝 Changes Made ({len(changes_made)} files):")
                for change in changes_made:
                    action = change.get("action", "unknown")
                    filepath = change.get("file_path", change.get("filepath", "unknown"))
                    description = change.get("description", "No description")
                    print(f"  • {filepath} ({action})")
                    print(f"    {description}")
            
            plan = results.get("plan", {})
            if plan:
                print(f"\n📋 Implementation Plan:")
                print(f"  Summary: {plan.get('summary', 'No summary')}")
                steps = plan.get("steps", [])
                if steps:
                    print("  Steps:")
                    for i, step in enumerate(steps, 1):
                        if isinstance(step, dict):
                            print(f"    {i}. {step.get('description', str(step))}")
                        else:
                            print(f"    {i}. {step}")
            
            repo_analysis = results.get("repo_analysis", {})
            if repo_analysis:
                print(f"\n📊 Repository Analysis:")
                print(f"  Files: {repo_analysis.get('file_count', 0)}")
                languages = repo_analysis.get("languages", [])
                if languages:
                    print(f"  Languages: {', '.join(languages)}")
            
            steps_completed = results.get("steps_completed", [])
            if steps_completed:
                print(f"\n🔄 Steps Completed:")
                for step in steps_completed:
                    print(f"  ✓ {step}")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Backspace Coding Agent - LangGraph CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_langgraph.py "https://github.com/user/repo.git" "Add a contact form"
  python cli_langgraph.py "https://github.com/user/repo.git" "Fix the login bug" --provider openai
  python cli_langgraph.py "https://github.com/user/repo.git" "Add unit tests" --output json
        """
    )
    
    parser.add_argument(
        "repo_url",
        help="GitHub repository URL"
    )
    
    parser.add_argument(
        "prompt",
        help="Coding prompt/request"
    )
    
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default=None,
        help="AI provider to use (default: from config)"
    )
    
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        "--token",
        help="GitHub Personal Access Token (or set GITHUB_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = BackspaceLangGraphCLI()
    
    # Process repository
    results = await cli.process_repository(
        repo_url=args.repo_url,
        prompt=args.prompt,
        ai_provider=args.provider,
        output_format=args.output,
        github_token=args.token
    )
    
    # Print results
    cli.print_results(results, args.output)
    
    # Exit with appropriate code
    sys.exit(0 if results.get("success", False) else 1)


if __name__ == "__main__":
    asyncio.run(main()) 