"""
Concrete implementation of the Backspace Coding Agent using LangGraph.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langsmith import Client, traceable

from app.agents.base_agent import BaseAgent, AgentState
from app.agents.tools import create_toolkit
from app.services.sandbox import SandboxService
from app.services.git_service import GitService
from app.core.config import settings
from app.core.telemetry import get_telemetry
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)


class CodingAgent(BaseAgent):
    """Concrete implementation of the coding agent."""
    
    def __init__(self):
        super().__init__()
        
        self.sandbox_service = SandboxService()
        self.git_service = GitService()
        
        self.tools = create_toolkit(self.sandbox_service, self.git_service)
        
        self.llm = self._initialize_llm()
        
        self.system_prompt = self._create_system_prompt()
        self.planning_prompt = self._create_planning_prompt()
        self.implementation_prompt = self._create_implementation_prompt()
        
    def _initialize_llm(self):
        """Initialize the language model based on provider."""
        if settings.ai_provider == "openai":
            return ChatOpenAI(
                model=settings.openai_model,
                temperature=0.1,
                api_key=settings.openai_api_key
            )
        elif settings.ai_provider == "anthropic":
            return ChatAnthropic(
                model=settings.anthropic_model,
                temperature=0.1,
                api_key=settings.anthropic_api_key
            )
        else:
            raise ValueError(f"Unsupported AI provider: {settings.ai_provider}")
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the agent."""
        return """You are Backspace, an AI coding agent that helps developers implement code changes in repositories.

Your capabilities:
- Analyze repository structure and understand codebases
- Create detailed implementation plans
- Write, modify, and create code files
- Execute git operations (clone, branch, commit, push)
- Run commands in a secure sandbox environment

CRITICAL PRINCIPLES:
1. **PRESERVE EXISTING CODE**: Never replace entire files unless explicitly requested. Make incremental changes.
2. **FOLLOW EXISTING PATTERNS**: Use the same file extensions (.js vs .jsx) and coding patterns as the existing codebase.
3. **MINIMAL CHANGES**: Only modify what's necessary to implement the requested feature.
4. **READ BEFORE WRITING**: Always read existing files to understand the current structure before making changes.
5. **COMPONENT-BASED APPROACH**: For React projects, create new components and import them into existing files rather than rewriting entire files.
6. **NO DUPLICATES**: Never create multiple versions of the same file (e.g., both .js and .jsx versions).
7. **SMART INTEGRATION**: When modifying existing files, preserve all existing functionality and only add new features.

IMPLEMENTATION RULES:
- Analyze the repository first to understand existing file extensions and patterns
- If the repo uses .jsx files, create new components as .jsx
- If the repo uses .js files, create new components as .js
- When modifying existing files, read the current content first and preserve it
- Only create ONE version of each file - never both .js and .jsx
- For React components, integrate new components into existing App structure without removing existing content

Your workflow:
1. Analyze the repository to understand its structure and patterns
2. Create a detailed implementation plan
3. Read existing files to understand current implementation
4. Make minimal, targeted changes to implement the feature
5. Commit and push the changes

Always:
- Think step by step before making changes
- Consider the existing codebase structure and patterns
- Write clean, maintainable code
- Follow best practices for the language/framework
- Test your changes when possible
- Provide clear commit messages
- Preserve existing functionality unless explicitly asked to change it

Use the available tools to accomplish your tasks. Be thorough and methodical in your approach."""

    def _create_planning_prompt(self) -> ChatPromptTemplate:
        """Create the prompt for planning."""
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Analyze the repository and create a detailed implementation plan for the following request:

Repository URL: {repo_url}
Request: {prompt}

Repository Analysis:
{repo_analysis}

Create a detailed plan with:
1. Summary of what needs to be done
2. List of files that need to be created or modified (be specific about file extensions)
3. Step-by-step implementation approach
4. Any considerations or potential issues

IMPORTANT GUIDELINES:
- Identify existing file patterns (.js vs .jsx, component structure, etc.)
- Plan to preserve existing code and functionality
- For React projects, plan to create new components and import them
- Specify exact file paths and extensions to match existing patterns

Think through this carefully and provide a comprehensive plan."""),
            MessagesPlaceholder(variable_name="chat_history")
        ])
    
    def _create_implementation_prompt(self) -> ChatPromptTemplate:
        """Create the prompt for implementation."""
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Implement the planned changes for the request: {prompt}. Repository: {repo_url}. Plan: {plan}. Repository analysis: {repo_analysis}. CRITICAL: Read existing files first, preserve all existing code, only add new features incrementally."),
            MessagesPlaceholder(variable_name="chat_history")
        ])
    
    @traceable(name="analyze_repository")
    async def _analyze_repository_node(self, state: AgentState) -> AgentState:
        """Analyze the repository structure."""
        try:
            if not state.get("correlation_id"):
                raise ValueError("correlation_id missing in state at analyze_repository_node")
            self._log_node_start("analyze_repository", state)
            
            state["current_step"] = "analyze_repository"
            state["last_update"] = datetime.utcnow()

            await self.sandbox_service.create_sandbox(
                correlation_id=state["correlation_id"]
            )
            
            analyze_tool = next(t for t in self.tools if t.name == "analyze_repository")
            analysis = await analyze_tool.ainvoke({
                "correlation_id": state["correlation_id"],
                "repo_url": state["repo_url"]
            })
            
            state["repo_path"] = analysis["repo_path"]
            state["repo_analysis"] = analysis
            
            state["messages"].append(
                SystemMessage(content=f"Repository analyzed: {json.dumps(analysis, indent=2)}")
            )
            
            state["steps_completed"].append("analyze_repository")
            self._log_node_success("analyze_repository", state)
            
        except Exception as e:
            self.telemetry.log_error(
                e,
                context={"step": "analyze_repository", **state},
                correlation_id=state.get("correlation_id")
            )
            raise
            
        return state
    
    @traceable(name="create_plan")
    async def _create_plan_node(self, state: AgentState) -> AgentState:
        """Create an implementation plan."""
        try:
            self._log_node_start("create_plan", state)
            
            state["current_step"] = "create_plan"
            state["last_update"] = datetime.utcnow()
            
            prompt = self.planning_prompt.format_messages(
                repo_url=state["repo_url"],
                prompt=state["prompt"],
                repo_analysis=json.dumps(state["repo_analysis"], indent=2),
                chat_history=state["messages"]
            )
            
            response = await self.llm.ainvoke(prompt)
            
            plan = self._parse_plan(response.content)
            
            state["plan"] = plan
            state["messages"].append(response)
            
            state["steps_completed"].append("create_plan")
            self._log_node_success("create_plan", state)
            
        except Exception as e:
            state = await self._handle_node_error("create_plan", state, e)
            
        return state
    
    @traceable(name="plan_changes")
    async def _plan_changes_node(self, state: AgentState) -> AgentState:
        try:
            if not state.get("correlation_id"):
                raise ValueError("correlation_id missing in state at plan_changes_node")
            self._log_node_start("plan_changes", state)
            plan_tool = next(t for t in self.tools if t.name == "plan_changes")
            plan = await plan_tool.ainvoke({
                "correlation_id": state["correlation_id"],
                "repo_path": state["repo_path"],
                "repo_analysis": state["repo_analysis"],
                "goal": state["goal"]
            })
            state["plan"] = plan
            return state
        except Exception as e:
            self.telemetry.log_error(
                e,
                context={"step": "plan_changes", **state},
                correlation_id=state.get("correlation_id")
            )
            raise
    
    @traceable(name="implement_changes")
    async def _implement_changes_node(self, state: AgentState) -> AgentState:
        """Implement the planned changes using available tools."""
        try:
            if not state.get("correlation_id"):
                raise ValueError("correlation_id missing in state at implement_changes_node")
            self._log_node_start("implement_changes", state)
            
            state["current_step"] = "implement_changes"
            state["last_update"] = datetime.utcnow()
            
            # Create branch first
            create_branch_tool = next(t for t in self.tools if t.name == "create_branch")
            
            # Create branch name
            branch_prompt = f"""Based on the following task description, generate a concise and descriptive branch name that follows git branch naming conventions.

Task: {state['prompt']}

Requirements:
- Use kebab-case (lowercase with hyphens)
- Be descriptive but concise (max 50 characters)
- Start with a type prefix like 'feature/', 'fix/', 'add/', etc.
- Avoid special characters except hyphens
- Make it clear what the branch is for

Examples:
- "Add contact form" → "feature/add-contact-form"
- "Fix navigation bug" → "fix/navigation-bug"
- "Update styling" → "feature/update-styling"

Branch name:"""
            
            branch_response = await self.llm.ainvoke(branch_prompt)
            base_branch_name = branch_response.content.strip()
            
            import re
            base_branch_name = re.sub(r'[^a-zA-Z0-9\-/]', '', base_branch_name)
            base_branch_name = base_branch_name.lower()
            
            if not any(base_branch_name.startswith(prefix) for prefix in ['feature/', 'fix/', 'add/', 'update/', 'improve/']):
                base_branch_name = f"feature/{base_branch_name}"
            
            branch_name = f"{base_branch_name}-{int(time.time())}"
            
            await create_branch_tool.ainvoke({
                "correlation_id": state["correlation_id"],
                "repo_path": state["repo_path"],
                "branch_name": branch_name
            })
            
            state["branch_name"] = branch_name
            
            # Create a simple prompt for the LLM to use tools directly
            implementation_prompt = f"""You are implementing the changes for: {state['prompt']}

Repository: {state['repo_url']}
Repository Path: {state['repo_path']}
Plan: {json.dumps(state['plan'], indent=2)}

TASK: Create the files and make the changes as described in the plan.

CRITICAL INSTRUCTIONS:
1. ALWAYS read existing files before modifying them using read_file
2. Create new files using write_file
3. When modifying existing files, preserve ALL existing content and only add new features
4. Follow the existing patterns in the codebase (.jsx extensions, Tailwind CSS, etc.)
5. **COMPLETE THE FULL INTEGRATION** - Don't just create components, integrate them into the app

IMPORTANT: When adding components (like header, footer, navbar, etc.), you MUST:
1. Create the new component file
2. Read the main App.jsx file to understand its structure
3. Modify App.jsx to import and use the new component
4. Ensure the component appears in the correct location (header at top, footer at bottom, etc.)

For this specific task:
- If creating a header: Add it at the top of the App component
- If creating a footer: Add it at the bottom of the App component  
- If creating other components: Add them in the appropriate location
- Always import the component at the top of App.jsx

You must use the tools to complete BOTH steps:
1. CREATE the component
2. INTEGRATE it into App.jsx

Do not stop after just creating the component file. Complete the full integration.

Start by reading App.jsx, then create the component, then modify App.jsx to include it."""

            # Use the LLM with tool calling in a conversation loop
            messages = [
                SystemMessage(content="You are a coding agent that implements changes by using tools. You have access to read_file, write_file, and execute_command tools. Use them to implement the requested changes."),
                HumanMessage(content=implementation_prompt)
            ]
            
            print(f"\n===== BACKSPACE DEBUG: Starting Implementation Loop =====")
            
            changes_made = []
            max_iterations = 10  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                print(f"\n===== ITERATION {iteration} =====")
                
                # Get the LLM response with tool calling
                response = await self.llm.bind_tools(self.tools).ainvoke(messages)
                
                print(f"Response: {response}")
                print(f"Tool calls: {response.tool_calls if hasattr(response, 'tool_calls') else 'No tool calls'}")
                
                # Add the response to messages
                messages.append(response)
                
                # Check if LLM made tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    tool_results = []
                    
                    # Execute each tool call
                    for tool_call in response.tool_calls:
                        print(f"\n===== EXECUTING TOOL: {tool_call['name']} =====")
                        print(f"Args: {tool_call['args']}")
                        
                        # Find the tool
                        tool = next((t for t in self.tools if t.name == tool_call['name']), None)
                        if tool:
                            try:
                                # Add correlation_id and repo_path to args if not present
                                tool_args = tool_call['args'].copy()
                                if 'correlation_id' not in tool_args:
                                    tool_args['correlation_id'] = state["correlation_id"]
                                else:
                                    # Override the correlation_id from LLM with the correct one
                                    tool_args['correlation_id'] = state["correlation_id"]
                                if 'repo_path' not in tool_args and tool_call['name'] in ['read_file', 'write_file']:
                                    tool_args['repo_path'] = state["repo_path"]
                                
                                result = await tool.ainvoke(tool_args)
                                print(f"Tool result: {result}")
                                
                                # Track changes
                                if tool_call['name'] == 'write_file':
                                    changes_made.append({
                                        "action": "created/modified",
                                        "file_path": tool_args.get('file_path', 'unknown'),
                                        "description": f"File {tool_args.get('file_path', 'unknown')} written"
                                    })
                                
                                # Create tool result message
                                tool_results.append({
                                    "tool_call_id": tool_call['id'],
                                    "name": tool_call['name'],
                                    "content": str(result)
                                })
                                
                            except Exception as e:
                                print(f"Error executing tool {tool_call['name']}: {e}")
                                tool_results.append({
                                    "tool_call_id": tool_call['id'],
                                    "name": tool_call['name'],
                                    "content": f"Error: {str(e)}"
                                })
                        else:
                            print(f"Tool {tool_call['name']} not found")
                    
                    # Add tool results to messages
                    from langchain_core.messages import ToolMessage
                    for tool_result in tool_results:
                        messages.append(ToolMessage(
                            content=tool_result["content"],
                            tool_call_id=tool_result["tool_call_id"]
                        ))
                    
                    # Ask the LLM if it wants to continue
                    continue_prompt = "Based on the tool results above, do you need to make any more changes to complete the task? If yes, continue using the tools. If the task is complete, respond with 'TASK COMPLETE' and summarize what was accomplished."
                    messages.append(HumanMessage(content=continue_prompt))
                    
                else:
                    # No tool calls made, check if task is complete
                    if "TASK COMPLETE" in response.content or "complete" in response.content.lower():
                        print("Task marked as complete by LLM")
                        break
                    else:
                        print("No tool calls made and task not marked complete")
                        break
            
            print(f"\n===== IMPLEMENTATION LOOP COMPLETED AFTER {iteration} ITERATIONS =====")
            print(f"Changes made: {changes_made}")
            
            state["implementation_result"] = {"output": f"Completed after {iteration} iterations", "tool_calls": []}
            state["changes_made"] = changes_made
            state["messages"].extend(messages)
            
            state["steps_completed"].append("implement_changes")
            self._log_node_success("implement_changes", state)
            
        except Exception as e:
            self.telemetry.log_error(
                e,
                context={"step": "implement_changes", **state},
                correlation_id=state.get("correlation_id")
            )
            raise
            
        return state
    
    @traceable(name="commit_changes")
    async def _commit_changes_node(self, state: AgentState) -> AgentState:
        try:
            if not state.get("correlation_id"):
                raise ValueError("correlation_id missing in state at commit_changes_node")
            self._log_node_start("commit_changes", state)
            
            state["current_step"] = "commit_changes"
            state["last_update"] = datetime.utcnow()
            
            # Create a better commit message based on the actual changes
            changes_summary = []
            if state.get("changes_made"):
                for change in state["changes_made"]:
                    action = change.get("action", "modified")
                    file_path = change.get("file_path", "unknown")
                    if action == "created":
                        changes_summary.append(f"Add {file_path}")
                    elif action == "modified":
                        changes_summary.append(f"Update {file_path}")
            
            if changes_summary:
                # Use the first change as the main message, limit to 50 chars
                commit_message = changes_summary[0]
                if len(changes_summary) > 1:
                    commit_message += f" and {len(changes_summary) - 1} more files"
                commit_message = commit_message[:50]
            else:
                # Fallback to prompt-based message
                prompt_summary = state.get("prompt", "")[:30]
                commit_message = f"feat: {prompt_summary}"
            
            # Clean up the commit message
            commit_message = commit_message.replace('"', '').replace("'", "").replace('`', '').strip()
            
            # Ensure it's not empty or just punctuation
            if not commit_message or commit_message in ['```', '`', '"', "'"]:
                commit_message = "feat: implement requested changes"
            
            commit_tool = next(t for t in self.tools if t.name == "commit_changes")
            commit_result = await commit_tool.ainvoke({
                "correlation_id": state["correlation_id"],
                "repo_path": state["repo_path"],
                "message": commit_message
            })
            state["commit_result"] = commit_result
            state["commit_hash"] = commit_result.get("commit_hash")
            
            state["steps_completed"].append("commit_changes")
            self._log_node_success("commit_changes", state)
            
        except Exception as e:
            self.telemetry.log_error(
                e,
                context={"step": "commit_changes", **state},
                correlation_id=state.get("correlation_id")
            )
            raise
            
        return state
    
    @traceable(name="push_changes")
    async def _push_changes_node(self, state: AgentState) -> AgentState:
        try:
            if not state.get("correlation_id"):
                raise ValueError("correlation_id missing in state at push_changes_node")
            self._log_node_start("push_changes", state)
            
            state["current_step"] = "push_changes"
            state["last_update"] = datetime.utcnow()
            
            push_tool = next(t for t in self.tools if t.name == "push_changes")
            push_result = await push_tool.ainvoke({
                "correlation_id": state["correlation_id"],
                "repo_path": state["repo_path"],
                "branch_name": state["branch_name"]
            })
            state["push_result"] = push_result
            state["push_success"] = True
            
            state["steps_completed"].append("push_changes")
            self._log_node_success("push_changes", state)
            
        except Exception as e:
            self.telemetry.log_error(
                e,
                context={"step": "push_changes", **state},
                correlation_id=state.get("correlation_id")
            )
            raise
            
        return state
    
    @traceable(name="create_pull_request")
    async def _create_pull_request_node(self, state: AgentState) -> AgentState:
        try:
            if not state.get("correlation_id"):
                raise ValueError("correlation_id missing in state at create_pull_request_node")
            self._log_node_start("create_pull_request", state)
            
            state["current_step"] = "create_pull_request"
            state["last_update"] = datetime.utcnow()
            
            # Create PR title and description
            changes_summary = []
            if state.get("changes_made"):
                for change in state["changes_made"]:
                    action = change.get("action", "modified")
                    file_path = change.get("file_path", "unknown")
                    if action == "created":
                        changes_summary.append(f"- Add {file_path}")
                    elif action == "modified":
                        changes_summary.append(f"- Update {file_path}")
            
            # Create PR title from prompt
            prompt = state.get("prompt", "Implement changes")
            pr_title = prompt[:50]  # GitHub PR title limit
            
            # Create PR description
            pr_body = f"""## Summary
{prompt}

## Changes Made
{chr(10).join(changes_summary) if changes_summary else "- Various improvements"}

## Implementation Plan
{state.get('plan', {}).get('summary', 'No plan summary available')}

---
*This pull request was automatically created by Backspace AI Coding Agent*
"""
            
            # Try to create the PR using GitService
            try:
                from app.services.git_service import GitService
                git_service = GitService()
                
                pr_url = await git_service.create_pull_request(
                    correlation_id=state["correlation_id"],
                    repo_url=state["repo_url"],
                    branch_name=state["branch_name"],
                    title=pr_title,
                    body=pr_body
                )
                
                state["pull_request_url"] = pr_url
                state["pull_request_created"] = True
                
                self.telemetry.log_event(
                    "Pull request created successfully",
                    correlation_id=state["correlation_id"],
                    pr_url=pr_url,
                    title=pr_title
                )
                
            except Exception as pr_error:
                # Log the error but don't fail the whole workflow
                self.telemetry.log_error(
                    pr_error,
                    context={"step": "create_pull_request", **state},
                    correlation_id=state.get("correlation_id")
                )
                state["pull_request_created"] = False
                state["pull_request_error"] = str(pr_error)
            
            state["steps_completed"].append("create_pull_request")
            self._log_node_success("create_pull_request", state)
            
        except Exception as e:
            self.telemetry.log_error(
                e,
                context={"step": "create_pull_request", **state},
                correlation_id=state.get("correlation_id")
            )
            # Don't raise here - PR creation failure shouldn't fail the whole workflow
            state["pull_request_created"] = False
            state["pull_request_error"] = str(e)
            
        return state
    
    def _parse_plan(self, content: str) -> Dict[str, Any]:
        """Parse the plan from the LLM response."""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Fallback: create a simple plan structure
            return {
                "summary": content[:200] + "..." if len(content) > 200 else content,
                "steps": content.split('\n'),
                "files_to_modify": [],
                "files_to_create": []
            }
        except Exception as e:
            logger.warning(f"Failed to parse plan: {e}")
            return {
                "summary": content,
                "steps": [content],
                "files_to_modify": [],
                "files_to_create": []
            }
    
    def _parse_implementation(self, content: str) -> Dict[str, Any]:
        """Parse the implementation from the LLM response and extract actual file changes."""
        try:
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            file_changes = []
            valid_extensions = ('.js', '.jsx', '.ts', '.tsx', '.css', '.html', '.json', '.md', '.txt', '.cjs', '.mjs', '.yml', '.yaml')
            
            # Track processed files to prevent duplicates
            processed_files = set()
            
            def is_valid_file_path(path: str) -> bool:
                path = path.strip()
                # Remove backticks and other formatting characters
                path = path.replace('`', '').replace('"', '').replace("'", "")
                return (
                    path.endswith(valid_extensions)
                    and not any(c in path for c in [' ', '\n', '\r'])
                    and not path.startswith('###')
                    and '/' in path
                )
            
            def clean_file_path(path: str) -> str:
                """Clean the file path by removing formatting characters."""
                # Remove backticks, quotes, and other formatting
                cleaned = path.replace('`', '').replace('"', '').replace("'", "").strip()
                
                # Remove leading/trailing slashes and dots
                if cleaned.startswith('./'):
                    cleaned = cleaned[2:]
                elif cleaned.startswith('/'):
                    cleaned = cleaned[1:]
                
                # Remove workspace prefixes dynamically (don't hardcode specific paths)
                # Look for patterns like workspace/repo-name/ or /workspace/repo-name/
                import re
                workspace_patterns = [
                    r'^workspace/[^/]+/',  # workspace/repo-name/
                    r'^/workspace/[^/]+/',  # /workspace/repo-name/
                    r'^workspace/',  # workspace/
                    r'^/workspace/',  # /workspace/
                ]
                
                for pattern in workspace_patterns:
                    cleaned = re.sub(pattern, '', cleaned)
                
                return cleaned
            
            def clean_code_content(content: str) -> str:
                """Clean the code content by removing instruction text and markdown."""
                # Remove common instruction patterns
                patterns_to_remove = [
                    r'// Create file.*?with the following content:\s*\n*',
                    r'// Modify.*?with the following content:\s*\n*',
                    r'Create file.*?with the following content:\s*\n*',
                    r'Modify.*?with the following content:\s*\n*',
                    r'```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n*',
                    r'```\s*\n*',
                    # Remove instruction lines that start with //
                    r'^//.*?file.*?`[^`]+`.*?\n*',
                    r'^//.*?content.*?\n*'
                ]
                
                cleaned = content
                for pattern in patterns_to_remove:
                    cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
                
                # Remove leading/trailing whitespace
                cleaned = cleaned.strip()
                
                return cleaned
            
            def add_file_change(action: str, file_path: str, file_content: str, description: str = None):
                """Add a file change, avoiding duplicates."""
                if not is_valid_file_path(file_path):
                    return
                
                file_path = clean_file_path(file_path)
                
                # Skip if we've already processed this file
                if file_path in processed_files:
                    return
                
                file_content = clean_code_content(file_content)
                
                if file_content:  # Only add if we have actual content
                    processed_files.add(file_path)
                    file_changes.append({
                        "action": action,
                        "file_path": file_path,
                        "content": file_content,
                        "description": description or f"{action.capitalize()} {file_path} with provided content"
                    })
            
            # Simplified patterns to avoid overlaps - prioritize more specific patterns first
            all_patterns = [
                # Create patterns (most specific first)
                (r'create.*?file.*?`([^`]+)`.*?```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n(.*?)```', 'create'),
                (r'new.*?file.*?`([^`]+)`.*?```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n(.*?)```', 'create'),
                # Modify patterns
                (r'modify.*?`([^`]+)`.*?```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n(.*?)```', 'modify'),
                (r'update.*?`([^`]+)`.*?```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n(.*?)```', 'modify'),
                # Generic patterns (less specific)
                (r'`([^`]+\.(?:js|jsx|ts|tsx|css|html|json|md|txt|cjs|mjs|yml|yaml))`.*?```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n(.*?)```', 'create'),
            ]
            
            for pattern, action in all_patterns:
                matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    file_path = match.group(1).strip()
                    file_content = match.group(2).strip()
                    add_file_change(action, file_path, file_content)
            
            if not file_changes:
                return {
                    "file_changes": [],
                    "description": "No actionable file changes found in the model output.",
                    "success": False,
                    "error": "No valid file changes detected. Please check the LLM output or prompt for clarification."
                }
            
            return {
                "file_changes": file_changes,
                "description": f"Implementation completed with {len(file_changes)} file changes",
                "success": True
            }
        except Exception as e:
            logger.warning(f"Failed to parse implementation: {e}")
            return {
                "file_changes": [],
                "description": f"Failed to parse implementation: {e}",
                "success": False,
                "error": str(e)
            }
    
    def _parse_implementation_with_incremental_support(self, content: str, existing_files_content: Dict[str, str]) -> Dict[str, Any]:
        """Parse implementation with support for incremental changes."""
        try:
            import re
            
            # First try the original parsing method
            original_result = self._parse_implementation(content)
            
            # If original parsing found changes, check if any are modifications to existing files
            if original_result.get("file_changes"):
                # Fix the action types based on existing files
                for change in original_result["file_changes"]:
                    file_path = change["file_path"]
                    # If the file exists in our existing_files_content, it should be a modify action
                    if file_path in existing_files_content:
                        print(f"CORRECTING: {file_path} should be 'modify' not '{change['action']}'")
                        change["action"] = "modify"
                        # Add a flag to indicate we need smart integration
                        change["needs_smart_integration"] = True
                
                return original_result
            
            # Try to parse the actual format the LLM is providing
            file_changes = []
            
            # Pattern for files with comments like "// File: path/to/file.jsx"
            file_comment_pattern = r'// File:\s*([^\n]+)\s*\n\n(.*?)```'
            
            # Pattern for standard code blocks with file paths
            code_block_pattern = r'```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n(.*?)```'
            
            # First, find all file comments and their associated code blocks
            file_matches = re.finditer(file_comment_pattern, content, re.DOTALL)
            for match in file_matches:
                file_path = match.group(1).strip()
                remaining_content = match.group(2)
                
                # Clean the file path
                file_path = file_path.replace('workspace/Personal-Website/', '').replace('`', '').strip()
                
                # Find the code block immediately after
                code_match = re.search(code_block_pattern, remaining_content, re.DOTALL)
                if code_match:
                    file_content = code_match.group(1).strip()
                    
                    # Determine if this should be create or modify
                    action = "modify" if file_path in existing_files_content else "create"
                    
                    file_changes.append({
                        "action": action,
                        "file_path": file_path,
                        "content": file_content,
                        "description": f"{'Modify' if action == 'modify' else 'Create'} {file_path}",
                        "needs_smart_integration": action == "modify"
                    })
            
            # Also try to find modification patterns
            modify_patterns = [
                r'### Step 2: Modify.*?App.*?Component.*?```jsx\s*\n(.*?)```',
                r'modify.*?App\.jsx.*?```jsx\s*\n(.*?)```',
                r'update.*?App\.jsx.*?```jsx\s*\n(.*?)```'
            ]
            
            for pattern in modify_patterns:
                modify_matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
                for match in modify_matches:
                    file_content = match.group(1).strip()
                    
                    file_changes.append({
                        "action": "modify",
                        "file_path": "src/App.jsx",
                        "content": file_content,
                        "description": "Modify App.jsx to include Footer component",
                        "needs_smart_integration": True
                    })
            
            if file_changes:
                return {
                    "file_changes": file_changes,
                    "description": f"Implementation with {len(file_changes)} file changes",
                    "success": True
                }
            else:
                # Fallback: try even more flexible patterns
                # Look for any code blocks that might be files
                all_code_blocks = re.finditer(r'```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n(.*?)```', content, re.DOTALL)
                
                for i, match in enumerate(all_code_blocks):
                    file_content = match.group(1).strip()
                    
                    # Try to infer the file path from context
                    # Look for file mentions in the preceding text
                    preceding_text = content[:match.start()]
                    
                    # Look for footer component creation
                    if 'footer' in preceding_text.lower() and 'component' in preceding_text.lower():
                        file_changes.append({
                            "action": "create",
                            "file_path": "src/components/Footer.jsx",
                            "content": file_content,
                            "description": "Create Footer component",
                            "needs_smart_integration": False
                        })
                    # Look for App.jsx modification
                    elif 'app' in preceding_text.lower() and ('modify' in preceding_text.lower() or 'import' in file_content.lower()):
                        file_changes.append({
                            "action": "modify",
                            "file_path": "src/App.jsx",
                            "content": file_content,
                            "description": "Modify App.jsx to include Footer component",
                            "needs_smart_integration": True
                        })
                
                if file_changes:
                    return {
                        "file_changes": file_changes,
                        "description": f"Implementation with {len(file_changes)} inferred file changes",
                        "success": True
                    }
                else:
                    return {
                        "file_changes": [],
                        "description": "No file changes detected in any format",
                        "success": False,
                        "error": "Could not parse any file changes from LLM response"
                    }
                
        except Exception as e:
            logger.warning(f"Failed to parse incremental implementation: {e}")
            return {
                "file_changes": [],
                "description": f"Failed to parse incremental implementation: {e}",
                "success": False,
                "error": str(e)
            }
    
    def _apply_incremental_changes(self, existing_content: str, modifications: List[Dict[str, Any]], fallback_content: str) -> str:
        """Apply incremental changes to existing content."""
        try:
            if not existing_content:
                return fallback_content
            
            modified_content = existing_content
            
            for mod in modifications:
                if mod["type"] == "add":
                    content_to_add = mod["content"]
                    
                    # If it's an import, add it at the top
                    if "import" in content_to_add.lower():
                        lines = modified_content.split('\n')
                        # Find the last import line
                        last_import_idx = -1
                        for i, line in enumerate(lines):
                            if line.strip().startswith('import'):
                                last_import_idx = i
                        
                        if last_import_idx >= 0:
                            lines.insert(last_import_idx + 1, content_to_add)
                        else:
                            lines.insert(0, content_to_add)
                        
                        modified_content = '\n'.join(lines)
                    
                    # If it's a component, add it before the last closing tag
                    elif '</' in content_to_add and '>' in content_to_add:
                        # Find the last closing div or main tag
                        import re
                        closing_patterns = [r'(\s*</div>\s*</div>\s*$)', r'(\s*</main>\s*</div>\s*$)', r'(\s*</div>\s*$)']
                        
                        for pattern in closing_patterns:
                            match = re.search(pattern, modified_content, re.MULTILINE)
                            if match:
                                insertion_point = match.start()
                                modified_content = (
                                    modified_content[:insertion_point] + 
                                    f"        {content_to_add}\n" + 
                                    modified_content[insertion_point:]
                                )
                                break
                        else:
                            # Fallback: add at the end
                            modified_content += f"\n{content_to_add}"
            
            return modified_content
            
        except Exception as e:
            logger.warning(f"Failed to apply incremental changes: {e}")
            return fallback_content if fallback_content else existing_content
    
    def _extract_changes_from_agent_result(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract changes from the agent execution result."""
        changes = []
        
        # Extract from intermediate steps if available
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                action, observation = step
                
                if hasattr(action, 'tool') and action.tool == "write_file":
                    # Extract file path from tool input
                    tool_input = action.tool_input
                    file_path = tool_input.get("file_path", "unknown")
                    
                    changes.append({
                        "action": "created/modified",
                        "file_path": file_path,
                        "description": f"File {file_path} written by agent"
                    })
        
        # If no intermediate steps, try to infer from output
        if not changes and "output" in result:
            output = result["output"]
            if "created" in output.lower() or "modified" in output.lower():
                changes.append({
                    "action": "implemented",
                    "file_path": "multiple files",
                    "description": "Changes implemented by agent"
                })
        
        return changes
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            pass
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def create_coding_agent() -> CodingAgent:
    """Create a new coding agent instance."""
    return CodingAgent() 