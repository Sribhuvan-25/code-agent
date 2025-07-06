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
    
    def __init__(self, streaming_service=None):
        super().__init__()
        
        self.streaming_service = streaming_service
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
    
    async def _send_streaming_update(self, correlation_id: str, message: str, progress: int = None, step: str = None):
        """Send a streaming update using the streaming service if available."""
        if self.streaming_service:
            try:
                # Use the appropriate streaming service method
                if progress is not None:
                    # Send progress update
                    await self.streaming_service.send_progress(
                        correlation_id=correlation_id,
                        progress=progress,
                        step=step or message
                    )
                else:
                    # Send AI message
                    await self.streaming_service.send_ai_message(
                        correlation_id=correlation_id,
                        message=message
                    )
            except Exception as e:
                # Fallback to logging if streaming fails
                self.telemetry.log_error(
                    e,
                    context={"streaming_update": message, "correlation_id": correlation_id},
                    correlation_id=correlation_id
                )
        else:
            # Log the streaming update for now
            self.telemetry.log_event(
                f"Streaming update: {message}",
                correlation_id=correlation_id,
                progress=progress,
                step=step
            )
        
        # Always log for debugging
        logger.info(f"[{correlation_id}] {message} (progress: {progress}%, step: {step})")
    
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
            
            # Send initial streaming update
            await self._send_streaming_update(
                state["correlation_id"], 
                "🔍 Analyzing repository structure...", 
                progress=10, 
                step="Analyzing Repository"
            )
            
            state["current_step"] = "analyze_repository"
            state["last_update"] = datetime.utcnow()

            await self.sandbox_service.create_sandbox(
                correlation_id=state["correlation_id"]
            )
            
            await self._send_streaming_update(
                state["correlation_id"], 
                "📥 Cloning repository...", 
                progress=20, 
                step="Cloning Repository"
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
            
            await self._send_streaming_update(
                state["correlation_id"], 
                "✅ Repository analysis complete", 
                progress=30, 
                step="Repository Analysis Complete"
            )
            
            state["steps_completed"].append("analyze_repository")
            self._log_node_success("analyze_repository", state)
            
        except Exception as e:
            await self._send_streaming_update(
                state["correlation_id"], 
                f"❌ Repository analysis failed: {str(e)}"
            )
            
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
            
            # Send initial streaming update
            await self._send_streaming_update(
                state["correlation_id"], 
                "🧠 Creating implementation plan...", 
                progress=35, 
                step="Creating Implementation Plan"
            )
            
            state["current_step"] = "create_plan"
            state["last_update"] = datetime.utcnow()
            
            await self._send_streaming_update(
                state["correlation_id"], 
                "📋 Analyzing requirements and planning approach...", 
                progress=40, 
                step="Analyzing Requirements"
            )
            
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
            
            await self._send_streaming_update(
                state["correlation_id"], 
                "✅ Implementation plan created", 
                progress=45, 
                step="Implementation Plan Complete"
            )
            
            state["steps_completed"].append("create_plan")
            self._log_node_success("create_plan", state)
            
        except Exception as e:
            await self._send_streaming_update(
                state["correlation_id"], 
                f"❌ Planning failed: {str(e)}"
            )
            
            state = await self._handle_node_error("create_plan", state, e)
            
        return state
    
    @traceable(name="implement_changes")
    async def _implement_changes_node(self, state: AgentState) -> AgentState:
        """Implement the planned changes using available tools."""
        try:
            if not state.get("correlation_id"):
                raise ValueError("correlation_id missing in state at implement_changes_node")
            self._log_node_start("implement_changes", state)
            
            # Send initial streaming update
            await self._send_streaming_update(
                state["correlation_id"], 
                "⚒️ Implementing changes...", 
                progress=50, 
                step="Implementing Changes"
            )
            
            state["current_step"] = "implement_changes"
            state["last_update"] = datetime.utcnow()
            
            # Create branch first
            await self._send_streaming_update(
                state["correlation_id"], 
                "🌿 Creating feature branch...", 
                progress=52, 
                step="Creating Branch"
            )
            
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
            
            await self._send_streaming_update(
                state["correlation_id"], 
                f"✅ Branch created: {branch_name}", 
                progress=55, 
                step="Branch Created"
            )
            
            await self._send_streaming_update(
                state["correlation_id"], 
                "📝 Writing code changes...", 
                progress=60, 
                step="Writing Code"
            )
            
            # Create a simple prompt for the LLM to use tools directly
            implementation_prompt = f"""You are implementing the changes for: {state['prompt']}

Repository: {state['repo_url']}
Repository Path: {state['repo_path']}
Plan: {json.dumps(state['plan'], indent=2)}

🎯 TASK: Create the files and make the changes as described in the plan.

⚠️ CRITICAL INSTRUCTIONS - FOLLOW THESE EXACTLY:
1. ALWAYS read existing files before modifying them using read_file
2. Create new files using write_file
3. When modifying existing files, preserve ALL existing content and only add new features
4. Follow the existing patterns in the codebase (.jsx extensions, Tailwind CSS, etc.)
5. **MANDATORY: COMPLETE THE FULL INTEGRATION** - Don't just create components, integrate them into the app

🔥 INTEGRATION REQUIREMENTS (NON-NEGOTIABLE):
When adding ANY new component (button, header, footer, navbar, etc.), you MUST complete BOTH steps:

✅ STEP 1: Create the component file
✅ STEP 2: Integrate it into the main application

Integration means:
1. Read the main application file (App.jsx, App.js, main.py, index.js, etc.)
2. Modify the main application file to:
   - Import the new component at the top
   - Add the component to the JSX/template/code
   - Position it correctly (header at top, footer at bottom, etc.)

⛔ FAILURE TO INTEGRATE = INCOMPLETE TASK
Creating a component without integrating it into the app is USELESS and INCOMPLETE.

🚀 WORKFLOW:
1. First: Read the main application file to understand its structure
2. Second: Create the new component file
3. Third: Modify the main application file to integrate the component
4. Verify: The component is imported and used in the main app

Start by reading the main application file, then create the component, then integrate it."""

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
                
                # Send progress update for each iteration
                progress_value = 60 + (iteration * 5)  # 60, 65, 70, etc.
                await self._send_streaming_update(
                    state["correlation_id"], 
                    f"🔄 Implementation iteration {iteration}...", 
                    progress=min(progress_value, 68), 
                    step=f"Implementation Step {iteration}"
                )
                
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
                    
                    # More specific continue prompt that enforces integration
                    component_creation_done = any(
                        'write_file' in str(tool_call.get('name', '')) and 
                        any(ext in str(tool_call.get('args', {}).get('file_path', '')) 
                            for ext in ['.jsx', '.js', '.tsx', '.ts', '.vue', '.py'])
                        for tool_call in response.tool_calls
                    )
                    
                    main_file_modification_done = any(
                        'write_file' in str(tool_call.get('name', '')) and 
                        any(main_file in str(tool_call.get('args', {}).get('file_path', '')) 
                            for main_file in ['App.', 'app.', 'main.', 'index.', 'Main.'])
                        for tool_call in response.tool_calls
                    )
                    
                    # Check if this was a component creation without integration
                    if component_creation_done and not main_file_modification_done:
                        continue_prompt = f"""CRITICAL: You created a new component but haven't integrated it into the main application yet!

You must complete BOTH steps:
✅ Step 1: Create the component (DONE)
❌ Step 2: Integrate it into the main application (NOT DONE)

To complete the integration:
1. Use read_file to examine the main application file (App.jsx, App.js, main.py, etc.)
2. Use write_file to modify the main application file to:
   - Import the new component
   - Add the component to the JSX/template/code
   - Ensure it appears in the correct location

The task is NOT complete until the component is integrated and visible in the main application.

Please continue with the integration step now."""
                    
                    elif component_creation_done and main_file_modification_done:
                        continue_prompt = """Great! You've created the component and integrated it into the main application. 

Please verify your work:
1. Did you import the component in the main file?
2. Did you add the component to the JSX/template in the correct location?
3. Is the component properly positioned (header at top, footer at bottom, etc.)?

If everything looks correct, respond with 'TASK COMPLETE' and summarize what was accomplished.
If you need to make any adjustments, continue using the tools."""
                    
                    else:
                        continue_prompt = """Based on the tool results above, analyze what you've accomplished so far:

1. Have you created any new components or files?
2. Have you integrated them into the main application?
3. Are there any remaining steps from the original plan?

Remember: If you're adding components (buttons, headers, footers, etc.), you MUST:
- Create the component file
- Integrate it into the main application file
- Ensure it's imported and used correctly

Continue with the next necessary step, or respond with 'TASK COMPLETE' if everything is done."""
                    
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
            
            # Send final implementation update
            await self._send_streaming_update(
                state["correlation_id"], 
                f"✅ Implementation complete - {len(changes_made)} files changed", 
                progress=68, 
                step="Implementation Complete"
            )
            
            # Final validation to ensure integration was completed
            if changes_made:
                component_files = [
                    change for change in changes_made 
                    if any(ext in change.get("file_path", "") for ext in ['.jsx', '.js', '.tsx', '.ts', '.vue', '.py'])
                    and not any(main_file in change.get("file_path", "") for main_file in ['App.', 'app.', 'main.', 'index.', 'Main.'])
                ]
                
                main_app_files = [
                    change for change in changes_made 
                    if any(main_file in change.get("file_path", "") for main_file in ['App.', 'app.', 'main.', 'index.', 'Main.'])
                ]
                
                if component_files and not main_app_files:
                    # Component was created but main app wasn't modified - this is incomplete
                    print("⚠️ WARNING: Component created but main application not modified - integration may be incomplete!")
                    self.telemetry.log_event(
                        "Potential incomplete integration detected",
                        correlation_id=state["correlation_id"],
                        component_files=[c.get("file_path") for c in component_files],
                        main_app_files=[c.get("file_path") for c in main_app_files],
                        level="warning"
                    )
                elif component_files and main_app_files:
                    print("✅ Integration appears complete - component created and main app modified")
                    self.telemetry.log_event(
                        "Integration completed successfully",
                        correlation_id=state["correlation_id"],
                        component_files=[c.get("file_path") for c in component_files],
                        main_app_files=[c.get("file_path") for c in main_app_files]
                    )
            
            state["implementation_result"] = {"output": f"Completed after {iteration} iterations", "tool_calls": []}
            state["changes_made"] = changes_made
            state["messages"].extend(messages)
            
            state["steps_completed"].append("implement_changes")
            self._log_node_success("implement_changes", state)
            
        except Exception as e:
            await self._send_streaming_update(
                state["correlation_id"], 
                f"❌ Implementation failed: {str(e)}"
            )
            
            self.telemetry.log_error(
                e,
                context={"step": "implement_changes", **state},
                correlation_id=state.get("correlation_id")
            )
            raise
            
        return state
    
    @traceable(name="commit_changes")
    async def _commit_changes_node(self, state: AgentState) -> AgentState:
        """Commit the changes to git."""
        try:
            self._log_node_start("commit_changes", state)
            
            await self._send_streaming_update(
                state["correlation_id"], 
                "📦 Committing changes...", 
                progress=70, 
                step="Committing changes"
            )
            
            state["current_step"] = "commit_changes"
            state["last_update"] = datetime.utcnow()
            
            # Use the existing branch name from implement_changes_node
            # If no branch was created yet, create one now
            if not state.get("branch_name"):
                branch_name = f"backspace-agent-{state['correlation_id'][:8]}"
                state["branch_name"] = branch_name
                
                # Create the branch first
                await self._send_streaming_update(
                    state["correlation_id"], 
                    f"🌿 Creating branch: {branch_name}...", 
                    progress=68, 
                    step="Creating branch"
                )
                
                create_branch_tool = next(t for t in self.tools if t.name == "create_branch")
                branch_result = await create_branch_tool.ainvoke({
                    "correlation_id": state["correlation_id"],
                    "repo_path": state["repo_path"],
                    "branch_name": branch_name
                })
                
                if not branch_result.get("success", False):
                    raise Exception(f"Failed to create branch: {branch_result.get('error', 'Unknown error')}")
            else:
                # Branch already exists from implement_changes_node
                branch_name = state["branch_name"]
                await self._send_streaming_update(
                    state["correlation_id"], 
                    f"📝 Using existing branch: {branch_name}...", 
                    progress=68, 
                    step="Using existing branch"
                )
            
            # Commit the changes
            await self._send_streaming_update(
                state["correlation_id"], 
                f"📝 Committing changes to branch: {branch_name}...", 
                progress=72, 
                step="Committing changes"
            )
            
            commit_tool = next(t for t in self.tools if t.name == "commit_changes")
            result = await commit_tool.ainvoke({
                "correlation_id": state["correlation_id"],
                "repo_path": state["repo_path"],
                "message": f"feat: {state['prompt'][:50]}..."
            })
            
            state["commit_hash"] = result.get("commit_hash")
            
            await self._send_streaming_update(
                state["correlation_id"], 
                f"✅ Changes committed to branch: {branch_name}"
            )
            
            state["steps_completed"].append("commit_changes")
            self._log_node_success("commit_changes", state)
            
        except Exception as e:
            state = await self._handle_node_error("commit_changes", state, e)
            
        return state
    
    @traceable(name="push_changes")
    async def _push_changes_node(self, state: AgentState) -> AgentState:
        """Push changes to the remote repository."""
        try:
            self._log_node_start("push_changes", state)
            
            await self._send_streaming_update(
                state["correlation_id"], 
                "🚀 Pushing changes to remote...", 
                progress=85, 
                step="Pushing changes"
            )
            
            state["current_step"] = "push_changes"
            state["last_update"] = datetime.utcnow()
            
            push_tool = next(t for t in self.tools if t.name == "push_changes")
            result = await push_tool.ainvoke({
                "correlation_id": state["correlation_id"],
                "repo_path": state["repo_path"],
                "branch_name": state["branch_name"]
            })
            
            state["push_success"] = result.get("success", False)
            
            await self._send_streaming_update(
                state["correlation_id"], 
                f"✅ Changes pushed to remote branch: {state['branch_name']}"
            )
            
            state["steps_completed"].append("push_changes")
            self._log_node_success("push_changes", state)
            
        except Exception as e:
            state = await self._handle_node_error("push_changes", state, e)
            
        return state
    
    @traceable(name="create_pull_request")
    async def _create_pull_request_node(self, state: AgentState) -> AgentState:
        try:
            if not state.get("correlation_id"):
                raise ValueError("correlation_id missing in state at create_pull_request_node")
            self._log_node_start("create_pull_request", state)
            
            # Send initial progress update
            await self._send_streaming_update(
                state["correlation_id"], 
                "📄 Creating pull request...", 
                progress=90, 
                step="Creating Pull Request"
            )
            
            state["current_step"] = "create_pull_request"
            state["last_update"] = datetime.utcnow()
            
            # Create PR title and description
            changes_summary = []
            files_created = []
            files_modified = []
            
            if state.get("changes_made"):
                for change in state["changes_made"]:
                    action = change.get("action", "modified")
                    file_path = change.get("file_path", "unknown")
                    description = change.get("description", "")
                    
                    if "created" in action.lower():
                        files_created.append(f"- **{file_path}** - {description}")
                        changes_summary.append(f"- Add {file_path}")
                    elif "modified" in action.lower():
                        files_modified.append(f"- **{file_path}** - {description}")
                        changes_summary.append(f"- Update {file_path}")
                    else:
                        changes_summary.append(f"- {action.title()} {file_path}")
            
            # Create PR title from prompt
            prompt = state.get("prompt", "Implement changes")
            pr_title = prompt[:50]  # GitHub PR title limit
            
            # Get plan details
            plan = state.get('plan', {})
            plan_summary = plan.get('summary', 'No plan summary available')
            plan_steps = plan.get('steps', [])
            
            # Create detailed PR description
            pr_body = f"""## 🎯 Task
{prompt}

## 🚀 What Was Done
This pull request implements the requested changes by completing the following actions:

"""
            
            # Add implementation details
            if plan_steps and isinstance(plan_steps, list):
                # Filter out agent workflow steps and focus on code changes
                code_related_steps = []
                for step in plan_steps[:10]:  # Look at more steps to filter
                    if isinstance(step, dict):
                        step_text = step.get('description', str(step))
                    else:
                        step_text = str(step)
                    
                    # Filter out agent workflow steps
                    workflow_keywords = [
                        'analyze', 'analysed', 'analyzed', 'repository', 'structure',
                        'understand', 'read', 'examine', 'review', 'study',
                        'plan', 'planning', 'create plan', 'implementation plan',
                        'workflow', 'process', 'approach', 'strategy'
                    ]
                    
                    # Keep steps that are about actual code implementation
                    step_lower = step_text.lower()
                    is_workflow_step = any(keyword in step_lower for keyword in workflow_keywords)
                    is_code_step = any(keyword in step_lower for keyword in [
                        'create', 'add', 'implement', 'build', 'write', 'modify', 
                        'update', 'integrate', 'import', 'component', 'function',
                        'class', 'file', 'code', 'style', 'css', 'jsx', 'js',
                        'html', 'python', 'install', 'configure', 'setup'
                    ])
                    
                    if is_code_step and not is_workflow_step:
                        code_related_steps.append(step_text)
                
                if code_related_steps:
                    pr_body += "### Implementation Steps:\n"
                    for i, step in enumerate(code_related_steps[:5], 1):  # Limit to 5 steps
                        pr_body += f"{i}. {step}\n"
                    pr_body += "\n"
            
            # Add file changes section
            pr_body += "## 📁 Files Changed\n\n"
            
            if files_created:
                pr_body += "### ✅ Files Created:\n"
                pr_body += "\n".join(files_created) + "\n\n"
            
            if files_modified:
                pr_body += "### 📝 Files Modified:\n"
                pr_body += "\n".join(files_modified) + "\n\n"
            
            if not files_created and not files_modified:
                pr_body += "- No specific file changes detected\n\n"
            
            # Add summary of changes
            total_changes = len(files_created) + len(files_modified)
            if total_changes > 0:
                pr_body += f"### 📊 Summary:\n"
                pr_body += f"- **{len(files_created)}** file(s) created\n"
                pr_body += f"- **{len(files_modified)}** file(s) modified\n"
                pr_body += f"- **{total_changes}** total changes\n\n"
            
            # Add implementation plan context
            if plan_summary and plan_summary != "No plan summary available":
                pr_body += f"## 🗺️ Implementation Plan\n{plan_summary}\n\n"
            
            pr_body += "---\n*This pull request was automatically created by **Backspace AI Coding Agent***"
            
            # Try to create the PR using GitService
            try:
                await self._send_streaming_update(
                    state["correlation_id"], 
                    f"🔗 Creating pull request: {pr_title}...", 
                    progress=95, 
                    step="Creating Pull Request"
                )
                
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
                
                # Send success message with PR URL
                await self._send_streaming_update(
                    state["correlation_id"], 
                    f"✅ Pull request created successfully: {pr_url}"
                )
                
                self.telemetry.log_event(
                    "Pull request created successfully",
                    correlation_id=state["correlation_id"],
                    pr_url=pr_url,
                    title=pr_title
                )
                
            except Exception as pr_error:
                # Log the error but don't fail the whole workflow
                await self._send_streaming_update(
                    state["correlation_id"], 
                    f"⚠️ PR creation failed: {str(pr_error)}"
                )
                
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
            await self._send_streaming_update(
                state["correlation_id"], 
                f"❌ Error creating pull request: {str(e)}"
            )
            
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
                r'modify.*?main.*?file.*?```jsx\s*\n(.*?)```',
                r'update.*?app.*?component.*?```jsx\s*\n(.*?)```'
            ]
            
            for pattern in modify_patterns:
                modify_matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
                for match in modify_matches:
                    file_content = match.group(1).strip()
                    
                    # Look for main app files in the existing files
                    app_files = [f for f in existing_files_content.keys() if 'app' in f.lower() and f.endswith(('.js', '.jsx', '.ts', '.tsx'))]
                    if app_files:
                        file_path = app_files[0]
                    else:
                        # Try common patterns
                        common_app_files = ["src/App.jsx", "src/App.js", "src/App.tsx", "src/App.ts", "App.jsx", "App.js"]
                        for common_file in common_app_files:
                            if common_file in existing_files_content:
                                file_path = common_file
                                break
                        else:
                            file_path = "src/App.jsx"  # Default fallback
                    
                    file_changes.append({
                        "action": "modify",
                        "file_path": file_path,
                        "content": file_content,
                        "description": f"Modify {file_path} to integrate component",
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
                        # Determine appropriate file path based on existing files
                        if any('components/' in f for f in existing_files_content.keys()):
                            file_path = "src/components/Footer.jsx"
                        elif any('src/' in f for f in existing_files_content.keys()):
                            file_path = "src/Footer.jsx"
                        else:
                            file_path = "Footer.jsx"
                            
                        file_changes.append({
                            "action": "create",
                            "file_path": file_path,
                            "content": file_content,
                            "description": f"Create Footer component at {file_path}",
                            "needs_smart_integration": False
                        })
                    # Look for main app file modification
                    elif ('app' in preceding_text.lower() or 'main' in preceding_text.lower()) and ('modify' in preceding_text.lower() or 'import' in file_content.lower()):
                        # Try to infer the actual file path from context
                        file_path_match = re.search(r'(src/[A-Za-z0-9_/]+\.jsx?)', preceding_text)
                        if file_path_match:
                            file_path = file_path_match.group(1)
                        else:
                            # Look for main app files in the existing files
                            app_files = [f for f in existing_files_content.keys() if 'app' in f.lower() and f.endswith(('.js', '.jsx', '.ts', '.tsx'))]
                            if app_files:
                                file_path = app_files[0]
                            else:
                                # Try common patterns
                                common_app_files = ["src/App.jsx", "src/App.js", "src/App.tsx", "src/App.ts", "App.jsx", "App.js"]
                                for common_file in common_app_files:
                                    if common_file in existing_files_content:
                                        file_path = common_file
                                        break
                                else:
                                    file_path = "src/App.jsx"  # Default fallback
                        
                        file_changes.append({
                            "action": "modify",
                            "file_path": file_path,
                            "content": file_content,
                            "description": f"Modify {file_path} to integrate component",
                            "needs_smart_integration": True
                        })
                    # Generic file creation/modification
                    else:
                        # Try to infer file type and path from content
                        file_path = f"generated_file_{i}.jsx"  # Default
                        
                        # If it looks like a component (has JSX), give it a component name
                        if 'export' in file_content and ('jsx' in file_content.lower() or '<' in file_content):
                            if any('components/' in f for f in existing_files_content.keys()):
                                file_path = f"src/components/GeneratedComponent_{i}.jsx"
                            elif any('src/' in f for f in existing_files_content.keys()):
                                file_path = f"src/GeneratedComponent_{i}.jsx"
                            else:
                                file_path = f"GeneratedComponent_{i}.jsx"
                        # If it looks like a Python file
                        elif 'def ' in file_content or 'import ' in file_content:
                            file_path = f"generated_module_{i}.py"
                        # If it looks like CSS
                        elif '{' in file_content and '}' in file_content and ('color' in file_content.lower() or 'font' in file_content.lower()):
                            file_path = f"generated_styles_{i}.css"
                        
                        file_changes.append({
                            "action": "create",
                            "file_path": file_path,
                            "content": file_content,
                            "description": f"Create {file_path}",
                            "needs_smart_integration": False
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