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

    🚨 CRITICAL PRINCIPLES - EXISTING CODE PRESERVATION:
    1. **NEVER TOUCH EXISTING CODE UNLESS ABSOLUTELY ESSENTIAL**: Existing code is sacred. Do not modify, refactor, or change ANY existing files unless it is 100% impossible to implement the requested feature without doing so.
    
    2. **CREATE NEW FILES FIRST**: Always try to implement new functionality by creating entirely new files/components/modules. Only modify existing files if the new functionality cannot possibly work without integration changes.
    
    3. **MINIMAL INTEGRATION ONLY**: If you must modify existing files, make only the absolute minimum changes required for integration:
       - Add a single import statement
       - Add a single component reference
       - Add a single route/endpoint
       - NO refactoring, NO style changes, NO "improvements"
    
    4. **READ BEFORE ANY MODIFICATION**: Always read existing files completely before making any changes to understand the current structure.
    
    5. **FOLLOW EXISTING PATTERNS**: Use the exact same file extensions (.js vs .jsx), naming conventions, and coding patterns as the existing codebase.
    
    6. **NO DUPLICATES**: Never create multiple versions of the same file (e.g., both .js and .jsx versions).
    
    7. **COMPONENT-BASED APPROACH**: For React/Vue/Angular projects, always create new components in separate files rather than modifying existing components.

    DEPENDENCY MANAGEMENT RULES (CRITICAL):
    8. **CHECK EXISTING DEPENDENCIES**: Always read package.json/requirements.txt/go.mod/Cargo.toml first to understand what packages are already installed.
    9. **SMART DEPENDENCY DECISIONS**: When new functionality requires packages not currently installed:
       - If the user explicitly requested functionality that commonly requires specific packages, explain why you need to install them
       - Use execute_command to install packages when necessary (npm install, pip install, etc.)
       - Choose well-established, popular packages for the required functionality
       - Document what packages you're installing and why
    10. **USE EXISTING PATTERNS**: Work with the existing tech stack - understand the current architecture before adding new dependencies.
    11. **SIMPLE SOLUTIONS FIRST**: Try to implement with existing dependencies first, but don't compromise functionality if new packages are clearly needed.
    12. **EXPLAIN YOUR DECISIONS**: When you install new packages, explain why they're necessary for the requested functionality.

    🔒 FORBIDDEN ACTIONS:
    - DO NOT modify existing components/functions unless absolutely required for integration
    - DO NOT refactor or "improve" existing code
    - DO NOT change existing file structures or move files
    - DO NOT install packages without understanding why they're needed
    - DO NOT use bleeding-edge or experimental packages without good reason
    - DO NOT install multiple packages that do the same thing
    - DO NOT install packages for functionality the user didn't request
    - DO NOT change existing styling or CSS unless specifically requested
    - DO NOT modify existing database schemas unless absolutely required
    - DO NOT change existing API endpoints unless absolutely required

    🎯 IMPLEMENTATION STRATEGY:
    - Analyze the repository first to understand existing languages, frameworks, and patterns
    - Read dependency files (package.json, requirements.txt, go.mod, Cargo.toml, etc.) to understand available packages
    - Create new files/components/modules for new functionality
    - Only modify existing files if integration is impossible without it
    - When integration is required, make only minimal changes (imports, single component reference, etc.)
    - Follow existing file naming conventions and directory structure exactly
    - Maintain consistency with existing code style and patterns
    - Install new dependencies when explicitly needed for requested functionality (with explanation)
    - Use only well-established, popular packages for the technology stack being used

    🚀 YOUR WORKFLOW:
    1. Analyze the repository to understand its structure, language, and framework patterns
    2. Read dependency files to understand available packages
    3. Create a detailed implementation plan using existing dependencies or explaining why new ones are needed
    4. Install any necessary dependencies with proper explanation
    5. Create new files for new functionality (components, modules, classes, etc.)
    6. Only if absolutely required: make minimal integration changes to existing files
    7. Commit and push the changes

    Always:
    - Think step by step before making changes
    - Consider the existing codebase structure and patterns
    - Write clean, maintainable code that works with existing dependencies
    - Follow best practices for the language/framework
    - Test your changes when possible
    - Provide clear commit messages
    - Preserve ALL existing functionality

    Use the available tools to accomplish your tasks. Be thorough and methodical in your approach.

    🚨 PACKAGE INSTALLATION RULES:
    - DO NOT install packages without explaining why they're needed for the requested functionality
    - DO NOT use experimental or bleeding-edge packages without strong justification
    - DO NOT install multiple packages that serve the same purpose
    - DO NOT install packages for features the user didn't explicitly requested
    - DO consider the technology stack when choosing packages (React packages for React apps, Flask packages for Flask apps, etc.)

    🔥 INTEGRATION REQUIREMENTS (NON-NEGOTIABLE):
    When adding ANY new functionality, you MUST complete FULL integration:

    ✅ STEP 1: Create the necessary files (components, modules, classes, etc.) - NEVER modify existing files for this
    ✅ STEP 2: Integrate them into the main application with MINIMAL changes to existing files

    Integration means:
    1. Read the main application files to understand current structure
    2. Make only the absolute minimum changes to existing files:
       - Add import/include statement
       - Add single component reference/route
       - NO other modifications unless 100% impossible to avoid

    ⛔ MODIFICATION GUIDELINES:
    - Existing files are SACRED - touch them only if absolutely essential
    - When you must modify existing files, explain WHY it's impossible to avoid
    - Make the smallest possible change that enables the new functionality
    - Never refactor, improve, or change existing code style
    - Preserve all existing functionality exactly as it is

    ⛔ FAILURE TO PRESERVE EXISTING CODE = UNACCEPTABLE
    Creating files without proper integration is incomplete, but modifying existing code unnecessarily is WORSE.
    """

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
            2. Available dependencies (check package.json/requirements.txt/go.mod/etc.)
            3. Required new dependencies (if any) and justification for why they're needed
            4. List of NEW files that need to be created (following existing conventions)
            5. Minimal changes to existing files (only if absolutely essential for integration)
            6. Step-by-step implementation approach
            7. Any considerations or potential issues

            🚨 CRITICAL PLANNING RULES - EXISTING CODE PRESERVATION:
            - Plan to create NEW files for new functionality - DO NOT plan to modify existing files unless absolutely essential
            - If you must modify existing files, justify WHY it's impossible to avoid and plan only minimal changes
            - Plan to preserve ALL existing functionality exactly as it is
            - Plan to follow existing patterns and conventions exactly
            - Plan integration with the smallest possible changes to existing code

            DEPENDENCY MANAGEMENT RULES:
            - First analyze dependency files to understand available packages
            - If the requested functionality commonly requires specific packages not currently installed, plan to install them with explanation
            - Choose well-established, popular packages for the technology stack
            - Explain why new packages are necessary for the requested functionality
            - Try to use existing dependencies first, but don't compromise functionality

            IMPLEMENTATION APPROACH:
            - Identify existing language, framework, and code patterns
            - Plan to create new components/modules/classes in separate files
            - Plan minimal integration points (imports, single references)
            - Follow existing file naming conventions and directory structure exactly
            - Ensure all planned packages are appropriate for the technology stack
            - Plan to preserve existing code structure and functionality completely

            Think through this carefully and provide a comprehensive plan that prioritizes creating new files over modifying existing ones.
            """),
            MessagesPlaceholder(variable_name="chat_history")
        ])
    
    def _create_implementation_prompt(self) -> ChatPromptTemplate:
        """Create the prompt for implementation."""
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Implement the planned changes for the request: {prompt}. Repository: {repo_url}. Plan: {plan}. Repository analysis: {repo_analysis}. 🚨 CRITICAL: Create new files for new functionality and make only minimal integration changes to existing files. Read existing files first if you must modify them."),
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

🎯 TASK: Create NEW files for new functionality and make only minimal integration changes to existing files.

🚨 CRITICAL INSTRUCTIONS - EXISTING CODE PRESERVATION:
1. **DEPENDENCY CHECK**: First read dependency files (package.json, requirements.txt, go.mod, etc.) to understand available packages
2. **SMART PACKAGE MANAGEMENT**: Install new packages when explicitly needed for requested functionality (with explanation)
3. **CREATE NEW FILES FIRST**: Always create new files/components/modules for new functionality - DO NOT modify existing files unless absolutely essential
4. **MINIMAL EXISTING FILE CHANGES**: Only modify existing files if integration is impossible without it, and then make only minimal changes:
   - Add a single import statement
   - Add a single component reference
   - Add a single route/endpoint
   - NO refactoring, NO style changes, NO "improvements"
5. **READ BEFORE MODIFYING**: If you must modify an existing file, ALWAYS read it first using read_file to understand current structure
6. Follow the existing patterns in the codebase (file extensions, naming conventions, code style, etc.) EXACTLY

🔧 PACKAGE INSTALLATION RULES:
- DO NOT install packages without explaining why they're needed for the requested functionality
- DO NOT use experimental or bleeding-edge packages without strong justification
- DO NOT install multiple packages that serve the same purpose
- DO NOT install packages for features the user didn't explicitly request
- DO consider the technology stack when choosing packages (React packages for React apps, Flask packages for Flask apps, etc.)

🔒 EXISTING CODE PRESERVATION RULES:
- Existing files are SACRED - touch them only if absolutely essential for integration
- When you must modify existing files, explain WHY it's impossible to avoid
- Make the smallest possible change that enables the new functionality
- Never refactor, improve, or change existing code style
- Preserve all existing functionality exactly as it is

🔥 INTEGRATION REQUIREMENTS (NON-NEGOTIABLE):
When adding ANY new functionality, you MUST complete FULL integration:

✅ STEP 1: Create the necessary files (components, modules, classes, etc.) - NEVER modify existing files for this
✅ STEP 2: Integrate them into the main application with MINIMAL changes to existing files

Integration means:
1. Read the main application files to understand current structure
2. Make only the absolute minimum changes to existing files:
   - Add import/include statement
   - Add single component reference/route
   - NO other modifications unless 100% impossible to avoid

⛔ MODIFICATION GUIDELINES:
- Existing files are SACRED - touch them only if absolutely essential
- When you must modify existing files, explain WHY it's impossible to avoid
- Make the smallest possible change that enables the new functionality
- Never refactor, improve, or change existing code style
- Preserve all existing functionality exactly as it is

🚀 WORKFLOW:
1. **First: Read dependency files to understand available packages**
2. Second: Read the main application files to understand current structure (BUT DO NOT MODIFY THEM YET)
3. Third: Install any necessary packages (with explanation) and create new files for functionality
4. Fourth: ONLY if absolutely essential for integration, make minimal changes to existing files
5. Verify: The new functionality is properly integrated and functional

Start by reading dependency files, then read the main application files, then create new files, and only modify existing files if absolutely essential."""

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
                                    file_path = tool_args.get('file_path', 'unknown')
                                    
                                    # Check if this file was read before (indicates it existed)
                                    file_existed = False
                                    
                                    # Check current tool calls for read_file
                                    if hasattr(response, 'tool_calls') and response.tool_calls:
                                        for prev_tool_call in response.tool_calls:
                                            if (prev_tool_call.get('name') == 'read_file' and 
                                                prev_tool_call.get('args', {}).get('file_path') == file_path):
                                                file_existed = True
                                                break
                                    
                                    # Also check previous iterations for read_file calls
                                    if not file_existed:
                                        for msg in messages:
                                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                                for prev_tool_call in msg.tool_calls:
                                                    if (prev_tool_call.get('name') == 'read_file' and 
                                                        prev_tool_call.get('args', {}).get('file_path') == file_path):
                                                        file_existed = True
                                                        break
                                    
                                    # Generate descriptive text based on the original prompt and file type
                                    prompt_lower = state.get('prompt', '').lower()
                                    
                                    # Determine file type and generate appropriate description
                                    if file_path.endswith(('.jsx', '.js', '.ts', '.tsx')):
                                        # JavaScript/TypeScript/React files
                                        if 'component' in file_path.lower() and not file_existed:
                                            # Infer component type from prompt
                                            if 'sign up' in prompt_lower or 'signup' in prompt_lower:
                                                description = f"Added new SignUp component with user registration form"
                                            elif 'login' in prompt_lower:
                                                description = f"Added new Login component with authentication form"
                                            elif 'contact' in prompt_lower:
                                                description = f"Added new Contact component with contact form"
                                            elif 'button' in prompt_lower:
                                                description = f"Added new interactive button component"
                                            elif 'nav' in prompt_lower or 'menu' in prompt_lower:
                                                description = f"Added new navigation component"
                                            elif 'footer' in prompt_lower:
                                                description = f"Added new footer component"
                                            elif 'header' in prompt_lower:
                                                description = f"Added new header component"
                                            elif 'form' in prompt_lower:
                                                description = f"Added new form component"
                                            elif 'modal' in prompt_lower or 'popup' in prompt_lower:
                                                description = f"Added new modal/popup component"
                                            else:
                                                component_name = file_path.split('/')[-1].replace('.jsx', '').replace('.js', '').replace('.tsx', '').replace('.ts', '')
                                                description = f"Added new {component_name} component"
                                        elif any(main_file in file_path.lower() for main_file in ['app.', 'main.', 'index.']) and file_existed:
                                            # Main application files
                                            if 'sign up' in prompt_lower or 'signup' in prompt_lower:
                                                description = f"Integrated SignUp functionality into main application"
                                            elif 'login' in prompt_lower:
                                                description = f"Integrated Login functionality into main application"
                                            elif 'contact' in prompt_lower:
                                                description = f"Integrated Contact form into main application"
                                            else:
                                                description = f"Enhanced main application with new functionality"
                                        else:
                                            # Other JS/TS files
                                            if file_existed:
                                                description = f"Updated {file_path} with new functionality"
                                            else:
                                                description = f"Created new {file_path} module"
                                    
                                    elif file_path.endswith('.py'):
                                        # Python files
                                        if 'api' in file_path.lower() or 'endpoint' in file_path.lower() or 'route' in file_path.lower():
                                            if 'auth' in prompt_lower or 'login' in prompt_lower:
                                                description = f"Added authentication API endpoints"
                                            elif 'user' in prompt_lower:
                                                description = f"Added user management API endpoints"
                                            elif 'contact' in prompt_lower:
                                                description = f"Added contact form API endpoint"
                                            else:
                                                description = f"Added new API endpoint functionality"
                                        elif 'model' in file_path.lower():
                                            if 'user' in prompt_lower:
                                                description = f"Added User data model"
                                            elif 'auth' in prompt_lower:
                                                description = f"Added authentication data model"
                                            else:
                                                description = f"Added new data model"
                                        elif 'service' in file_path.lower():
                                            description = f"Added new service functionality"
                                        elif 'test' in file_path.lower():
                                            description = f"Added test cases"
                                        elif any(main_file in file_path.lower() for main_file in ['app.py', 'main.py', '__init__.py']) and file_existed:
                                            description = f"Enhanced main application with new functionality"
                                        else:
                                            if file_existed:
                                                description = f"Enhanced Python module {file_path}"
                                            else:
                                                description = f"Created new Python module {file_path}"
                                    
                                    elif file_path.endswith(('.go')):
                                        # Go files
                                        if 'main.go' in file_path:
                                            description = f"Enhanced main Go application"
                                        elif 'handler' in file_path.lower() or 'route' in file_path.lower():
                                            description = f"Added new HTTP handlers"
                                        elif 'model' in file_path.lower():
                                            description = f"Added new data structures"
                                        elif 'service' in file_path.lower():
                                            description = f"Added new service functionality"
                                        else:
                                            if file_existed:
                                                description = f"Enhanced Go module {file_path}"
                                            else:
                                                description = f"Created new Go module {file_path}"
                                    
                                    elif file_path.endswith(('.rs')):
                                        # Rust files
                                        if 'main.rs' in file_path or 'lib.rs' in file_path:
                                            description = f"Enhanced main Rust application"
                                        elif 'mod.rs' in file_path:
                                            description = f"Added new Rust module"
                                        else:
                                            if file_existed:
                                                description = f"Enhanced Rust module {file_path}"
                                            else:
                                                description = f"Created new Rust module {file_path}"
                                    
                                    elif file_path.endswith(('.java', '.kt')):
                                        # Java/Kotlin files
                                        if 'Controller' in file_path:
                                            description = f"Added new REST controller"
                                        elif 'Service' in file_path:
                                            description = f"Added new service class"
                                        elif 'Repository' in file_path:
                                            description = f"Added new data repository"
                                        elif 'Model' in file_path or 'Entity' in file_path:
                                            description = f"Added new data model/entity"
                                        else:
                                            if file_existed:
                                                description = f"Enhanced {file_path}"
                                            else:
                                                description = f"Created new {file_path}"
                                    
                                    elif file_path.endswith(('.css', '.scss', '.sass', '.less')):
                                        # Styling files
                                        if file_existed:
                                            description = f"Updated styling and visual design"
                                        else:
                                            description = f"Added new CSS styles and layout"
                                    
                                    elif file_path.endswith(('.html', '.htm')):
                                        # HTML files
                                        if file_existed:
                                            description = f"Updated HTML template and structure"
                                        else:
                                            description = f"Created new HTML page template"
                                    
                                    elif file_path.endswith(('.php')):
                                        # PHP files
                                        if 'index.php' in file_path:
                                            description = f"Enhanced main PHP application"
                                        elif 'api' in file_path.lower() or 'endpoint' in file_path.lower():
                                            description = f"Added new PHP API endpoint"
                                        else:
                                            if file_existed:
                                                description = f"Enhanced PHP module {file_path}"
                                            else:
                                                description = f"Created new PHP module {file_path}"
                                    
                                    elif file_path.endswith(('.rb')):
                                        # Ruby files
                                        if 'controller' in file_path.lower():
                                            description = f"Added new Rails controller"
                                        elif 'model' in file_path.lower():
                                            description = f"Added new Rails model"
                                        elif 'view' in file_path.lower():
                                            description = f"Added new Rails view"
                                        else:
                                            if file_existed:
                                                description = f"Enhanced Ruby module {file_path}"
                                            else:
                                                description = f"Created new Ruby module {file_path}"
                                    
                                    elif file_path.endswith(('.cs', '.vb')):
                                        # .NET files
                                        if 'Controller' in file_path:
                                            description = f"Added new .NET controller"
                                        elif 'Service' in file_path:
                                            description = f"Added new .NET service"
                                        elif 'Model' in file_path:
                                            description = f"Added new .NET model"
                                        else:
                                            if file_existed:
                                                description = f"Enhanced .NET module {file_path}"
                                            else:
                                                description = f"Created new .NET module {file_path}"
                                    
                                    else:
                                        # Generic fallback
                                        if file_existed:
                                            description = f"Modified {file_path}"
                                        else:
                                            description = f"Created {file_path}"
                                    
                                    changes_made.append({
                                        "action": "modified" if file_existed else "created",
                                        "file_path": file_path,
                                        "description": description
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
                            for ext in ['.jsx', '.js', '.tsx', '.ts', '.vue', '.py', '.go', '.rs', '.java', '.kt', '.php', '.rb', '.cs'])
                        for tool_call in response.tool_calls
                    )
                    
                    main_file_modification_done = any(
                        'write_file' in str(tool_call.get('name', '')) and 
                        any(main_file in str(tool_call.get('args', {}).get('file_path', '')) 
                            for main_file in ['App.', 'app.', 'main.', 'index.', 'Main.', '__init__.py'])
                        for tool_call in response.tool_calls
                    )
                    
                    # Check if this was a component creation without integration
                    if component_creation_done and not main_file_modification_done:
                        continue_prompt = f"""🚨 CRITICAL: You created new files but haven't integrated them into the main application yet!

You must complete BOTH steps:
✅ Step 1: Create the new files/modules (DONE)
❌ Step 2: Minimal integration into the main application (NOT DONE)

⚠️ INTEGRATION RULES - EXISTING CODE PRESERVATION:
- Existing files are SACRED - make only minimal changes for integration
- ONLY add what's absolutely essential: import statement + single component reference
- NO refactoring, NO style changes, NO "improvements" to existing code
- Preserve ALL existing functionality exactly as it is

To complete the minimal integration:
1. Use read_file to examine the main application files (App.jsx, main.py, main.go, etc.)
2. Use write_file to make MINIMAL changes to main application files:
   - Add import/include statement for new functionality
   - Add single component reference/route where needed
   - NO other modifications

Explain WHY any modification to existing files is absolutely necessary for integration.

Please continue with the minimal integration step now."""
                    
                    elif component_creation_done and main_file_modification_done:
                        continue_prompt = """✅ Excellent! You've created new files and made minimal integration changes to the main application. 

Please verify your minimal integration:
1. Did you add only essential imports/includes for the new functionality?
2. Did you add single component references/routes where needed?
3. Did you preserve ALL existing functionality exactly as it was?
4. Did you avoid any refactoring or "improvements" to existing code?

If everything looks correct and you made only minimal necessary changes, respond with 'TASK COMPLETE' and summarize what was accomplished.
If you need to make any adjustments, continue using the tools."""
                    
                    else:
                        continue_prompt = """Based on the tool results above, analyze what you've accomplished so far:

1. Have you created any new files or modules?
2. Have you made minimal integration changes to existing files?
3. Are there any remaining steps from the original plan?

🚨 REMEMBER - EXISTING CODE PRESERVATION:
- CREATE new files/modules for new functionality (preferred approach)
- Only modify existing files if integration is impossible without it
- When modifying existing files, make only minimal changes (imports + single references)
- Always read existing files first to understand current structure
- Preserve ALL existing functionality exactly as it is
- NO refactoring, NO style changes, NO "improvements"

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
                new_functionality_files = [
                    change for change in changes_made 
                    if any(ext in change.get("file_path", "") for ext in ['.jsx', '.js', '.tsx', '.ts', '.vue', '.py', '.go', '.rs', '.java', '.kt', '.php', '.rb', '.cs'])
                    and not any(main_file in change.get("file_path", "") for main_file in ['App.', 'app.', 'main.', 'index.', 'Main.', '__init__.py'])
                ]
                
                main_app_files = [
                    change for change in changes_made 
                    if any(main_file in change.get("file_path", "") for main_file in ['App.', 'app.', 'main.', 'index.', 'Main.', '__init__.py'])
                ]
                
                if new_functionality_files and not main_app_files:
                    # Files were created but main app wasn't modified - this may be incomplete
                    print("⚠️ WARNING: New files created but main application not modified - integration may be incomplete!")
                    self.telemetry.log_event(
                        "Potential incomplete integration detected",
                        correlation_id=state["correlation_id"],
                        new_files=[c.get("file_path") for c in new_functionality_files],
                        main_app_files=[c.get("file_path") for c in main_app_files],
                        level="warning"
                    )
                elif new_functionality_files and main_app_files:
                    print("✅ Integration appears complete - new functionality created and main app modified appropriately")
                    self.telemetry.log_event(
                        "Integration completed successfully",
                        correlation_id=state["correlation_id"],
                        new_files=[c.get("file_path") for c in new_functionality_files],
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
            
            # Commit the changes
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
"""
            
            # Add what was actually accomplished in simple bullet points
            if state.get("changes_made"):
                for change in state["changes_made"]:
                    file_path = change.get("file_path", "unknown")
                    description = change.get("description", "")
                    action = change.get("action", "modified")
                    
                    # Use the descriptive text from the change description
                    if description and description != f"File {file_path} written":
                        pr_body += f"• {description}\n"
                    else:
                        # Fallback to action-based description
                        if action == "created":
                            pr_body += f"• Created {file_path}\n"
                        elif action == "modified":
                            pr_body += f"• Modified {file_path}\n"
                        else:
                            pr_body += f"• {action.title()} {file_path}\n"
            else:
                pr_body += "• Implemented requested changes\n"
            
            # Add files changed section
            pr_body += f"""
## 📁 Files Changed
"""
            
            if files_created:
                pr_body += "**Files Created:**\n"
                for file_info in files_created:
                    # Extract just the file path from the full description
                    file_path = file_info.split("**")[1].split("**")[0] if "**" in file_info else file_info.replace("- ", "")
                    pr_body += f"• {file_path}\n"
                pr_body += "\n"
            
            if files_modified:
                pr_body += "**Files Modified:**\n"
                for file_info in files_modified:
                    # Extract just the file path from the full description  
                    file_path = file_info.split("**")[1].split("**")[0] if "**" in file_info else file_info.replace("- ", "")
                    pr_body += f"• {file_path}\n"
                pr_body += "\n"
            
            if not files_created and not files_modified:
                pr_body += "• No specific files detected\n\n"
            
            pr_body += "---\n*This pull request was automatically created by **Backspace AI Coding Agent***"
            
            # Try to create the PR using GitService
            try:
                await self._send_streaming_update(
                    state["correlation_id"], 
                    f"🔗 Creating pull request: {pr_title}...", 
                    progress=90, 
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
                file_path = clean_file_path(file_path.strip())
                
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
            
            if file_changes:
                return {
                    "file_changes": file_changes,
                    "description": f"Implementation with {len(file_changes)} file changes",
                    "success": True
                }
            else:
                return {
                    "file_changes": [],
                    "description": "No file changes detected",
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
    
    async def cleanup(self, correlation_id: str = None):
        """Cleanup resources including sandbox containers."""
        try:
            if correlation_id and self.sandbox_service:
                await self.sandbox_service.cleanup_sandbox(correlation_id)
                self.telemetry.log_event(
                    "Sandbox cleaned up after workflow completion",
                    correlation_id=correlation_id
                )
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            self.telemetry.log_error(
                e,
                context={"correlation_id": correlation_id, "operation": "cleanup"},
                correlation_id=correlation_id
            )


def create_coding_agent() -> CodingAgent:
    """Create a new coding agent instance."""
    return CodingAgent()