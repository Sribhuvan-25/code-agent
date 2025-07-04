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
            ("human", """Implement the planned changes for the following request:

Repository URL: {repo_url}
Request: {prompt}
Plan: {plan}

Current repository state:
{repo_analysis}

CRITICAL: You MUST generate actual file changes, not just describe what to do.

For each file you need to create or modify, use this exact format:

**For creating a new file:**
```
Create file `src/components/NewComponent.jsx` with the following content:

```jsx
import React from 'react';

const NewComponent = () => {{
  return (
    <div>
      <h2>New Component</h2>
      <p>This is a new component.</p>
    </div>
  );
}};

export default NewComponent;
```

**For modifying an existing file:**
```
Modify `src/App.jsx` to add the new component:

```jsx
import React from 'react';
import NewComponent from './components/NewComponent';

function App() {{
  return (
    <div className="App">
      <header className="App-header">
        <h1>My Website</h1>
      </header>
      <main>
        <NewComponent />
      </main>
    </div>
  );
}}

export default App;
```

IMPORTANT RULES:
1. Use relative paths (e.g., 'src/components/Component.jsx', NOT 'workspace/Personal-Website/src/components/Component.jsx')
2. Follow existing file extensions (.js vs .jsx)
3. Preserve existing code structure when modifying files
4. Generate complete, working code
5. Use the exact format shown above with backticks and file paths

Now implement the changes:"""),
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
            
            read_tool = next(t for t in self.tools if t.name == "read_file")
            write_tool = next(t for t in self.tools if t.name == "write_file")
            execute_tool = next(t for t in self.tools if t.name == "execute_command")
            
            create_branch_tool = next(t for t in self.tools if t.name == "create_branch")
            
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
            
            implementation_prompt = self.implementation_prompt.format_messages(
                repo_url=state["repo_url"],
                prompt=state["prompt"],
                plan=json.dumps(state["plan"], indent=2),
                repo_analysis=json.dumps(state["repo_analysis"], indent=2),
                tools="Available tools: read_file, write_file, execute_command",
                chat_history=state["messages"],
                agent_scratchpad=""
            )
            
            response = await self.llm.ainvoke(implementation_prompt)
            
            print(f"\n===== BACKSPACE DEBUG: LLM Implementation Response =====")
            print(f"Response length: {len(response.content)} characters")
            print(f"First 500 chars: {response.content[:500]}...")
            print(f"Last 500 chars: {response.content[-500:]}...")
            print(f"\n===== FULL LLM RESPONSE =====")
            print(response.content)
            print(f"===== END FULL LLM RESPONSE =====")
            
            implementation_result = self._parse_implementation(response.content)
            
            print(f"\n===== BACKSPACE DEBUG: Parsed Implementation Result =====")
            print(f"Implementation result keys: {list(implementation_result.keys())}")
            print(f"File changes count: {len(implementation_result.get('file_changes', []))}")
            for i, change in enumerate(implementation_result.get("file_changes", [])):
                print(f"Change {i+1}: {change}")
            
            changes_made = []
            changed_files = []
            
            for change in implementation_result.get("file_changes", []):
                print(f"\n===== BACKSPACE DEBUG: Processing change {change['action']} for {change['file_path']} =====")
                
                if not change["file_path"].startswith("/"):
                    absolute_file_path = os.path.join(state["repo_path"], change["file_path"])
                else:
                    absolute_file_path = change["file_path"]
                
                print(f"Relative path: {change['file_path']}")
                print(f"Absolute path: {absolute_file_path}")
                
                if change["action"] == "create":
                    print(f"Creating file: {change['file_path']}")
                    print(f"Content length: {len(change['content'])} characters")
                    print(f"First 100 chars: {change['content'][:100]}...")
                    
                    await write_tool.ainvoke({
                        "correlation_id": state["correlation_id"],
                        "repo_path": state["repo_path"],
                        "file_path": change["file_path"],  # Use relative path for git service
                        "content": change["content"]
                    })
                    
                    print(f"\n===== BACKSPACE DEBUG: After creating {change['file_path']} =====")
                    await execute_tool.ainvoke({
                        "correlation_id": state["correlation_id"],
                        "repo_path": state["repo_path"],
                        "command": "pwd",
                        "working_dir": state["repo_path"]
                    })
                    await execute_tool.ainvoke({
                        "correlation_id": state["correlation_id"],
                        "repo_path": state["repo_path"],
                        "command": f"ls -la '{os.path.dirname(absolute_file_path)}'",
                        "working_dir": state["repo_path"]
                    })
                    await execute_tool.ainvoke({
                        "correlation_id": state["correlation_id"],
                        "repo_path": state["repo_path"],
                        "command": f"test -f '{absolute_file_path}' && echo 'File exists' || echo 'File does not exist'",
                        "working_dir": state["repo_path"]
                    })
                    
                    changes_made.append({
                        "action": "created",
                        "file_path": change["file_path"],
                        "description": change.get("description", "File created by agent")
                    })
                    changed_files.append(change["file_path"])
                    
                elif change["action"] == "modify":
                    print(f"Modifying file: {change['file_path']}")
                    
                    # Try to read existing content, but if file doesn't exist, treat it as create
                    try:
                        existing_content = await read_tool.ainvoke({
                            "correlation_id": state["correlation_id"],
                            "repo_path": state["repo_path"],
                            "file_path": change["file_path"]
                        })
                        print(f"Existing content length: {len(existing_content)}")
                    except Exception as e:
                        print(f"File {change['file_path']} doesn't exist, treating as create: {e}")
                        change["action"] = "create"
                        existing_content = ""
                    
                    new_content = change["content"]
                    print(f"New content length: {len(new_content)}")
                    
                    await write_tool.ainvoke({
                        "correlation_id": state["correlation_id"],
                        "repo_path": state["repo_path"],
                        "file_path": change["file_path"],
                        "content": new_content
                    })
                    
                    print(f"File modified successfully: {change['file_path']}")
                    
                    await execute_tool.ainvoke({
                        "correlation_id": state["correlation_id"],
                        "repo_path": state["repo_path"],
                        "command": f"ls -la '{absolute_file_path}'",
                        "working_dir": state["repo_path"]
                    })
                    
                    changes_made.append({
                        "action": "modified",
                        "file_path": change["file_path"],
                        "description": change.get("description", "File modified by agent")
                    })
                    changed_files.append(change["file_path"])
            
            print("\n===== BACKSPACE DEBUG: COMPREHENSIVE FILE SYSTEM INSPECTION =====")
            
            print("\n--- Current Working Directory ---")
            await execute_tool.ainvoke({
                "correlation_id": state["correlation_id"],
                "repo_path": state["repo_path"],
                "command": "pwd",
                "working_dir": state["repo_path"]
            })
            
            print("\n--- All Files in Repository ---")
            await execute_tool.ainvoke({
                "correlation_id": state["correlation_id"],
                "repo_path": state["repo_path"],
                "command": "find . -type f | head -20",
                "working_dir": state["repo_path"]
            })
            
            print("\n--- Git Status ---")
            await execute_tool.ainvoke({
                "correlation_id": state["correlation_id"],
                "repo_path": state["repo_path"],
                "command": "git status",
                "working_dir": state["repo_path"]
            })
            
            print("\n--- Git Diff ---")
            await execute_tool.ainvoke({
                "correlation_id": state["correlation_id"],
                "repo_path": state["repo_path"],
                "command": "git diff",
                "working_dir": state["repo_path"]
            })
            
            print("\n--- Git Diff --cached ---")
            await execute_tool.ainvoke({
                "correlation_id": state["correlation_id"],
                "repo_path": state["repo_path"],
                "command": "git diff --cached",
                "working_dir": state["repo_path"]
            })
            
            for file_path in changed_files:
                print(f"\n--- Checking {file_path} ---")
                await execute_tool.ainvoke({
                    "correlation_id": state["correlation_id"],
                    "repo_path": state["repo_path"],
                    "command": f"ls -la '{file_path}'",
                    "working_dir": state["repo_path"]
                })
                await execute_tool.ainvoke({
                    "correlation_id": state["correlation_id"],
                    "repo_path": state["repo_path"],
                    "command": f"wc -l '{file_path}'",
                    "working_dir": state["repo_path"]
                })
                await execute_tool.ainvoke({
                    "correlation_id": state["correlation_id"],
                    "repo_path": state["repo_path"],
                    "command": f"head -5 '{file_path}'",
                    "working_dir": state["repo_path"]
                })
            
            print("\n--- Git Tracked Files ---")
            await execute_tool.ainvoke({
                "correlation_id": state["correlation_id"],
                "repo_path": state["repo_path"],
                "command": "git ls-files | grep -E '(ContactForm|Contact|App|index)' || echo 'No matching files found'",
                "working_dir": state["repo_path"]
            })
            
            print("\n===== END COMPREHENSIVE DEBUG =====\n")
            
            state["implementation_result"] = implementation_result
            state["changes_made"] = changes_made
            state["messages"].append(response)
            
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
            
            commit_prompt = f"""Based on the following implementation plan and changes made, generate a concise and descriptive commit message:

Plan Summary: {state['plan'].get('summary', 'No summary available')}
Changes Made: {state.get('changes_made', [])}

Generate a commit message that follows conventional commit format and describes what was implemented. Keep it under 50 characters for the subject line.

Commit message:"""
            
            commit_response = await self.llm.ainvoke(commit_prompt)
            commit_message = commit_response.content.strip()
            commit_message = commit_message.replace('"', '').replace("'", "").split('\n')[0][:50]
            
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
            state["push_success"] = push_result.get("success", False)
            
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
            
            # Patterns to match file creation
            create_patterns = [
                r'create.*?file.*?`([^`]+)`.*?```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n(.*?)```',
                r'create.*?`([^`]+)`.*?with.*?content.*?```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n(.*?)```',
                r'new.*?file.*?`([^`]+)`.*?```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n(.*?)```',
                # Also match without backticks
                r'create.*?file.*?([^\s]+\.(?:js|jsx|ts|tsx|css|html|json|md|txt|cjs|mjs|yml|yaml)).*?```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n(.*?)```',
                # Handle case where instruction is inside the code block
                r'```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n.*?create.*?file.*?`([^`]+)`.*?\n(.*?)```',
                r'```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n.*?// Create file.*?`([^`]+)`.*?\n(.*?)```'
            ]
            
            for pattern in create_patterns:
                matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    file_path = match.group(1).strip()
                    file_content = match.group(2).strip()
                    
                    if not is_valid_file_path(file_path):
                        continue
                    
                    file_path = clean_file_path(file_path)
                    
                    file_content = clean_code_content(file_content)
                    
                    if file_content:  # Only add if we have actual content
                        file_changes.append({
                            "action": "create",
                            "file_path": file_path,
                            "content": file_content,
                            "description": f"Create {file_path} with provided content"
                        })
            
            # Patterns to match file modifications
            modify_patterns = [
                r'modify.*?`([^`]+)`.*?```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n(.*?)```',
                r'update.*?`([^`]+)`.*?```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n(.*?)```',
                r'change.*?`([^`]+)`.*?```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n(.*?)```',
                # Also match without backticks
                r'modify.*?([^\s]+\.(?:js|jsx|ts|tsx|css|html|json|md|txt|cjs|mjs|yml|yaml)).*?```(?:jsx?|js|tsx?|ts|css|html|json|md|txt|cjs|mjs|yml|yaml)\s*\n(.*?)```'
            ]
            
            for pattern in modify_patterns:
                matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    file_path = match.group(1).strip()
                    file_content = match.group(2).strip()
                    
                    if not is_valid_file_path(file_path):
                        continue
                    
                    file_path = clean_file_path(file_path)
                    
                    file_content = clean_code_content(file_content)
                    
                    if file_content:
                        file_changes.append({
                            "action": "modify",
                            "file_path": file_path,
                            "content": file_content,
                            "description": f"Modify {file_path} with new content"
                        })
            
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
    
    def _apply_modifications(self, existing_content: str, modifications: List[Dict[str, Any]]) -> str:
        """Apply modifications to existing content."""
        return existing_content
    
    def _extract_changes(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract changes from the agent execution result."""
        changes = []
        
        # This would need to be implemented based on how the agent returns results
        # For now, return a simple structure
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if step[0].tool == "write_file":
                    changes.append({
                        "action": "created",
                        "file_path": step[1].get("file_path", "unknown"),
                        "description": "File created by agent"
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