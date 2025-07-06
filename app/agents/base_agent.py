"""
Base LangGraph agent for the Backspace Coding Agent.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langsmith import Client

from app.core.config import settings
from app.core.telemetry import get_telemetry

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the LangGraph agent."""
    
    correlation_id: str
    repo_url: str
    prompt: str
    ai_provider: str
    
    repo_path: Optional[str]
    repo_analysis: Optional[Dict[str, Any]]
    
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    plan: Optional[Dict[str, Any]]
    current_step: Optional[str]
    steps_completed: List[str]
    
    changes_made: List[Dict[str, Any]]
    files_created: List[str]
    files_modified: List[str]
    
    branch_name: Optional[str]
    commit_hash: Optional[str]
    push_success: Optional[bool]
    
    pull_request_url: Optional[str]
    pull_request_created: Optional[bool]
    pull_request_error: Optional[str]
    
    errors: List[Dict[str, Any]]
    retry_count: int
    
    start_time: datetime
    last_update: datetime


class BaseAgent:
    """Base class for LangGraph-based agents."""
    
    def __init__(self):
        self.telemetry = get_telemetry()
        self.langsmith_client = Client() if settings.langsmith_api_key else None
        
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        
        workflow = StateGraph(AgentState)
        
        workflow.add_node("analyze_repository", self._analyze_repository_node)
        workflow.add_node("create_plan", self._create_plan_node)
        workflow.add_node("implement_changes", self._implement_changes_node)
        workflow.add_node("commit_changes", self._commit_changes_node)
        workflow.add_node("push_changes", self._push_changes_node)
        workflow.add_node("create_pull_request", self._create_pull_request_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Set entry point
        workflow.set_entry_point("analyze_repository")
        
        # Add conditional edges for the workflow
        workflow.add_conditional_edges(
            "analyze_repository",
            self._should_continue,
            {
                "continue": "create_plan",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "create_plan",
            self._should_continue,
            {
                "continue": "implement_changes",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "implement_changes",
            self._should_continue,
            {
                "continue": "commit_changes",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "commit_changes",
            self._should_continue,
            {
                "continue": "push_changes",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "push_changes",
            self._should_continue,
            {
                "continue": "create_pull_request",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "create_pull_request",
            self._should_continue,
            {
                "continue": END,
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "handle_error",
            self._should_retry,
            {
                "retry": "analyze_repository",
                "end": END
            }
        )
        
        return workflow.compile()
    
    async def _analyze_repository_node(self, state: AgentState) -> AgentState:
        """Analyze the repository structure."""
        try:
            self._log_node_start("analyze_repository", state)
            
            state["current_step"] = "analyze_repository"
            state["last_update"] = datetime.utcnow()
            
            state["steps_completed"].append("analyze_repository")
            self._log_node_success("analyze_repository", state)
            
        except Exception as e:
            state = await self._handle_node_error("analyze_repository", state, e)
            
        return state
    
    async def _create_plan_node(self, state: AgentState) -> AgentState:
        """Create an implementation plan."""
        try:
            self._log_node_start("create_plan", state)
            
            state["current_step"] = "create_plan"
            state["last_update"] = datetime.utcnow()
            
            state["steps_completed"].append("create_plan")
            self._log_node_success("create_plan", state)
            
        except Exception as e:
            state = await self._handle_node_error("create_plan", state, e)
            
        return state
    
    async def _implement_changes_node(self, state: AgentState) -> AgentState:
        """Implement the planned changes."""
        try:
            self._log_node_start("implement_changes", state)
            
            state["current_step"] = "implement_changes"
            state["last_update"] = datetime.utcnow()
            
            state["steps_completed"].append("implement_changes")
            self._log_node_success("implement_changes", state)
            
        except Exception as e:
            state = await self._handle_node_error("implement_changes", state, e)
            
        return state
    
    async def _commit_changes_node(self, state: AgentState) -> AgentState:
        """Commit the changes to git."""
        try:
            self._log_node_start("commit_changes", state)
            
            state["current_step"] = "commit_changes"
            state["last_update"] = datetime.utcnow()
            
            state["steps_completed"].append("commit_changes")
            self._log_node_success("commit_changes", state)
            
        except Exception as e:
            state = await self._handle_node_error("commit_changes", state, e)
            
        return state
    
    async def _push_changes_node(self, state: AgentState) -> AgentState:
        """Push changes to the remote repository."""
        try:
            self._log_node_start("push_changes", state)
            
            state["current_step"] = "push_changes"
            state["last_update"] = datetime.utcnow()
            
            state["steps_completed"].append("push_changes")
            self._log_node_success("push_changes", state)
            
        except Exception as e:
            state = await self._handle_node_error("push_changes", state, e)
            
        return state
    
    async def _create_pull_request_node(self, state: AgentState) -> AgentState:
        """Create a pull request."""
        try:
            self._log_node_start("create_pull_request", state)
            
            state["current_step"] = "create_pull_request"
            state["last_update"] = datetime.utcnow()
            
            state["steps_completed"].append("create_pull_request")
            self._log_node_success("create_pull_request", state)
            
        except Exception as e:
            state = await self._handle_node_error("create_pull_request", state, e)
            
        return state
    
    async def _handle_error_node(self, state: AgentState) -> AgentState:
        """Handle errors and decide on retry strategy."""
        try:
            self._log_node_start("handle_error", state)
            
            state["current_step"] = "handle_error"
            state["last_update"] = datetime.utcnow()
            
            if state["errors"]:
                latest_error = state["errors"][-1]
                logger.error(f"Agent error: {latest_error}")
                
                state["retry_count"] += 1
                
                if state["retry_count"] < 3:
                    state["errors"] = []
                    logger.info(f"Retrying operation (attempt {state['retry_count']})")
                else:
                    logger.error("Max retries exceeded")
            
            state["steps_completed"].append("handle_error")
            self._log_node_success("handle_error", state)
            
        except Exception as e:
            logger.error(f"Error in error handler: {e}")
            
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if we should continue or handle an error."""
        if state["errors"]:
            return "error"
        return "continue"
    
    def _should_retry(self, state: AgentState) -> str:
        """Determine if we should retry or end."""
        if state["retry_count"] < 3:
            return "retry"
        return "end"
    
    async def _handle_node_error(self, node_name: str, state: AgentState, error: Exception) -> AgentState:
        """Handle errors in a node."""
        error_info = {
            "node": node_name,
            "error": str(error),
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": state["correlation_id"]
        }
        
        state["errors"].append(error_info)
        state["last_update"] = datetime.utcnow()
        
        self._log_node_error(node_name, state, error)
        
        return state
    
    def _log_node_start(self, node_name: str, state: AgentState):
        """Log the start of a node execution."""
        self.telemetry.log_event(
            f"Node started: {node_name}",
            correlation_id=state["correlation_id"],
            node=node_name,
            step=state["current_step"]
        )
        
        if self.langsmith_client:
            pass
    
    def _log_node_success(self, node_name: str, state: AgentState):
        """Log the successful completion of a node."""
        self.telemetry.log_event(
            f"Node completed: {node_name}",
            correlation_id=state["correlation_id"],
            node=node_name,
            step=state["current_step"]
        )
        
        if self.langsmith_client:
            pass
    
    def _log_node_error(self, node_name: str, state: AgentState, error: Exception):
        """Log an error in a node."""
        self.telemetry.log_error(
            error,
            context={
                "correlation_id": state["correlation_id"],
                "node": node_name,
                "step": state["current_step"]
            },
            correlation_id=state["correlation_id"]
        )
        
        if self.langsmith_client:
            pass
    
    async def run(self, correlation_id: str, repo_url: str, prompt: str, ai_provider: str = "openai") -> Dict[str, Any]:
        """Run the agent workflow."""
        
        initial_state = AgentState(
            correlation_id=correlation_id,
            repo_url=repo_url,
            prompt=prompt,
            ai_provider=ai_provider,
            repo_path=None,
            repo_analysis=None,
            messages=[],
            plan=None,
            current_step=None,
            steps_completed=[],
            changes_made=[],
            files_created=[],
            files_modified=[],
            branch_name=None,
            commit_hash=None,
            push_success=None,
            pull_request_url=None,
            pull_request_created=None,
            pull_request_error=None,
            errors=[],
            retry_count=0,
            start_time=datetime.utcnow(),
            last_update=datetime.utcnow()
        )
        
        self.telemetry.log_event(
            "Agent workflow started",
            correlation_id=correlation_id,
            repo_url=repo_url,
            prompt=prompt,
            ai_provider=ai_provider
        )
        
        try:
            final_state = await self.graph.ainvoke(initial_state)
            
            self.telemetry.log_event(
                "Agent workflow completed",
                correlation_id=correlation_id,
                success=len(final_state["errors"]) == 0,
                steps_completed=len(final_state["steps_completed"]),
                errors_count=len(final_state["errors"])
            )
            
            return {
                "success": len(final_state["errors"]) == 0,
                "correlation_id": correlation_id,
                "final_state": final_state,
                "errors": final_state["errors"]
            }
            
        except Exception as e:
            self.telemetry.log_error(
                e,
                context={"correlation_id": correlation_id},
                correlation_id=correlation_id
            )
            
            return {
                "success": False,
                "correlation_id": correlation_id,
                "error": str(e)
            } 