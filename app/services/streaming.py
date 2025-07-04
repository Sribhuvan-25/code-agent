"""
Streaming service for Server-Sent Events.
"""

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Any, Optional

from fastapi import Request
from sse_starlette.sse import EventSourceResponse
import janus

from app.core.telemetry import get_telemetry
from app.models.schemas import StreamEvent, StreamEventType

# Singleton instance and getter
_streaming_service_instance = None

def get_streaming_service():
    """Get the singleton StreamingService instance."""
    global _streaming_service_instance
    if _streaming_service_instance is None:
        _streaming_service_instance = StreamingService()
    return _streaming_service_instance

class StreamingService:
    """Service for handling Server-Sent Events streaming. Singleton pattern enforced via get_streaming_service(). Uses janus.Queue for thread-safe cross-event-loop streaming."""
    
    def __init__(self):
        self.telemetry = get_telemetry()
        self.active_connections: Dict[str, janus.Queue] = {}
    
    async def create_event_stream(
        self, 
        correlation_id: str, 
        request: Request
    ) -> EventSourceResponse:
        """
        Create a Server-Sent Events stream using janus.Queue for thread-safe event delivery.
        
        Args:
            correlation_id: Unique identifier for the stream
            request: FastAPI request object
            
        Returns:
            EventSourceResponse for streaming
        """
        event_queue = janus.Queue()
        self.active_connections[correlation_id] = event_queue
        
        # File-based debug output for stream creation
        try:
            with open(f"/tmp/streaming_create_{correlation_id[:8]}.txt", "w") as f:
                f.write(f"{time.time()}: Event stream created for {correlation_id}\n")
                f.write(f"Active connections count: {len(self.active_connections)}\n")
        except Exception as debug_error:
            print(f"DEBUG: Failed to write create debug file: {debug_error}")
        
        async def event_generator() -> AsyncGenerator[str, None]:
            """Generate events for the stream."""
            # File-based debug output for stream start
            try:
                with open(f"/tmp/streaming_start_{correlation_id[:8]}.txt", "w") as f:
                    f.write(f"{time.time()}: Event generator started for {correlation_id}\n")
                    f.write(f"Stream will stay open until explicitly closed\n")
            except Exception as debug_error:
                print(f"DEBUG: Failed to write start debug file: {debug_error}")
            
            try:
                while True:
                    try:
                        # Wait for next event with timeout
                        event = await asyncio.wait_for(
                            event_queue.async_q.get(), 
                            timeout=1.0
                        )
                        print(f"[DEBUG] Event dequeued for {correlation_id}: {event}")
                        
                        # File-based debug output for event dequeue
                        try:
                            with open(f"/tmp/streaming_dequeue_{correlation_id[:8]}.txt", "a") as f:
                                f.write(f"{time.time()}: Event dequeued for {correlation_id}\n")
                                f.write(f"Event type: {event.type if event else 'None'}\n")
                                f.write(f"Event message: {event.message if event else 'None'}\n")
                                f.write(f"Queue size after get: {event_queue.async_q.qsize()}\n")
                        except Exception as debug_error:
                            print(f"DEBUG: Failed to write dequeue debug file: {debug_error}")
                        
                        # Check for stream termination
                        if event is None:
                            # File-based debug output for explicit termination
                            try:
                                with open(f"/tmp/streaming_terminate_{correlation_id[:8]}.txt", "w") as f:
                                    f.write(f"{time.time()}: Stream explicitly terminated for {correlation_id}\n")
                            except Exception as debug_error:
                                print(f"DEBUG: Failed to write terminate debug file: {debug_error}")
                            break
                        
                        # Serialize event
                        event_data = self._serialize_event(event)
                        yield event_data
                        
                    except asyncio.TimeoutError:
                        # Send keepalive
                        yield "data: {\"type\": \"keepalive\", \"timestamp\": \"" + \
                              str(asyncio.get_event_loop().time()) + "\"}\n\n"
                        
            except Exception as e:
                # File-based error debug output
                try:
                    with open(f"/tmp/streaming_dequeue_{correlation_id[:8]}_error.txt", "w") as f:
                        f.write(f"{time.time()}: event_generator failed for {correlation_id}\n")
                        f.write(f"Error: {str(e)}\n")
                        f.write(f"Error type: {type(e).__name__}\n")
                except Exception as debug_error:
                    print(f"DEBUG: Failed to write dequeue error debug file: {debug_error}")
                
                self.telemetry.log_error(
                    e,
                    context={"correlation_id": correlation_id},
                    correlation_id=correlation_id
                )
                
                # Send error event
                error_event = StreamEvent(
                    type=StreamEventType.ERROR,
                    message=f"Stream error: {str(e)}",
                    error=str(e),
                    error_type=type(e).__name__,
                    correlation_id=correlation_id
                )
                yield self._serialize_event(error_event)
                
            finally:
                # Cleanup
                if correlation_id in self.active_connections:
                    self.active_connections[correlation_id].close()
                    del self.active_connections[correlation_id]
                    
                    # File-based debug output for cleanup
                    try:
                        with open(f"/tmp/streaming_cleanup_{correlation_id[:8]}.txt", "w") as f:
                            f.write(f"{time.time()}: Stream cleaned up for {correlation_id}\n")
                    except Exception as debug_error:
                        print(f"DEBUG: Failed to write cleanup debug file: {debug_error}")
        
        return EventSourceResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            }
        )
    
    async def send_event(
        self, 
        correlation_id: str, 
        event: StreamEvent
    ) -> None:
        """
        Send an event to a specific stream.
        
        Args:
            correlation_id: Stream identifier
            event: Event to send
        """
        if correlation_id in self.active_connections:
            try:
                print(f"[DEBUG] Event enqueued for {correlation_id}: {event}")
                
                # File-based debug output
                try:
                    with open(f"/tmp/streaming_send_{correlation_id[:8]}.txt", "a") as f:
                        f.write(f"{time.time()}: send_event called for {correlation_id}\n")
                        f.write(f"Event type: {event.type}\n")
                        f.write(f"Event message: {event.message}\n")
                        f.write(f"Queue size before put: {self.active_connections[correlation_id].sync_q.qsize()}\n")
                except Exception as debug_error:
                    print(f"DEBUG: Failed to write send debug file: {debug_error}")
                
                self.active_connections[correlation_id].sync_q.put(event)
                
                # File-based debug output after put
                try:
                    with open(f"/tmp/streaming_send_{correlation_id[:8]}.txt", "a") as f:
                        f.write(f"Queue size after put: {self.active_connections[correlation_id].sync_q.qsize()}\n")
                        f.write(f"Event put successfully\n")
                except Exception as debug_error:
                    print(f"DEBUG: Failed to write send debug file: {debug_error}")
                
                # Log the event
                self.telemetry.log_event(
                    "Event sent",
                    correlation_id=correlation_id,
                    event_type=event.type,
                    message=event.message
                )
                
            except Exception as e:
                # File-based error debug output
                try:
                    with open(f"/tmp/streaming_send_{correlation_id[:8]}_error.txt", "w") as f:
                        f.write(f"{time.time()}: send_event failed for {correlation_id}\n")
                        f.write(f"Error: {str(e)}\n")
                        f.write(f"Error type: {type(e).__name__}\n")
                except Exception as debug_error:
                    print(f"DEBUG: Failed to write send error debug file: {debug_error}")
                
                self.telemetry.log_error(
                    e,
                    context={
                        "correlation_id": correlation_id,
                        "event_type": event.type
                    },
                    correlation_id=correlation_id
                )
        else:
            # File-based debug output for missing connection
            try:
                with open(f"/tmp/streaming_send_{correlation_id[:8]}_missing.txt", "w") as f:
                    f.write(f"{time.time()}: send_event called but connection {correlation_id} not found\n")
                    f.write(f"Active connections: {list(self.active_connections.keys())}\n")
            except Exception as debug_error:
                print(f"DEBUG: Failed to write missing connection debug file: {debug_error}")
    
    async def broadcast_event(self, event: StreamEvent) -> None:
        """
        Broadcast an event to all active streams.
        
        Args:
            event: Event to broadcast
        """
        for correlation_id in list(self.active_connections.keys()):
            await self.send_event(correlation_id, event)
    
    async def close_stream(self, correlation_id: str) -> None:
        """
        Close a specific stream.
        
        Args:
            correlation_id: Stream identifier
        """
        if correlation_id in self.active_connections:
            try:
                # Send termination signal
                self.active_connections[correlation_id].sync_q.put(None)
                
                self.telemetry.log_event(
                    "Stream closed",
                    correlation_id=correlation_id
                )
                
            except Exception as e:
                self.telemetry.log_error(
                    e,
                    context={"correlation_id": correlation_id},
                    correlation_id=correlation_id
                )
    
    def _serialize_event(self, event: StreamEvent) -> str:
        """
        Serialize an event for Server-Sent Events format.
        
        Args:
            event: Event to serialize
            
        Returns:
            Formatted SSE string
        """
        try:
            event_dict = event.model_dump(exclude_none=True)
            
            # Convert datetime to ISO string
            if "timestamp" in event_dict:
                event_dict["timestamp"] = event_dict["timestamp"].isoformat()
            
            # Format as SSE
            data = json.dumps(event_dict, default=str)
            return f"data: {data}\n\n"
            
        except Exception as e:
            self.telemetry.log_error(
                e,
                context={"event": event.model_dump()},
                correlation_id=event.correlation_id
            )
            # Return a fallback SSE message
            return f"data: {{\"type\": \"error\", \"message\": \"Event serialization failed: {str(e)}\"}}\n\n"
    
    def get_active_connections(self) -> Dict[str, Any]:
        """
        Get information about active connections.
        
        Returns:
            Dictionary of active connections
        """
        return {
            correlation_id: {
                "queue_size": queue.sync_q.qsize(),
                "connected": True
            }
            for correlation_id, queue in self.active_connections.items()
        }
    
    async def send_ai_message(
        self, 
        correlation_id: str, 
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send an AI message event.
        
        Args:
            correlation_id: Stream identifier
            message: AI message
            context: Additional context
        """
        event = StreamEvent(
            type=StreamEventType.AI_MESSAGE,
            message=message,
            correlation_id=correlation_id,
            context=context
        )
        await self.send_event(correlation_id, event)
    
    async def send_tool_event(
        self,
        correlation_id: str,
        tool_type: str,
        command: Optional[str] = None,
        output: Optional[str] = None,
        filepath: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send a tool execution event.
        
        Args:
            correlation_id: Stream identifier
            tool_type: Type of tool (bash, read, edit, git)
            command: Command executed
            output: Tool output
            filepath: File path (for file operations)
            context: Additional context
        """
        event_type_map = {
            "bash": StreamEventType.TOOL_BASH,
            "read": StreamEventType.TOOL_READ,
            "edit": StreamEventType.TOOL_EDIT,
            "git": StreamEventType.TOOL_GIT,
        }
        
        event = StreamEvent(
            type=event_type_map.get(tool_type, StreamEventType.TOOL_BASH),
            command=command,
            output=output,
            filepath=filepath,
            correlation_id=correlation_id,
            context=context
        )
        await self.send_event(correlation_id, event)
    
    async def send_progress(
        self,
        correlation_id: str,
        progress: float,
        step: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send a progress update event.
        
        Args:
            correlation_id: Stream identifier
            progress: Progress percentage (0-100)
            step: Current step description
            context: Additional context
        """
        event = StreamEvent(
            type=StreamEventType.PROGRESS,
            progress=progress,
            step=step,
            correlation_id=correlation_id,
            context=context
        )
        await self.send_event(correlation_id, event)
    
    async def send_error(
        self,
        correlation_id: str,
        error: str,
        error_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send an error event.
        
        Args:
            correlation_id: Stream identifier
            error: Error message
            error_type: Error type
            context: Additional context
        """
        event = StreamEvent(
            type=StreamEventType.ERROR,
            error=error,
            error_type=error_type,
            correlation_id=correlation_id,
            context=context
        )
        await self.send_event(correlation_id, event)
    
    async def send_success(
        self,
        correlation_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send a success event.
        
        Args:
            correlation_id: Stream identifier
            message: Success message
            context: Additional context
        """
        event = StreamEvent(
            type=StreamEventType.SUCCESS,
            message=message,
            correlation_id=correlation_id,
            context=context
        )
        await self.send_event(correlation_id, event) 