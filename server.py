"""
FastAPI Server for Logistics Agent
Provides REST API endpoints to interact with the LangGraph agent
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, AsyncGenerator
from langchain_core.messages import HumanMessage, AIMessage
import uvicorn
import json
import asyncio
from logistics_agent import graph

app = FastAPI(
    title="SeeSea Logistics Agent API",
    description="API for querying vessel arrivals data through an AI agent",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User's question or query", min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "How many container ships passed through in January 2026?",
                "session_id": "user-123"
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Agent's response")
    session_id: Optional[str] = Field(None, description="Session ID")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tools called during processing")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "In January 2026, there were 123 container ships...",
                "session_id": "user-123",
                "tool_calls": [
                    {"tool": "query_vessel_data", "args": {"start_date": "2026-01-01", "end_date": "2026-01-31"}}
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    agent_ready: bool


# ============================================================================
# Session Management (Simple in-memory store)
# ============================================================================

# In production, use Redis or a database
sessions: Dict[str, List[Any]] = {}


def get_session_messages(session_id: Optional[str]) -> List[Any]:
    """Get messages for a session"""
    if not session_id:
        return []
    return sessions.get(session_id, [])


def save_session_messages(session_id: Optional[str], messages: List[Any]):
    """Save messages for a session"""
    if session_id:
        sessions[session_id] = messages


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "SeeSea Logistics Agent API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test that the graph is compiled and ready
        agent_ready = graph is not None
        return HealthResponse(
            status="healthy",
            agent_ready=agent_ready
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            agent_ready=False
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the logistics agent (non-streaming).

    The agent can answer questions about vessel traffic data at the Bab el-Mandeb strait,
    including historical trends, comparisons between time periods, and vessel type breakdowns.
    """
    try:
        # Get session messages
        previous_messages = get_session_messages(request.session_id)

        # Add new user message
        new_message = HumanMessage(content=request.message)
        all_messages = previous_messages + [new_message]

        # Invoke the agent
        result = graph.invoke({
            "messages": all_messages
        })

        # Extract response
        messages = result.get("messages", [])
        if not messages:
            raise HTTPException(status_code=500, detail="No response from agent")

        # Get the final AI message
        final_message = messages[-1]
        response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)

        # Extract tool calls for transparency
        tool_calls = []
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "tool": tc.get("name", "unknown"),
                        "args": tc.get("args", {})
                    })

        # Save session messages
        save_session_messages(request.session_id, messages)

        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            tool_calls=tool_calls if tool_calls else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Chat with the logistics agent using Server-Sent Events (SSE) for streaming responses.

    Returns a stream of events:
    - event: tool_call - When agent calls a tool
    - event: tool_result - Result from tool execution
    - event: content - AI response content (streamed token by token)
    - event: done - Stream complete
    - event: error - Error occurred
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Get session messages
            previous_messages = get_session_messages(request.session_id)

            # Add new user message
            new_message = HumanMessage(content=request.message)
            all_messages = previous_messages + [new_message]

            # Stream events from the graph
            async for event in graph.astream_events(
                {"messages": all_messages},
                version="v2"
            ):
                event_type = event.get("event")

                # Tool call started
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk", {})

                    # Handle tool calls
                    if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                        for tc_chunk in chunk.tool_call_chunks:
                            yield f"event: tool_call\n"
                            yield f"data: {json.dumps({'name': tc_chunk.get('name', ''), 'args': tc_chunk.get('args', {})})}\n\n"

                    # Handle content streaming
                    if hasattr(chunk, "content") and chunk.content:
                        yield f"event: content\n"
                        yield f"data: {json.dumps({'content': chunk.content})}\n\n"
                        await asyncio.sleep(0)  # Allow other tasks to run

                # Tool execution result
                elif event_type == "on_tool_end":
                    tool_output = event.get("data", {}).get("output", "")
                    tool_name = event.get("name", "unknown")
                    yield f"event: tool_result\n"
                    yield f"data: {json.dumps({'tool': tool_name, 'result': str(tool_output)})}\n\n"

            # Get final state to save session
            final_result = graph.invoke({"messages": all_messages})
            save_session_messages(request.session_id, final_result.get("messages", []))

            # Send completion event
            yield f"event: done\n"
            yield f"data: {json.dumps({'session_id': request.session_id})}\n\n"

        except Exception as e:
            yield f"event: error\n"
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.post("/clear-session")
async def clear_session(session_id: str):
    """Clear a session's conversation history"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    return {"message": "Session not found"}


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "sessions": list(sessions.keys()),
        "count": len(sessions)
    }


# ============================================================================
# Direct Query Endpoints (Optional - for non-conversational access)
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for direct data query"""
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    vessel_type: Optional[str] = Field(None, description="Vessel type: container, dry_bulk, tanker, general_cargo, roro")
    aggregate: Optional[str] = Field("none", description="Aggregation: none, daily_avg, monthly_avg, total")


@app.post("/query/direct")
async def query_direct(request: QueryRequest):
    """
    Direct data query without conversational context.
    Useful for programmatic access to vessel data.
    """
    try:
        from logistics_agent import query_vessel_data

        result = query_vessel_data(
            start_date=request.start_date,
            end_date=request.end_date,
            vessel_type=request.vessel_type,
            aggregate=request.aggregate
        )

        return {
            "data": result,
            "query_params": request.model_dump()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying data: {str(e)}")


@app.get("/query/summary")
async def get_summary():
    """Get dataset summary"""
    try:
        from logistics_agent import get_data_summary

        summary = get_data_summary()
        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting summary: {str(e)}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the FastAPI server"""
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
