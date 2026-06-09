"""
app/main.py — FastAPI Entry Point (Architecture v2 with SSE & Persistence)

This version adds:
- SQLite persistence for Chat Sessions and Messages
- Server-Sent Events (SSE) streaming endpoint (/api/query/stream)
- REST endpoints to fetch history
"""

import json
import logging
import uuid
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .config import settings
from .database import engine, Base, get_db
from .models import ChatSession, ChatMessage

logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Elytics Analytics API",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # If none, creates a new session

class SessionCreate(BaseModel):
    title: str

# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------
@app.get("/", tags=["Health"])
def health_check():
    from . import graph as graph_module
    return {"status": "online", "langgraph": "ready" if graph_module.analytics_graph else "not_ready"}

@app.get("/api/sessions", tags=["History"])
def get_sessions(db: Session = Depends(get_db)):
    sessions = db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()
    return [{"id": s.id, "title": s.title, "created_at": s.created_at} for s in sessions]

@app.post("/api/sessions", tags=["History"])
def create_session(req: SessionCreate, db: Session = Depends(get_db)):
    session_id = str(uuid.uuid4())
    db_session = ChatSession(id=session_id, title=req.title)
    db.add(db_session)
    db.commit()
    return {"id": session_id, "title": req.title}

@app.get("/api/sessions/{session_id}/messages", tags=["History"])
def get_messages(session_id: str, db: Session = Depends(get_db)):
    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc()).all()
    res = []
    for m in messages:
        res.append({
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "metadata": m.metadata_json,
            "created_at": m.created_at
        })
    return res

@app.delete("/api/sessions/{session_id}", tags=["History"])
def delete_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        db.delete(session)
        db.commit()
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.post("/api/query/stream", tags=["Analytics"])
async def submit_query_stream(req: Request, db: Session = Depends(get_db)):
    """
    SSE Endpoint. Expects JSON body with 'query' and optional 'session_id'.
    Yields Server-Sent Events showing progress, and finally the results.
    """
    body = await req.json()
    user_query = body.get("query", "").strip()
    session_id = body.get("session_id")

    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not session_id:
        session_id = str(uuid.uuid4())
        # Auto-generate a title from the query
        title = user_query[:30] + "..." if len(user_query) > 30 else user_query
        db_session = ChatSession(id=session_id, title=title)
        db.add(db_session)
        db.commit()

    # Save User Message
    user_msg = ChatMessage(session_id=session_id, role="user", content=user_query)
    db.add(user_msg)
    db.commit()

    from . import graph as graph_module
    graph = graph_module.analytics_graph

    if graph is None:
        try:
            graph_module.analytics_graph = graph_module.build_graph()
            graph = graph_module.analytics_graph
        except Exception as e:
            raise HTTPException(status_code=500, detail="Backend graph startup failed.")

    initial_state = {
        "user_query": user_query,
        "intent": "",
        "plan": {},
        "schema_context": {},
        "generated_sql": "",
        "sql_validation": {},
        "query_results": [],
        "generated_python": "",
        "execution_results": {},
        "charts": [],
        "insights": [],
        "error": "",
        "step_log": [f"🚀 Pipeline started | query='{user_query[:60]}'"],
    }

    async def event_generator():
        # Stream events from LangGraph
        final_state = None
        
        # We use graph.stream which yields updates from each node
        for event in graph.stream(initial_state):
            # event is a dict mapping node_name -> state_updates
            for node_name, state_update in event.items():
                final_state = state_update # keep track of the latest state update
                
                # Yield a progress event
                if "step_log" in state_update and state_update["step_log"]:
                    latest_log = state_update["step_log"][-1]
                    yield f"event: progress\ndata: {json.dumps({'node': node_name, 'log': latest_log})}\n\n"
                
                if "error" in state_update and state_update["error"]:
                    yield f"event: error\ndata: {json.dumps({'error': state_update['error']})}\n\n"

        # Graph execution is complete. Find the final accumulated state from the last node
        # Wait, graph.stream yields partial updates. We need to invoke it or accumulate state.
        # It's safer to just run graph.invoke() in a thread and yield progress, OR manually accumulate.
        # Let's just use graph.stream and accumulate the state manually.
        pass

    # Wait, graph.stream in LangGraph v0.0.30 returns the FULL state updates, but we need to accumulate them.
    # To keep it simple and robust, let's just use `invoke` but yield the final result via SSE? 
    # No, user wants streaming. Let's fix the event_generator to properly accumulate state.
    
    async def true_event_generator():
        yield f"event: session\ndata: {json.dumps({'session_id': session_id})}\n\n"
        
        current_state = dict(initial_state)
        try:
            for event in graph.stream(initial_state):
                for node_name, state_update in event.items():
                    # Accumulate state
                    for k, v in state_update.items():
                        if k == "step_log":
                            current_state[k].extend(v)
                            # Emit the new logs
                            for log_msg in v:
                                yield f"event: progress\ndata: {json.dumps({'node': node_name, 'log': log_msg})}\n\n"
                        else:
                            current_state[k] = v
            
            # Execution finished
            insights_str = "\\n\\n".join(current_state.get("insights", []))
            
            # Send the final result
            final_data = {
                "status": "success" if not current_state.get("error") else "error",
                "insights": current_state.get("insights", []),
                "generated_sql": current_state.get("generated_sql"),
                "generated_python": current_state.get("generated_python"),
                "charts": current_state.get("charts", []),
                "error": current_state.get("error")
            }
            yield f"event: complete\ndata: {json.dumps(final_data)}\n\n"
            
            # Save Assistant Message to DB
            meta = {
                "generated_sql": current_state.get("generated_sql"),
                "generated_python": current_state.get("generated_python"),
                "charts": current_state.get("charts", []),
            }
            db_msg = ChatMessage(session_id=session_id, role="assistant", content=insights_str, metadata_json=meta)
            db.add(db_msg)
            db.commit()

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(true_event_generator(), media_type="text/event-stream")
