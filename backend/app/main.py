"""
app/main.py — FastAPI Entry Point (Architecture v2)

ARCHITECTURE v2 CHANGES:
    - QueryResponse now returns:
        * charts: List[Dict]  (was plotly_chart_json: Dict — single chart)
        * insights: List[str] (was insights: str — single string)
        * generated_python: str (NEW — LLM-written code, shown in frontend)
        * execution_results: Dict (NEW — stats and findings from executor)
    - Initial state includes new v2 fields: intent, generated_python, execution_results, charts

ENDPOINTS:
    GET  /                  → Health check (shows graph status)
    POST /api/query         → Main endpoint: NL query → insights + charts + SQL + Python code

HOW TO RUN:
    Mac:     source backend/venv/bin/activate && uvicorn app.main:app --reload
    Windows: backend\\venv\\Scripts\\activate && uvicorn app.main:app --reload

API DOCS:
    Once running, open http://127.0.0.1:8000/docs for interactive Swagger UI.
    You can test the /api/query endpoint directly there without a frontend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging

from .config import settings

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# FastAPI app initialization
# -----------------------------------------------------------------------
app = FastAPI(
    title="Elytics Analytics API",
    description=(
        "AI-Powered Natural Language Data Analytics Platform. "
        "Submit natural language questions and receive SQL, Python analysis, "
        "interactive Plotly charts, and business insights."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS: Allow the React + TypeScript frontend (Vite runs on 5173 by default)
# In production, replace "*" origins with your actual frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------
# Request / Response Pydantic Models
# -----------------------------------------------------------------------

class QueryRequest(BaseModel):
    """
    JSON body sent by the React frontend to POST /api/query.
    Pydantic validates this automatically — missing fields return HTTP 422.
    """
    query: str

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What were the top 10 products by total revenue last quarter?"
            }
        }


class QueryResponse(BaseModel):
    """
    Structured JSON response returned to the React + TypeScript frontend.

    Architecture v2 Fields:
    - status:           "success" | "partial" | "error"
    - query:            Echo of the original user question
    - insights:         List[str] — multiple insight strings (v2: was a single string)
    - generated_sql:    The Redshift SQL query (shown in SQL viewer panel)
    - generated_python: The LLM-written analysis code (shown in code viewer panel)
    - charts:           List[Dict] — multiple Plotly JSON charts (v2: was a single chart)
    - execution_results: Dict with statistics and text_outputs from the executor
    - step_log:         List[str] — execution trace for progress display
    - error:            Error message if anything failed

    WHY return generated_python to the frontend?
        Business users (and developers!) benefit from seeing exactly what analysis
        was performed. Showing the code makes the system transparent and debuggable.
    """
    status: str
    query: str
    insights: List[str]                               # v2: List[str] (was str)
    generated_sql: Optional[str] = None
    generated_python: Optional[str] = None            # v2: NEW
    charts: Optional[List[Dict[str, Any]]] = None     # v2: List (was single Dict)
    execution_results: Optional[Dict[str, Any]] = None  # v2: NEW
    step_log: Optional[List[str]] = None
    error: Optional[str] = None


# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health_check():
    """
    Health check endpoint. Returns backend status and LangGraph readiness.
    Used by the frontend to verify the API is available before sending queries.
    """
    from . import graph as graph_module
    graph_status = "ready" if graph_module.analytics_graph is not None else "not_initialized"

    return {
        "status": "online",
        "service": "Elytics Analytics API",
        "version": "2.0.0",
        "architecture": "v2 — 8-agent pipeline",
        "environment": settings.environment,
        "pipeline": [
            "1_planner", "2_schema", "3_sql", "4_validator",
            "5_redshift", "6_python_agent", "7_executor", "8_insights"
        ],
        "langgraph_status": graph_status,
    }


@app.post("/api/query", response_model=QueryResponse, tags=["Analytics"])
def submit_query(request: QueryRequest):
    """
    Main analytics endpoint.

    Accepts a natural language question and returns:
    - insights:         Business-friendly findings (list of strings)
    - generated_sql:    The Redshift SQL that was run
    - generated_python: The Python analysis code that was executed
    - charts:           Plotly chart JSON dicts (render with React-Plotly)
    - execution_results: Statistics and text findings from the analysis
    - step_log:         Step-by-step trace of what each agent did

    Full pipeline flow:
        1. PlannerAgent    → identifies intent, builds plan
        2. SchemaAgent     → discovers relevant Redshift tables/columns
        3. SQLAgent        → generates the SQL query
        4. ValidationAgent → validates SQL (rules + LLM review)
        5. Redshift node   → executes SQL, retrieves rows
        6. PythonAgent     → generates Pandas/Plotly analysis code
        7. ExecutorAgent   → runs code safely in sandbox
        8. InsightsAgent   → generates business narrative

    Conditional routing:
        If validation fails → skip Redshift + analysis + insights, return error
    """
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"[/api/query] Received: '{user_query[:100]}'")

    # Get or build the graph (lazy initialization if startup failed)
    from . import graph as graph_module
    graph = graph_module.analytics_graph

    if graph is None:
        try:
            graph_module.analytics_graph = graph_module.build_graph()
            graph = graph_module.analytics_graph
            logger.info("analytics_graph built on first request")
        except Exception as e:
            logger.error(f"Could not build analytics_graph: {e}")
            return QueryResponse(
                status="error",
                query=user_query,
                insights=[
                    "The backend is not fully configured. "
                    "Please check your .env file for AWS and Redshift credentials."
                ],
                error=f"Graph initialization failed: {e}",
                step_log=["❌ Backend startup failed — check .env configuration"],
            )

    # -----------------------------------------------------------------------
    # Build initial state
    # -----------------------------------------------------------------------
    # All fields in QueryState must be present with appropriate defaults.
    # Agents fill in their respective keys as the graph executes.
    # The Annotated step_log field uses operator.add so agents append to it.
    initial_state = {
        # INPUT
        "user_query": user_query,
        # Planner will fill:
        "intent": "",
        "plan": {},
        # Schema will fill:
        "schema_context": {},
        # SQL will fill:
        "generated_sql": "",
        # Validator will fill:
        "sql_validation": {},
        # Redshift node will fill:
        "query_results": [],
        # PythonAgent will fill (v2 new):
        "generated_python": "",
        # ExecutorAgent will fill (v2 new):
        "execution_results": {},
        "charts": [],
        # InsightsAgent will fill (v2: now a list):
        "insights": [],
        # System fields:
        "error": "",
        "step_log": [f"🚀 Pipeline started | query='{user_query[:60]}'"],
    }

    try:
        # Invoke the full 8-agent LangGraph pipeline
        final_state = graph.invoke(initial_state)

        # Determine response status
        has_error    = bool(final_state.get("error"))
        has_insights = bool(final_state.get("insights"))
        has_charts   = bool(final_state.get("charts"))

        if has_insights and not has_error:
            status = "success"
        elif has_insights or has_charts:
            status = "partial"
        else:
            status = "error"

        step_log = final_state.get("step_log", [])
        logger.info(
            f"[/api/query] Complete | status={status} | "
            f"steps={len(step_log)} | "
            f"charts={len(final_state.get('charts', []))} | "
            f"insights={len(final_state.get('insights', []))}"
        )

        return QueryResponse(
            status=status,
            query=user_query,
            insights=final_state.get("insights", []),
            generated_sql=final_state.get("generated_sql") or None,
            generated_python=final_state.get("generated_python") or None,
            charts=final_state.get("charts") or None,
            execution_results=final_state.get("execution_results") or None,
            step_log=step_log,
            error=final_state.get("error") or None,
        )

    except Exception as e:
        error_msg = f"Pipeline execution error: {e}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
