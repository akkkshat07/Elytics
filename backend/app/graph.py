"""
graph.py — LangGraph Workflow Orchestrator (Architecture v2)

ARCHITECTURE v2 CHANGES:
    - Removed: `analyst` node (old AnalystAgent — retired)
    - Added: `python_code` node (PythonAgent — LLM generates analysis code)
    - Added: `executor` node (ExecutorAgent — safely runs that code)
    - New 8-step pipeline:
        planner → schema → sql → validator → redshift_query → python_code → executor → insights

PIPELINE OVERVIEW:
    Step 1: planner        — Identifies intent, builds structured plan
    Step 2: schema         — Discovers relevant Redshift tables/columns
    Step 3: sql            — Generates Redshift SQL query
    Step 4: validator      — Rule-based + LLM SQL safety check
    [conditional]          — Only continues if SQL is valid
    Step 5: redshift_query — Executes SQL against Redshift, stores raw rows
    Step 6: python_code    — LLM writes Pandas/NumPy/Plotly analysis code
    Step 7: executor       — Runs code safely in restricted sandbox
    Step 8: insights       — Generates business narrative from results

CONDITIONAL ROUTING:
    After the validator, we check sql_validation.is_valid:
    - True  → proceed to Redshift
    - False → route straight to END (don't run bad SQL)

LangGraph CONCEPTS (for reference):
    - StateGraph: The directed graph container
    - add_node(name, fn): Register an agent function as a named node
    - set_entry_point: Which node runs first
    - add_edge(A, B): A always runs before B
    - add_conditional_edges(A, fn, {str→node}): fn(state) decides next node

SINGLETON PATTERN:
    `analytics_graph` is built once at module import time.
    FastAPI's main.py imports it and reuses it for every request.
    This avoids re-initializing agents (and boto3 connections) per request.
"""

import logging
from typing import Dict, Any, Literal

from langgraph.graph import StateGraph, END

from .state import QueryState
from .utils.bedrock import BedrockClient
from .utils.redshift import RedshiftClient
from .agents.planner import PlannerAgent
from .agents.schema import SchemaAgent
from .agents.sql import SQLAgent
from .agents.validator import ValidationAgent
from .agents.python_agent import PythonAgent
from .agents.executor import ExecutorAgent
from .agents.insights import InsightsAgent

logger = logging.getLogger(__name__)


def build_graph():
    """
    Constructs and compiles the complete 8-node LangGraph workflow.

    Returns:
        CompiledGraph: Ready to invoke with initial QueryState.
    """

    # -----------------------------------------------------------------------
    # Initialize shared clients (created ONCE, shared across ALL agents)
    # -----------------------------------------------------------------------
    # WHY share? Creating a boto3 Bedrock client involves TLS negotiation.
    # One client per process is the same pattern as the reference code's shared LLMClient.
    llm_client = BedrockClient()
    redshift_client = RedshiftClient()

    # -----------------------------------------------------------------------
    # Initialize agents (all receive shared clients via constructor injection)
    # -----------------------------------------------------------------------
    planner_agent   = PlannerAgent(llm_client=llm_client)
    schema_agent    = SchemaAgent(llm_client=llm_client, redshift_client=redshift_client)
    sql_agent       = SQLAgent(llm_client=llm_client)
    validator_agent = ValidationAgent(llm_client=llm_client)
    python_agent    = PythonAgent(llm_client=llm_client)   # NEW in v2
    executor_agent  = ExecutorAgent()                       # NEW in v2 — no LLM needed
    insights_agent  = InsightsAgent(llm_client=llm_client)

    # -----------------------------------------------------------------------
    # Define the Redshift query execution node
    # -----------------------------------------------------------------------
    # This sits between validation and python_code.
    # It's a plain function (not an Agent class) because it has no LLM —
    # just a database call. Keeping it separate from ExecutorAgent maintains
    # clean separation between data RETRIEVAL and data ANALYSIS.
    def redshift_query_node(state: QueryState) -> Dict[str, Any]:
        """
        Execute the validated SQL against Amazon Redshift.

        Reads:  state["generated_sql"]
        Writes: state["query_results"]
        """
        sql = state.get("generated_sql", "")
        logger.info(f"redshift_query_node: Executing SQL | {sql[:150]}...")

        try:
            rows = redshift_client.execute_query(sql)
            logger.info(f"Redshift returned {len(rows)} rows")
            return {
                "query_results": rows,
                "step_log": [f"✅ Redshift Query: {len(rows)} rows returned"],
            }
        except RuntimeError as e:
            error_msg = f"Redshift execution failed: {e}"
            logger.error(error_msg)
            return {
                "query_results": [],
                "error": error_msg,
                "step_log": [f"❌ Redshift Query: {e}"],
            }

    # -----------------------------------------------------------------------
    # Build the StateGraph
    # -----------------------------------------------------------------------
    workflow = StateGraph(QueryState)

    # --- Register all nodes ---
    workflow.add_node("planner",        planner_agent.process)
    workflow.add_node("schema",         schema_agent.process)
    workflow.add_node("sql",            sql_agent.process)
    workflow.add_node("validator",      validator_agent.process)
    workflow.add_node("redshift_query", redshift_query_node)
    workflow.add_node("python_code",    python_agent.process)    # NEW v2
    workflow.add_node("executor",       executor_agent.process)  # NEW v2
    workflow.add_node("insights",       insights_agent.process)

    # --- Entry point ---
    workflow.set_entry_point("planner")

    # --- Sequential edges (steps 1–4: planner through validator) ---
    workflow.add_edge("planner", "schema")
    workflow.add_edge("schema",  "sql")
    workflow.add_edge("sql",     "validator")

    # --- Conditional edge after validation ---
    # The routing function inspects state to decide: run SQL or bail out.
    def route_after_validation(state: QueryState) -> Literal["redshift_query", "__end__"]:
        """
        Routing function called after the ValidationAgent.

        Rules:
        - If SQL is valid AND no error: proceed to Redshift
        - Otherwise: route to END to return error response

        WHY Literal return type?
            LangGraph uses the returned string to look up the destination node name.
            "__end__" is LangGraph's special name for the END node.
        """
        validation = state.get("sql_validation", {})
        is_valid   = validation.get("is_valid", False)
        has_error  = bool(state.get("error", ""))
        has_sql    = bool(state.get("generated_sql", "").strip())

        if is_valid and has_sql and not has_error:
            logger.info("Routing: validator PASSED → redshift_query")
            return "redshift_query"
        else:
            logger.warning(
                f"Routing: validator FAILED → END | "
                f"is_valid={is_valid} | has_sql={has_sql} | has_error={has_error}"
            )
            return "__end__"

    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "redshift_query": "redshift_query",
            "__end__": END,
        }
    )

    # --- Sequential edges (steps 5–8: data through insights) ---
    workflow.add_edge("redshift_query", "python_code")   # NEW v2: was → analyst
    workflow.add_edge("python_code",    "executor")      # NEW v2
    workflow.add_edge("executor",       "insights")
    workflow.add_edge("insights",       END)

    # --- Compile ---
    compiled = workflow.compile()
    logger.info(
        "LangGraph v2 compiled successfully | "
        "pipeline: planner→schema→sql→validator→redshift→python_code→executor→insights"
    )
    return compiled


# -----------------------------------------------------------------------
# Module-level singleton — built ONCE at server startup
# -----------------------------------------------------------------------
try:
    analytics_graph = build_graph()
    logger.info("analytics_graph ready")
except Exception as e:
    logger.warning(
        f"analytics_graph could not be built at startup (likely missing credentials): {e}. "
        "Will retry on first /api/query request."
    )
    analytics_graph = None
