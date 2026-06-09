"""
agents/planner.py — Planner Agent

ROLE IN THE PIPELINE:
    Step 1. The very first agent to process the user's question.
    
WHAT IT DOES:
    Takes the raw natural language query and identifies:
    - The user's INTENT (e.g., trend analysis, aggregation, comparison)
    - The ANALYTICAL OBJECTIVE (what business question is being asked)
    - Any KEY FILTERS mentioned (e.g., "last 6 months", "in California")
    - A STRUCTURED PLAN for how to answer the question

WHY THIS MATTERS:
    Without a plan, the SQL Agent might generate a query that technically runs
    but doesn't answer what the user actually wanted. The planner translates
    ambiguous human language into a precise, structured goal.

PATTERN ADAPTED FROM:
    router_agent.py's intent classification and business_agent.py's planner context.
    Instead of routing to different systems, our Planner creates a structured plan
    for a single analytics pipeline.
"""

import json
import logging
from typing import Dict, Any

from ..utils.bedrock import BedrockClient
from ..state import QueryState

logger = logging.getLogger(__name__)

# The system prompt tells Claude what role it plays in this step.
# WHY a detailed prompt? Because the quality of the plan directly determines
# the quality of every downstream agent's output. Garbage in = garbage out.
PLANNER_SYSTEM_PROMPT = """You are an expert Data Analytics Planner. Your job is to analyze
a user's natural language question about business data and produce a structured analytical plan.

You must respond with ONLY a valid JSON object (no markdown, no extra text) with this structure:
{
  "intent": "One of: trend_analysis | aggregation | comparison | ranking | distribution | lookup | other",
  "analytical_objective": "A single, clear sentence describing exactly what the user wants to know",
  "key_filters": ["filter1", "filter2"],
  "time_period": "Any time range mentioned, or null if none",
  "grouping_dimensions": ["dimension1", "dimension2"],
  "expected_output_type": "One of: chart | table | number | text | chart_and_table",
  "complexity": "One of: simple | moderate | complex",
  "plan_summary": "2-3 sentence plain English plan of how to answer the question using SQL and data analysis"
}"""


class PlannerAgent:
    """
    Identifies user intent and builds a structured analytical plan.
    
    Adapted from the RouterAgent pattern in the reference code:
    - Takes user query as input
    - Makes one LLM call to classify and plan
    - Returns structured JSON consumed by downstream agents
    """

    def __init__(self, llm_client: BedrockClient):
        """
        Args:
            llm_client: Shared BedrockClient instance. We share it across agents
                        to avoid creating multiple AWS connections. Same pattern
                        as the reference code passing llm_client in constructors.
        """
        self.llm_client = llm_client
        logger.info("PlannerAgent initialized")

    def process(self, state: QueryState) -> Dict[str, Any]:
        """
        LangGraph node function. Called by graph.py during workflow execution.

        WHY it takes 'state' and returns a dict?
            LangGraph nodes receive the full current state and return ONLY the
            keys they want to update. LangGraph merges the returned dict into
            the state automatically. This is the core LangGraph pattern.

        Args:
            state: The current shared QueryState object

        Returns:
            dict with 'plan' key containing the structured plan, and a log entry.
        """
        user_query = state["user_query"]
        logger.info(f"PlannerAgent processing query: '{user_query[:100]}...'")

        user_message = f"""Analyze this business data question and produce a structured plan:

USER QUESTION: {user_query}

Remember: Respond with ONLY valid JSON. No markdown. No extra text."""

        try:
            # Use Claude 3.5 Sonnet (our most capable model) for planning
            # temperature=0.0 for consistency and reproducibility
            response_text = self.llm_client.generate(
                system_prompt=PLANNER_SYSTEM_PROMPT,
                user_message=user_message,
                model_id=BedrockClient.SONNET_MODEL_ID,
                temperature=0.0,
            )
            # Parse the JSON response safely
            plan = self.llm_client.parse_json_response(response_text)

            logger.info(
                f"PlannerAgent success | intent={plan.get('intent')} "
                f"| complexity={plan.get('complexity')}"
            )

            # Extract intent as a top-level state field (new in Architecture v2)
            # This lets downstream agents read state["intent"] directly
            # without having to do state["plan"]["intent"] every time.
            intent = plan.get("intent", "aggregation")

            return {
                "plan": plan,
                "intent": intent,
                "step_log": [f"✅ Planner: Intent='{intent}' | '{plan.get('analytical_objective')[:80]}'"],
            }

        except Exception as e:
            error_msg = f"PlannerAgent failed: {e}"
            logger.error(error_msg)
            return {
                "plan": {},
                "error": error_msg,
                "step_log": [f"❌ Planner failed: {e}"],
            }
