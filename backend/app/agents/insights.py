"""
agents/insights.py — Insights Agent (Architecture v2)

ROLE IN THE PIPELINE:
    Step 8. The FINAL agent. Runs after ExecutorAgent.

ARCHITECTURE v2 CHANGES:
    - Reads from `execution_results` and `charts` (not the old `analysis_results`)
    - Returns `insights` as a List[str] (not a single string — matching design doc)
    - Has access to richer data: full statistics dict + executor text_outputs

WHAT IT DOES:
    Takes the execution results (statistics dict, text_outputs, charts list)
    plus the full analytical context and generates a human-readable, business-
    friendly narrative that:
    - Directly answers the user's question
    - Highlights the most important findings (top values, trends, anomalies)
    - Provides business context and implications
    - References ONLY exact numbers from the provided statistics

CRITICAL RULE — "NO LLM MATH":
    The LLM must NEVER compute new numbers. It must ONLY cite numbers that
    appear in the `statistics` dict or `text_outputs` list.
    
    WHY? LLMs can hallucinate numbers ("Sales grew by 34%") that are simply
    wrong. By providing exact computed statistics and instructing the LLM to
    only reference those, we guarantee numeric accuracy.

    This principle is adapted from the reference business_agent.py which
    passes exact metrics and explicitly says: "NO LLM math".

PATTERN ADAPTED FROM:
    business_agent.py's _summarize_executor_results() — formats execution results
    as key=value pairs for the LLM prompt to avoid token waste on raw JSON.
"""

import logging
from typing import Dict, Any, List

from ..utils.bedrock import BedrockClient
from ..state import QueryState

logger = logging.getLogger(__name__)

INSIGHTS_SYSTEM_PROMPT = """You are a Business Intelligence Analyst providing insights to executive-level business users.
Your job is to translate data analysis results into clear, actionable business insights.

CRITICAL RULES:
1. ONLY cite numbers that are explicitly provided in the statistics or text_outputs — NEVER invent or compute new numbers
2. Use plain business language — avoid technical jargon like "DataFrame" or "Pandas"
3. Be specific and concrete — "Revenue totaled $4.2M" not "Revenue was high"
4. Directly answer the user's original question in the first insight
5. Each insight string should be a complete, standalone sentence or short paragraph

Respond with ONLY valid JSON:
{
  "insights": [
    "Headline insight that directly answers the user's question (1-2 sentences)",
    "Key finding #1 with specific numbers from the data",
    "Key finding #2 with specific numbers from the data",
    "Key finding #3 or trend observation",
    "Optional: Business recommendation or action (only if clearly supported by data)"
  ],
  "data_notes": "Any important caveats (e.g., date range, row count, missing data)"
}"""


class InsightsAgent:
    """
    The final agent: converts execution results into executive-level narrative insights.
    Returns insights as a List[str] matching the Architecture v2 design doc schema.
    """

    def __init__(self, llm_client: BedrockClient):
        self.llm_client = llm_client
        logger.info("InsightsAgent initialized (v2)")

    def _format_execution_results_for_prompt(
        self, execution_results: Dict, text_outputs: List[str]
    ) -> str:
        """
        Format execution_results as compact key=value pairs for the LLM prompt.

        WHY key=value format?
            Adapted directly from business_agent.py's _format_dataframes() approach.
            "revenue_sum=4250000, revenue_mean=85000" uses far fewer tokens
            than serializing the full JSON dict. The LLM reads it just as well.
        """
        if not execution_results or execution_results.get("error"):
            error = execution_results.get("error", "Unknown error") if execution_results else "No results"
            return f"Execution error: {error}"

        lines = []
        statistics = execution_results.get("statistics", {})

        if statistics:
            lines.append("COMPUTED STATISTICS (use ONLY these numbers in insights):")
            for key, value in statistics.items():
                if isinstance(value, float):
                    lines.append(f"  {key.replace('_', ' ')}: {value:,.2f}")
                elif isinstance(value, int):
                    lines.append(f"  {key.replace('_', ' ')}: {value:,}")
                else:
                    lines.append(f"  {key.replace('_', ' ')}: {value}")

        if text_outputs:
            lines.append("\nANALYSIS FINDINGS (from Python analysis):")
            for finding in text_outputs:
                lines.append(f"  • {finding}")

        return "\n".join(lines) if lines else "No statistical results available"

    def _describe_charts(self, charts: List[Dict]) -> str:
        """Describe generated charts concisely for the LLM context."""
        if not charts:
            return "No charts generated"

        descriptions = []
        for i, chart in enumerate(charts, 1):
            chart_data = chart.get("data", [])
            layout = chart.get("layout", {})
            chart_type = chart_data[0].get("type", "chart") if chart_data else "chart"
            title_obj = layout.get("title", {})
            # Title can be a string or a dict with "text" key
            if isinstance(title_obj, dict):
                title = title_obj.get("text", "Untitled")
            else:
                title = str(title_obj) if title_obj else "Untitled"
            descriptions.append(f"  Chart {i}: {chart_type.capitalize()} — '{title}'")

        return "CHARTS GENERATED:\n" + "\n".join(descriptions)

    def process(self, state: QueryState) -> Dict[str, Any]:
        """
        LangGraph node function. The final step of the 8-agent pipeline.

        Reads:
            state["user_query"]          — original question to directly answer
            state["plan"]                — for context on what was asked
            state["intent"]              — top-level intent
            state["execution_results"]   — statistics + text_outputs from ExecutorAgent
            state["charts"]              — list of Plotly charts for context
            state["generated_sql"]       — for transparency in response
            state["generated_python"]    — for transparency in response

        Writes:
            state["insights"]  — List[str] of insight strings (v2: was a single string)
        """
        user_query = state["user_query"]
        plan = state.get("plan", {})
        intent = state.get("intent", plan.get("intent", "unknown"))
        execution_results = state.get("execution_results", {})
        charts = state.get("charts", [])
        sql = state.get("generated_sql", "")
        query_results = state.get("query_results", [])
        text_outputs = execution_results.get("text_outputs", []) if execution_results else []

        logger.info(
            f"InsightsAgent v2 generating insights | "
            f"intent={intent} | charts={len(charts)} | "
            f"stats={len(execution_results.get('statistics', {}))}"
        )

        stats_text = self._format_execution_results_for_prompt(execution_results, text_outputs)
        charts_text = self._describe_charts(charts)
        total_rows = len(query_results)

        user_message = f"""Generate business insights for this analysis:

ORIGINAL USER QUESTION: {user_query}

ANALYTICAL INTENT: {intent}
OBJECTIVE: {plan.get('analytical_objective', 'Data analysis')}

{stats_text}

{charts_text}

CONTEXT:
- Total rows of data analyzed: {total_rows}
- SQL was executed on Amazon Redshift
- Python analysis was performed using Pandas and Plotly

IMPORTANT: Only cite numbers from the COMPUTED STATISTICS section above.
Do NOT invent numbers or percentages that are not shown in the data.

Respond with ONLY valid JSON."""

        try:
            response_text = self.llm_client.generate(
                system_prompt=INSIGHTS_SYSTEM_PROMPT,
                user_message=user_message,
                model_id=BedrockClient.SONNET_MODEL_ID,
                temperature=0.3,  # Slightly creative for natural language, grounded by data
                max_tokens=2048,
            )
            result = self.llm_client.parse_json_response(response_text)

            # Architecture v2: insights is a List[str]
            insights_list = result.get("insights", [])
            data_notes = result.get("data_notes", "")

            # Add data notes as a final insight if present
            if data_notes:
                insights_list.append(f"📋 Data note: {data_notes}")

            # Ensure we always have at least one insight
            if not insights_list:
                insights_list = [f"Analysis of {total_rows} records complete. Please review the charts for visual findings."]

            logger.info(f"InsightsAgent generated {len(insights_list)} insight strings")

            return {
                "insights": insights_list,
                "step_log": [
                    f"✅ Insights: Generated {len(insights_list)} insights | "
                    f"intent={intent} | rows={total_rows}"
                ],
            }

        except Exception as e:
            error_msg = f"InsightsAgent failed: {e}"
            logger.error(error_msg)

            # Fallback: build minimal insights from raw execution results
            fallback_insights = [
                f"Analysis complete. {total_rows} records were retrieved from the database.",
            ]
            # Add any text_outputs from the executor as fallback insights
            for finding in text_outputs[:3]:
                fallback_insights.append(str(finding))

            return {
                "insights": fallback_insights,
                "error": error_msg,
                "step_log": [f"⚠️ Insights: LLM failed, using fallback | {e}"],
            }
