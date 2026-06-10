import json
import logging
from typing import Dict, Any
from ..utils.bedrock import BedrockClient
from ..state import QueryState
logger = logging.getLogger(__name__)
PLANNER_SYSTEM_PROMPT = 'You are an expert Data Analytics Planner. Your job is to analyze\na user\'s natural language question about business data and produce a structured analytical plan.\n\nYou must respond with ONLY a valid JSON object (no markdown, no extra text) with this structure:\n{\n  "intent": "One of: trend_analysis | aggregation | comparison | ranking | distribution | lookup | other",\n  "analytical_objective": "A single, clear sentence describing exactly what the user wants to know",\n  "key_filters": ["filter1", "filter2"],\n  "time_period": "Any time range mentioned, or null if none",\n  "grouping_dimensions": ["dimension1", "dimension2"],\n  "expected_output_type": "One of: chart | table | number | text | chart_and_table",\n  "complexity": "One of: simple | moderate | complex",\n  "plan_summary": "2-3 sentence plain English plan of how to answer the question using SQL and data analysis"\n}'

class PlannerAgent:

    def __init__(self, llm_client: BedrockClient):
        self.llm_client = llm_client
        logger.info('PlannerAgent initialized')

    def process(self, state: QueryState) -> Dict[str, Any]:
        user_query = state['user_query']
        logger.info(f"PlannerAgent processing query: '{user_query[:100]}...'")
        user_message = f'Analyze this business data question and produce a structured plan:\n\nUSER QUESTION: {user_query}\n\nRemember: Respond with ONLY valid JSON. No markdown. No extra text.'
        try:
            response_text = self.llm_client.generate(system_prompt=PLANNER_SYSTEM_PROMPT, user_message=user_message, model_id=BedrockClient.SONNET_MODEL_ID, temperature=0.0)
            plan = self.llm_client.parse_json_response(response_text)
            logger.info(f"PlannerAgent success | intent={plan.get('intent')} | complexity={plan.get('complexity')}")
            intent = plan.get('intent', 'aggregation')
            return {'plan': plan, 'intent': intent, 'step_log': [f"✅ Planner: Intent='{intent}' | '{plan.get('analytical_objective')[:80]}'"]}
        except Exception as e:
            error_msg = f'PlannerAgent failed: {e}'
            logger.error(error_msg)
            return {'plan': {}, 'error': error_msg, 'step_log': [f'❌ Planner failed: {e}']}