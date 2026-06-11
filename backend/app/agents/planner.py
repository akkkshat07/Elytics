import json
import logging
from typing import Dict, Any
from ..utils.bedrock import BedrockClient
from ..state import QueryState
logger = logging.getLogger(__name__)
PLANNER_SYSTEM_PROMPT = 'You are an expert Data Analytics Planner and Query Router for a business intelligence system.\nYour job is to analyze a user\'s natural language question about business data and produce a structured analytical plan, including an enhanced, normalized question.\n\nCRITICAL RULES:\n1. ALWAYS err on the side of relevance. If a business interpretation exists, mark it relevant.\n2. Produce an "enhanced_question" which is a SINGLE, clean, standalone English sentence.\n  - For follow-ups (e.g. "what about 2024?"), merge it with the previous context into a complete question.\n  - NEVER use meta-phrases like "Following from:" or "Previously:".\n3. Provide a clear analytical plan for downstream SQL and Python agents.\n\nYou must respond with ONLY a valid JSON object (no markdown, no extra text) with this exact structure:\n{\n "enhanced_question": "Normalized, complete English sentence representing the full intent",\n "intent": "One of: trend_analysis | aggregation | comparison | ranking | distribution | lookup | other",\n "analytical_objective": "A single, clear sentence describing exactly what needs to be queried and calculated",\n "key_filters": ["filter1", "filter2"],\n "time_period": "Any time range mentioned, or null if none",\n "grouping_dimensions": ["dimension1", "dimension2"],\n "expected_output_type": "One of: chart | table | number | text | chart_and_table",\n "complexity": "One of: simple | moderate | complex",\n "plan_summary": "2-3 sentence plain English plan of how to answer the question using SQL and data analysis"\n}'

class PlannerAgent:

  def __init__(self, llm_client: BedrockClient):
    self.llm_client = llm_client
    logger.info('PlannerAgent initialized (CoreSight Upgrade)')

  def process(self, state: QueryState) -> Dict[str, Any]:
    user_query = state.get('user_query', '')
    previous_context = state.get('previous_context', '')
    logger.info(f"PlannerAgent processing query: '{user_query[:100]}...'")
    user_message = f'Analyze this business data question and produce a structured plan:\n\nUSER QUESTION: {user_query}\n\n'
    if previous_context:
      user_message += f'PREVIOUS CONTEXT (for follow-ups): {previous_context}\n\n'
    user_message += 'Remember: Respond with ONLY valid JSON. No markdown. No extra text.'
    try:
      response_text = self.llm_client.generate(system_prompt=PLANNER_SYSTEM_PROMPT, user_message=user_message, model_id=BedrockClient.SONNET_MODEL_ID, temperature=0.0)
      plan = self.llm_client.parse_json_response(response_text)
      enhanced_question = plan.get('enhanced_question', user_query)
      intent = plan.get('intent', 'aggregation')
      logger.info(f"PlannerAgent success | intent={intent} | complexity={plan.get('complexity')}")
      return {'user_query': enhanced_question, 'plan': plan, 'intent': intent, 'step_log': [f" Planner: Intent='{intent}' | Enhanced: '{enhanced_question[:80]}...'"]}
    except Exception as e:
      error_msg = f'PlannerAgent failed: {e}'
      logger.error(error_msg)
      return {'plan': {}, 'error': error_msg, 'step_log': [f' Planner failed: {e}']}