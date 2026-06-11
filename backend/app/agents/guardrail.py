import logging
import re
from typing import Dict, Any, Tuple, Optional
from ..state import QueryState
logger = logging.getLogger(__name__)
QUESTION_PATTERNS = ['^how\\s+many\\b', '^show\\b', '^list\\b', '^top\\s+\\d+\\b', '^what\\s+(is|are|was|were)\\b', '^give\\s+me\\b', '^find\\b', '^summarize\\b', '^analyze\\b', '^compare\\b', '\\btotal\\b', '\\baverage\\b', '\\bcount\\b', '\\bby\\s+(month|year|quarter|week|day|category|region|location|department|policy|premium|customer)\\b']
GREETING_PATTERNS = ['^\\s*(hi|hello|hey|howdy|namaste|greetings)\\s*[!.,?]*\\s*$', '^\\s*(good\\s+(morning|afternoon|evening|night|day))\\s*[!.,?]*\\s*$']
MATH_PATTERNS = ['^\\s*\\d+\\s*[\\+\\-\\*\\/\\%\\^]\\s*\\d+\\s*$', '\\b(calculate|compute|solve|evaluate)\\s+\\d', '\\b(sqrt|square\\s+root|cube\\s+root|factorial)\\s*(\\(|\\s*of)?\\s*\\d+']
GENERAL_KNOWLEDGE_PATTERNS = ['\\b(who\\s+(was|invented|discovered|founded|created))\\b', '\\bcapital\\s+(of|city)\\b', '\\b(what\\s+is\\s+the\\s+(meaning|definition)\\s+of)\\b', '\\bwho\\s+is\\b(?!.*(customer|supplier|vendor|employee|user|manager|client|policyholder|agent))']
CODING_PATTERNS = ['\\b(write|create|generate|build|make)\\s+(a\\s+)?(code|program|script|function|class|app|website)\\b', '\\b(python|javascript|java|c\\+\\+|rust|golang|ruby|php|swift|kotlin)\\s+(code|program|function|script)\\b']
ENTERTAINMENT_PATTERNS = ['\\b(tell\\s+me\\s+a\\s+(joke|story|riddle|poem|fact))\\b', '\\b(make\\s+me\\s+laugh|something\\s+funny|entertain\\s+me)\\b', '\\b(play\\s+a\\s+game|trivia|quiz|word\\s+game)\\b']
RISKY_SQL_PATTERNS = ['\\bdrop\\s+table\\b', '\\bdrop\\s+database\\b', '\\btruncate\\s+table\\b', '\\bdelete\\s+from\\b', '\\bupdate\\s+\\w+\\s+set\\b', '\\binsert\\s+into\\b', '\\balter\\s+table\\b']

def _matches_any(patterns: list[str], text: str) -> bool:
  return any((re.search(p, text, flags=re.IGNORECASE) for p in patterns))

class GuardrailAgent:

  def __init__(self):
    pass

  def process(self, state: QueryState) -> Dict[str, Any]:
    query = state.get('user_query', '').strip()
    logger.info(f'Guardrail processing query: {query}')
    if _matches_any(RISKY_SQL_PATTERNS, query):
      msg = 'I can only process read-only queries. Operations modifying the database are prohibited.'
      return {'guardrail_status': 'rejected', 'error': msg, 'insights': [msg], 'step_log': [f' Guardrail Rejected: {msg}']}
    rejection_msg = None
    if _matches_any(GREETING_PATTERNS, query):
      rejection_msg = 'Hello! I am your Edelweiss Life Analytics Assistant. Please ask me questions about your data.'
    elif _matches_any(MATH_PATTERNS, query):
      rejection_msg = 'I specialize in data analytics, not general math. Ask me about trends or totals in your dataset.'
    elif _matches_any(GENERAL_KNOWLEDGE_PATTERNS, query):
      rejection_msg = 'I analyze business data. Ask me about your policies, premiums, or agent performance.'
    elif _matches_any(CODING_PATTERNS, query):
      rejection_msg = "I'm a data intelligence assistant, not a code generator. How can I help you explore your data?"
    elif _matches_any(ENTERTAINMENT_PATTERNS, query):
      rejection_msg = "I'm focused on helping you with data intelligence. Try asking me for a performance summary!"
    if rejection_msg:
      if not _matches_any(QUESTION_PATTERNS, query) and len(query.split()) < 10:
        return {'guardrail_status': 'rejected', 'error': rejection_msg, 'insights': [rejection_msg], 'step_log': [f' Guardrail Rejected: {rejection_msg}']}
    return {'guardrail_status': 'passed', 'step_log': [' Guardrail Passed: Query is relevant to analytics']}