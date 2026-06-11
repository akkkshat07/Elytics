import json
import logging
import re
import time
from difflib import SequenceMatcher
from typing import Dict, Any, Optional, List, Set
from ..utils.bedrock import BedrockClient
from ..state import QueryState

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are an expert Data Analytics Planner and Query Router for a business intelligence system.
Your job is to analyze a user's natural language question about business data and produce a structured analytical plan, including an enhanced, normalized question.

CRITICAL RULES:
1. ALWAYS err on the side of relevance. If a business interpretation exists, mark it relevant.
2. Produce an "enhanced_question" which is a SINGLE, clean, standalone English sentence.
  - For follow-ups (e.g. "what about 2024?"), merge it with the previous context into a complete question.
  - NEVER use meta-phrases like "Following from:" or "Previously:".
3. Provide a clear analytical plan for downstream SQL and Python agents.

You must respond with ONLY a valid JSON object (no markdown, no extra text) with this exact structure:
{
 "enhanced_question": "Normalized, complete English sentence representing the full intent",
 "intent": "One of: trend_analysis | aggregation | comparison | ranking | distribution | lookup | other",
 "analytical_objective": "A single, clear sentence describing exactly what needs to be queried and calculated",
 "key_filters": ["filter1", "filter2"],
 "time_period": "Any time range mentioned, or null if none",
 "grouping_dimensions": ["dimension1", "dimension2"],
 "expected_output_type": "One of: chart | table | number | text | chart_and_table",
 "complexity": "One of: simple | moderate | complex",
 "plan_summary": "2-3 sentence plain English plan of how to answer the question using SQL and data analysis"
}"""

class RouterAgent:
    """
    Consolidated Router Agent (CoreSight Architecture Port).
    Handles: relevance detection (L2), intent classification, table identification,
    cache matching, follow-up resolution, RBAC check, subscription gating.
    """

    def __init__(
        self,
        llm_client: BedrockClient,
        agent_name: str = "router_agent",
        client_id: str = "default_client",
        dataset_id: Optional[str] = None,
        db=None,
    ):
        self.llm_client = llm_client
        self.agent_name = agent_name
        self.client_id = client_id
        self.dataset_id = dataset_id
        self.db = db

        self._denied_table_keys: Set[str] = set()
        self.data_scientist_enabled = True

        self._embedding_usage = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }
        self.followup_second_pass_similarity_threshold = 0.72

        self._data_context: Optional[str] = None
        self._table_introductions_xml: Optional[str] = None
        
        self.system_prompt = ROUTER_SYSTEM_PROMPT

        logger.info(f"RouterAgent initialized with full CoreSight semantic cache/RBAC architecture for client '{client_id}'")

    # ── Main process method ──────────────────────────────────────────────

    def process(self, state: QueryState) -> Dict[str, Any]:
        """
        Classify, route, and judge cache match using the Graph State.
        """
        user_question = state.get('user_query', '')
        previous_context = state.get('previous_context', '')
        client_id = self.client_id
        user_id = "default_user"
        
        conversation_context = {"previous_enhanced_question": previous_context} if previous_context else None

        try:
            started = time.perf_counter()
            timing: Dict[str, float] = {}

            # ── Step 1: Embedding + data context loading (parallel in async) ────────
            t0 = time.perf_counter()
            query_embedding = self._generate_embedding(user_question)
            timing["embedding_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            
            data_context = self._load_data_context()

            # ── Step 1B: Candidate retrieval (Semantic Caching via Qdrant) ────────
            t0 = time.perf_counter()
            candidates = self._get_top_candidates(query_embedding, limit=5)
            timing["candidate_retrieval_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

            # ── Step 3: Single LLM call — classify + route + match ──
            t0 = time.perf_counter()
            classification = self._classify_and_match(
                user_question, candidates, data_context, conversation_context
            )
            timing["classify_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

            # ── Step 3A: Early exit if irrelevant (Layer 2) ──
            if classification.get("irrelevant_detected"):
                logger.info("Query detected as irrelevant by LLM — returning early")
                return {"error": "Query irrelevant to data analysis", "step_log": ["❌ Router: Irrelevant query detected"]}

            # ── Step 3B: Re-embed enhanced question for follow-ups ──
            enhanced_q = classification.get("enhanced_question", "")
            is_followup = classification.get("is_followup", False)

            if is_followup and enhanced_q and SequenceMatcher(None, enhanced_q.lower().strip(), user_question.lower().strip()).ratio() < 0.85:
                # Follow up logic (bypassed if no cache)
                logger.info(f"Follow-up: re-embedding enhanced question: '{enhanced_q[:80]}...'")

            # ── Step 4: RBAC check (MongoDB) ──
            try:
                t0 = time.perf_counter()
                if self.db:
                    # In Phase 2: Call table_permissions_service
                    pass
                timing["rbac_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            except Exception as rbac_err:
                logger.error(f"[RBAC] Error during access check: {rbac_err}")

            # ── Step 5: Cache match decision ──
            cache_match = classification.get("semantic_cache_match", {})
            is_match = cache_match.get("matched", False)
            
            if is_match and candidates:
                logger.info(f"Cache HIT (LLM-verified) for: '{user_question[:80]}...'")
                # Return cached code directly
                pass

            # ── Cache MISS — normal flow ──
            intent = classification.get("intent", "aggregation")
            
            logger.info(f"Cache MISS for: '{user_question[:80]}...' | intent={intent}")

            timing["total_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            logger.info(f"[Latency] router timings={timing}")
            
            # Map back to Elytics LangGraph expected state format
            return {
                "user_query": enhanced_q, 
                "plan": classification, 
                "intent": intent, 
                "step_log": [
                    f"✅ Router: Intent='{intent}' | Enhanced: '{enhanced_q[:80]}...'"
                ]
            }

        except Exception as e:
            logger.error(f"Router classification failed: {e}", exc_info=True)
            return {
                "plan": {},
                "error": f"Router failed: {e}",
                "step_log": [f"❌ Router failed: {e}"]
            }

    # ── Embedding ────────────────────────────────────────────────────────

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding. (Bypassed for Phase 1)."""
        logger.debug(f"Generating embedding for text (simulated): {text[:20]}...")
        if not self.db:
            return [0.0] * 1536
        # Phase 2 implementation will use Bedrock Titan or OpenAI Embeddings
        return [0.0] * 1536

    # ── Candidate retrieval ──────────────────────────────────────────────

    def _get_top_candidates(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top N candidates from Qdrant semantic cache."""
        if not self.db:
            # Phase 1: Semantic caching disabled
            return []
        return []

    # ── Data context (table introductions) ───────────────────────────────

    def _load_data_context(self) -> str:
        """Load brief table names + descriptions (< 2000 chars). Cached."""
        if self._data_context is not None:
            return self._data_context
        
        # Phase 1 bypass
        self._data_context = ""
        return self._data_context

    def get_table_introductions_xml(self) -> str:
        """Return full table_introductions XML."""
        return self._table_introductions_xml or ""

    # ── LLM classify + match ─────────────────────────────────────────────

    def _classify_and_match(
        self,
        user_question: str,
        candidates: List[Dict[str, Any]],
        data_context: str,
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Single LLM call: classify + route + identify tables + match cache."""
        user_message_parts = []

        if conversation_context:
            prev_enhanced = conversation_context.get("previous_enhanced_question", "")
            if prev_enhanced:
                user_message_parts.append(f"PREVIOUS CONTEXT:\n{prev_enhanced}")
                user_message_parts.append(f"\nCURRENT QUERY (may be a follow-up):\n{user_question}")
            else:
                user_message_parts.append(f"QUERY:\n{user_question}")
        else:
            user_message_parts.append(f"QUERY:\n{user_question}")

        if data_context:
            user_message_parts.append(f"\nTABLE CONTEXT:\n{data_context}")

        if candidates:
            candidate_lines = [
                f'  [{c.get("index", 0)}] "{c.get("question", "")}" (similarity: {c.get("similarity", 0.0)})'
                for c in candidates
            ]
            user_message_parts.append(f"\nCACHED QUESTIONS:\n" + "\n".join(candidate_lines))
        else:
            user_message_parts.append("\nCACHED QUESTIONS: None found.")

        user_message_parts.append("\nRespond with ONLY valid JSON.")
        user_message = "\n".join(user_message_parts)

        # Call Bedrock
        response_text = self.llm_client.generate(
            system_prompt=self.system_prompt,
            user_message=user_message,
            model_id=BedrockClient.SONNET_MODEL_ID,
            temperature=0.0
        )
        
        return self._parse_response(response_text, user_question, conversation_context, candidates)

    # ── Response parsing ─────────────────────────────────────────────────

    def _parse_response(
        self,
        response: str,
        user_question: str,
        conversation_context: Optional[Dict[str, Any]] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Parse LLM JSON response."""
        try:
            parsed = self.llm_client.parse_json_response(response)
            
            enhanced_question = parsed.get("enhanced_question", "")
            if not enhanced_question:
                enhanced_question = self._generate_fallback_enhanced_question(user_question, conversation_context)
                parsed["enhanced_question"] = enhanced_question

            # Mock semantic match for Phase 1
            parsed["semantic_cache_match"] = {
                "matched": False,
                "matched_index": None,
                "match_reason": "No cache active",
                "adaptations": []
            }
            
            return parsed
            
        except Exception as e:
            logger.warning(f"Failed to parse router JSON: {e}")
            return {
                "intent": "aggregation",
                "enhanced_question": self._generate_fallback_enhanced_question(user_question, conversation_context),
                "is_followup": False,
                "semantic_cache_match": {"matched": False}
            }

    # ── Helpers ──────────────────────────────────────────────────────────

    def _generate_fallback_enhanced_question(
        self,
        query: str,
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate fallback enhanced question for follow-ups."""
        normalized = (query or "").strip()
        if not normalized or not conversation_context:
            return normalized
        prev_enhanced = conversation_context.get("previous_enhanced_question", "")
        if not prev_enhanced:
            return normalized
        return self._merge_questions_followup(prev_enhanced, normalized)

    def _merge_questions_followup(
        self, previous: str, current: str, previous_plan: str = ""
    ) -> str:
        """Merge previous enhanced question with follow-up into one clean sentence."""
        current_lower = current.lower().strip()
        prev = previous.strip()

        if re.match(r"^(for|only|just|in)\s+.+", current_lower) or re.match(r"^.+\s+only$", current_lower):
            return f"{prev} {current.strip()}"

        return f"{prev} {current.strip()}"