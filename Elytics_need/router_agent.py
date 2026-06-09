"""
Router Agent for CoreSight v2 Graph.

Consolidates guard-L2 + intent classification + table identification +
cache judgment + reference lookup + subscription gating into a SINGLE agent.

Makes exactly 2 external API calls:
1. Embedding call — generate query embedding, retrieve top N candidates from Qdrant
2. LLM call — classify, route, identify tables, judge cache match

On cache HIT: Returns cached code for re-execution by coder_node.
On cache MISS: Returns classification + relevant_tables for scout/coder routing.
"""

import json
import logging
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict
from langsmith import traceable

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from util.llm_utils import LLMClient
from config.system_config import AGENT_CONFIG, DEFAULT_LLM_PROVIDER
from util.xml_prompt_loader import load_xml_prompt_raw, load_client_prompt
from util.dataset_paths import resolve_xml_data_sources_dir
from response_caching.models import SemanticSignature
from response_caching.semantic_cache_manager import SemanticCacheManager

logger = logging.getLogger(__name__)


class RouterAgent:
    """
    Consolidated Router Agent replacing IntentClassifierAgent.

    Handles: relevance detection (L2), intent classification, table identification,
    cache matching, follow-up resolution, RBAC check, subscription gating.
    """

    def __init__(
        self,
        agent_name: str = "router_agent",
        provided_config: Optional[Dict] = None,
        client_id: str = None,
        db=None,
        llm_client: Optional[LLMClient] = None,
        resolved_prompt: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ):
        if not client_id:
            raise ValueError("client_id is REQUIRED for multi-tenant operation")

        self.agent_name = agent_name
        self.config = provided_config or AGENT_CONFIG.get(self.agent_name, {})
        self.client_id = client_id
        self.dataset_id = dataset_id
        self.db = db

        # Data scientist routing enabled by default (can be toggled per client)
        self.data_scientist_enabled = True

        if llm_client is None:
            raise ValueError(
                f"llm_client is REQUIRED for {self.agent_name}. "
                "Pass the shared LLMClient from graph state."
            )
        self.llm_client = llm_client
        self._last_inputs: Optional[Dict[str, Any]] = None
        self._last_usage: Optional[Dict[str, Any]] = None
        self._last_timing: Optional[Dict[str, Any]] = None
        self._embedding_usage: Dict[str, Any] = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }
        self.followup_second_pass_similarity_threshold = float(
            self.config.get("followup_second_pass_similarity_threshold", 0.72)
        )

        self._resolved_prompt = resolved_prompt

        # Load prompt
        self._load_prompt()

        # Semantic cache manager (embedding + Qdrant candidate retrieval)
        self.cache_manager = SemanticCacheManager(client_id=self.client_id, dataset_id=self.dataset_id)

        # Cached data context (table introductions, loaded once)
        self._data_context: Optional[str] = None
        # Cached full table introductions XML (for scout/coder table briefs)
        self._table_introductions_xml: Optional[str] = None

        self._llm_provider = self.config.get("llm_provider", DEFAULT_LLM_PROVIDER)
        self._model_name = self.config.get("model_name")
        self._temperature = self.config.get("temperature", 0.0)

        logger.info(
            f"RouterAgent initialized for client '{client_id}' "
            f"with provider={self._llm_provider}, model={self._model_name}, "
            f"temperature={self._temperature}"
        )

    # ── Prompt loading ───────────────────────────────────────────────────

    def _load_prompt(self) -> None:
        """Load router prompt with client override support."""
        if self._resolved_prompt:
            self.system_prompt = self._resolved_prompt
            self._merge_data_scientist_route()
            return

        client_prompt_path = (
            PROJECT_ROOT / "xml_prompts" / "clients" / self.client_id
            / "agents" / "router.xml"
        )
        base_prompt_path = (
            PROJECT_ROOT / "xml_prompts" / "base" / "agents" / "router.xml"
        )
        config_prompt_path = self.config.get("prompt_file")

        if client_prompt_path.exists():
            self.prompt_path = client_prompt_path
            logger.info(f"Using client-specific router prompt: {self.prompt_path}")
        elif config_prompt_path and Path(config_prompt_path).exists():
            self.prompt_path = Path(config_prompt_path)
            logger.info(f"Using config router prompt: {self.prompt_path}")
        elif base_prompt_path.exists():
            self.prompt_path = base_prompt_path
            logger.info(f"Using base router prompt: {self.prompt_path}")
        else:
            logger.warning("Router XML prompt not found, using fallback")
            self.prompt_path = None

        if self.prompt_path:
            try:
                if self.db is not None:
                    try:
                        import asyncio
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            self.system_prompt = load_xml_prompt_raw(self.prompt_path)
                        else:
                            # load_client_prompt expects path relative to base/, e.g. "agents/router.xml"
                            try:
                                relative_path = str(
                                    self.prompt_path.relative_to(PROJECT_ROOT / "xml_prompts" / "base")
                                )
                            except ValueError:
                                relative_path = str(
                                    self.prompt_path.relative_to(PROJECT_ROOT / "xml_prompts")
                                )
                            self.system_prompt = loop.run_until_complete(
                                load_client_prompt(
                                    relative_path, self.client_id, self.db,
                                    use_formatting=False,
                                )
                            )
                    except Exception as e:
                        logger.warning(f"Client-aware loading failed, using base: {e}")
                        self.system_prompt = load_xml_prompt_raw(self.prompt_path)
                else:
                    self.system_prompt = load_xml_prompt_raw(self.prompt_path)

                # Merge data scientist route if enabled
                self._merge_data_scientist_route()
            except Exception as e:
                logger.error(f"Error loading router prompt: {e}")
                self.system_prompt = self._get_fallback_prompt()
        else:
            self.system_prompt = self._get_fallback_prompt()

    def _get_fallback_prompt(self) -> str:
        """Minimal fallback prompt if XML file is not available."""
        return """You are a query router for a data analysis system.
Classify the query, identify relevant tables, and judge cache matches.
Respond with ONLY valid JSON:
{
  "is_relevant": true,
  "query_routing_type": "data_analyst",
  "relevant_tables": [],
  "enhanced_question": "The user's question",
  "is_followup": false,
  "confidence": 0.5,
  "operation_type": "aggregate",
  "semantic_signature": {},
  "semantic_cache_match": {"matched": false, "matched_index": null, "match_reason": "No cached questions"}
}"""

    def _merge_data_scientist_route(self) -> None:
        """Merge data scientist route into router prompt if enabled."""
        if not self.data_scientist_enabled:
            self.system_prompt = self.system_prompt.replace(
                "<!-- DS_ROUTE_PLACEHOLDER -->", ""
            )
            logger.info("Data scientist routing disabled - placeholder removed")
            return

        ds_route_path = (
            PROJECT_ROOT / "xml_prompts" / "base" / "agents"
            / "intent_classifier.data_scientist_route.xml"
        )

        if not ds_route_path.exists():
            logger.warning(f"Data scientist route XML not found: {ds_route_path}")
            return

        try:
            ds_route_content = load_xml_prompt_raw(ds_route_path)
            route_match = re.search(
                r'<route name="data_scientist">.*?</route>',
                ds_route_content,
                re.DOTALL,
            )
            if route_match:
                route_xml = route_match.group(0)
                self.system_prompt = self.system_prompt.replace(
                    "<!-- DS_ROUTE_PLACEHOLDER -->", route_xml
                )
                logger.info("Data scientist route merged into router prompt")
            else:
                logger.warning("Could not extract data_scientist route from XML")
        except Exception as e:
            logger.error(f"Failed to merge data scientist route: {e}")

    # ── Main process method ──────────────────────────────────────────────

    async def process(
        self,
        user_question: str,
        user_id: str,
        client_id: str,
        conversation_context: Optional[Dict[str, Any]] = None,
        skip_rbac: bool = False,
    ) -> Dict[str, Any]:
        """
        Classify, route, and judge cache match.

        Returns:
            On CACHE HIT: {"cache_hit": True, "cached_code": "...", ...}
            On CACHE MISS: {"cache_hit": False, "relevant_tables": [...], ...}
        """
        if client_id != self.client_id:
            raise ValueError(
                f"client_id mismatch: expected {self.client_id}, got {client_id}"
            )

        try:
            started = time.perf_counter()
            timing: Dict[str, float] = {}

            # ── Step 1: Embedding + data context loading (parallel) ────────
            # Embedding is an async network call; data context is sync file
            # I/O (cached after first call). Overlap them so data context
            # doesn't wait for the embedding round-trip.
            import asyncio as _aio

            async def _embed_with_timing():
                _t = time.perf_counter()
                emb = await self._generate_embedding(user_question)
                return emb, round((time.perf_counter() - _t) * 1000.0, 2)

            async def _load_ctx():
                return self._load_data_context()

            (query_embedding, embed_ms), data_context = await _aio.gather(
                _embed_with_timing(), _load_ctx()
            )
            timing["embedding_ms"] = embed_ms

            # ── Step 1B: Candidate retrieval (requires embedding) ────────
            t0 = time.perf_counter()
            candidates = await self._get_top_candidates(query_embedding, limit=5)
            timing["candidate_retrieval_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            logger.info(
                f"Retrieved {len(candidates)} candidates for: '{user_question[:80]}...'"
            )

            # ── Step 3: Single LLM call — classify + route + match ──
            t0 = time.perf_counter()
            classification = await self._classify_and_match(
                user_question, candidates, data_context, conversation_context
            )
            timing["classify_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

            # ── Step 3A: Early exit if irrelevant (Layer 2) ──
            if classification.get("irrelevant_detected"):
                logger.info(
                    "Query detected as irrelevant by LLM — returning early: '%s'",
                    user_question[:80],
                )
                timing["total_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
                self._last_timing = timing
                return classification

            # ── Step 3B: Re-embed enhanced question for follow-ups ──
            enhanced_q = classification.get("enhanced_question", "")
            is_followup = classification.get("is_followup", False)

            if (
                is_followup
                and enhanced_q
                and SequenceMatcher(
                    None, enhanced_q.lower().strip(), user_question.lower().strip()
                ).ratio() < 0.85
            ):
                top_similarity = max(
                    (c.get("similarity", 0.0) for c in candidates), default=0.0
                )
                has_match = bool(
                    (classification.get("semantic_cache_match") or {}).get("matched")
                )
                allow_second_pass = (not has_match) and (
                    top_similarity < self.followup_second_pass_similarity_threshold
                )
                logger.info(
                    "Follow-up second pass gate | allow=%s | top_similarity=%.3f | matched=%s",
                    allow_second_pass, top_similarity, has_match,
                )
                if allow_second_pass:
                    logger.info(
                        "Follow-up: re-embedding enhanced question: '%s...'",
                        enhanced_q[:80],
                    )
                    try:
                        t0 = time.perf_counter()
                        enhanced_embedding = await self._generate_embedding(enhanced_q)
                        timing["followup_reembed_ms"] = round(
                            (time.perf_counter() - t0) * 1000.0, 2
                        )
                        if enhanced_embedding and any(
                            v != 0.0 for v in enhanced_embedding[:5]
                        ):
                            query_embedding = enhanced_embedding
                            t0 = time.perf_counter()
                            new_candidates = await self._get_top_candidates(
                                query_embedding, limit=5
                            )
                            timing["followup_candidate_retrieval_ms"] = round(
                                (time.perf_counter() - t0) * 1000.0, 2
                            )
                            if new_candidates:
                                candidates = new_candidates
                                t0 = time.perf_counter()
                                classification = await self._classify_and_match(
                                    enhanced_q, candidates, data_context,
                                    conversation_context,
                                )
                                timing["followup_reclassify_ms"] = round(
                                    (time.perf_counter() - t0) * 1000.0, 2
                                )
                    except Exception as e:
                        logger.warning(
                            "Follow-up re-embedding failed: %s — keeping original", e
                        )

            # ── Step 4: RBAC check (skipped when caller already handled it) ──
            if not skip_rbac:
                try:
                    t0 = time.perf_counter()
                    from services.table_permissions_service import (
                        get_denied_tables_for_user,
                        check_tables_access,
                    )
                    denied_tables = await get_denied_tables_for_user(
                        user_id, client_id, self.db, dataset_id=self.dataset_id
                    )
                    if denied_tables:
                        relevant_tables = classification.get("relevant_tables") or []
                        violations = check_tables_access(relevant_tables, denied_tables)
                        if violations:
                            logger.warning(
                                f"[RBAC] Access denied for user '{user_id}' — "
                                f"restricted tables: {violations}"
                            )
                            timing["rbac_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
                            timing["total_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
                            self._last_timing = timing
                            return {
                                "cache_hit": False,
                                "access_denied": True,
                                "denied_tables_violated": violations,
                                "semantic_signature": self._safe_sig_dict(
                                    classification.get("semantic_signature")
                                ),
                                "query_embedding": None,
                                "terminate_graph": True,
                                "cached_executor_response": {
                                    "console_output": "",
                                    "dataframes": [],
                                    "plotly_charts": [],
                                    "access_denied_message": "This data is not accessible to you.",
                                },
                                "cached_business_response": {
                                    "analysis": "This data is not accessible to you.",
                                },
                                "semantic_cache_match": {
                                    "matched": False,
                                    "matched_question": None,
                                    "matched_question_id": None,
                                    "match_type": None,
                                    "similarity_score": None,
                                    "match_reason": "Access denied by table-level RBAC",
                                },
                            }
                    timing["rbac_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
                except Exception as rbac_err:
                    logger.error(f"[RBAC] Error during access check: {rbac_err}", exc_info=True)

            # ── Step 5: Cache match decision ──
            cache_match = classification.get("semantic_cache_match", {})
            is_match = cache_match.get("matched", False)
            matched_index = cache_match.get("matched_index")

            if (
                is_match
                and matched_index is not None
                and 0 <= matched_index < len(candidates)
            ):
                matched_candidate = candidates[matched_index]
                cached_result = matched_candidate.get("_cached_result")
                _qid = matched_candidate.get("question_id")
                if not cached_result and _qid is not None:
                    ref = self.cache_manager.response_matcher.get_reference_responses(_qid)
                    if ref:
                        planner_text = ref.get("planner_agent_response", "")
                        planner_resp = ref.get("cached_planner_response") or {
                            "plan": planner_text
                        }
                        cached_result = {
                            "planner_response": planner_resp,
                            "code": ref.get("cached_code", ""),
                            "executor_response": ref.get("cached_executor_response", {}),
                            "business_response": ref.get("cached_business_response", {}),
                        }

                # Use explicit None/length check — empty string code is a legitimate
                # cache entry (e.g. narrator-only response); only skip if truly absent.
                if cached_result and cached_result.get("code") is not None:
                    logger.info(
                        f"Cache HIT (LLM-verified) for: '{user_question[:80]}...' "
                        f"(matched: '{matched_candidate['question'][:60]}...')"
                    )
                    result = {
                        "cache_hit": True,
                        "cached_code": cached_result["code"],
                        "semantic_signature": self._safe_sig_dict(
                            classification.get("semantic_signature")
                        ),
                        "query_embedding": query_embedding,
                        "semantic_cache_match": {
                            "matched": True,
                            "matched_question": matched_candidate["question"],
                            "matched_question_id": matched_candidate["question_id"],
                            "match_type": "llm_verified",
                            "similarity_score": matched_candidate["similarity"],
                            "match_reason": cache_match.get(
                                "match_reason", "LLM verified match"
                            ),
                        },
                        "query_routing_type": classification.get(
                            "query_routing_type", "data_analyst"
                        ),
                        "relevant_tables": classification.get("relevant_tables", []),
                        "enhanced_question": classification.get(
                            "enhanced_question", user_question
                        ),
                    }
                    if "llm_usage" in classification:
                        result["usage"] = classification["llm_usage"]
                    timing["total_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
                    self._last_timing = timing
                    logger.info("[Latency] router timings=%s", timing)
                    return result
                else:
                    logger.warning(
                        f"LLM matched candidate [{matched_index}] but no cached code — "
                        f"falling through to cache MISS"
                    )
            elif is_match:
                logger.warning(
                    f"LLM said matched=true but matched_index={matched_index} "
                    f"out of range ({len(candidates)} candidates) — treating as MISS"
                )

            # ── Cache MISS — normal flow ──
            routing_type = classification.get("query_routing_type", "data_analyst")
            relevant_tables = classification.get("relevant_tables", [])

            logger.info(
                f"Cache MISS for: '{user_question[:80]}...' | "
                f"route={routing_type} | tables={relevant_tables}"
            )

            # Build adaptation-aware reference (if available)
            planner_reference = ""
            adaptations = cache_match.get("adaptations", [])
            adapted_index = cache_match.get("matched_index")
            if (
                adaptations
                and adapted_index is not None
                and 0 <= adapted_index < len(candidates)
            ):
                adapted_candidate = candidates[adapted_index]
                planner_reference = self._format_enhanced_planner_reference(
                    adapted_candidate, adaptations
                )

            # Best candidate info for reference fast-path
            best_candidate_info = None
            if candidates:
                best = max(candidates, key=lambda c: c.get("similarity", 0))
                best_candidate_info = {
                    "question_id": best.get("question_id"),
                    "question": best.get("question", ""),
                    "similarity": best.get("similarity", 0.0),
                }

            result = {
                "cache_hit": False,
                "semantic_signature": self._safe_sig_dict(
                    classification.get("semantic_signature")
                ),
                "query_embedding": query_embedding,
                "query_routing_type": routing_type,
                "relevant_tables": relevant_tables,
                "enhanced_question": classification.get(
                    "enhanced_question", user_question
                ),
                "is_followup": classification.get("is_followup", False),
                "intents": classification.get("intents", ["analytical"]),
                "terminate_graph": False,
                "semantic_cache_match": {
                    "matched": False,
                    "matched_question": cache_match.get("matched_question"),
                    "matched_question_id": cache_match.get("matched_question_id"),
                    "match_type": None,
                    "similarity_score": cache_match.get("similarity_score"),
                    "match_reason": cache_match.get(
                        "match_reason", "No cache match found"
                    ),
                    "adaptations": adaptations,
                },
                "planner_reference": planner_reference,
                "best_candidate_info": best_candidate_info,
            }
            if "llm_usage" in classification:
                result["usage"] = classification["llm_usage"]
            timing["total_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            self._last_timing = timing
            logger.info("[Latency] router timings=%s", timing)
            return result

        except Exception as e:
            logger.error(f"Router classification failed: {e}", exc_info=True)
            return {
                "cache_hit": False,
                "semantic_signature": SemanticSignature().to_dict(),
                "query_routing_type": "data_analyst",
                "relevant_tables": [],
                "enhanced_question": user_question,
                "terminate_graph": False,
                "semantic_cache_match": {
                    "matched": False,
                    "matched_question": None,
                    "matched_question_id": None,
                    "match_type": None,
                    "similarity_score": None,
                    "match_reason": f"Classification failed: {str(e)}",
                },
                "best_candidate_info": None,
            }

    # ── Embedding ────────────────────────────────────────────────────────

    @traceable(name="router_embedding")
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding via asyncio.to_thread (non-blocking). Tracks token usage."""
        try:
            import asyncio
            from util.embedding_utils import generate_embedding_with_usage
            embedding, usage = await asyncio.to_thread(generate_embedding_with_usage, text)
            # Accumulate embedding usage (may be called multiple times per request)
            if usage:
                self._embedding_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self._embedding_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                self._embedding_usage["total_tokens"] += usage.get("total_tokens", 0)
                if "provider" not in self._embedding_usage or not self._embedding_usage.get("provider"):
                    self._embedding_usage["provider"] = usage.get("provider")
                    self._embedding_usage["model"] = usage.get("model")
                if usage.get("estimated"):
                    self._embedding_usage["estimated"] = True
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            from util.embedding_utils import get_embedding_dimension
            return [0.0] * get_embedding_dimension()

    # ── Candidate retrieval ──────────────────────────────────────────────

    async def _get_top_candidates(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve top N candidates from Qdrant."""
        return await self.cache_manager.get_top_candidates(
            query_embedding=query_embedding, limit=limit
        )

    # ── Data context (table introductions) ───────────────────────────────

    def _load_data_context(self) -> str:
        """Load brief table names + descriptions (< 2000 chars). Cached."""
        if self._data_context is not None:
            return self._data_context

        try:
            intro_path = (
                resolve_xml_data_sources_dir(self.client_id, self.dataset_id)
                / "meta_information"
                / "table_introductions.xml"
            )
            if not intro_path.exists():
                self._data_context = ""
                return self._data_context

            xml_content = load_xml_prompt_raw(intro_path)
            self._table_introductions_xml = xml_content

            from util.knowledge_filter import _extract_table_descriptions
            descriptions = _extract_table_descriptions(xml_content)

            if not descriptions:
                self._data_context = ""
                return self._data_context

            lines = []
            for table_name, description in descriptions:
                first_sentence = description.split(".")[0].strip() if description else ""
                if first_sentence:
                    lines.append(f"- {table_name}: {first_sentence}")
                else:
                    lines.append(f"- {table_name}")

            context = "Available tables:\n" + "\n".join(lines)
            if len(context) > 2000:
                context = context[:2000] + "\n... (truncated)"

            self._data_context = context
            logger.info(
                f"Loaded table context for client '{self.client_id}': "
                f"{len(descriptions)} tables, {len(context)} chars"
            )
            return self._data_context

        except Exception as e:
            logger.warning(f"Failed to load data context: {e}")
            self._data_context = ""
            return self._data_context

    def get_table_introductions_xml(self) -> str:
        """Return full table_introductions XML (loads on demand). Used by graph nodes."""
        if self._table_introductions_xml is not None:
            return self._table_introductions_xml
        # Trigger load
        self._load_data_context()
        return self._table_introductions_xml or ""

    # ── LLM classify + match ─────────────────────────────────────────────

    @traceable(name="router_classify")
    async def _classify_and_match(
        self,
        user_question: str,
        candidates: List[Dict[str, Any]],
        data_context: str,
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Single LLM call: classify + route + identify tables + match cache."""
        # Build user message
        user_message_parts = []

        # Conversation context for follow-ups
        if conversation_context:
            prev_enhanced = conversation_context.get("previous_enhanced_question", "")
            prev_plan = conversation_context.get("previous_plan", "")
            if prev_enhanced:
                user_message_parts.append(f"PREVIOUS CONTEXT:\n{prev_enhanced}")
                if prev_plan:
                    user_message_parts.append(f"PREVIOUS PLAN:\n{prev_plan}")
                user_message_parts.append(
                    f"\nCURRENT QUERY (may be a follow-up):\n{user_question}"
                )
            else:
                user_message_parts.append(f"QUERY:\n{user_question}")
        else:
            user_message_parts.append(f"QUERY:\n{user_question}")

        # Table context
        if data_context:
            user_message_parts.append(f"\nTABLE CONTEXT:\n{data_context}")

        # Cached candidates
        if candidates:
            candidate_lines = []
            for c in candidates:
                candidate_lines.append(
                    f'  [{c["index"]}] "{c["question"]}" '
                    f'(similarity: {c["similarity"]}, '
                    f'has_cached_code: {c["has_cached_code"]})'
                )
            user_message_parts.append(
                f"\nCACHED QUESTIONS (top {len(candidates)} by embedding similarity):\n"
                + "\n".join(candidate_lines)
            )
        else:
            user_message_parts.append("\nCACHED QUESTIONS: None found.")

        user_message_parts.append("\nRespond with ONLY valid JSON.")
        user_message = "\n".join(user_message_parts)

        # Logging
        prompt_metrics = {
            "user_message_chars": len(user_message),
            "data_context_chars": len(data_context or ""),
            "candidate_count": len(candidates or []),
            "user_message_tokens_est": len(user_message) // 4,
        }
        logger.info("[PromptSize] router metrics=%s", prompt_metrics)

        self._last_inputs = {
            "system_prompt": self.system_prompt,
            "user_message": user_message,
            "prompt_metrics": prompt_metrics,
        }

        # LLM call — use configured model and temperature
        result = await self.llm_client.generate_completion(
            system_prompt=self.system_prompt,
            user_message=user_message,
            provider=self._llm_provider,
            model=self._model_name,
            temperature=self._temperature,
            json_mode=True,
        )
        self._last_usage = result.get("usage")
        response = result.get("content", "")
        usage = result.get("usage")

        # Parse response
        parsed = self._parse_response(
            response, user_question, conversation_context, candidates
        )

        if usage:
            parsed["llm_usage"] = usage

        return parsed

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
            response_text = response.strip()

            # Strip markdown code fences
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            if not response_text.startswith("{"):
                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)

            data = json.loads(response_text)

            # Build SemanticSignature
            qs = data.get("semantic_signature", {})
            semantic_signature = SemanticSignature(
                semantic_variant=data.get("operation_type", "analytical"),
                operation_type=data.get("operation_type", "analytical"),
                modifiers={},
                query_type=qs.get("query_type", "analytical"),
                primary_entity=qs.get("primary_entity", "item"),
                aggregation=qs.get("aggregation", "none"),
                grouping_dimensions=qs.get("grouping_dimensions", []),
                filter_types=qs.get("filter_types", []),
                query_routing_type=data.get("query_routing_type", "data_analyst"),
            )

            # Enhanced question
            enhanced_question = data.get("enhanced_question", "")
            if not enhanced_question:
                enhanced_question = self._generate_fallback_enhanced_question(
                    user_question, conversation_context
                )

            # Relevance check (Layer 2)
            is_relevant = data.get("is_relevant", True)
            routing_type = data.get("query_routing_type", "data_analyst")

            if not is_relevant or routing_type == "irrelevant":
                logger.info(
                    "Router LLM flagged query as irrelevant: '%s'",
                    user_question[:80],
                )
                return {
                    "irrelevant_detected": True,
                    "semantic_signature": semantic_signature.to_dict()
                    if hasattr(semantic_signature, "to_dict") else {},
                    "intents": [],
                    "relevant_tables": [],
                    "enhanced_question": enhanced_question,
                    "query_routing_type": "irrelevant",
                    "is_followup": data.get("is_followup", False),
                    "confidence": data.get("confidence", 0.0),
                    "semantic_cache_match": {
                        "matched": False,
                        "matched_index": None,
                        "matched_question": None,
                        "matched_question_id": None,
                        "similarity_score": None,
                        "match_reason": "Query irrelevant to data analysis",
                        "adaptations": [],
                    },
                }

            # Cache match
            cache_match_raw = data.get("semantic_cache_match", {})
            matched = cache_match_raw.get("matched", False)
            matched_index = cache_match_raw.get("matched_index")
            match_reason = cache_match_raw.get("match_reason", "")
            adaptations = cache_match_raw.get("adaptations", [])

            # Populate from candidates
            matched_question = None
            matched_question_id = None
            similarity_score = None
            if (
                matched and matched_index is not None
                and candidates and 0 <= matched_index < len(candidates)
            ):
                matched_question = candidates[matched_index]["question"]
                matched_question_id = candidates[matched_index]["question_id"]
                similarity_score = candidates[matched_index]["similarity"]
            elif candidates and not matched:
                matched_question = candidates[0]["question"]
                matched_question_id = candidates[0]["question_id"]
                similarity_score = candidates[0]["similarity"]

            return {
                "semantic_signature": semantic_signature,
                "intents": data.get("intents", ["analytical"]) if "intents" in data else [],
                "relevant_tables": data.get("relevant_tables", []),
                "enhanced_question": enhanced_question,
                "query_routing_type": data.get("query_routing_type", "data_analyst"),
                "is_followup": data.get("is_followup", False),
                "confidence": data.get("confidence", 0.5),
                "semantic_cache_match": {
                    "matched": matched,
                    "matched_index": matched_index,
                    "matched_question": matched_question,
                    "matched_question_id": matched_question_id,
                    "similarity_score": similarity_score,
                    "match_reason": match_reason,
                    "adaptations": adaptations,
                },
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse router JSON: {e}")
            return {
                "semantic_signature": SemanticSignature(),
                "intents": [],
                "relevant_tables": [],
                "enhanced_question": self._generate_fallback_enhanced_question(
                    user_question, conversation_context
                ),
                "query_routing_type": "data_analyst",
                "is_followup": False,
                "confidence": 0.0,
                "semantic_cache_match": {
                    "matched": False,
                    "matched_index": None,
                    "matched_question": None,
                    "matched_question_id": None,
                    "similarity_score": None,
                    "match_reason": f"JSON parse error: {str(e)}",
                    "adaptations": [],
                },
            }

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _safe_sig_dict(sig) -> Dict[str, Any]:
        """Safely convert SemanticSignature to dict."""
        if sig is None:
            return {}
        if hasattr(sig, "to_dict"):
            return sig.to_dict()
        if isinstance(sig, dict):
            return sig
        return {}

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
        prev_plan = conversation_context.get("previous_plan", "")
        return self._merge_questions_followup(prev_enhanced, normalized, prev_plan)

    def _merge_questions_followup(
        self, previous: str, current: str, previous_plan: str = ""
    ) -> str:
        """Merge previous enhanced question with follow-up into one clean sentence."""
        current_lower = current.lower().strip()
        prev = previous.strip()

        # Filter / scope
        if re.match(r"^(for|only|just|in)\s+.+", current_lower) or re.match(
            r"^.+\s+only$", current_lower
        ):
            add = current_lower.startswith("for ") and " for " not in prev.lower()
            if add:
                return f"{prev} for {current.strip()}"
            return f"{prev} {current.strip()}"

        # Time
        if re.match(r"^(for\s+)?(the\s+)?(year\s+)?\d{4}$", current_lower) or re.match(
            r"^(for\s+)?(last|past)\s+\d+\s*(months?|years?|quarters?)$", current_lower,
        ):
            time_phrase = current.strip()
            if not time_phrase.lower().startswith("for "):
                time_phrase = (
                    f"for {time_phrase}" if re.match(r"^\d", time_phrase) else time_phrase
                )
            return f"{prev} {time_phrase}"

        # Addition
        if (
            re.match(r"^(also|and|with|plus)\s+", current_lower)
            or "trend" in current_lower
            or "breakdown" in current_lower
        ):
            return f"{prev} {current.strip()}"

        # Grouping
        if re.match(r"^(by|group\s*by|breakdown\s*by)\s+", current_lower):
            return f"{prev} {current.strip()}"

        # Operation-changing
        if re.match(
            r"^(predict|forecast|compare|correlate|summarize|explain)\b", current_lower,
        ):
            return f"{current.strip()} — {prev}"

        return f"{prev} {current.strip()}"

    def _format_enhanced_planner_reference(
        self,
        candidate: Dict[str, Any],
        adaptations: List[Dict[str, str]],
    ) -> str:
        """Format adaptation-aware reference for coder guidance."""
        if not candidate or not adaptations:
            return ""

        similarity = candidate.get("similarity", 0.0)
        ref_question = candidate.get("question", "")

        planner_plan_text = ""
        planner_plan = candidate.get("_planner_plan")
        if planner_plan:
            planner_plan_text = (
                planner_plan.get("plan", "")
                if isinstance(planner_plan, dict) else str(planner_plan)
            )
        if not planner_plan_text:
            cached_result = candidate.get("_cached_result")
            if cached_result:
                pr = cached_result.get("planner_response", {})
                planner_plan_text = (
                    pr.get("plan", "") if isinstance(pr, dict) else str(pr) if pr else ""
                )

        adaptation_lines = []
        for a in adaptations:
            field = a.get("field", "unknown")
            ref_val = a.get("reference_value", "?")
            cur_val = a.get("current_value", "?")
            adaptation_lines.append(f"- **{field}**: `{ref_val}` -> `{cur_val}`")

        parts = [
            f"## REFERENCE EXAMPLE ({similarity:.1%} similarity)",
            "",
            f"**Similar Question:** {ref_question}",
            "",
            "### Required Adaptations",
            "The reference used values that MUST be changed for the current query:",
            "\n".join(adaptation_lines),
        ]

        if planner_plan_text:
            parts.extend([
                "",
                "### Reference Plan (Adapt values as marked above)",
                "```",
                planner_plan_text.strip(),
                "```",
            ])

        parts.extend([
            "",
            "**Instructions:** Follow the reference structure closely. "
            "Make ALL required adaptations. Do not reinvent the approach.",
        ])

        return "\n".join(parts)
