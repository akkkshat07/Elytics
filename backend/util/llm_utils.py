from __future__ import annotations
import os
import logging
import json
import asyncio
import re
import time
from functools import cache
from typing import Dict, Any, Optional, List, AsyncGenerator, Tuple
import httpx

def _get_metrics_service():
    try:
        from services.llm_metrics_service import llm_metrics_service
        return llm_metrics_service
    except Exception:
        return None
try:
    from openai import AsyncOpenAI as OpenAIClient
except ImportError:
    OpenAIClient = None
    logging.warning('OpenAI SDK not installed. Run: pip install openai')
try:
    from groq import AsyncGroq as GroqClient
except ImportError:
    GroqClient = None
    logging.warning('Groq SDK not installed. Run: pip install groq')
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None
    logging.warning('Google GenAI SDK not installed. Run: pip install google-genai')
try:
    from anthropic import AsyncAnthropic as AnthropicClient
except ImportError:
    AnthropicClient = None
    logging.warning('Anthropic SDK not installed. Run: pip install anthropic')
from config.system_config import LLM_PROVIDERS, DEFAULT_LLM_PROVIDER, AGENT_CONFIG, gemini_use_vertex_ai
from util.knowledge_filter import _approx_token_count
from util.llm_errors import LLMErrorCategory, classify_error, _extract_retry_after, _retry_after_exceeds_threshold
logger = logging.getLogger(__name__)
CLIENT_INITIALIZED = False
SDK_CLIENTS = {}
_HEALTHCHECK_CACHE: Dict[str, Dict[str, Any]] = {}
_HEALTHCHECK_CACHE_TTL = 15
_HEALTHCHECK_FAILURE_TTL = 300
_CLIENT_LLM_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}
_CLIENT_LLM_CONFIG_CACHE_TTL = 60
_GEMINI_CONTEXT_CACHE: Dict[str, Dict[str, Any]] = {}
_GEMINI_CONTEXT_CACHE_TTL = 3540

@cache
def _get_agent_config(agent_name: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if agent_name and agent_name in AGENT_CONFIG:
        agent_cfg = AGENT_CONFIG[agent_name]
        provider = agent_cfg.get('llm_provider', DEFAULT_LLM_PROVIDER)
        model = agent_cfg.get('model_name')
    else:
        provider = DEFAULT_LLM_PROVIDER
        model = None
    if not model and provider:
        provider_cfg = LLM_PROVIDERS.get(provider, {})
        model = provider_cfg.get('default_model')
    return (provider, model)

def _initialize_sdk_clients() -> Dict[str, Any]:
    global CLIENT_INITIALIZED, SDK_CLIENTS
    if CLIENT_INITIALIZED:
        logger.info('SDK clients already initialized, returning cached clients.')
        return SDK_CLIENTS
    logger.info('Performing one-time initialization of all configured LLM SDK clients...')
    for provider_name, config in LLM_PROVIDERS.items():
        api_key_env_var = config.get('api_key_env_var', '')
        api_key = os.environ.get(api_key_env_var) if api_key_env_var else None
        try:
            if provider_name == 'openai' and OpenAIClient and api_key:
                SDK_CLIENTS['openai'] = OpenAIClient(api_key=api_key, timeout=45.0)
                logger.info('AsyncOpenAI client configured successfully.')
            elif provider_name == 'groq' and GroqClient and api_key:
                SDK_CLIENTS['groq'] = GroqClient(api_key=api_key, timeout=45.0)
                logger.info('AsyncGroq client configured successfully.')
            elif provider_name == 'gemini' and genai and api_key:
                use_v = gemini_use_vertex_ai()
                SDK_CLIENTS['gemini'] = genai.Client(api_key=api_key, vertexai=use_v)
                logger.info('Gemini client configured successfully (%s).', 'Vertex AI' if use_v else 'Gemini API')
            elif provider_name == 'claude' and AnthropicClient and api_key:
                SDK_CLIENTS['claude'] = AnthropicClient(api_key=api_key, timeout=45.0)
                logger.info('AsyncAnthropic client configured successfully.')
        except Exception as e:
            logger.error(f'Failed to initialize {provider_name.capitalize()} client: {e}')
    CLIENT_INITIALIZED = True
    logger.info('All SDK clients initialization completed.')
    return SDK_CLIENTS

def get_sdk_clients() -> Dict[str, Any]:
    return _initialize_sdk_clients()
_sdk_clients_registry = get_sdk_clients()

class LLMClient:

    def __init__(self, agent_name: Optional[str]=None, client_id: Optional[str]=None, db: Any=None, session_id: str='', run_id: str='', user_id: str='', is_load_test: bool=False, load_test_id: Optional[str]=None):
        self.sdk_clients = _sdk_clients_registry
        self.agent_name = agent_name
        self.client_id = client_id
        self.db = db
        self._session_id = session_id
        self._run_id = run_id
        self._user_id = user_id
        self._is_load_test = is_load_test
        self._load_test_id = load_test_id
        self.default_provider, self.default_model = _get_agent_config(agent_name)
        self.client_config = None
        self._provider_chain: List[Dict[str, Any]] = []
        self._provider_chain_index: int = 0
        self.client_specific_sdk_clients = {}
        self._initialized = False
        self._init_error = None
        logger.info(f"LLMClient instance for '{agent_name or 'default'}' | client_id={client_id or 'system'} | provider: {self.default_provider}, model: {self.default_model or 'provider default'}")

    def _is_mongo_manager(self) -> bool:
        return type(self.db).__name__ == 'MongoDBManager'

    async def _fetch_llm_meta_config(self) -> Optional[Dict[str, Any]]:
        if self.db is None:
            return None
        try:
            if self._is_mongo_manager():
                return await self.db.get_llm_meta_config()
            else:
                collection = self.db.llm_meta_config
                return await collection.find_one({'is_active': True})
        except Exception as e:
            logger.error(f'Error fetching LLM meta config: {e}', exc_info=True)
            return None

    async def _fetch_client_configurations(self) -> Optional[Dict[str, Any]]:
        if not self.client_id or self.db is None:
            return None
        try:
            logger.info(f'Fetching LLM config from DB | client_id={self.client_id} | agent={self.agent_name}')
            if self._is_mongo_manager():
                return await self.db.get_llm_configurations(self.client_id)
            else:
                collection = self.db.llm_configurations
                return await collection.find_one({'client_id': self.client_id})
        except Exception as e:
            logger.error(f'Error fetching LLM config | client_id={self.client_id} | error={e}', exc_info=True)
            return None

    def _resolve_api_key(self, config: Dict[str, Any]) -> Optional[str]:
        if config.get('is_platform', False):
            provider_config = LLM_PROVIDERS.get(config.get('provider'), {})
            api_key_env_var = provider_config.get('api_key_env_var')
            return os.getenv(api_key_env_var) if api_key_env_var else None
        return config.get('api_key')

    async def _healthcheck_with_cache(self, provider: str, model: str, api_key: str, config_id: str=None) -> Dict[str, Any]:
        cache_key = f"{provider}:{model}:{config_id or 'default'}"
        now = time.time()
        cached = _HEALTHCHECK_CACHE.get(cache_key)
        if cached and cached.get('expires_at', 0) > now:
            return {**cached['result'], 'cached': True}
        from services.llm_config_service import llm_config_service
        result = await llm_config_service.healthcheck(provider=provider, api_key=api_key, model=model, timeout=5.0)
        ttl = _HEALTHCHECK_CACHE_TTL if result.get('healthy') else _HEALTHCHECK_FAILURE_TTL
        _HEALTHCHECK_CACHE[cache_key] = {'result': result, 'expires_at': now + ttl}
        return {**result, 'cached': False}

    async def _load_client_llm_config(self) -> Optional[Dict[str, Any]]:
        if self._initialized:
            if self._init_error:
                raise self._init_error
            return self.client_config
        cache_key = f"{self.client_id or 'none'}"
        now = time.time()
        cached = _CLIENT_LLM_CONFIG_CACHE.get(cache_key)
        if cached and cached.get('expires_at', 0) > now:
            self.client_config = cached.get('config')
            self._provider_chain = cached.get('provider_chain', [])
            self._provider_chain_index = 0
            self._initialized = True
            return self.client_config
        if not self.client_id or self.db is None:
            self._initialized = True
            return None
        from services.subscription_service import get_client_subscription
        meta_config = await self._fetch_llm_meta_config()
        if not meta_config:
            error = RuntimeError('LLM meta configuration not available from database')
            self._init_error = error
            self._initialized = True
            raise error
        PLATFORM_FALLBACK_ORDER = meta_config.get('platform_fallback_order', [])
        PLATFORM_DEFAULT_CONFIGS = meta_config.get('platform_configs', [])
        MODEL_TIER = meta_config.get('model_tier', {})
        config_doc = await self._fetch_client_configurations()
        configurations = config_doc.get('configurations', []) if config_doc else []
        subscription = await get_client_subscription(self.client_id, self.db)
        plan_name = subscription.get('plan_name', 'freemium').lower()
        plan_features = subscription.get('features', {})
        allow_premium_models = bool(plan_features.get('advanced_agents', False))

        def is_premium_model(model_name: str) -> bool:
            return MODEL_TIER.get(model_name) == 'premium'
        candidates = []
        default_config_id = None
        for cfg in configurations:
            if cfg.get('is_deleted', False) or not cfg.get('is_active', True):
                continue
            if cfg.get('is_default'):
                default_config_id = cfg.get('config_id')
                if not allow_premium_models and is_premium_model(cfg.get('model')):
                    logger.info(f"Skipping premium model {cfg.get('model')} | client_id={self.client_id} | plan={plan_name}")
                    continue
                candidates.append({**cfg, '_source': 'client_default'})
                break
        for fallback_config_id in PLATFORM_FALLBACK_ORDER:
            platform_cfg = next((c for c in PLATFORM_DEFAULT_CONFIGS if c['config_id'] == fallback_config_id), None)
            if platform_cfg:
                if not allow_premium_models and is_premium_model(platform_cfg.get('model')):
                    logger.info(f"Skipping premium platform model {platform_cfg.get('model')} | client_id={self.client_id} | plan={plan_name}")
                    continue
                candidates.append({**platform_cfg, '_source': 'platform_fallback'})
        failed_default_provider = None
        user_message = None
        for candidate_index, candidate in enumerate(candidates):
            provider = candidate.get('provider')
            model = candidate.get('model')
            config_id = candidate.get('config_id', 'unknown')
            is_platform = candidate.get('is_platform', False)
            source = candidate.get('_source')
            if failed_default_provider and provider == failed_default_provider and (source != 'client_default'):
                logger.info(f'Skipping {config_id} | same provider as failed default ({failed_default_provider})')
                continue
            api_key = self._resolve_api_key(candidate)
            if not api_key:
                logger.warning(f'Skipping {config_id} | no API key for {provider}')
                continue
            result = await self._healthcheck_with_cache(provider=provider, model=model, api_key=api_key, config_id=config_id)
            if result.get('healthy'):
                self.client_config = {'provider': provider, 'model': model, 'api_key': api_key, 'is_platform': is_platform, 'is_fallback': source != 'client_default', 'user_message': user_message}
                self._provider_chain = candidates[candidate_index:]
                self._provider_chain_index = 0
                _CLIENT_LLM_CONFIG_CACHE[cache_key] = {'config': dict(self.client_config), 'provider_chain': list(self._provider_chain), 'expires_at': now + _CLIENT_LLM_CONFIG_CACHE_TTL}
                self._initialized = True
                log_level = logging.INFO if source == 'client_default' else logging.WARNING
                logger.log(log_level, f"Using LLM config | client_id={self.client_id} | source={source} | config_id={config_id} | provider={provider} | model={model} | latency_ms={result.get('latency_ms')} | cached={result.get('cached')} | fallback_chain_length={len(self._provider_chain)}")
                return self.client_config
            else:
                error_msg = result.get('error', '')
                if source == 'client_default':
                    failed_default_provider = provider
                    if not is_platform:
                        if error_msg:
                            user_message = f'Your configured LLM ({provider}/{model}) returned an error: "{error_msg}". Please check your configuration in the admin dashboard. Using fallback LLM.'
                        else:
                            user_message = f'Your configured LLM ({provider}/{model}) is not responding. Please check your API key in the admin dashboard. Using fallback LLM.'
                    logger.error(f"Default LLM healthcheck failed | client_id={self.client_id} | config_id={config_id} | provider={provider} | model={model} | error={error_msg} | latency_ms={result.get('latency_ms')}")
                else:
                    logger.warning(f'Fallback LLM healthcheck failed | source={source} | config_id={config_id} | provider={provider} | model={model} | error={error_msg}')
        error = RuntimeError('No LLM configuration is available. All configured LLMs failed healthcheck. Please check your API keys and network connectivity.')
        self._init_error = error
        self._initialized = True
        raise error

    @staticmethod
    async def check_llm_health(client_id: str, db: Any) -> Dict[str, Any]:
        temp_client = LLMClient(agent_name='healthcheck', client_id=client_id, db=db)
        try:
            config = await temp_client._load_client_llm_config()
            return {'healthy': True, 'is_fallback': config.get('is_fallback', False) if config else False, 'user_message': config.get('user_message') if config else None, 'provider': config.get('provider') if config else None, 'model': config.get('model') if config else None}
        except RuntimeError as e:
            return {'healthy': False, 'is_fallback': False, 'user_message': str(e), 'provider': None, 'model': None}

    def _create_dynamic_client(self, provider: str, api_key: str) -> Any:
        cache_key = f'{provider}_{api_key[:10]}'
        if cache_key in self.client_specific_sdk_clients:
            return self.client_specific_sdk_clients[cache_key]
        try:
            client = None
            if provider == 'openai' and OpenAIClient:
                client = OpenAIClient(api_key=api_key, timeout=45.0)
                logger.info('Created dynamic OpenAI client')
            elif provider == 'groq' and GroqClient:
                client = GroqClient(api_key=api_key, timeout=45.0)
                logger.info('Created dynamic Groq client')
            elif provider == 'gemini' and genai:
                use_v = gemini_use_vertex_ai()
                client = genai.Client(api_key=api_key, vertexai=use_v)
                logger.info('Created dynamic Gemini client (%s)', 'Vertex AI' if use_v else 'Gemini API')
            elif provider == 'claude' and AnthropicClient:
                client = AnthropicClient(api_key=api_key, timeout=45.0)
                logger.info('Created dynamic Claude client')
            if client:
                self.client_specific_sdk_clients[cache_key] = client
            return client
        except Exception as e:
            logger.error(f'Failed to create dynamic client for {provider}: {e}')
            return None

    async def _get_provider_and_model(self, provider: Optional[str]=None, model: Optional[str]=None, reasoning_effort: Optional[str]=None) -> Tuple[str, str, Optional[str]]:
        client_cfg = await self._load_client_llm_config()
        agent_cfg = AGENT_CONFIG.get(self.agent_name, {})
        default_provider = self.default_provider
        default_model = self.default_model
        default_effort = None
        if client_cfg and client_cfg.get('provider') and client_cfg.get('model'):
            current_provider = client_cfg['provider']
            current_model = client_cfg['model']
        else:
            current_provider = provider or agent_cfg.get('llm_provider') or default_provider
            current_model = model or agent_cfg.get('model_name') or default_model
        current_reasoning_effort = reasoning_effort or agent_cfg.get('reasoning_effort') or default_effort
        logger.info(f"LLM config resolved | client_id={self.client_id} | agent={self.agent_name} | provider={current_provider} | model={current_model} | source={('client_db' if client_cfg else 'system_default')}")
        return (current_provider, current_model, current_reasoning_effort)

    async def _get_sdk_client(self, provider: str) -> Any:
        if self.client_config and self.client_config.get('provider') == provider:
            api_key = self.client_config.get('api_key')
            if api_key:
                dynamic_client = self._create_dynamic_client(provider, api_key)
                if dynamic_client:
                    return dynamic_client
        return self.sdk_clients.get(provider)

    def _advance_to_next_provider(self, failed_provider: str, reason: str) -> Optional[Dict[str, Any]]:
        if not self._provider_chain:
            return None
        self._provider_chain_index += 1
        while self._provider_chain_index < len(self._provider_chain):
            next_candidate = self._provider_chain[self._provider_chain_index]
            next_provider = next_candidate.get('provider')
            if next_provider == failed_provider:
                logger.info(f"Runtime fallback: skipping {next_candidate.get('config_id', 'unknown')} (same provider as failed: {failed_provider})")
                self._provider_chain_index += 1
                continue
            api_key = self._resolve_api_key(next_candidate)
            if not api_key:
                logger.warning(f"Runtime fallback: skipping {next_candidate.get('config_id', 'unknown')} (no API key for {next_provider})")
                self._provider_chain_index += 1
                continue
            self.client_config = {'provider': next_provider, 'model': next_candidate.get('model'), 'api_key': api_key, 'is_platform': next_candidate.get('is_platform', False), 'is_fallback': True, 'user_message': self.client_config.get('user_message') if self.client_config else None}
            cache_key = f"{self.client_id or 'none'}"
            _CLIENT_LLM_CONFIG_CACHE.pop(cache_key, None)
            logger.warning(f"Runtime provider fallback | client_id={self.client_id} | from={failed_provider} | to={next_provider} | model={next_candidate.get('model')} | reason={reason}")
            return next_candidate
        logger.error(f'Runtime provider fallback exhausted | client_id={self.client_id} | failed_provider={failed_provider} | reason={reason} | chain_length={len(self._provider_chain)}')
        return None

    def _prepare_openai_messages(self, system_prompt: str, user_message: str, prior_messages: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
        messages = [{'role': 'system', 'content': system_prompt}]
        if prior_messages:
            for m in prior_messages:
                if isinstance(m, dict) and 'role' in m and ('content' in m):
                    messages.append({'role': m['role'], 'content': m['content']})
        messages.append({'role': 'user', 'content': user_message})
        return messages

    def _prepare_common_api_params(self, model: str, temperature: float, max_tokens: Optional[int], reasoning_effort: Optional[str]=None) -> Dict[str, Any]:
        api_params = {}
        is_reasoning_model = str(model).lower().startswith(('o1', 'o3', 'o4', 'gpt-5'))
        if not is_reasoning_model:
            api_params['temperature'] = temperature
        VALID_REASONING_EFFORTS = {'low', 'medium', 'high', 'minimal'}
        if reasoning_effort and is_reasoning_model:
            if reasoning_effort not in VALID_REASONING_EFFORTS:
                logger.warning(f"Invalid reasoning_effort '{reasoning_effort}'. Defaulting to 'medium'.")
                reasoning_effort = 'medium'
            api_params['reasoning'] = {'effort': reasoning_effort}
        if max_tokens:
            api_params['max_output_tokens'] = max_tokens
        return api_params

    def _estimate_usage(self, system_prompt: str, user_message: str, prior_messages: Optional[List[Dict[str, str]]], content: Optional[str], provider: str, model: Optional[str]) -> Dict[str, Any]:
        prompt_parts = [system_prompt or '', user_message or '']
        if prior_messages:
            for msg in prior_messages:
                if isinstance(msg, dict):
                    prompt_parts.extend([str(msg.get('role', '')), str(msg.get('content', ''))])
        prompt_text = '\n'.join([p for p in prompt_parts if p])
        completion_text = content or ''
        prompt_tokens = _approx_token_count(prompt_text)
        completion_tokens = _approx_token_count(completion_text)
        return {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': prompt_tokens + completion_tokens, 'provider': provider, 'model': model, 'estimated': True}

    async def _emit_metric(self, *, provider: str, model: str, latency_ms: int, usage: Optional[Dict[str, Any]], success: bool, error_type: Optional[str]=None, error_msg: Optional[str]=None, is_load_test: bool=False, load_test_id: Optional[str]=None, session_id: str='', run_id: str='', user_id: str=''):
        svc = _get_metrics_service()
        if svc is None:
            return
        try:
            from services.llm_metrics_service import LLMCallEvent
            event = LLMCallEvent(session_id=session_id or getattr(self, '_session_id', ''), run_id=run_id or getattr(self, '_run_id', ''), client_id=self.client_id or '', user_id=user_id or getattr(self, '_user_id', ''), agent=self.agent_name or 'unknown', provider=provider, model=model or '', latency_ms=latency_ms, prompt_tokens=int((usage or {}).get('prompt_tokens', 0)), completion_tokens=int((usage or {}).get('completion_tokens', 0)), total_tokens=int((usage or {}).get('total_tokens', 0)), success=success, error_type=error_type, error_msg=error_msg, is_load_test=is_load_test, source='load_test' if is_load_test else 'ui', load_test_id=load_test_id)
            asyncio.ensure_future(svc.emit(event))
        except Exception as exc:
            logger.debug('Metric emit failed (non-fatal): %s', exc)

    @staticmethod
    def _safe_int(value: Any) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return 0

    def _normalize_usage(self, usage: Any, *, provider: str, model: Optional[str]) -> Dict[str, Any]:

        def _usage_get(name: str) -> Any:
            if usage is None:
                return None
            if isinstance(usage, dict):
                return usage.get(name)
            return getattr(usage, name, None)
        prompt_tokens = self._safe_int(_usage_get('prompt_tokens') or _usage_get('input_tokens') or _usage_get('prompt_token_count'))
        completion_tokens = self._safe_int(_usage_get('completion_tokens') or _usage_get('output_tokens') or _usage_get('candidates_token_count'))
        total_tokens_provider = self._safe_int(_usage_get('total_tokens') or _usage_get('total_token_count'))
        total_tokens = prompt_tokens + completion_tokens
        usage_info: Dict[str, Any] = {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': total_tokens, 'provider': provider, 'model': model}
        if total_tokens_provider:
            usage_info['total_tokens_provider'] = total_tokens_provider
        extra_fields = ['reasoning_tokens', 'cached_input_tokens', 'cache_creation_input_tokens', 'audio_input_tokens', 'audio_output_tokens', 'image_input_tokens', 'accepted_prediction_tokens', 'rejected_prediction_tokens', 'text_input_tokens', 'text_output_tokens']
        for field in extra_fields:
            value = self._safe_int(_usage_get(field))
            if value:
                usage_info[field] = value
        return usage_info

    async def _get_or_create_gemini_context_cache(self, client, model: str, system_instruction: str) -> Optional[str]:
        if not system_instruction:
            return None
        if len(system_instruction) < 10000:
            return None
        import hashlib
        prompt_hash = hashlib.md5(system_instruction.encode()).hexdigest()[:12]
        client_id = getattr(self, 'client_id', None) or 'default'
        agent_name = getattr(self, 'agent_name', None) or 'default'
        cache_key = f'gemini:{client_id}:{agent_name}:{prompt_hash}'
        now = time.time()
        cached = _GEMINI_CONTEXT_CACHE.get(cache_key)
        if cached and cached.get('expires_at', 0) > now:
            logger.debug(f'Gemini context cache hit | key={cache_key}')
            return cached['cache_name']
        try:
            cache = await client.aio.caches.create(model=model, config=genai_types.CreateCachedContentConfig(system_instruction=system_instruction, ttl='3600s'))
            _GEMINI_CONTEXT_CACHE[cache_key] = {'cache_name': cache.name, 'expires_at': now + _GEMINI_CONTEXT_CACHE_TTL}
            logger.info(f'Gemini context cache created | key={cache_key} | name={cache.name}')
            return cache.name
        except Exception as e:
            logger.debug(f'Gemini context cache creation skipped: {e}')
            return None

    async def _call_gemini(self, client, api_params: Dict[str, Any]) -> Dict[str, Any]:
        max_retries = 3
        base_delay = 2.0
        response = None
        model = api_params.get('model')
        contents = []
        system_instruction = None
        messages = api_params.get('messages', [])
        filtered_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                system_instruction = content
            elif role == 'user':
                filtered_messages.append(genai_types.Content(role='user', parts=[genai_types.Part.from_text(text=content)]))
            elif role == 'assistant' or role == 'model':
                filtered_messages.append(genai_types.Content(role='model', parts=[genai_types.Part.from_text(text=content)]))
        safety_settings = [genai_types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='OFF'), genai_types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='OFF'), genai_types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='OFF'), genai_types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='OFF')]
        model_str = str(model).lower()
        thinking_config = None
        cached_content_name = await self._get_or_create_gemini_context_cache(client, model, system_instruction)
        use_json_mode = api_params.get('json_mode', False) and 'gemini-3' not in model_str
        if cached_content_name:
            config = genai_types.GenerateContentConfig(temperature=api_params.get('temperature', 0.7), top_p=0.95, max_output_tokens=api_params.get('max_tokens', 65535), cached_content=cached_content_name, safety_settings=safety_settings, thinking_config=thinking_config, response_mime_type='application/json' if use_json_mode else None)
        else:
            config = genai_types.GenerateContentConfig(temperature=api_params.get('temperature', 0.7), top_p=0.95, max_output_tokens=api_params.get('max_tokens', 65535), system_instruction=system_instruction, safety_settings=safety_settings, thinking_config=thinking_config, response_mime_type='application/json' if use_json_mode else None)
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(client.aio.models.generate_content(model=model, contents=filtered_messages, config=config), timeout=120.0)
                break
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    delay = min(base_delay * 2 ** attempt, 30.0)
                    logger.warning(f'Gemini request timed out after 120s (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s')
                    await asyncio.sleep(delay)
                else:
                    raise Exception(f'Gemini API timed out after {max_retries} attempts (120s each)')
            except Exception as e:
                category = classify_error('gemini', e)
                if category == LLMErrorCategory.HARD_FAILURE:
                    logger.error(f'Gemini hard failure (not retrying): {e}')
                    raise
                if category == LLMErrorCategory.PROVIDER_FALLBACK:
                    logger.warning(f'Gemini provider-fallback error (not retrying): {e}')
                    raise
                if attempt < max_retries - 1:
                    if _retry_after_exceeds_threshold(e):
                        logger.warning(f'Gemini Retry-After exceeds 30s — falling back to next provider.')
                        raise
                    delay = min(_extract_retry_after(e) or base_delay * 2 ** attempt, 30.0)
                    logger.warning(f'Gemini retryable error (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s: {e}')
                    await asyncio.sleep(delay)
                else:
                    raise
        if not response:
            raise Exception('No response from Gemini API.')
        content_text = ''
        finish_reason = None
        if response.candidates:
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, 'finish_reason', None)
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if getattr(part, 'thought', False):
                        continue
                    if part.text:
                        content_text += part.text
        if not content_text:
            logger.warning(f'Gemini returned empty content. model={model}, finish_reason={finish_reason}')
        usage_info = self._normalize_usage(response.usage_metadata, provider='gemini', model=model)
        return {'content': content_text, 'usage': usage_info}

    async def _call_gemini_stream(self, client, api_params: Dict[str, Any]) -> AsyncGenerator[Tuple[str, Optional[Dict[str, Any]]], None]:
        model = api_params.get('model')
        messages = api_params.get('messages', [])
        system_instruction = None
        filtered_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                system_instruction = content
            elif role == 'user':
                filtered_messages.append(genai_types.Content(role='user', parts=[genai_types.Part.from_text(text=content)]))
            elif role == 'assistant' or role == 'model':
                filtered_messages.append(genai_types.Content(role='model', parts=[genai_types.Part.from_text(text=content)]))
        safety_settings = [genai_types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='OFF'), genai_types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='OFF'), genai_types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='OFF'), genai_types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='OFF')]
        model_str = str(model).lower()
        thinking_config = None
        use_json_mode = api_params.get('json_mode', False) and 'gemini-3' not in model_str
        cached_content_name = await self._get_or_create_gemini_context_cache(client, model, system_instruction)
        if cached_content_name:
            config = genai_types.GenerateContentConfig(temperature=api_params.get('temperature', 0.7), top_p=0.95, max_output_tokens=api_params.get('max_tokens', 65535), cached_content=cached_content_name, safety_settings=safety_settings, thinking_config=thinking_config, response_mime_type='application/json' if use_json_mode else None)
        else:
            config = genai_types.GenerateContentConfig(temperature=api_params.get('temperature', 0.7), top_p=0.95, max_output_tokens=api_params.get('max_tokens', 65535), system_instruction=system_instruction, safety_settings=safety_settings, thinking_config=thinking_config, response_mime_type='application/json' if use_json_mode else None)
        try:
            async for chunk in await client.aio.models.generate_content_stream(model=model, contents=filtered_messages, config=config):
                text_content = ''
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if getattr(part, 'thought', False):
                            continue
                        if part.text:
                            text_content += part.text
                if text_content:
                    yield (text_content, None)
                if chunk.usage_metadata:
                    usage_info = self._normalize_usage(chunk.usage_metadata, provider='gemini', model=model)
                    yield ('__USAGE__', usage_info)
        except Exception as e:
            logger.error(f'Error during Gemini stream: {e}', exc_info=True)
            yield (f'ERROR: {str(e)}', None)

    async def _call_groq(self, client, api_params: Dict[str, Any]) -> Dict[str, Any]:
        response = None
        max_retries = 3
        base_delay = 2.0
        model = api_params.get('model')
        messages = api_params.get('input') or []
        temperature = api_params.get('temperature', 0.7)
        max_tokens = api_params.get('max_output_tokens', 4096)
        use_json_mode = api_params.get('json_mode', False)
        _temp_retried = False
        _json_mode_retried = False
        for attempt in range(max_retries):
            try:
                create_kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, timeout=180.0)
                if use_json_mode and (not _json_mode_retried):
                    create_kwargs['response_format'] = {'type': 'json_object'}
                response = await client.chat.completions.create(**create_kwargs)
                break
            except Exception as e:
                category = classify_error('groq', e)
                err_text = str(e).lower()
                if category == LLMErrorCategory.HARD_FAILURE and (not _temp_retried) and ('temperature' in err_text) and ('unsupported_value' in err_text or 'does not support' in err_text):
                    temperature = None
                    _temp_retried = True
                    logger.warning("Groq model rejected 'temperature'. Retrying without it.")
                    continue
                if category == LLMErrorCategory.HARD_FAILURE and use_json_mode and (not _json_mode_retried) and ('response_format' in err_text or 'json' in err_text):
                    _json_mode_retried = True
                    logger.warning('Groq model rejected response_format=json_object. Retrying without it.')
                    continue
                if category == LLMErrorCategory.HARD_FAILURE:
                    logger.error(f'Groq hard failure (not retrying): {e}')
                    raise
                if category == LLMErrorCategory.PROVIDER_FALLBACK:
                    logger.warning(f'Groq provider-fallback error (not retrying): {e}')
                    raise
                if attempt < max_retries - 1:
                    if _retry_after_exceeds_threshold(e):
                        logger.warning(f'Groq Retry-After exceeds 30s — falling back to next provider.')
                        raise
                    delay = min(_extract_retry_after(e) or base_delay * 2 ** attempt, 30.0)
                    logger.warning(f'Groq retryable error (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s: {e}')
                    await asyncio.sleep(delay)
                else:
                    raise
        if not response:
            raise Exception('No response from Groq API after retries.')
        content_text = None
        try:
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if getattr(choice, 'message', None) and getattr(choice.message, 'content', None):
                    content_text = choice.message.content
                elif getattr(choice, 'text', None):
                    content_text = choice.text
        except Exception:
            content_text = None
        if content_text is None:
            try:
                content_text = str(response)
            except Exception:
                content_text = ''
        usage = getattr(response, 'usage', None)
        usage_info = self._normalize_usage(usage, provider='groq', model=model)
        return {'content': content_text, 'usage': usage_info}

    async def _call_claude(self, client, api_params: Dict[str, Any]) -> Dict[str, Any]:
        response = None
        max_retries = 3
        base_delay = 2.0
        model = api_params.get('model')
        system = api_params.get('system', '')
        messages = api_params.get('messages', [])
        temperature = api_params.get('temperature', 0.7)
        max_tokens = api_params.get('max_tokens', 4096)
        try:
            system_payload = [{'type': 'text', 'text': system, 'cache_control': {'type': 'ephemeral'}}] if system else system
        except Exception:
            system_payload = system
        for attempt in range(max_retries):
            try:
                response = await client.messages.create(model=model, system=system_payload, messages=messages, temperature=temperature, max_tokens=max_tokens)
                break
            except Exception as e:
                category = classify_error('claude', e)
                if category == LLMErrorCategory.HARD_FAILURE:
                    logger.error(f'Claude hard failure (not retrying): {e}')
                    raise
                if category == LLMErrorCategory.PROVIDER_FALLBACK:
                    logger.warning(f'Claude provider-fallback error (not retrying): {e}')
                    raise
                if attempt < max_retries - 1:
                    if _retry_after_exceeds_threshold(e):
                        logger.warning(f'Claude Retry-After exceeds 30s — falling back to next provider.')
                        raise
                    delay = min(_extract_retry_after(e) or base_delay * 2 ** attempt, 30.0)
                    logger.warning(f'Claude retryable error (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s: {e}')
                    await asyncio.sleep(delay)
                else:
                    raise
        if not response:
            raise Exception('No response from Claude API after retries.')
        content_text = None
        try:
            if hasattr(response, 'content') and response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content_text = block.text
                        break
                    elif hasattr(block, 'type') and block.type == 'text':
                        content_text = getattr(block, 'text', None)
                        break
        except Exception:
            content_text = None
        if content_text is None:
            try:
                content_text = str(response)
            except Exception:
                content_text = ''
        usage = getattr(response, 'usage', None)
        usage_info = self._normalize_usage(usage, provider='claude', model=model)
        return {'content': content_text, 'usage': usage_info}

    async def _call_openai(self, client: OpenAIClient, api_params: Dict[str, Any]) -> Dict[str, Any]:
        response = None
        max_retries = 3
        base_delay = 2.0
        use_json_mode = api_params.pop('json_mode', False)
        if use_json_mode:
            api_params['text'] = {'format': {'type': 'json_object'}}
        _temp_retried = False
        for attempt in range(max_retries):
            try:
                response = await client.responses.create(**api_params, timeout=180.0)
                api_params.pop('input', None)
                break
            except Exception as e:
                category = classify_error('openai', e)
                err_text = str(e).lower()
                if category == LLMErrorCategory.HARD_FAILURE and (not _temp_retried) and ('temperature' in err_text) and ('unsupported_value' in err_text or 'does not support' in err_text):
                    api_params.pop('temperature', None)
                    _temp_retried = True
                    logger.warning("OpenAI model rejected 'temperature'. Retrying without it.")
                    continue
                if category == LLMErrorCategory.HARD_FAILURE:
                    logger.error(f'OpenAI hard failure (not retrying): {e}')
                    raise
                if category == LLMErrorCategory.PROVIDER_FALLBACK:
                    logger.warning(f'OpenAI provider-fallback error (not retrying): {e}')
                    raise
                if attempt < max_retries - 1:
                    if _retry_after_exceeds_threshold(e):
                        logger.warning(f'OpenAI Retry-After exceeds 30s — falling back to next provider.')
                        raise
                    delay = min(_extract_retry_after(e) or base_delay * 2 ** attempt, 30.0)
                    logger.warning(f'OpenAI retryable error (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s: {e}')
                    await asyncio.sleep(delay)
                else:
                    raise
        if not response:
            raise Exception('No response from OpenAI API after retries.')
        content_text = None
        try:
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if getattr(choice, 'message', None) is not None and getattr(choice.message, 'content', None) is not None:
                    content_text = choice.message.content
                elif getattr(choice, 'text', None) is not None:
                    content_text = choice.text
        except Exception:
            content_text = None
        if content_text is None and getattr(response, 'output_text', None) is not None:
            content_text = response.output_text
        if content_text is None and getattr(response, 'output', None):
            try:
                pieces = []
                for out_item in response.output:
                    content_list = getattr(out_item, 'content', None) or []
                    for msg in content_list:
                        if getattr(msg, 'text', None) is not None:
                            pieces.append(msg.text)
                        elif getattr(msg, 'value', None) is not None:
                            pieces.append(str(msg.value))
                        else:
                            try:
                                pieces.append(str(msg))
                            except Exception:
                                pass
                if pieces:
                    content_text = '\n'.join(pieces)
            except Exception:
                content_text = None
        if content_text is None:
            try:
                content_text = str(response)
            except Exception:
                content_text = ''
        usage = getattr(response, 'usage', None)
        usage_info = self._normalize_usage(usage, provider='openai', model=api_params.get('model'))
        return {'content': content_text, 'usage': usage_info}

    async def _call_openai_stream(self, client: OpenAIClient, api_params: Dict[str, Any]) -> AsyncGenerator[Tuple[str, Optional[Dict[str, Any]]], None]:
        api_params['stream'] = True
        usage_info = None
        try:
            if hasattr(client, 'responses'):
                stream = await client.responses.create(**api_params, timeout=180.0)
                if 'input' in api_params:
                    del api_params['input']
            elif hasattr(client, 'chat'):
                stream = await client.chat.completions.create(**api_params, timeout=180.0)
                if 'messages' in api_params:
                    del api_params['messages']
            else:
                raise ValueError(f'Unsupported client type: {type(client)}')
            async for chunk in stream:
                text_content = None
                usage = None
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = chunk.usage
                elif hasattr(chunk, 'response') and getattr(chunk.response, 'usage', None):
                    usage = chunk.response.usage
                if usage:
                    usage_info = self._normalize_usage(usage, provider='openai' if hasattr(client, 'responses') else 'groq', model=api_params.get('model'))
                if hasattr(chunk, 'delta') and chunk.delta:
                    text_content = chunk.delta
                elif hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'content') and choice.delta.content:
                        text_content = choice.delta.content
                elif hasattr(chunk, 'output') and chunk.output:
                    try:
                        for output_item in chunk.output:
                            if hasattr(output_item, 'content') and output_item.content:
                                for content_item in output_item.content:
                                    if hasattr(content_item, 'text') and content_item.text:
                                        text_content = content_item.text
                                        break
                    except Exception as e:
                        pass
                if text_content:
                    yield (str(text_content), None)
            if usage_info:
                yield ('__USAGE__', usage_info)
        except asyncio.TimeoutError:
            logger.error('OpenAI stream read timeout after 120 seconds')
            yield ('ERROR: Request timeout occurred while reading the stream.', None)
        except httpx.ReadTimeout:
            logger.error('HTTPX Read timeout during OpenAI stream')
            yield ('ERROR: HTTP read timeout occurred during streaming.', None)
        except Exception as e:
            logger.error(f'Error during OpenAI stream: {e}', exc_info=True)
            yield ('ERROR: An unexpected error occurred with the OpenAI stream.', None)

    async def generate_completion(self, system_prompt: str, user_message: str, reasoning_effort: Optional[str]=None, temperature: float=0.7, provider: Optional[str]=None, model: Optional[str]=None, max_tokens: Optional[int]=None, prior_messages: Optional[List[Dict[str, str]]]=None, json_mode: bool=False) -> Dict[str, Any]:
        try:
            current_provider, current_model, current_reasoning_effort = await self._get_provider_and_model(provider, model, reasoning_effort)
        except Exception as e:
            logger.exception(f'Failed to resolve LLM provider/model: {e}')
            return {'content': None, 'error': str(e), 'usage': None}
        providers_tried: List[str] = []
        last_error: Optional[str] = None
        while True:
            if current_provider in providers_tried:
                break
            providers_tried.append(current_provider)
            client = await self._get_sdk_client(current_provider)
            if not client:
                logger.warning(f"No SDK client for provider '{current_provider}', skipping.")
                next_candidate = self._advance_to_next_provider(current_provider, 'no_sdk_client')
                if not next_candidate:
                    last_error = f"Provider '{current_provider}' SDK is unavailable and no fallback succeeded."
                    break
                current_provider = next_candidate['provider']
                current_model = next_candidate['model']
                continue
            logger.info(f"Generating completion | provider={current_provider} | model={current_model} | agent={self.agent_name or 'default'} | temperature={temperature} | max_tokens={max_tokens} | reasoning_effort={current_reasoning_effort or 'N/A'}")
            _t0 = time.perf_counter()
            try:
                if current_provider == 'openai':
                    messages = self._prepare_openai_messages(system_prompt, user_message, prior_messages)
                    api_params = self._prepare_common_api_params(current_model, temperature, max_tokens, current_reasoning_effort)
                    api_params.update({'model': current_model, 'input': messages})
                    if json_mode:
                        api_params['json_mode'] = True
                    response_data = await self._call_openai(client, api_params)
                elif current_provider == 'groq':
                    messages = self._prepare_openai_messages(system_prompt, user_message, prior_messages)
                    api_params = self._prepare_common_api_params(current_model, temperature, max_tokens, current_reasoning_effort)
                    api_params.update({'model': current_model, 'input': messages})
                    if json_mode:
                        api_params['json_mode'] = True
                    response_data = await self._call_groq(client, api_params)
                elif current_provider == 'gemini':
                    messages = [{'role': 'system', 'content': system_prompt}]
                    if prior_messages:
                        messages.extend(prior_messages)
                    messages.append({'role': 'user', 'content': user_message})
                    api_params = {'model': current_model, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'reasoning_effort': current_reasoning_effort}
                    if json_mode:
                        api_params['json_mode'] = True
                    response_data = await self._call_gemini(client, api_params)
                elif current_provider == 'claude':
                    messages = []
                    if prior_messages:
                        messages.extend(prior_messages)
                    messages.append({'role': 'user', 'content': user_message})
                    api_params = {'model': current_model, 'system': system_prompt, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens or 4096}
                    response_data = await self._call_claude(client, api_params)
                else:
                    raise NotImplementedError(f"Provider '{current_provider}' non-streaming completion not implemented.")
                _latency = int((time.perf_counter() - _t0) * 1000)
                await self._emit_metric(provider=current_provider, model=current_model, latency_ms=_latency, usage=response_data.get('usage'), success=True, is_load_test=getattr(self, '_is_load_test', False), load_test_id=getattr(self, '_load_test_id', None), session_id=getattr(self, '_session_id', ''), run_id=getattr(self, '_run_id', ''), user_id=getattr(self, '_user_id', ''))
                return {'content': response_data.get('content'), 'error': None, 'usage': response_data.get('usage')}
            except Exception as e:
                _latency = int((time.perf_counter() - _t0) * 1000)
                last_error = str(e)
                category = classify_error(current_provider, e)
                _etype = 'hard_failure' if category == LLMErrorCategory.HARD_FAILURE else 'auth' if category == LLMErrorCategory.PROVIDER_FALLBACK else 'retryable'
                await self._emit_metric(provider=current_provider, model=current_model, latency_ms=_latency, usage=None, success=False, error_type=_etype, error_msg=last_error[:300], is_load_test=getattr(self, '_is_load_test', False), load_test_id=getattr(self, '_load_test_id', None), session_id=getattr(self, '_session_id', ''), run_id=getattr(self, '_run_id', ''), user_id=getattr(self, '_user_id', ''))
                if category == LLMErrorCategory.HARD_FAILURE:
                    logger.error(f'Hard failure on {current_provider} (no fallback): {e}')
                    return {'content': None, 'error': last_error, 'usage': None}
                logger.warning(f'Provider {current_provider} failed [{category.value}], attempting runtime fallback: {e}')
                next_candidate = self._advance_to_next_provider(current_provider, f'{category.value}: {type(e).__name__}')
                if not next_candidate:
                    break
                current_provider = next_candidate['provider']
                current_model = next_candidate['model']
        error_msg = last_error or 'All configured LLM providers failed. Please try again later.'
        logger.error(f'All LLM providers exhausted | client_id={self.client_id} | agent={self.agent_name} | tried={providers_tried} | last_error={last_error}')
        return {'content': None, 'error': error_msg, 'usage': None}

    async def generate_completion_stream(self, system_prompt: str, user_message: str, reasoning_effort: Optional[str]=None, temperature: float=0.7, provider: Optional[str]=None, model: Optional[str]=None, max_tokens: Optional[int]=None, prior_messages: Optional[List[Dict[str, str]]]=None) -> AsyncGenerator[Tuple[str, Optional[Dict[str, Any]]], None]:
        try:
            current_provider, current_model, current_reasoning_effort = await self._get_provider_and_model(provider, model, reasoning_effort)
        except Exception as e:
            logger.exception(f'Failed to resolve LLM provider/model for stream: {e}')
            yield (f'ERROR: {str(e)}', None)
            return
        providers_tried: List[str] = []
        while True:
            if current_provider in providers_tried:
                yield ('ERROR: All configured LLM providers failed.', None)
                return
            providers_tried.append(current_provider)
            client = await self._get_sdk_client(current_provider)
            if not client:
                logger.warning(f"No SDK client for provider '{current_provider}' (stream), skipping.")
                next_candidate = self._advance_to_next_provider(current_provider, 'no_sdk_client')
                if not next_candidate:
                    yield (f"ERROR: Provider '{current_provider}' SDK unavailable and no fallback succeeded.", None)
                    return
                current_provider = next_candidate['provider']
                current_model = next_candidate['model']
                continue
            logger.info(f"Streaming completion | provider={current_provider} | model={current_model} | agent={self.agent_name or 'default'}")
            tokens_yielded = 0
            failure_exc: Optional[Exception] = None
            failure_category: Optional[LLMErrorCategory] = None
            _stream_t0 = time.perf_counter()
            _stream_usage: Optional[Dict[str, Any]] = None
            try:
                if current_provider in ('openai', 'groq'):
                    if current_provider == 'openai':
                        messages = self._prepare_openai_messages(system_prompt, user_message, prior_messages)
                        api_params = self._prepare_common_api_params(current_model, temperature, max_tokens, current_reasoning_effort)
                        api_params.update({'model': current_model, 'input': messages})
                    else:
                        messages = self._prepare_openai_messages(system_prompt, user_message, prior_messages)
                        api_params = self._prepare_common_api_params(current_model, temperature, max_tokens, current_reasoning_effort)
                        api_params.update({'model': current_model, 'messages': messages})
                    async for token, usage in self._call_openai_stream(client, api_params):
                        if isinstance(token, str) and token.startswith('ERROR:'):
                            failure_exc = Exception(token)
                            failure_category = classify_error(current_provider, failure_exc)
                            break
                        if token == '__USAGE__' and usage:
                            _stream_usage = usage
                        tokens_yielded += 1
                        yield (token, usage)
                elif current_provider == 'gemini':
                    messages = [{'role': 'system', 'content': system_prompt}]
                    if prior_messages:
                        messages.extend(prior_messages)
                    messages.append({'role': 'user', 'content': user_message})
                    api_params = {'model': current_model, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'reasoning_effort': current_reasoning_effort}
                    async for token, usage in self._call_gemini_stream(client, api_params):
                        if isinstance(token, str) and token.startswith('ERROR:'):
                            failure_exc = Exception(token)
                            failure_category = classify_error(current_provider, failure_exc)
                            break
                        if token == '__USAGE__' and usage:
                            _stream_usage = usage
                        tokens_yielded += 1
                        yield (token, usage)
                elif current_provider == 'claude':
                    claude_messages = []
                    if prior_messages:
                        claude_messages.extend(prior_messages)
                    claude_messages.append({'role': 'user', 'content': user_message})
                    api_params = {'model': current_model, 'system': system_prompt, 'messages': claude_messages, 'temperature': temperature, 'max_tokens': max_tokens or 4096}
                    async with client.messages.stream(**api_params) as stream:
                        async for text in stream.text_stream:
                            tokens_yielded += 1
                            yield (text, None)
                        try:
                            final_message = await stream.get_final_message()
                            if hasattr(final_message, 'usage'):
                                usage_info = self._normalize_usage(final_message.usage, provider='claude', model=current_model)
                                _stream_usage = usage_info
                                yield ('__USAGE__', usage_info)
                        except Exception:
                            pass
                else:
                    yield (f"ERROR: Provider '{current_provider}' does not support streaming.", None)
                    return
            except Exception as e:
                failure_exc = e
                failure_category = classify_error(current_provider, e)
            if failure_exc is None:
                _latency_s = int((time.perf_counter() - _stream_t0) * 1000)
                await self._emit_metric(provider=current_provider, model=current_model, latency_ms=_latency_s, usage=_stream_usage, success=True, is_load_test=getattr(self, '_is_load_test', False), load_test_id=getattr(self, '_load_test_id', None), session_id=getattr(self, '_session_id', ''), run_id=getattr(self, '_run_id', ''), user_id=getattr(self, '_user_id', ''))
                return
            _latency_f = int((time.perf_counter() - _stream_t0) * 1000)
            _etype_f = 'hard_failure' if failure_category == LLMErrorCategory.HARD_FAILURE else 'auth' if failure_category == LLMErrorCategory.PROVIDER_FALLBACK else 'retryable'
            await self._emit_metric(provider=current_provider, model=current_model, latency_ms=_latency_f, usage=None, success=False, error_type=_etype_f, error_msg=str(failure_exc)[:300] if failure_exc else None, is_load_test=getattr(self, '_is_load_test', False), load_test_id=getattr(self, '_load_test_id', None), session_id=getattr(self, '_session_id', ''), run_id=getattr(self, '_run_id', ''), user_id=getattr(self, '_user_id', ''))
            if tokens_yielded > 0:
                logger.error(f'Stream provider {current_provider} failed after {tokens_yielded} tokens. Cannot fall back mid-stream.')
                yield (f'ERROR: Stream interrupted after partial content from {current_provider}.', None)
                return
            if failure_category == LLMErrorCategory.HARD_FAILURE:
                yield (f'ERROR: {str(failure_exc)}', None)
                return
            logger.warning(f"Stream provider {current_provider} failed before yielding content [{(failure_category.value if failure_category else 'unknown')}], attempting fallback: {failure_exc}")
            next_candidate = self._advance_to_next_provider(current_provider, f"{(failure_category.value if failure_category else 'unknown')}: {type(failure_exc).__name__}")
            if not next_candidate:
                yield ('ERROR: All configured LLM providers failed.', None)
                return
            current_provider = next_candidate['provider']
            current_model = next_candidate['model']

    def extract_content(self, response: Dict[str, Any]) -> Optional[str]:
        if response.get('error'):
            logger.error(f"Error in API response: {response['error']}")
            return None
        return response.get('content')