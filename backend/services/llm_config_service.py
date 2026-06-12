import logging
import asyncio
import time
from typing import Dict, Any, Optional, List
from config.system_config import gemini_use_vertex_ai
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None
try:
    from groq import AsyncGroq
except ImportError:
    AsyncGroq = None
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None
try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None
try:
    import oci
    from oci.generative_ai_inference import GenerativeAiInferenceClient
    from oci.auth.signers import InstancePrincipalsSecurityTokenSigner
except ImportError:
    oci = None
    GenerativeAiInferenceClient = None
logger = logging.getLogger(__name__)
PROVIDER_MODELS = {'openai': ['gpt-5-chat-latest', 'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'o1', 'o1-mini', 'o3', 'o3-mini'], 'groq': ['openai/gpt-oss-120b', 'llama-3.3-70b-versatile', 'llama-3.1-70b-versatile', 'mixtral-8x7b-32768', 'gemma2-9b-it'], 'gemini': ['gemini-3-pro-preview', 'gemini-3-flash-preview', 'gemini-2.5-pro-preview-03-25', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-pro-latest', 'gemini-1.5-flash', 'gemini-1.5-flash-latest'], 'claude': ['claude-sonnet-4-20250514', 'claude-opus-4-20250514', 'claude-sonnet-3.5-20241022', 'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229']}

class LLMConfigService:

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_available_models(self, provider: str) -> List[str]:
        return PROVIDER_MODELS.get(provider, [])

    def get_all_providers(self) -> Dict[str, List[str]]:
        return PROVIDER_MODELS.copy()

    async def validate_api_key(self, provider: str, api_key: str, model: Optional[str]=None) -> Dict[str, Any]:
        try:
            if provider == 'openai':
                return await self._validate_openai(api_key, model)
            elif provider == 'groq':
                return await self._validate_groq(api_key, model)
            elif provider == 'gemini':
                return await self._validate_gemini(api_key, model)
            elif provider == 'claude':
                return await self._validate_claude(api_key, model)
            else:
                return {'valid': False, 'message': f'Unknown provider: {provider}'}
        except Exception as e:
            self.logger.error(f'Validation error for {provider}: {e}')
            return {'valid': False, 'message': 'Validation failed', 'error': str(e)}

    async def _validate_openai(self, api_key: str, model: Optional[str]=None) -> Dict[str, Any]:
        if not AsyncOpenAI:
            return {'valid': False, 'message': 'OpenAI SDK not installed'}
        try:
            client = AsyncOpenAI(api_key=api_key, timeout=10.0)
            try:
                models = await client.models.list()
                return {'valid': True, 'message': 'API key validated successfully'}
            except Exception as list_error:
                test_model = model or 'gpt-4o-mini'
                response = await client.chat.completions.create(model=test_model, messages=[{'role': 'user', 'content': 'Hi'}], max_tokens=5)
                return {'valid': True, 'message': 'API key validated successfully'}
        except Exception as e:
            error_msg = str(e).lower()
            if 'authentication' in error_msg or 'api_key' in error_msg or 'unauthorized' in error_msg:
                return {'valid': False, 'message': 'Invalid API key'}
            elif 'not_found' in error_msg or 'model' in error_msg:
                return {'valid': False, 'message': f"Model '{model}' not found or not accessible"}
            else:
                return {'valid': False, 'message': f'Validation failed: {str(e)}'}

    async def _validate_groq(self, api_key: str, model: Optional[str]=None) -> Dict[str, Any]:
        if not AsyncGroq:
            return {'valid': False, 'message': 'Groq SDK not installed'}
        try:
            client = AsyncGroq(api_key=api_key, timeout=10.0)
            test_model = model or 'llama-3.3-70b-versatile'
            response = await client.chat.completions.create(model=test_model, messages=[{'role': 'user', 'content': 'Hi'}], max_tokens=5)
            return {'valid': True, 'message': 'API key validated successfully'}
        except Exception as e:
            error_msg = str(e).lower()
            if 'authentication' in error_msg or 'api_key' in error_msg or 'unauthorized' in error_msg:
                return {'valid': False, 'message': 'Invalid API key'}
            elif 'not_found' in error_msg or 'model' in error_msg:
                return {'valid': False, 'message': f"Model '{model}' not found or not accessible"}
            else:
                return {'valid': False, 'message': f'Validation failed: {str(e)}'}

    async def _validate_gemini(self, api_key: str, model: Optional[str]=None) -> Dict[str, Any]:
        if not genai:
            return {'valid': False, 'message': 'Google GenAI SDK not installed'}
        try:
            client = genai.Client(api_key=api_key, vertexai=gemini_use_vertex_ai())
            test_model = model or 'gemini-1.5-flash'
            response = await client.aio.models.generate_content(model=test_model, contents='Hi', config=genai_types.GenerateContentConfig(max_output_tokens=1))
            return {'valid': True, 'message': 'API key validated successfully'}
        except Exception as e:
            error_msg = str(e).lower()
            if 'api_key' in error_msg or 'authentication' in error_msg or 'unauthorized' in error_msg:
                return {'valid': False, 'message': 'Invalid API key'}
            elif 'not found' in error_msg or 'model' in error_msg:
                return {'valid': False, 'message': f"Model '{model}' not found or not accessible"}
            else:
                return {'valid': False, 'message': f'Validation failed: {str(e)}'}

    async def _validate_claude(self, api_key: str, model: Optional[str]=None) -> Dict[str, Any]:
        if not AsyncAnthropic:
            return {'valid': False, 'message': 'Anthropic SDK not installed'}
        try:
            client = AsyncAnthropic(api_key=api_key, timeout=10.0)
            test_model = model or 'claude-3-5-sonnet-20241022'
            response = await client.messages.create(model=test_model, max_tokens=5, messages=[{'role': 'user', 'content': 'Hi'}])
            return {'valid': True, 'message': 'API key validated successfully'}
        except Exception as e:
            error_msg = str(e).lower()
            if 'authentication' in error_msg or 'api_key' in error_msg or 'unauthorized' in error_msg:
                return {'valid': False, 'message': 'Invalid API key'}
            elif 'not_found' in error_msg or 'model' in error_msg:
                return {'valid': False, 'message': f"Model '{model}' not found or not accessible"}
            else:
                return {'valid': False, 'message': f'Validation failed: {str(e)}'}

    async def healthcheck(self, provider: str, api_key: str, model: str, timeout: float=5.0) -> Dict[str, Any]:
        start = time.time()
        try:
            if provider == 'openai':
                result = await self._healthcheck_openai(api_key, model, timeout)
            elif provider == 'groq':
                result = await self._healthcheck_groq(api_key, model, timeout)
            elif provider == 'gemini':
                result = await self._healthcheck_gemini(api_key, model, timeout)
            elif provider == 'claude':
                result = await self._healthcheck_claude(api_key, model, timeout)
            else:
                return {'healthy': False, 'latency_ms': 0, 'error': f'Unknown provider: {provider}'}
            latency_ms = int((time.time() - start) * 1000)
            if result.get('valid'):
                return {'healthy': True, 'latency_ms': latency_ms}
            else:
                return {'healthy': False, 'latency_ms': latency_ms, 'error': result.get('message', 'Unknown error')}
        except asyncio.TimeoutError:
            latency_ms = int((time.time() - start) * 1000)
            return {'healthy': False, 'latency_ms': latency_ms, 'error': 'Healthcheck timed out'}
        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            self.logger.error(f'Healthcheck error for {provider}/{model}: {e}')
            return {'healthy': False, 'latency_ms': latency_ms, 'error': f'Healthcheck error for LLM: {provider}/{model}'}

    async def _healthcheck_openai(self, api_key: str, model: str, timeout: float) -> Dict[str, Any]:
        if not AsyncOpenAI:
            return {'valid': False, 'message': 'OpenAI SDK not installed'}
        try:
            client = AsyncOpenAI(api_key=api_key, timeout=timeout)
            model_lower = model.lower()
            request_params = {'model': model, 'messages': [{'role': 'user', 'content': 'Hi'}]}
            response = await asyncio.wait_for(client.chat.completions.create(**request_params), timeout=timeout)
            return {'valid': True, 'message': 'Healthy'}
        except asyncio.TimeoutError:
            return {'valid': False, 'message': 'Request timed out'}
        except Exception as e:
            error_message = str(e)
            if hasattr(e, 'body') and isinstance(e.body, dict):
                error_message = e.body.get('message', str(e))
            elif hasattr(e, 'message'):
                error_message = e.message
            return {'valid': False, 'message': error_message}

    async def _healthcheck_groq(self, api_key: str, model: str, timeout: float) -> Dict[str, Any]:
        if not AsyncGroq:
            return {'valid': False, 'message': 'Groq SDK not installed'}
        try:
            client = AsyncGroq(api_key=api_key, timeout=timeout)
            response = await asyncio.wait_for(client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': 'Hi'}], max_tokens=1), timeout=timeout)
            return {'valid': True, 'message': 'Healthy'}
        except asyncio.TimeoutError:
            return {'valid': False, 'message': 'Request timed out'}
        except Exception as e:
            error_message = str(e)
            if hasattr(e, 'body') and isinstance(e.body, dict):
                error_message = e.body.get('error', {}).get('message', str(e))
            elif hasattr(e, 'message'):
                error_message = e.message
            return {'valid': False, 'message': error_message}

    async def _healthcheck_gemini(self, api_key: str, model: str, timeout: float) -> Dict[str, Any]:
        if not genai:
            return {'valid': False, 'message': 'Google GenAI SDK not installed'}
        try:
            client = genai.Client(api_key=api_key, vertexai=gemini_use_vertex_ai())
            response = await asyncio.wait_for(client.aio.models.generate_content(model=model, contents='Hi', config=genai_types.GenerateContentConfig(max_output_tokens=1)), timeout=timeout)
            return {'valid': True, 'message': 'Healthy'}
        except asyncio.TimeoutError:
            return {'valid': False, 'message': 'Request timed out'}
        except Exception as e:
            error_message = str(e)
            if hasattr(e, 'message'):
                error_message = e.message
            return {'valid': False, 'message': error_message}

    async def _healthcheck_claude(self, api_key: str, model: str, timeout: float) -> Dict[str, Any]:
        if not AsyncAnthropic:
            return {'valid': False, 'message': 'Anthropic SDK not installed'}
        try:
            client = AsyncAnthropic(api_key=api_key, timeout=timeout)
            response = await asyncio.wait_for(client.messages.create(model=model, max_tokens=1, messages=[{'role': 'user', 'content': 'Hi'}]), timeout=timeout)
            return {'valid': True, 'message': 'Healthy'}
        except asyncio.TimeoutError:
            return {'valid': False, 'message': 'Request timed out'}
        except Exception as e:
            error_message = str(e)
            if hasattr(e, 'body') and isinstance(e.body, dict):
                error_message = e.body.get('error', {}).get('message', str(e))
            elif hasattr(e, 'message'):
                error_message = e.message
            return {'valid': False, 'message': error_message}
llm_config_service = LLMConfigService()