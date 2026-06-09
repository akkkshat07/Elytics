"""
utils/bedrock.py — Amazon Bedrock LLM Client

PURPOSE:
    Wraps the AWS Bedrock API (Claude 3.5 Sonnet/Haiku) into a simple class that our agents
    can call with a system_prompt + user_message and get back a plain string response.

WHY WE DO IT THIS WAY:
    - All 6 agents share this one client. We instantiate it once in graph.py and pass it through state.
    - This follows the same "shared LLMClient" pattern from the reference code (business_agent.py, router_agent.py).
    - Keeps AWS credentials in one place, easy to change the model in one file.

WINDOWS NOTE:
    This works on Windows with no changes — boto3 is cross-platform.
"""

import json
import logging
import boto3
from botocore.exceptions import ClientError

from ..config import settings

logger = logging.getLogger(__name__)


class BedrockClient:
    """
    Thin wrapper around AWS Bedrock's invoke_model API.
    Supports Claude 3.5 Sonnet and Claude 3 Haiku via Amazon Bedrock.
    """

    # Default model IDs — can be overridden per call
    SONNET_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    HAIKU_MODEL_ID  = "anthropic.claude-3-haiku-20240307-v1:0"

    def __init__(self):
        """
        Initialize the Bedrock runtime client using credentials from config.

        WHY boto3.client?
            boto3 is the official AWS SDK for Python. 'bedrock-runtime' is the specific
            service endpoint for calling LLM inference on Bedrock.
        """
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=settings.aws_default_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
        logger.info(f"BedrockClient initialized | region={settings.aws_default_region}")

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        model_id: str = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """
        Call Claude via Bedrock and return the response text.

        Args:
            system_prompt: The agent's role/instructions (e.g., "You are a SQL expert...")
            user_message:  The specific task or query for this call
            model_id:      Optional model override. Defaults to Claude 3.5 Sonnet.
            temperature:   0.0 = deterministic (good for SQL). 0.7+ = creative (good for insights).
            max_tokens:    Max tokens in the response.

        Returns:
            str: The LLM's response text, stripped of whitespace.

        WHY temperature=0.0 for most agents?
            SQL generation must be exact and reproducible. We use temperature=0.0 for
            Planner, Schema, SQL, and Validation agents. The Insights agent can use
            a slightly higher value like 0.3 for more natural language.
        """
        if model_id is None:
            model_id = self.SONNET_MODEL_ID

        # Bedrock requires messages in Anthropic's messages API format
        # System prompt is passed separately from user messages
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_message}
            ],
        }

        try:
            response = self.client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            # The response body is a streaming blob — read and parse it
            response_body = json.loads(response["body"].read())
            # Claude's response is inside content[0].text
            text = response_body["content"][0]["text"]
            logger.debug(f"Bedrock response received | model={model_id} | chars={len(text)}")
            return text.strip()

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            logger.error(f"Bedrock API error | code={error_code} | msg={e}")
            raise RuntimeError(f"Bedrock call failed: {error_code} — {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error calling Bedrock: {e}")
            raise RuntimeError(f"Bedrock call failed: {e}") from e

    def parse_json_response(self, response_text: str) -> dict:
        """
        Safely parse a JSON response from the LLM, stripping markdown code fences.

        WHY:
            LLMs often wrap JSON in ```json ... ``` blocks. This strips those fences
            before parsing — exactly as the reference router_agent.py does in _parse_response().

        Args:
            response_text: Raw LLM response string

        Returns:
            dict: Parsed JSON dictionary
        """
        import re
        text = response_text.strip()

        # Strip markdown JSON code fences
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Use regex to find the first JSON object if response has extra text
        if not text.startswith("{"):
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}\nRaw text: {text[:500]}")
            raise ValueError(f"LLM returned invalid JSON: {e}") from e
