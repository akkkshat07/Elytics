import json
import logging
import boto3
from botocore.exceptions import ClientError
from ..config import settings
logger = logging.getLogger(__name__)

class BedrockClient:
  SONNET_MODEL_ID = 'anthropic.claude-3-5-sonnet-20241022-v2:0'
  HAIKU_MODEL_ID = 'anthropic.claude-3-haiku-20240307-v1:0'

  def __init__(self):
    self.client = boto3.client(service_name='bedrock-runtime', region_name=settings.aws_default_region, aws_access_key_id=settings.aws_access_key_id, aws_secret_access_key=settings.aws_secret_access_key)
    logger.info(f'BedrockClient initialized | region={settings.aws_default_region}')

  def generate(self, system_prompt: str, user_message: str, model_id: str=None, temperature: float=0.0, max_tokens: int=4096) -> str:
    if model_id is None:
      model_id = self.SONNET_MODEL_ID
    body = {'anthropic_version': 'bedrock-2023-05-31', 'max_tokens': max_tokens, 'temperature': temperature, 'system': system_prompt, 'messages': [{'role': 'user', 'content': user_message}]}
    try:
      response = self.client.invoke_model(modelId=model_id, contentType='application/json', accept='application/json', body=json.dumps(body))
      response_body = json.loads(response['body'].read())
      text = response_body['content'][0]['text']
      logger.debug(f'Bedrock response received | model={model_id} | chars={len(text)}')
      return text.strip()
    except ClientError as e:
      error_code = e.response['Error']['Code']
      logger.error(f'Bedrock API error | code={error_code} | msg={e}')
      raise RuntimeError(f'Bedrock call failed: {error_code} — {e}') from e
    except Exception as e:
      logger.error(f'Unexpected error calling Bedrock: {e}')
      raise RuntimeError(f'Bedrock call failed: {e}') from e

  def parse_json_response(self, response_text: str) -> dict:
    import re
    text = response_text.strip()
    if '```json' in text:
      text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
      text = text.split('```')[1].split('```')[0].strip()
    if not text.startswith('{'):
      match = re.search('\\{.*\\}', text, re.DOTALL)
      if match:
        text = match.group(0)
    try:
      return json.loads(text)
    except json.JSONDecodeError as e:
      logger.error(f'Failed to parse LLM JSON response: {e}\nRaw text: {text[:500]}')
      raise ValueError(f'LLM returned invalid JSON: {e}') from e