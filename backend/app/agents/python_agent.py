import ast
import difflib
import logging
from typing import Dict, Any
from ..utils.bedrock import BedrockClient
from ..state import QueryState

logger = logging.getLogger(__name__)

PYTHON_AGENT_SYSTEM_PROMPT = """You are an expert Python Data Analyst. Your task is to write
Python code using Pandas, NumPy, and Plotly that analyzes data and answers a business question.

The code will be executed in a sandbox environment with these pre-loaded variables:
- `query_results`: List[Dict] — raw data rows from Amazon Redshift (already loaded)
- `pd`: pandas module
- `np`: numpy module
- `px`: plotly.express module
- `go`: plotly.graph_objects module
- `charts`: List — append Plotly chart dicts here: charts.append(fig.to_dict())
- `text_outputs`: List — append string findings here: text_outputs.append("...")
- `statistics`: Dict — store computed metrics here: statistics["key"] = value

RULES:
1. Start with: df = pd.DataFrame(query_results)
2. Handle edge cases: check if df is empty before analysis
3. Every Plotly figure MUST be appended to `charts`: charts.append(fig.to_dict())
4. Store all key numeric findings in `statistics` dict for the Insights agent
5. Append text summaries or findings to `text_outputs`
6. Use plotly.express (px) for most charts — it is simpler and cleaner
7. NEVER use: open(), import os, import sys, import subprocess, exec(), eval()
8. Apply professional chart styling: template="plotly_white", descriptive titles

Respond with ONLY valid JSON:
{
 "python_code": "your complete Python code as a single string",
 "code_explanation": "Plain English explanation of what the code does step by step"
}"""

PYTHON_REPAIR_PROMPT = """The previous Python code you generated crashed during execution.
Review the FAILED CODE and the ERROR TRACEBACK below, and write corrected Python code.

Remember:
- Only return valid Python code inside the JSON payload.
- Fix the logic or syntax error indicated by the traceback.

Respond with ONLY valid JSON:
{
 "python_code": "your complete corrected Python code",
 "code_explanation": "Explanation of what you fixed"
}"""

class PythonAgent:
    def __init__(self, llm_client: BedrockClient):
        self.llm_client = llm_client
        self.max_retries = 3
        self.doom_loop_threshold = 3
        logger.info("PythonAgent initialized (CoreSight Architecture)")

    def _detect_doom_loop(self, current_code: str, failed_codes: list) -> bool:
        if len(failed_codes) < self.doom_loop_threshold:
            return False
            
        last_n = failed_codes[-self.doom_loop_threshold:]
        for prev in last_n:
            ratio = difflib.SequenceMatcher(None, current_code.strip(), prev.strip()).ratio()
            if ratio < 0.92:
                return False
        return True

    def _validate_code_syntax(self, code: str) -> str:
        try:
            ast.parse(code)
            return ""
        except SyntaxError as e:
            return str(e)

    def _describe_data(self, query_results: list) -> str:
        if not query_results:
            return "Dataset: EMPTY — no rows returned"
        columns = list(query_results[0].keys())
        total_rows = len(query_results)
        sample_rows = query_results[:5]
        sample_lines = []
        for i, row in enumerate(sample_rows, 1):
            row_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}' for k, v in row.items())
            sample_lines.append(f" Row {i}: {row_str}")
        return f"Dataset shape: {total_rows} rows × {len(columns)} columns\nColumns: {', '.join(columns)}\nSample rows:\n" + "\n".join(sample_lines)

    def process(self, state: QueryState) -> Dict[str, Any]:
        query_results = state.get('query_results', [])
        plan = state.get('plan', {})
        intent = state.get('intent', plan.get('intent', 'aggregation'))
        user_query = state.get('user_query', '')
        
        # State tracking for retries
        attempts = state.get('python_attempts', 0)
        failed_codes = state.get('failed_codes', [])
        last_error = state.get('error', '')
        previous_code = state.get('generated_python', '')
        
        attempts += 1
        
        if not query_results:
            logger.warning("PythonAgent: No query results to analyze")
            return {
                'generated_python': "# No data\ntext_outputs.append('No data found.')\nstatistics['total_rows']=0", 
                'step_log': ["Python Agent: No data — generating empty-result code"]
            }

        if attempts == 1:
            # First attempt
            logger.info(f"PythonAgent Attempt {attempts}: Generating initial code")
            data_description = self._describe_data(query_results)
            user_message = f"""Write Python code to analyze this business data:

ORIGINAL USER QUESTION: {user_query}

ANALYTICAL INTENT: {intent}
OBJECTIVE: {plan.get('analytical_objective', '')}

DATA AVAILABLE:
{data_description}

Respond with ONLY valid JSON."""
            prompt_to_use = PYTHON_AGENT_SYSTEM_PROMPT
            step_log_msg = f"✅ Python Agent: Generated code (Attempt {attempts})"
        else:
            # Retry after failure
            logger.warning(f"PythonAgent Attempt {attempts}: Regenerating after error: {last_error[:100]}")
            failed_codes.append(previous_code)
            
            user_message = f"""FAILED CODE:
{previous_code}

ERROR TRACEBACK:
{last_error}

ORIGINAL QUESTION: {user_query}

Please fix the error and rewrite the code."""
            prompt_to_use = PYTHON_REPAIR_PROMPT
            step_log_msg = f"🔄 Python Agent: Regenerated code after execution error (Attempt {attempts})"

        # DOOM LOOP CHECK
        if len(failed_codes) >= self.doom_loop_threshold and self._detect_doom_loop(previous_code, failed_codes):
            doom_msg = f"Doom loop detected: The last {self.doom_loop_threshold} failed attempts used nearly identical code. Aborting."
            logger.error(doom_msg)
            return {
                'error': doom_msg,
                'python_attempts': attempts,
                'step_log': [f"❌ Python Agent: {doom_msg}"]
            }

        try:
            response_text = self.llm_client.generate(
                system_prompt=prompt_to_use, 
                user_message=user_message, 
                model_id=BedrockClient.SONNET_MODEL_ID, 
                temperature=0.2, 
                max_tokens=4096
            )
            result = self.llm_client.parse_json_response(response_text)
            python_code = result.get('python_code', '').strip()
            
            if not python_code:
                raise ValueError("LLM returned empty Python code")

            # AST Validation (CoreSight feature)
            syntax_err = self._validate_code_syntax(python_code)
            if syntax_err:
                logger.warning(f"AST syntax validation failed: {syntax_err}")
                return {
                    'generated_python': python_code,
                    'error': f"SyntaxError: {syntax_err}",
                    'python_attempts': attempts,
                    'failed_codes': failed_codes,
                    'step_log': [f"⚠️ Python Agent: AST Syntax Error -> Retrying"]
                }
            
            # Successful generation (clear error so executor runs it)
            return {
                'generated_python': python_code, 
                'error': "",  
                'python_attempts': attempts,
                'failed_codes': failed_codes,
                'step_log': [f"{step_log_msg} | {result.get('code_explanation', '')[:60]}..."]
            }
            
        except Exception as e:
            error_msg = f"PythonAgent failed: {e}"
            logger.error(error_msg)
            return {
                'generated_python': previous_code, 
                'error': error_msg, 
                'python_attempts': attempts,
                'step_log': [f"⚠️ Python Agent: LLM generation failed | {e}"]
            }