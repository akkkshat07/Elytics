from __future__ import annotations
import asyncio
import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from domains.dashboard.model import DashboardReport, DashboardReportResult
from domains.dashboard.repository import DashboardRepository
from util.llm_utils import LLMClient
from util.time_utils import utcnow
logger = logging.getLogger(__name__)

def _normalize_dashboard_dataset_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None

class DuplicateReportFromConversationError(Exception):
    pass

class AdhocQuickUploadReportBlockedError(Exception):
    pass

def _strip_iteration_headers(code: str) -> str:
    lines = code.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if re.match('#\\s*---.*Code Summary\\s*---', stripped):
            continue
        if re.match('# Iteration \\d+\\b', stripped):
            continue
        cleaned.append(line)
    result: list[str] = []
    blank_run = 0
    for line in cleaned:
        if line.strip() == '':
            blank_run += 1
            if blank_run <= 1:
                result.append(line)
        else:
            blank_run = 0
            result.append(line)
    return '\n'.join(result).strip()

def _build_executable_code(code: str) -> str:
    final_result_match = re.search('\\bFINAL_RESULT\\s*=\\s*\\{', code)
    if not final_result_match:
        logger.debug('_build_executable_code: no FINAL_RESULT found, returning as-is')
        return code.strip()
    pre_code = code[:final_result_match.start()].rstrip()
    pre_code = re.sub('\\n[ \\t]*(?:if|elif|else|for|while|with|try|except|finally|def|class)[^\\n]*:\\s*$', '', pre_code)
    fr_substr = code[final_result_match.start():]
    table_expr: Optional[str] = None
    table_match = re.search('[\'\\"]table[\'\\"]\\s*:\\s*((?:[a-zA-Z_][a-zA-Z0-9_]*)(?:\\.[a-zA-Z_]\\w*(?:\\([^)]*\\))?)*)\\s*\\.to_dict\\(', fr_substr)
    if table_match:
        table_expr = table_match.group(1).strip()
    else:
        plain_match = re.search('[\'\\"]table[\'\\"]\\s*:\\s*([a-zA-Z_][a-zA-Z0-9_]*)\\s*[,\\n\\}]', fr_substr)
        if plain_match:
            var_name = plain_match.group(1).strip()
            assign_match = re.search('\\b' + re.escape(var_name) + '\\s*=\\s*((?:[a-zA-Z_][a-zA-Z0-9_]*)(?:\\.[a-zA-Z_]\\w*(?:\\([^)]*\\))?)*)\\s*\\.to_dict\\(', pre_code)
            if assign_match:
                table_expr = assign_match.group(1).strip()
            else:
                table_expr = None
    kpis_block = ''
    kpis_match = re.search('[\'\\"]kpis[\'\\"]\\s*:\\s*\\{([^}]+)\\}', fr_substr, re.DOTALL)
    if kpis_match:
        kpis_body = kpis_match.group(1).strip()
        kpis_block = f'\n_generated_text_output_0_ = str({{\n    {kpis_body}\n}})'
    has_fig = bool(re.search('\\bfig\\s*=\\s*(px|go)\\.', pre_code))
    chart_line = '_generated_plotly_fig_0_ = fig' if has_fig else ''
    table_line = f'_generated_dataframe_0_ = {table_expr}' if table_expr else ''
    if not chart_line and (not table_line):
        logger.warning('_build_executable_code: could not extract chart or table variable; returning as-is')
        return code.strip()
    output_lines = ['', '# ── Dashboard outputs — regenerated from live data on every execution ──']
    if not has_fig and table_expr:
        output_lines.append('import plotly.express as _px_auto')
        output_lines.append(f'_auto_df = {table_expr}.reset_index(drop=True)')
        output_lines.append('_auto_cols = _auto_df.columns.tolist()')
        output_lines.append("_auto_fig = _px_auto.bar(_auto_df, x=_auto_cols[0], y=_auto_cols[1] if len(_auto_cols) > 1 else _auto_cols[0], title=f'{_auto_cols[1] if len(_auto_cols) > 1 else _auto_cols[0]} by {_auto_cols[0]}') if len(_auto_df) > 0 else None")
        output_lines.append('_generated_plotly_fig_0_ = _auto_fig')
    if chart_line:
        output_lines.append(chart_line)
    if table_line:
        output_lines.append(table_line)
    elif chart_line:
        output_lines.append('')
        output_lines.append(code[final_result_match.start():].strip())
        logger.warning('_build_executable_code: table extraction failed; preserving FINAL_RESULT as fallback')
    if kpis_block:
        output_lines.append(kpis_block)
    result = pre_code + '\n' + '\n'.join(output_lines)
    logger.debug('_build_executable_code: replaced FINAL_RESULT, chart=%s table_expr=%s', bool(chart_line), table_expr)
    return result.strip()

def _normalize_dataframes(raw_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for df in raw_frames:
        if not isinstance(df, dict):
            continue
        frame: Dict[str, Any] = {'name': df.get('name', ''), 'column_mapping': df.get('column_mapping', {}), 'column_metadata': df.get('column_metadata', {})}
        if df.get('json_data'):
            frame['json_data'] = df['json_data']
        elif isinstance(df.get('data'), list) and df['data']:
            columns = list(df['data'][0].keys()) if df['data'] else []
            rows = [[row.get(c) for c in columns] for row in df['data']]
            frame['json_data'] = json.dumps({'columns': columns, 'data': rows})
        else:
            continue
        out.append(frame)
    return out

async def execute_report_code(code: str, client_id: str, db: AsyncIOMotorDatabase, dataset_id: Optional[str]=None) -> DashboardReportResult:
    try:
        compile(code, '<dashboard_report>', 'exec')
    except SyntaxError as syn_err:
        logger.error('Dashboard report has invalid code (syntax error): %s', syn_err)
        return DashboardReportResult(dataframes=[], plotly_charts=[], text_outputs=[], executed_at=utcnow(), status='error', error=f'Report code has a syntax error and needs to be re-saved: {syn_err}')
    try:
        from agents.executor_agent import ExecutorAgent
        executor = ExecutorAgent(client_id=client_id, db=db, dataset_id=dataset_id)
        result = await executor.process(generated_code=code, python_agent_task_id=f'dashboard_{uuid.uuid4().hex[:8]}')
        error_text: str = result.get('error_text', '')
        raw_response: str = result.get('generated_response', '')
        if error_text:
            logger.warning('Dashboard code execution error client=%s: %s', client_id, error_text[:300])
            return DashboardReportResult(dataframes=[], plotly_charts=[], text_outputs=[], executed_at=utcnow(), status='error', error=error_text[:1000])
        parsed: Dict[str, Any] = {}
        if isinstance(raw_response, str) and raw_response.strip().startswith('{'):
            try:
                parsed = json.loads(raw_response)
            except json.JSONDecodeError:
                parsed = {}
        elif isinstance(raw_response, dict):
            parsed = raw_response
        return DashboardReportResult(dataframes=_normalize_dataframes(parsed.get('dataframes', [])), plotly_charts=parsed.get('plotly_charts', []), text_outputs=parsed.get('text_outputs', []), executed_at=utcnow(), status='success')
    except Exception as e:
        logger.error('Dashboard execution failed client=%s: %s', client_id, e, exc_info=True)
        return DashboardReportResult(dataframes=[], plotly_charts=[], text_outputs=[], executed_at=utcnow(), status='error', error=str(e)[:1000])

async def _generate_report_description(query: str, title: str, client_id: str, db) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    try:
        llm = LLMClient(agent_name='narrator_agent', client_id=client_id, db=db)
        await llm._load_client_llm_config()
        response = await llm.generate_completion(system_prompt="You write descriptions for business dashboard report cards. Given the user's question, write EXACTLY 1-2 sentences (minimum 8 words, maximum 20 words total) that clearly explain: (1) what metric or data is being measured, (2) how it is grouped or segmented, (3) any filters, scope, or conditions applied — such as customer status, time period, region, or sort order. Be specific and complete. A reader must understand exactly how the report was built just from your description. Do NOT start with 'This report' or 'Displays'. Do NOT mention numbers or results. Output only the description text — no labels, no quotes, no extra commentary.", user_message=f'''User's question: "{query}"\n\nWrite a 1-2 sentence description (8-20 words) explaining what data this report covers and how it is organized.''', temperature=0.4, max_tokens=1000)
        text = (response.get('content') or '').strip()
        return (text[:400] if text else None, response.get('usage'))
    except Exception as e:
        logger.warning('Failed to generate report description: %s', e)
        return (None, None)

async def create_report_from_run(run_id: str, title: str, user_id: str, client_id: str, db: AsyncIOMotorDatabase) -> Optional[DashboardReport]:
    return None
    try:
        conv_doc = await db['conversations'].find_one({'run_id': run_id})
        if not conv_doc:
            logger.warning('create_report_from_run: run_id=%s not found', run_id)
            return None
        conv_id = str(conv_doc.get('_id') or '')
        raw_code: str = (conv_doc.get('agent_responses') or {}).get('python') or conv_doc.get('coder_response') or ''
        if not raw_code:
            logger.warning('create_report_from_run: no code found for run_id=%s', run_id)
            return None
        code = _strip_iteration_headers(raw_code)
        code = _build_executable_code(code)
        logger.info('create_report_from_run: code reduced from %d to %d chars for run_id=%s', len(raw_code), len(code), run_id)
        repo = DashboardRepository(db)
        await repo.get_or_create(user_id, client_id)
        existing_doc = await repo.get_by_user(user_id, client_id)
        if existing_doc:
            reports = existing_doc.get('reports', [])
            if any(((r.get('source_run_id') or '') == run_id for r in reports)):
                logger.info('Duplicate dashboard report prevented user=%s run_id=%s', user_id, run_id)
                raise DuplicateReportFromConversationError('This response is already saved as a dashboard report.')
        report_title = title or conv_doc.get('input', 'Untitled Report')[:120]
        original_query = conv_doc.get('input', '')
        cached_question = conv_doc.get('enhanced_question') or None
        ds_id = _normalize_dashboard_dataset_id(conv_doc.get('dataset_id'))
        last_result, (description, description_usage) = await asyncio.gather(execute_report_code(code=code, client_id=client_id, db=db, dataset_id=ds_id), _generate_report_description(original_query, report_title, client_id, db))
        now = utcnow()
        report = DashboardReport(report_id=str(uuid.uuid4()), title=report_title, description=description, original_query=original_query, cached_question=cached_question, code=code, dataset_id=ds_id, source_run_id=run_id, source_conversation_id=conv_id or None, order=0, created_at=now, updated_at=now, last_result=last_result)
        if existing_doc:
            updated_reports = [{**r, 'order': r.get('order', 0) + 1} for r in existing_doc.get('reports', [])]
            if updated_reports:
                await db['dashboard_reports'].update_one({'user_id': user_id, 'client_id': client_id}, {'$set': {'reports': updated_reports}})
        saved = await repo.add_report(user_id, client_id, report)
        if not saved:
            logger.error('Failed to persist report for user=%s run_id=%s', user_id, run_id)
            return None
        if description_usage:
            try:
                from util.token_usage_utils import normalize_agent_token_usage
                agent_usage = {'narrator_agent': description_usage}
                normalized_agent_usage, total_token_usage = normalize_agent_token_usage(agent_usage)
                await repo.append_report_usage_event(user_id=user_id, client_id=client_id, report_id=report.report_id, event={'at': utcnow(), 'action': 'save_description', 'agent_token_usage': normalized_agent_usage, 'total_token_usage': total_token_usage})
            except Exception as e:
                logger.warning('Failed to append dashboard usage event (save_description) report=%s: %s', report.report_id, e)
        logger.info('Report created user=%s report_id=%s', user_id, report.report_id)
        return report
    except DuplicateReportFromConversationError:
        raise
    except AdhocQuickUploadReportBlockedError:
        raise
    except Exception as e:
        logger.error('create_report_from_run failed run_id=%s: %s', run_id, e, exc_info=True)
        return None

async def refresh_single_report(user_id: str, client_id: str, report_id: str, db: AsyncIOMotorDatabase) -> Optional[DashboardReportResult]:
    repo = DashboardRepository(db)
    doc = await repo.get_by_user(user_id, client_id)
    if not doc:
        return None
    report = next((r for r in doc.get('reports', []) if r['report_id'] == report_id), None)
    if not report:
        return None
    code = report.get('code') or report.get('templatized_code') or report.get('original_code') or ''
    ds_id = _normalize_dashboard_dataset_id(report.get('dataset_id'))
    result = await execute_report_code(code=code, client_id=client_id, db=db, dataset_id=ds_id)
    result_dict = result.model_dump(mode='python')
    await repo.update_report_fields(user_id, client_id, report_id, {'last_result': result_dict, 'updated_at': utcnow()})
    return result

async def run_interact_query(report_id: str, user_id: str, client_id: str, question: str, db: AsyncIOMotorDatabase) -> Optional[DashboardReportResult]:
    repo = DashboardRepository(db)
    doc = await repo.get_by_user(user_id, client_id)
    if not doc:
        return None
    report = next((r for r in doc.get('reports', []) if r['report_id'] == report_id), None)
    if not report:
        return None
    original_code = report.get('code') or report.get('templatized_code') or ''
    original_query = report.get('original_query', '')
    ds_id = _normalize_dashboard_dataset_id(report.get('dataset_id'))
    try:
        llm = LLMClient(agent_name='narrator_agent', client_id=client_id, db=db)
        await llm._load_client_llm_config()
        response = await llm.generate_completion(system_prompt='You are a Python data analyst. You are given existing Python code that answers a specific question. Adapt the code to answer the NEW question provided by the user. Keep the database connection, imports, and data loading logic intact. Modify only the query/filter/aggregation/visualization parts as needed. The code MUST end with these exact output variable assignments (do not use FINAL_RESULT):\n  _generated_plotly_fig_0_ = <your plotly figure>\n  _generated_dataframe_0_ = <your result dataframe>\nOutput ONLY valid Python code — no markdown fences, no explanations.', user_message=f'Original question: {original_query}\n\nExisting code:\n```python\n{original_code}\n```\n\nNew question: {question}\n\nAdapt the code to answer the new question. Output only Python code.', temperature=0.2, max_tokens=4000)
        interact_usage = response.get('usage')
        adapted_code = (response.get('content') or '').strip()
        adapted_code = re.sub('^```(?:python)?\\n?', '', adapted_code, flags=re.MULTILINE)
        adapted_code = re.sub('\\n?```$', '', adapted_code, flags=re.MULTILINE)
        adapted_code = adapted_code.strip()
    except Exception as e:
        logger.error('Interact LLM call failed report_id=%s: %s', report_id, e)
        return DashboardReportResult(dataframes=[], plotly_charts=[], text_outputs=[], executed_at=utcnow(), status='error', error=str(e)[:500])
    if not adapted_code:
        return DashboardReportResult(dataframes=[], plotly_charts=[], text_outputs=[], executed_at=utcnow(), status='error', error='LLM returned empty code.')
    if interact_usage:
        try:
            from util.token_usage_utils import normalize_agent_token_usage
            agent_usage = {'narrator_agent': interact_usage}
            normalized_agent_usage, total_token_usage = normalize_agent_token_usage(agent_usage)
            await repo.append_report_usage_event(user_id=user_id, client_id=client_id, report_id=report_id, event={'at': utcnow(), 'action': 'interact', 'agent_token_usage': normalized_agent_usage, 'total_token_usage': total_token_usage})
        except Exception as e:
            logger.warning('Failed to append dashboard usage event (interact) report=%s: %s', report_id, e)
    return await execute_report_code(code=adapted_code, client_id=client_id, db=db, dataset_id=ds_id)

async def refresh_all_reports(user_id: str, client_id: str, db: AsyncIOMotorDatabase, max_concurrency: int=4) -> List[Dict[str, Any]]:
    repo = DashboardRepository(db)
    doc = await repo.get_by_user(user_id, client_id)
    if not doc:
        return []
    reports: List[Dict[str, Any]] = sorted(doc.get('reports', []), key=lambda r: r.get('order', 0))
    if not reports:
        return []
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _refresh_one(report: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            code = report.get('code') or report.get('templatized_code') or report.get('original_code') or ''
            ds_id = _normalize_dashboard_dataset_id(report.get('dataset_id'))
            result = await execute_report_code(code=code, client_id=client_id, db=db, dataset_id=ds_id)
            result_dict = result.model_dump(mode='python')
            await repo.update_report_fields(user_id, client_id, report['report_id'], {'last_result': result_dict, 'updated_at': utcnow()})
            return {**report, 'last_result': result_dict}
    refreshed = await asyncio.gather(*[_refresh_one(r) for r in reports])
    return list(refreshed)