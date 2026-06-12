from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Set
import re
import logging
logger = logging.getLogger(__name__)
from services.schema_mapper import SchemaMapper
CORE_KEYWORDS_BASE = {'value', 'volume', 'quantity', 'qty', 'amount', 'count', 'total', 'percentage', 'ratio', 'average', 'mean', 'median', 'sum', 'trend', 'growth', 'decline', 'variance', 'difference', 'category', 'group', 'segment', 'type', 'status', 'class', 'id', 'code', 'name', 'description', 'date', 'time', 'fiscal year', 'fy', 'period', 'month', 'year month', 'closing', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'daily', 'weekly', 'monthly', 'yearly', 'annual', 'vendor', 'supplier', 'customer', 'client', 'buyer', 'seller', 'employee', 'staff', 'department', 'division', 'branch', 'location', 'warehouse', 'plant', 'site', 'facility', 'store', 'office', 'organization', 'org', 'company', 'entity', 'order', 'purchase', 'sale', 'invoice', 'receipt', 'payment', 'transaction', 'entry', 'record', 'line item', 'shipment', 'rfq', 'quotation', 'quote', 'bid', 'tender', 'contract', 'po', 'purchase order', 'delivery', 'dispatch', 'receipt', 'product', 'item', 'material', 'sku', 'part', 'component', 'stock', 'inventory', 'consumption', 'issued', 'received', 'revenue', 'cost', 'price', 'profit', 'margin', 'expense', 'budget', 'actual', 'forecast', 'target', 'plan', 'breakdown', 'distribution', 'summary', 'report', 'analysis', 'ranking', 'top', 'bottom', 'highest', 'lowest', 'most', 'least', 'performance', 'comparison', 'benchmark'}

async def _get_client_keywords(client_id: str, db) -> Set[str]:
    keywords = CORE_KEYWORDS_BASE.copy()
    try:
        schema_mapper = await SchemaMapper.create(client_id, db)
        guardrails_config = schema_mapper.get_guardrails_config()
        keywords.update(guardrails_config.get('domain_keywords', []))
        keywords.update(guardrails_config.get('facility_names', []))
        keywords.update(guardrails_config.get('product_terms', []))
        try:
            for table in schema_mapper.schema.get('tables', []):
                keywords.add(table.get('logical_name', '').replace('_', ' '))
                for col in table.get('columns', []):
                    keywords.add(col.get('logical_name', '').replace('_', ' '))
                    if col.get('display_name'):
                        keywords.add(col.get('display_name', '').lower())
        except Exception:
            pass
        logger.info(f"[Guardrails] Loaded {len(keywords)} keywords for client '{client_id}'")
    except Exception as e:
        logger.warning(f"[Guardrails] Error loading schema for '{client_id}': {e}, using base keywords only")
    return keywords
QUESTION_PATTERNS = ['^how\\s+many\\b', '^show\\b', '^list\\b', '^top\\s+\\d+\\b', '^which\\b', '^what\\s+(is|are|was|were)\\b', '^who\\s+(is|are|has|had|was|were)\\s+(the\\s+)?(top|best|worst|highest|lowest|most|least)\\b', '^where\\s+(is|are|do|does)\\b', '^when\\s+(did|was|were|is)\\b', '^give\\s+me\\b', '^get\\s+me\\b', '^find\\b', '^fetch\\b', '^display\\b', '^summarize\\b', '^analyze\\b', '^compare\\b', '^predict\\b', '^forecast\\b', '^calculate\\s+(the\\s+)?(total|average|sum|count|mean|median)\\b', '\\bversus\\b|\\bvs\\.?\\b', '\\btrend\\b', '\\bdifference\\b', '\\bpercent(age)?\\b', '\\btotal\\b', '\\baverage\\b', '\\bcount\\b', '\\bmaximum\\b|\\bmax\\b', '\\bminimum\\b|\\bmin\\b', '\\bmost\\b', '\\bleast\\b', '\\bhighest\\b', '\\blowest\\b', '\\btop\\s+\\d+\\b', '\\bbottom\\s+\\d+\\b', '\\branking\\b|\\brank\\b', '\\bbreakdown\\b', '\\bby\\s+(month|year|quarter|week|day|category|region|location|department|vendor|supplier|customer|site|plant|type|status|group)\\b', '\\bper\\s+(month|year|quarter|week|day|unit|item|vendor|supplier|customer|site|organization)\\b', '\\b\\d+\\s*(to|-|–|—)\\s*\\d+\\b', '\\b\\d{4}\\b', '\\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\\b', '\\b(q[1-4]|quarter\\s*[1-4])\\b', '\\breceiving\\b|\\bsending\\b|\\bprocessing\\b', '\\binvitation\\w*\\b|\\binvite\\w*\\b']
IRRELEVANT_PATTERNS = ['\\bstock\\s+price\\b']
RISKY_SQL_PATTERNS = ['\\bdrop\\s+table\\b', '\\bdrop\\s+database\\b', '\\btruncate\\s+table\\b', '\\bdelete\\s+from\\b', '\\bupdate\\s+\\w+\\s+set\\b', '\\binsert\\s+into\\b', '\\balter\\s+table\\b', '\\bgrant\\b|\\brevoke\\b']
RISKY_OS_PATTERNS = ['rm\\s+-rf\\s+/', 'del\\s+/s', 'powershell\\s+remove-item', 'curl\\s+.*\\|\\s*bash', 'format\\s+c:']
RISKY_SECRET_PATTERNS = ['\\bpassword\\b', '\\bapi\\s*key\\b|\\bapikey\\b|\\btoken\\b', '\\bsecret\\b', '\\bcredential\\b|\\blogin\\b', '\\botp\\b|\\bone\\s*time\\s*password\\b', '\\baadhaar\\b|\\bpan\\b|\\bssn\\b']
RISKY_INJECTION_PATTERNS = ['ignore\\s+previous\\s+instructions', 'reveal\\s+(system|hidden)\\s+prompt', 'jailbreak', 'disable\\s+guard|turn\\s+off\\s+guard', 'prompt\\s+injection']
RISKY_EXFIL_PATTERNS = ['email\\s+.*\\ball\\s+data\\b', 'send\\s+me\\s+the\\s+dataset', 'export\\s+all\\s+records', 'download\\s+entire\\s+database']
RISKY_NL_DESTRUCTIVE = ['\\b(delete|remove|erase|wipe|purge|clear|destroy)\\b.*\\b(data|dataset|records|table|tables|database)\\b', '\\b(data|dataset|records|table|tables|database)\\b.*\\b(delete|remove|erase|wipe|purge|clear|destroy)\\b', '\\bempty\\b.*\\b(table|tables|database)\\b', '\\breset\\b.*\\b(database|tables)\\b']
GREETING_PATTERNS = ['^\\s*(hi|hello|hey|howdy|hola|namaste|greetings)\\s*[!.,?]*\\s*$', '^\\s*(good\\s+(morning|afternoon|evening|night|day))\\s*[!.,?]*\\s*$', '^\\s*(thank\\s*you|thanks|thx|cheers)\\s*[!.,?]*\\s*$', '^\\s*(bye|goodbye|see\\s+you|take\\s+care)\\s*[!.,?]*\\s*$', '^\\s*(who\\s+are\\s+you|what\\s+are\\s+you|what\\s+can\\s+you\\s+do)\\s*[!.,?]*\\s*$', '^\\s*(nice\\s+to\\s+meet|how\\s+do\\s+you\\s+do)\\b']
MATH_PATTERNS = ['^\\s*\\d+\\s*[\\+\\-\\*\\/\\%\\^]\\s*\\d+\\s*$', '\\b(calculate|compute|solve|evaluate)\\s+\\d', '\\b(what\\s+is|whats)\\s+\\d+\\s*[\\+\\-\\*\\/\\%\\^]\\s*\\d+', '\\b(sqrt|square\\s+root|cube\\s+root|factorial)\\s*(\\(|\\s*of)?\\s*\\d+', '\\b(derivative|integral|limit|matrix|determinant|eigenvector)\\b', '\\b(sin|cos|tan|arctan|arcsin)\\s*\\(?\\s*\\d+']
GENERAL_KNOWLEDGE_PATTERNS = ['\\b(who\\s+(was|invented|discovered|founded|created))\\b', '\\bcapital\\s+(of|city)\\b', '\\b(history\\s+of|biography\\s+of)\\b', '\\b(what\\s+is\\s+the\\s+(meaning|definition)\\s+of)\\b', '\\b(when\\s+was|when\\s+did)\\b.*\\b(born|die|happen|start|end|found)\\b', '\\b(where\\s+is|where\\s+was)\\b.*\\b(located|born|built|founded)\\b', '\\b(how\\s+(tall|old|long|far|big|heavy)\\s+is)\\b', '\\b(population\\s+of|area\\s+of|gdp\\s+of)\\b', '\\b(president|prime\\s+minister|king|queen|emperor)\\s+of\\b', '\\b(planet|galaxy|universe|solar\\s+system|constellation)\\b', '\\bwho\\s+is\\b(?!.*(customer|supplier|vendor|employee|user|manager|client|top|best|worst|highest|lowest))']
WEATHER_PATTERNS = ['\\b(temperature|forecast|rain|snow|humidity|wind\\s+speed)\\b.*\\b(today|tomorrow|this\\s+week)\\b', '\\b(is\\s+it|will\\s+it)\\s+(be\\s+)?(rain|rainy|snow|snowy|cold|hot|sunny|cloudy|warm|freezing)\\b', '\\bweather\\b(?!.*(data|dataset|table|column|report|analysis|impact|related|supply|disruption))']
CODING_PATTERNS = ['\\b(write|create|generate|build|make)\\s+(a\\s+)?(code|program|script|function|class|app|website)\\b', '\\b(python|javascript|java|c\\+\\+|rust|golang|ruby|php|swift|kotlin)\\s+(code|program|function|script)\\b', '\\b(how\\s+to\\s+(code|program|implement|build|develop|deploy))\\b', '\\b(debug|compile|runtime\\s+error|syntax\\s+error|segfault)\\b', '\\b(html|css|react|angular|vue|django|flask|node\\.?js|express)\\b.*\\b(tutorial|example|how)\\b']
ENTERTAINMENT_PATTERNS = ['\\b(tell\\s+me\\s+a\\s+(joke|story|riddle|poem|fact))\\b', '\\b(make\\s+me\\s+laugh|something\\s+funny|entertain\\s+me)\\b', '\\b(sing|song|lyrics|music|movie|book)\\s*(recommend|suggestion)\\b', '\\b(play\\s+a\\s+game|trivia|quiz|word\\s+game)\\b', '\\b(horoscope|zodiac|fortune)\\b']
SMALL_TALK_PATTERNS = ["^\\s*(how\\s+are\\s+you|how'?s\\s+it\\s+going|what'?s\\s+(up|new))\\s*[!.,?]*\\s*$", '\\b(do\\s+you\\s+(like|love|hate|think|believe|feel|prefer))\\b(?!.*(data|sales|stock|inventory|consumption|report|trend|analysis))', '\\b(what\\s+is\\s+your\\s+(name|age|favorite|opinion))\\b', '\\b(are\\s+you\\s+(a\\s+|an\\s+)?(real|human|ai|robot|alive|sentient|bot|machine|computer))\\b', '\\b(meaning\\s+of\\s+life|purpose\\s+of\\s+existence)\\b']
NSFW_PATTERNS = ['\\b(nude|naked|porn|xxx|nsfw|adult|sexual|explicit|erotic|hentai|sexy)\\b.*\\b(pic|picture|photo|video|image|content|film|clip|show)\\b', '\\b(show|send|display|give|get|find)\\b.*\\b(nude|naked|porn|xxx|nsfw|sexual|explicit|erotic|hentai|sexy)\\b', '\\b(nude|naked|porn|xxx|nsfw|sexual|explicit|erotic|hentai)\\b']
MALICIOUS_EXTRA_PATTERNS = ['\\b(hack|exploit|crack|bypass|brute\\s*force)\\b.*\\b(system|server|network|database|account)\\b', '\\b(phishing|malware|ransomware|virus|trojan)\\b', '\\b(steal|exfiltrate|dump)\\s+(data|credentials|passwords|info)\\b', '\\b(ddos|denial\\s+of\\s+service|sql\\s+injection|xss|csrf)\\b', '\\b(reverse\\s+shell|backdoor|rootkit|keylogger)\\b']
IRRELEVANT_CATEGORIES = {'greeting': (GREETING_PATTERNS, 'Hello! I\'m your data intelligence assistant. Here are some things I can help with:\n\n- **Spot trends** - "Show me the trend over the last 6 months"\n- **Compare segments** - "Compare performance across regions"\n- **Forecast** - "Predict next quarter\'s performance"\n\nJust ask a question about your data and I\'ll analyze it for you!'), 'math': (MATH_PATTERNS, "I'm designed to analyze your business data, not solve math problems. Try asking about trends, totals, or breakdowns in your dataset."), 'general_knowledge': (GENERAL_KNOWLEDGE_PATTERNS, "I specialize in analyzing your organization's data, not general knowledge. Ask me about your dataset -- for example, 'Show me the top 10 items' or 'What was the total last month?'"), 'weather': (WEATHER_PATTERNS, "I don't have access to weather data. I'm here to help you analyze your business data. Try asking about your dataset."), 'coding': (CODING_PATTERNS, "I'm a data intelligence assistant, not a code generator. I can help you explore your business data -- try asking about trends, summaries, or comparisons."), 'entertainment': (ENTERTAINMENT_PATTERNS, "I'm focused on helping you with data intelligence. Ask me about your data -- for example, 'Show me a summary' or 'Compare sales by region.'"), 'small_talk': (SMALL_TALK_PATTERNS, "I appreciate the conversation! I'm your data intelligence assistant -- ask me about your dataset and I'll help you find insights."), 'nsfw': (NSFW_PATTERNS, "I'm a data intelligence assistant and can only help with business data queries. Please ask questions about your dataset."), 'malicious': (MALICIOUS_EXTRA_PATTERNS, "I'm a data intelligence assistant. I can only help with legitimate data queries. Please ask questions about your dataset.")}

def _score_keywords(text: str, keywords: Set[str]) -> int:
    t = text.lower()
    return sum((1 for k in keywords if k in t))

def _matches_any(patterns, text: str) -> bool:
    return any((re.search(p, text, flags=re.IGNORECASE) for p in patterns))

def _check_irrelevant_category(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = text.strip()
    for category, (patterns, message) in IRRELEVANT_CATEGORIES.items():
        if _matches_any(patterns, t):
            return (category, message)
    return (None, None)

@dataclass
class GuardrailResult:
    is_relevant: bool
    score: float
    reason: str
    category: str
    user_message: str
_RISKY_MESSAGE = "I'm a data intelligence assistant. I can only help with legitimate data queries. Please ask questions about your dataset."
_RISKY_CHECKS = [(RISKY_NL_DESTRUCTIVE, 'risky_nl_destructive', 'risky: destructive request (natural language)'), (RISKY_SQL_PATTERNS, 'risky_sql', 'risky: destructive SQL operation requested'), (RISKY_OS_PATTERNS, 'risky_os', 'risky: destructive OS command requested'), (RISKY_SECRET_PATTERNS, 'risky_secret', 'risky: secrets/credentials access requested'), (RISKY_INJECTION_PATTERNS, 'risky_injection', 'risky: prompt-injection/control attempt'), (RISKY_EXFIL_PATTERNS, 'risky_exfil', 'risky: bulk data exfiltration requested')]

async def classify_question(question: str, client_id: Optional[str]=None, db=None) -> GuardrailResult:
    if not question or not question.strip():
        return GuardrailResult(False, 0.0, 'Empty question', 'off_topic', 'Please type a question about your data.')
    q = question.strip()
    for patterns, category, reason in _RISKY_CHECKS:
        if _matches_any(patterns, q):
            return GuardrailResult(False, 0.0, reason, category, _RISKY_MESSAGE)
    cat, user_msg = _check_irrelevant_category(q)
    if cat:
        if cat not in ('malicious', 'nsfw'):
            keywords = await _get_client_keywords(client_id, db) if client_id and db is not None else CORE_KEYWORDS_BASE
            kw_score = _score_keywords(q, keywords)
            intent_match = _matches_any(QUESTION_PATTERNS, q)
            if kw_score >= 2 or (kw_score >= 1 and intent_match):
                logger.info(f"[Guardrails] Rescued query from '{cat}' — {kw_score} domain keywords, intent_match={intent_match}")
                return GuardrailResult(True, min(kw_score / 5.0, 1.0), f'Rescued from {cat}: domain keywords override', 'relevant', '')
        return GuardrailResult(False, 0.0, f'Matches irrelevant category: {cat}', cat, user_msg)
    if _matches_any(IRRELEVANT_PATTERNS, q):
        keywords = await _get_client_keywords(client_id, db) if client_id and db is not None else CORE_KEYWORDS_BASE
        kw_score = _score_keywords(q, keywords)
        intent_match = _matches_any(QUESTION_PATTERNS, q)
        if kw_score >= 2 or (kw_score >= 1 and intent_match):
            logger.info(f'[Guardrails] Rescued query from legacy irrelevant — {kw_score} domain keywords')
            return GuardrailResult(True, min(kw_score / 5.0, 1.0), 'Rescued from irrelevant: domain keywords override', 'relevant', '')
        return GuardrailResult(False, 0.0, 'Matches known irrelevant topic', 'off_topic', "That question doesn't seem related to your data. I can help you explore your dataset -- try asking about available records or summaries.")
    keywords = await _get_client_keywords(client_id, db) if client_id and db is not None else CORE_KEYWORDS_BASE
    kw_score = _score_keywords(q, keywords)
    intent_match = _matches_any(QUESTION_PATTERNS, q)
    raw = kw_score + (2 if intent_match else 0)
    norm = min(raw / 5.0, 1.0)
    word_count = len(q.split())
    is_relevant = kw_score >= 1 or intent_match or word_count >= 4
    client_label = f" for client '{client_id}'" if client_id else ''
    if is_relevant:
        return GuardrailResult(True, max(float(norm), 0.3), f'Sufficient domain signals detected{client_label}', 'relevant', '')
    else:
        return GuardrailResult(False, float(norm), f'Insufficient domain signals for business context{client_label}', 'off_topic', "I couldn't find a connection to your data in that question. Try asking about your dataset -- for example, 'What data is available?' or 'Show me a summary.'")

async def is_question_relevant(question: str, client_id: Optional[str]=None, db=None) -> Tuple[bool, float, str]:
    result = await classify_question(question, client_id, db)
    return (result.is_relevant, result.score, result.reason)