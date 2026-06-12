from __future__ import annotations
import re
from typing import Any, Dict, List

class PreferenceExtractor:
    _PATTERNS: List[tuple] = [(re.compile('\\b(?:show\\s+(?:as\\s+)?|use\\s+(?:a\\s+)?|make\\s+(?:a\\s+)?|display\\s+(?:as\\s+)?)bar\\s+(?:chart|graph|plot)\\b', re.I), 'chart_type', 'bar'), (re.compile('\\bbar\\s+(?:chart|graph|plot)\\b', re.I), 'chart_type', 'bar'), (re.compile('\\b(?:show\\s+(?:as\\s+)?|use\\s+(?:a\\s+)?|make\\s+(?:a\\s+)?|display\\s+(?:as\\s+)?)pie\\s+(?:chart|graph)\\b', re.I), 'chart_type', 'pie'), (re.compile('\\bpie\\s+(?:chart|graph)\\b', re.I), 'chart_type', 'pie'), (re.compile('\\b(?:show\\s+(?:as\\s+)?|use\\s+(?:a\\s+)?|make\\s+(?:a\\s+)?|display\\s+(?:as\\s+)?)line\\s+(?:chart|graph|plot)\\b', re.I), 'chart_type', 'line'), (re.compile('\\bline\\s+(?:chart|graph|plot)\\b', re.I), 'chart_type', 'line'), (re.compile('\\btrend\\s+(?:line|chart|graph)\\b', re.I), 'chart_type', 'line'), (re.compile('\\b(?:show\\s+(?:as\\s+)?|use\\s+(?:a\\s+)?|make\\s+(?:a\\s+)?|display\\s+(?:as\\s+)?)(?:scatter|scatter\\s+plot)\\b', re.I), 'chart_type', 'scatter'), (re.compile('\\bheatmap\\b', re.I), 'chart_type', 'heatmap'), (re.compile('\\b(?:stacked\\s+bar|stacked\\s+chart)\\b', re.I), 'chart_type', 'stacked_bar'), (re.compile('\\b(?:show\\s+(?:as\\s+)?|display\\s+(?:as\\s+)?)(?:table|tabular)\\b', re.I), 'viz_style', 'table_only'), (re.compile('\\bjust\\s+(?:the\\s+)?numbers\\b', re.I), 'viz_style', 'table_only'), (re.compile('\\b(?:on\\s+a\\s+)?dashboard\\b', re.I), 'viz_style', 'dashboard'), (re.compile('\\bmultiple\\s+charts\\b', re.I), 'viz_style', 'dashboard'), (re.compile('\\b(?:detailed|granular)\\s+(?:breakdown|analysis|view)\\b', re.I), 'detail_level', 'detailed'), (re.compile('\\bdrill\\s+down\\b', re.I), 'detail_level', 'detailed'), (re.compile('\\b(?:high\\s+level|summary|overview|brief)\\b', re.I), 'detail_level', 'summary'), (re.compile('\\bin\\s+(?:lakhs?|crores?)\\b', re.I), 'number_format', 'indian'), (re.compile('\\blakh\\s+format\\b', re.I), 'number_format', 'indian'), (re.compile('\\bin\\s+(?:millions?|thousands?|billions?)\\b', re.I), 'number_format', 'us'), (re.compile('\\b(?:monthly|month\\s*wise|month[\\s-]over[\\s-]month|MoM)\\b', re.I), 'time_granularity', 'monthly'), (re.compile('\\b(?:weekly|week\\s*wise|week[\\s-]over[\\s-]week|WoW)\\b', re.I), 'time_granularity', 'weekly'), (re.compile('\\b(?:daily|day\\s*wise|day[\\s-]over[\\s-]day|DoD)\\b', re.I), 'time_granularity', 'daily'), (re.compile('\\b(?:quarterly|quarter\\s*wise|quarter[\\s-]over[\\s-]quarter|QoQ)\\b', re.I), 'time_granularity', 'quarterly'), (re.compile('\\b(?:year\\s*over\\s*year|YoY|yearly|annual|annually)\\b', re.I), 'time_granularity', 'yearly'), (re.compile('\\b(?:compare|break\\s*down|group)\\s+by\\s+region\\b', re.I), 'grouping_preference', 'region'), (re.compile('\\bregion\\s*wise\\b', re.I), 'grouping_preference', 'region'), (re.compile('\\b(?:compare|break\\s*down|group)\\s+by\\s+product\\b', re.I), 'grouping_preference', 'product'), (re.compile('\\bproduct\\s*wise\\b', re.I), 'grouping_preference', 'product'), (re.compile('\\b(?:compare|break\\s*down|group)\\s+by\\s+category\\b', re.I), 'grouping_preference', 'category'), (re.compile('\\bcategory\\s*wise\\b', re.I), 'grouping_preference', 'category')]
    _LIMIT_PATTERN = re.compile('\\btop\\s+(\\d+)\\b', re.I)

    @classmethod
    def extract(cls, query: str) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        results: Dict[str, Dict[str, Any]] = {}
        for pattern, ptype, value in cls._PATTERNS:
            match = pattern.search(query)
            if match and ptype not in results:
                results[ptype] = {'preference_type': ptype, 'value': value, 'learned_from': match.group(0).strip()}
        limit_match = cls._LIMIT_PATTERN.search(query)
        if limit_match and 'limit' not in results:
            results['limit'] = {'preference_type': 'limit', 'value': limit_match.group(1), 'learned_from': limit_match.group(0).strip()}
        return list(results.values())

    @classmethod
    def extract_as_dict(cls, query: str) -> Dict[str, str]:
        prefs = cls.extract(query)
        return {p['preference_type']: p['value'] for p in prefs}