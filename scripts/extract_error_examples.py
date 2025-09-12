#!/usr/bin/env python3
"""
Extract verbatim quotes from JSONL to illustrate error patterns:

Categories:
- experiments_inference: 'in two experiments', 'experiment', 'within-subject', 'manipulated', etc.
- covariate_adjustments: 'adjusted for age/sex/gender', 'controlled for ...'
- temporal_ordering: 'cross-sectional', 'temporal ordering', 'measured at time', 'longitudinal'
- sensitivity_analysis: 'sensitivity analysis', 'robustness checks', 'bias analysis', 'E-value'

Output: results/error_pattern_examples.csv with columns:
pmid,title,category,match,quote
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple


JSONL_PATH = Path('data/processed/PMID_all_text.jsonl')
OUT_CSV = Path('results/error_pattern_examples.csv')


PATTERNS = {
    'experiments_inference': [
        re.compile(r"\bIn\s+(one|two|three|four|five)\s+experiments\b", re.I),
        re.compile(r"\bexperiment(s)?\b", re.I),
        re.compile(r"within-?subjects?", re.I),
        re.compile(r"within-?participants?", re.I),
        re.compile(r"between-?subjects?", re.I),
        re.compile(r"manipulat(ed|ion)", re.I),
        re.compile(r"random(ized| assignment)", re.I),
        re.compile(r"instruction", re.I),
        re.compile(r"study\s*(1|2|3)", re.I),
    ],
    'covariate_adjustments': [
        re.compile(r"adjust(ed|ment)?\s+for\s+(age|sex|gender|education|income|race|ethnicity|depress|anx|mood)", re.I),
        re.compile(r"controlled\s+for\s+(age|sex|gender|education|income|race|ethnicity|depress|anx|mood)", re.I),
        re.compile(r"covariate(s)?\s+(included|adjusted)", re.I),
        re.compile(r"demographic\s+(variables|covariates)", re.I),
    ],
    'temporal_ordering': [
        re.compile(r"cross[-\s]?sectional", re.I),
        re.compile(r"temporal\s+(order|ordering)", re.I),
        re.compile(r"measured\s+at\s+time", re.I),
        re.compile(r"longitudinal", re.I),
    ],
    'sensitivity_analysis': [
        re.compile(r"sensitivity\s+analys", re.I),
        re.compile(r"robustness\s+(check|analys)", re.I),
        re.compile(r"bias\s+analysis", re.I),
        re.compile(r"\bE-?value\b", re.I),
    ],
}


def get_title(text: str) -> str:
    # Title often appears as a line starting with '# '
    for line in text.splitlines():
        if line.startswith('# '):
            t = line[2:].strip()
            # filter generic '# Author's Accepted Manuscript'
            if not t.lower().startswith("author's accepted manuscript"):
                return t
    return ''


def collect_examples(limit_per_category: int = 50, per_pmid_per_cat: int = 2) -> List[Tuple[str, str, str, str, str]]:
    rows: List[Tuple[str, str, str, str, str]] = []  # pmid,title,category,match,quote
    if not JSONL_PATH.exists():
        raise FileNotFoundError(f"JSONL not found: {JSONL_PATH}")

    with JSONL_PATH.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pmid = str(obj.get('pmid', '')).strip()
            text = obj.get('text', '')
            if not pmid or not text:
                continue
            title = get_title(text)

            for cat, patterns in PATTERNS.items():
                added_for_this_pmid = 0
                for pat in patterns:
                    for m in pat.finditer(text):
                        start = max(0, m.start() - 140)
                        end = min(len(text), m.end() + 140)
                        snippet = text[start:end].replace('\n', ' ')
                        match_txt = text[m.start():m.end()]
                        rows.append((pmid, title, cat, match_txt, snippet))
                        added_for_this_pmid += 1
                        if added_for_this_pmid >= per_pmid_per_cat:
                            break
                    if added_for_this_pmid >= per_pmid_per_cat:
                        break

    # Limit rows per category
    by_cat: Dict[str, List[Tuple[str, str, str, str, str]]] = {}
    for r in rows:
        by_cat.setdefault(r[2], []).append(r)

    final: List[Tuple[str, str, str, str, str]] = []
    for cat, lst in by_cat.items():
        final.extend(lst[:limit_per_category])
    return final


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = collect_examples()
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['pmid', 'title', 'category', 'match', 'quote'])
        for r in rows:
            w.writerow(r)
    print(f"Saved {len(rows)} rows to {OUT_CSV}")


if __name__ == '__main__':
    main()

