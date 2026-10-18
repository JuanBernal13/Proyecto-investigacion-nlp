"""Integridad de las anotaciones y regresiones de casos científicos delicados."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Permite importar el script sin convertirlo en un paquete de aplicación.
SPEC = importlib.util.spec_from_file_location('pilot_review', ROOT / 'scripts/validate_pilot_review.py')
assert SPEC is not None and SPEC.loader is not None
sys.path.insert(0, str(ROOT / 'scripts'))
try:
    review = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(review)
finally:
    sys.path.pop(0)


@pytest.fixture(scope='module')
def records() -> tuple[Any, ...]:
    return (
        review.read_jsonl(ROOT / 'salida/pilot_40_prompts.jsonl'),
        review.read_jsonl(ROOT / 'salida/pilot_40_gold.jsonl'),
        review.read_jsonl(ROOT / 'data/processed/pilot_40_review_v1.jsonl'),
        review.read_jsonl(ROOT / 'data/processed/pilot_40_reference_audit_v1.jsonl'),
        json.loads((ROOT / 'configs/pilot_review_policy.json').read_text(encoding='utf-8')),
    )


def test_real_review_and_unchanged_originals() -> None:
    result = review.inspect_review(ROOT)
    assert result['original_files_unchanged'] is True
    assert result['summary']['datasets']['GSM8K']['contrast_candidates'] == {
        'B_vs_O': 30, 'N_vs_B': 11, 'D_vs_B': 29, 'W_vs_O': 30}
    assert result['summary']['datasets']['MATH']['contrast_candidates'] == {
        'B_vs_O': 2, 'N_vs_B': 1, 'D_vs_B': 2, 'W_vs_O': 10}
    assert result['summary']['source_issues'] == [
        {'id': 'GSM8K-train-03515', 'issue': 'ambiguous_wording'}]
    assert result['summary']['human_reviews_completed'] == 0


@pytest.mark.parametrize('damage', [
    'duplicate_id', 'offset', 'omit_date', 'value', 'gold_leak',
    'unprotect_math', 'space_inside_math', 'contrast', 'reference', 'hide_ambiguity',
    'human_claim',
])
def test_reject_damaged_annotations(records: tuple[Any, ...], damage: str) -> None:
    prompts, gold, reviews, audits, policy = deepcopy(records)
    if damage == 'duplicate_id':
        reviews[1]['id'] = reviews[0]['id']
    elif damage == 'offset':
        reviews[0]['numeric_decisions'][0]['end'] += 1
    elif damage == 'omit_date':
        reviews[5]['numeric_decisions'].pop(1)
    elif damage == 'value':
        reviews[0]['numeric_decisions'][0]['value'] = '999'
    elif damage == 'gold_leak':
        reviews[0]['answer'] = gold[0]['answer']
    elif damage == 'unprotect_math':
        reviews[30]['w_protected_spans'] = [
            s for s in reviews[30]['w_protected_spans'] if s['reason'] == 'numeric_literal']
    elif damage == 'space_inside_math':
        text = prompts[30]['problem']
        span = next(s for s in reviews[30]['w_protected_spans']
                    if s['reason'] == 'math_notation' and ' ' in s['text'])
        reviews[30]['w_candidate_space_offsets'].append(text.index(' ', span['start'], span['end']))
    elif damage == 'contrast':
        reviews[0]['contrast_candidates']['N_vs_B'] = False
    elif damage == 'reference':
        audits[0]['expected_answer'] = '999'
    elif damage == 'hide_ambiguity':
        audits[7]['status'] = 'consistent_by_case_reasoning'
    else:
        policy['human_audit_complete'] = True
    with pytest.raises(ValueError):
        review.validate_review(prompts, gold, reviews, audits, policy)


def test_dates_fractions_lists_and_length_have_distinct_semantics(records: tuple[Any, ...]) -> None:
    annotations = {r['id']: r for r in records[2]}
    years = [d for d in annotations['GSM8K-train-02756']['numeric_decisions']
             if len(d['text']) == 4]
    assert len(years) == 4 and all(d['reason'] == 'date' for d in years)
    length = annotations['MATH-train-06217']['numeric_decisions'][0]
    assert length['text'] == '2002' and length['decision'] == 'approve'
    for ident in ('MATH-train-09539', 'MATH-train-03841'):
        assert all(d['decision'] == 'protect' for d in annotations[ident]['numeric_decisions'])
    grouped = annotations['GSM8K-train-05634']['numeric_decisions']
    assert all(d['decision'] == 'protect' for d in grouped if d['text'] in {'32', '000', '15'})
    compact = annotations['MATH-train-01558']['numeric_decisions']
    assert all(d['decision'] == 'protect' for d in compact if d['text'] in {'12', '13', '14'})


def test_cli_writes_new_report_and_rejects_overwrite(tmp_path: Path) -> None:
    output = tmp_path / 'review.json'
    command = [sys.executable, str(ROOT / 'scripts/validate_pilot_review.py'), '--output', str(output)]
    first = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert first.returncode == 0, first.stderr
    assert json.loads(output.read_text(encoding='utf-8'))['structural_validation'] == 'ok'
    before = output.read_bytes()
    second = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert second.returncode == 1
    assert json.loads(second.stderr)['status'] == 'failed'
    assert output.read_bytes() == before
