"""Integración real: trazabilidad, entradas sin gold y fallos persistentes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
try:
    SPEC = importlib.util.spec_from_file_location('conditions', ROOT / 'scripts/prepare_pilot_conditions.py')
    assert SPEC is not None and SPEC.loader is not None
    conditions = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(conditions)
finally:
    sys.path.remove(str(ROOT / 'scripts'))


@pytest.fixture(scope='module')
def prepared(tmp_path_factory: Any) -> Path:
    output = tmp_path_factory.mktemp('conditions') / 'batch'
    conditions.prepare(ROOT, output)
    return output


def test_real_batch_and_reference_separation(prepared: Path) -> None:
    result = conditions.verify_artifacts(ROOT, prepared)
    summary = result['summary']
    assert summary['records'] == 200
    assert summary['numeric_span_checks'] == 282
    assert summary['human_reviews_completed'] == 0
    assert summary['datasets']['GSM8K']['applicable'] == {'O': 30, 'B': 30, 'N': 11, 'D': 29, 'W': 30}
    assert summary['datasets']['MATH']['applicable'] == {'O': 10, 'B': 2, 'N': 1, 'D': 2, 'W': 10}
    rows = conditions.read_jsonl(prepared / conditions.INPUTS)
    assert len({row['variant_id'] for row in rows}) == 200
    assert all(set(row) == {'run_id', 'variant_id', 'id', 'dataset', 'condition', 'problem'} for row in rows)
    manifest = json.loads((prepared / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['original_files_unchanged'] is True
    assert manifest['model_generations'] == 0


def test_existing_directory_never_overwritten(prepared: Path) -> None:
    before = {path.name: path.read_bytes() for path in prepared.iterdir()}
    with pytest.raises(FileExistsError):
        conditions.prepare(ROOT, prepared)
    assert before == {path.name: path.read_bytes() for path in prepared.iterdir()}


@pytest.mark.parametrize('corruption', ['value', 'gold_field', 'audit_applicability'])
def test_tampering_rejected_even_with_updated_hashes(
    prepared: Path, tmp_path: Path, corruption: str,
) -> None:
    for path in prepared.iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    filename = conditions.AUDITS if corruption == 'audit_applicability' else conditions.INPUTS
    rows = conditions.read_jsonl(tmp_path / filename)
    if corruption == 'value':
        target = next(row for row in rows if '<num>' in row.get('problem', ''))
        target['problem'] = target['problem'].replace('<num>', '<num>9', 1)
    elif corruption == 'gold_field':
        rows[0]['answer'] = 'unexpected reference'
    else:
        rows[1]['applicable'] = not rows[1]['applicable']
    (tmp_path / filename).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    manifest_path = tmp_path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['output_hashes'][filename] = conditions.sha256(tmp_path / filename)
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError):
        conditions.verify_artifacts(ROOT, tmp_path)


def test_interrupted_batch_preserved_and_not_successful(tmp_path: Path, monkeypatch: Any) -> None:
    original = conditions.records_for_problem
    calls = 0

    def fail_on_second(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError('simulated interruption')
        return original(*args)

    monkeypatch.setattr(conditions, 'records_for_problem', fail_on_second)
    output = tmp_path / 'interrupted'
    with pytest.raises(ValueError, match='simulated interruption'):
        conditions.prepare(ROOT, output)
    assert len(conditions.read_jsonl(output / conditions.INPUTS)) == 5
    assert len(conditions.read_jsonl(output / conditions.AUDITS)) == 5
    assert not (output / 'manifest.json').exists()
    assert json.loads((output / 'failure.json').read_text(encoding='utf-8'))['status'] == 'failed'
    with pytest.raises(ValueError, match='fallido'):
        conditions.verify_artifacts(ROOT, output)


def test_partial_batch_without_manifest_rejected(tmp_path: Path) -> None:
    (tmp_path / conditions.INPUTS).write_text('{}\n', encoding='utf-8')
    with pytest.raises(FileNotFoundError):
        conditions.verify_artifacts(ROOT, tmp_path)
