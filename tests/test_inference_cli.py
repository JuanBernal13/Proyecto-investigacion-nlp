"""Comprobación pública sin red ni pesos y plan real del piloto."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
try:
    SPEC = importlib.util.spec_from_file_location('inference_cli', ROOT / 'scripts/run_inference.py')
    assert SPEC is not None and SPEC.loader is not None
    cli = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(cli)
finally:
    sys.path.remove(str(ROOT / 'scripts'))


def test_real_plan_does_not_copy_gold_or_audit_notes() -> None:
    config, models = cli.load_config(ROOT)
    tasks, hashes = cli.prepare_plan(ROOT, config)
    assert len(tasks) == 200 and sum(t['applicable'] for t in tasks) == 155
    assert all(set(t) == {'variant_id', 'id', 'dataset', 'condition', 'problem', 'applicable',
                          'baseline', 'non_applicable_reason'} for t in tasks)
    assert len(models) == 2 and 'configs/inference.json' in hashes
    assert config['max_new_tokens'] == 1024 and config['max_context_tokens'] == 4096


def test_public_check_creates_report_without_overwriting(tmp_path: Path) -> None:
    report = tmp_path / 'check.json'
    command = [sys.executable, str(ROOT / 'scripts/run_inference.py'), '--check', '--report', str(report)]
    first = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert first.returncode == 0, first.stderr
    result = json.loads(report.read_text(encoding='utf-8'))
    assert result['study_model_generations'] == 0 and result['experiment_ready'] is False
    assert result['variants'] == 200 and result['applicable'] == 155
    before = report.read_bytes()
    second = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert second.returncode == 1 and report.read_bytes() == before


@pytest.mark.parametrize('option', ['--allow-download', '--resume'])
def test_check_rejects_execution_options(option: str) -> None:
    command = [sys.executable, str(ROOT / 'scripts/run_inference.py'), '--check', option]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and '--check' in result.stderr
