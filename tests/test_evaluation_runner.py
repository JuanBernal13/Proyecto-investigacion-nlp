"""Integración con el diario real de inferencia; backend de respuestas controladas."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_results as runner
from src.inference.engine import execute

CONFIG = runner.load_config(ROOT)
GOLD = {'GSM8K-fixture': {'answer': '2', 'solution': '#### 2'}}


class FixtureBackend:
    metadata = {'fixture': True}

    def generate(self, task: dict) -> dict:
        return {'status': 'generated', 'text': '<final>2</final>', 'reached_token_limit': False}

    def close(self) -> None:
        pass


def source(tmp_path: Path, *, partial: bool = False) -> Path:
    tasks = [{'variant_id': str(i), 'id': 'GSM8K-fixture', 'dataset': 'GSM8K',
              'condition': condition, 'problem': 'fixture', 'applicable': i != 2}
             for i, condition in enumerate(('O', 'B', 'N'))]
    execute(tmp_path / 'inference', {'fixture': True}, tasks, FixtureBackend, limit=1 if partial else 3)
    return tmp_path / 'inference/results.sqlite3'


def test_runner_readonly_resume_and_no_overwrite(tmp_path: Path) -> None:
    database = source(tmp_path)
    before = database.read_bytes()
    out = tmp_path / 'evaluation'
    result = runner.evaluate_run(database, out, GOLD, CONFIG, {'fixture': True})
    assert result['status'] == 'completed' and result['counts'] == {'correct': 2, 'not_applicable': 1}
    with sqlite3.connect(out / 'evaluation.sqlite3') as connection:
        saved = connection.execute('SELECT * FROM evaluations').fetchall()
    assert runner.evaluate_run(database, out, GOLD, CONFIG, {'fixture': True}, resume=True)['counts'] == result['counts']
    with sqlite3.connect(out / 'evaluation.sqlite3') as connection:
        assert connection.execute('SELECT * FROM evaluations').fetchall() == saved
    assert database.read_bytes() == before
    with pytest.raises(FileExistsError):
        runner.evaluate_run(database, out, GOLD, CONFIG, {'fixture': True})
    with pytest.raises(ValueError):
        runner.evaluate_run(database, out, GOLD, CONFIG, {'fixture': False}, resume=True)


def test_partial_inference_is_not_mathematical_error(tmp_path: Path) -> None:
    database = source(tmp_path, partial=True)
    result = runner.evaluate_run(database, tmp_path / 'eval', GOLD, CONFIG, {})
    assert result['status'] == 'partial_source'
    assert result['counts'] == {'correct': 1, 'indeterminate': 1, 'not_applicable': 1}


def test_interrupted_evaluation_preserves_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = source(tmp_path)
    real_evaluate = runner.evaluate
    count = 0

    def interrupt(*args: object, **kwargs: object) -> dict:
        nonlocal count
        count += 1
        if count == 2:
            raise KeyboardInterrupt
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(runner, 'evaluate', interrupt)
    out = tmp_path / 'eval'
    with pytest.raises(KeyboardInterrupt):
        runner.evaluate_run(database, out, GOLD, CONFIG, {})
    with sqlite3.connect(out / 'evaluation.sqlite3') as connection:
        assert connection.execute('SELECT COUNT(*) FROM evaluations').fetchone()[0] == 1
    monkeypatch.setattr(runner, 'evaluate', real_evaluate)
    assert runner.evaluate_run(database, out, GOLD, CONFIG, {}, resume=True)['counts']['correct'] == 2


def test_modified_source_and_unknown_reference_rejected(tmp_path: Path) -> None:
    database = source(tmp_path)
    out = tmp_path / 'eval'
    runner.evaluate_run(database, out, GOLD, CONFIG, {})
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE tasks SET result=? WHERE variant_id='0'",
                           (json.dumps({'status': 'generated', 'text': '<final>3</final>'}),))
    with pytest.raises(ValueError):
        runner.evaluate_run(database, out, GOLD, CONFIG, {}, resume=True)
    with pytest.raises(ValueError):
        runner.evaluate_run(database, tmp_path / 'bad', {}, CONFIG, {})
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('arguments', [[], ['--check', '--resume'], ['--evaluate'], ['--check', '--input-db', 'x']])
def test_cli_rejects_invalid_invocation(arguments: list[str]) -> None:
    completed = subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_results.py'), *arguments],
                               capture_output=True, text=True, timeout=15)
    assert completed.returncode != 0
