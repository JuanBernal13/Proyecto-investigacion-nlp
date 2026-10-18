"""Diario real, interrupciones y reanudación sin repetir intentos."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from src.inference.engine import execute, exclusive_lock


class FakeBackend:
    metadata: dict[str, Any] = {'fixture': 'fake_backend_not_a_study_model'}

    def __init__(self, calls: list[str], failure: BaseException | None = None) -> None:
        self.calls, self.failure = calls, failure

    def generate(self, task: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(task['variant_id'])
        if self.failure:
            raise self.failure
        return {'status': 'generated', 'text': 'fixture only'}

    def close(self) -> None:
        pass


@pytest.fixture
def tasks() -> list[dict[str, Any]]:
    return [{'variant_id': str(i), 'problem': f'fixture {i}', 'applicable': i != 2} for i in range(4)]


def test_resume_and_completed_run_never_repeat(tmp_path: Path, tasks: list[dict[str, Any]]) -> None:
    output, calls = tmp_path / 'run', []
    first = execute(output, {'seed': 2026}, tasks, lambda: FakeBackend(calls), limit=1)
    assert first['status'] == 'incomplete' and calls == ['0']
    second = execute(output, {'seed': 2026}, tasks, lambda: FakeBackend(calls), resume=True, limit=10)
    assert second['run_id'] == first['run_id'] and second['status'] == 'completed'
    assert second['counts'] == {'generated': 3, 'not_applicable': 1} and calls == ['0', '1', '3']
    execute(output, {'seed': 2026}, tasks, lambda: pytest.fail('No cargar modelo terminado'), resume=True)
    assert calls == ['0', '1', '3']


def test_existing_output_rejected(tmp_path: Path, tasks: list[dict[str, Any]]) -> None:
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'sentinel'
    marker.write_text('preserve')
    with pytest.raises(FileExistsError):
        execute(output, {}, tasks, lambda: FakeBackend([]))
    assert marker.read_text() == 'preserve'


@pytest.mark.parametrize('change', ['identity', 'task', 'runtime', 'persisted_plan'])
def test_resume_rejects_changes(tmp_path: Path, tasks: list[dict[str, Any]], change: str) -> None:
    output, calls, identity = tmp_path / 'run', [], {'seed': 2026}
    execute(output, identity, tasks, lambda: FakeBackend(calls))
    changed = deepcopy(tasks)
    backend = FakeBackend(calls)
    if change == 'identity':
        identity = {'seed': 99}
    elif change == 'task':
        changed[1]['problem'] = 'changed'
    elif change == 'runtime':
        backend.metadata = {'different': 'template'}
    else:
        with sqlite3.connect(output / 'results.sqlite3') as db:
            db.execute("UPDATE tasks SET payload='{}' WHERE variant_id='1'")
    with pytest.raises(ValueError):
        execute(output, identity, changed, lambda: backend, resume=True)
    assert calls == ['0']


def test_abrupt_interruption_remains_in_doubt(tmp_path: Path, tasks: list[dict[str, Any]]) -> None:
    output, calls = tmp_path / 'run', []
    with pytest.raises(KeyboardInterrupt):
        execute(output, {}, tasks, lambda: FakeBackend(calls, KeyboardInterrupt()))
    with sqlite3.connect(output / 'results.sqlite3') as db:
        assert db.execute("SELECT status FROM tasks WHERE variant_id='0'").fetchone() == ('started',)
    result = execute(output, {}, tasks, lambda: FakeBackend(calls), resume=True, limit=10)
    assert result['status'] == 'partial_failure' and result['in_doubt'] == 1
    assert calls == ['0', '1', '3']


def test_failed_attempt_retained_without_retry(tmp_path: Path, tasks: list[dict[str, Any]]) -> None:
    output, calls = tmp_path / 'run', []
    first = execute(output, {}, tasks, lambda: FakeBackend(calls, RuntimeError('simulated OOM')))
    assert first['status'] == 'partial_failure' and first['counts']['failed'] == 1
    result = execute(output, {}, tasks, lambda: FakeBackend(calls), resume=True, limit=10)
    assert result['status'] == 'partial_failure' and calls == ['0', '1', '3']
    with sqlite3.connect(output / 'results.sqlite3') as db:
        saved = json.loads(db.execute("SELECT result FROM tasks WHERE variant_id='0'").fetchone()[0])
        assert 'simulated OOM' in saved['error']


def test_model_loading_failure_allows_safe_resume(tmp_path: Path, tasks: list[dict[str, Any]]) -> None:
    output, calls = tmp_path / 'run', []

    def unavailable() -> Any:
        raise OSError('weights unavailable')

    with pytest.raises(OSError):
        execute(output, {}, tasks, unavailable)
    result = execute(output, {}, tasks, lambda: FakeBackend(calls), resume=True, limit=10)
    assert result['status'] == 'completed' and calls == ['0', '1', '3']


def test_context_overflow_is_not_success(tmp_path: Path, tasks: list[dict[str, Any]]) -> None:
    backend = FakeBackend([])
    backend.generate = lambda task: {'status': 'context_overflow', 'generation_attempted': False}  # type: ignore[method-assign]
    result = execute(tmp_path / 'run', {}, tasks, lambda: backend, limit=10)
    assert result['status'] == 'partial_failure' and result['counts']['context_overflow'] == 3


def test_process_lock_rejects_second_writer_and_releases(tmp_path: Path) -> None:
    path = tmp_path / 'inference.lock'
    with exclusive_lock(path):
        with pytest.raises(OSError):
            with exclusive_lock(path):
                pytest.fail('Segundo escritor adquirió el bloqueo')
    with exclusive_lock(path):
        assert path.exists()
