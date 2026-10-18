"""Diario SQLite: un intento por variante y ninguna repetición implícita."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Protocol
from uuid import uuid4

from src.transformations.core import require


class Backend(Protocol):
    metadata: dict[str, Any]

    def generate(self, task: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Bloqueo del SO, liberado incluso al morir el proceso; no borrar el archivo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if path.stat().st_size == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def execute(
    output: Path, identity: dict[str, Any], tasks: list[dict[str, Any]],
    backend_factory: Callable[[], Backend], *, resume: bool = False, limit: int = 1,
    emit: Callable[[dict[str, Any]], None] = lambda event: None,
) -> dict[str, Any]:
    """Solo procesa tareas nunca intentadas; commits individuales antes/después."""
    require(type(limit) is int and limit > 0, 'El límite debe ser positivo')
    require(len({t['variant_id'] for t in tasks}) == len(tasks), 'Variantes duplicadas')
    signature = hashlib.sha256(canonical({'identity': identity, 'tasks': tasks}).encode('utf-8')).hexdigest()
    db_path = output / 'results.sqlite3'
    if resume:
        require(db_path.is_file(), 'No existe un diario para reanudar')
    else:
        output.mkdir(parents=True, exist_ok=False)
    connection = sqlite3.connect(db_path, timeout=0)
    backend: Backend | None = None
    try:
        connection.execute('PRAGMA journal_mode=DELETE')
        connection.execute('PRAGMA synchronous=FULL')
        if not resume:
            with connection:
                connection.execute('CREATE TABLE run (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
                connection.execute('CREATE TABLE tasks (variant_id TEXT PRIMARY KEY, payload TEXT NOT NULL, '
                                   'status TEXT NOT NULL, result TEXT, started_at TEXT, finished_at TEXT)')
                connection.executemany('INSERT INTO run VALUES (?, ?)', [
                    ('run_id', canonical(f'inference-{uuid4()}')), ('signature', canonical(signature)),
                    ('identity', canonical(identity)), ('created_at', canonical(utc_now()))])
                connection.executemany('INSERT INTO tasks VALUES (?, ?, ?, NULL, NULL, NULL)', [
                    (task['variant_id'], canonical(task), 'pending' if task['applicable'] else 'not_applicable')
                    for task in tasks])
        saved = {key: json.loads(value) for key, value in connection.execute('SELECT key, value FROM run')}
        require(saved['signature'] == signature, 'Configuración, entradas o entorno distintos; no reanudar')
        stored = connection.execute('SELECT variant_id, payload FROM tasks ORDER BY rowid').fetchall()
        require(stored == [(t['variant_id'], canonical(t)) for t in tasks], 'Plan persistido alterado')
        pending = connection.execute("SELECT variant_id, payload FROM tasks WHERE status='pending' ORDER BY rowid").fetchall()
        if pending:
            backend = backend_factory()
            runtime = canonical(backend.metadata)
            if 'runtime' in saved:
                require(canonical(saved['runtime']) == runtime, 'Modelo, plantilla o hardware distintos al reanudar')
            else:
                with connection:
                    connection.execute('INSERT INTO run VALUES (?, ?)', ('runtime', runtime))
            for variant_id, payload in pending[:limit]:
                with connection:
                    connection.execute("UPDATE tasks SET status='started', started_at=? WHERE variant_id=? AND status='pending'",
                                       (utc_now(), variant_id))
                emit({'event': 'generation_started', 'variant_id': variant_id, 'run_id': saved['run_id']})
                try:
                    result = backend.generate(json.loads(payload))
                    require(result['status'] in {'generated', 'context_overflow'}, 'Resultado incompatible del backend')
                    encoded = canonical(result)
                except Exception as exc:
                    # Un fallo durante la generación tampoco autoriza un segundo intento.
                    result = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
                    with connection:
                        connection.execute('UPDATE tasks SET status=?, result=?, finished_at=? WHERE variant_id=?',
                                           ('failed', canonical(result), utc_now(), variant_id))
                    emit({'event': 'generation_failed', 'variant_id': variant_id, **result})
                    break
                with connection:
                    connection.execute('UPDATE tasks SET status=?, result=?, finished_at=? WHERE variant_id=?',
                                       (result['status'], encoded, utc_now(), variant_id))
                emit({'event': 'generation_saved', 'variant_id': variant_id, 'status': result['status']})
        counts = dict(connection.execute('SELECT status, COUNT(*) FROM tasks GROUP BY status'))
        has_error = any(counts.get(s, 0) for s in ('started', 'failed', 'context_overflow'))
        status = 'partial_failure' if has_error else ('incomplete' if counts.get('pending', 0) else 'completed')
        return {'run_id': saved['run_id'], 'status': status, 'counts': counts,
                'in_doubt': counts.get('started', 0), 'database': str(db_path)}
    finally:
        try:
            if backend is not None:
                backend.close()
        finally:
            connection.close()
