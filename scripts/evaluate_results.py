"""Evalúa inferencias existentes en otro diario; --check no ejecuta modelos."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation import evaluate  # noqa: E402
from src.inference.engine import canonical, exclusive_lock, utc_now  # noqa: E402
from src.transformations.core import require  # noqa: E402
from prepare_pilot_conditions import read_jsonl, sha256, verify_artifacts, write_json  # noqa: E402


def load_config(root: Path) -> dict[str, Any]:
    config = json.loads((root / 'configs/evaluation.json').read_text(encoding='utf-8'))
    require(config['schema_version'] == 1
            and config['protocol_frozen'] is False and config['numeric_tolerance'] is None
            and config['gsm8k_fallback'] == 'last_nonempty_line_final_answer_is'
            and config['math_fallback'] == 'last_complete_boxed', 'Reglas de evaluación incompatibles')
    require(type(config['timeout_seconds']) in {int, float} and 0 < config['timeout_seconds'] <= 60
            and all(type(config[k]) is int and 0 < config[k] <= 100000
                    for k in ('max_answer_chars', 'max_response_chars')), 'Límites inválidos')
    return config


def conditions_dir(root: Path) -> Path:
    config = json.loads((root / 'configs/transformations.json').read_text(encoding='utf-8'))
    return root / config['default_output_dir']


def identity(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    paths = [config['gold_path'], 'configs/evaluation.json', 'scripts/evaluate_results.py',
             'src/evaluation/__init__.py', 'src/evaluation/core.py', 'src/evaluation/worker.py',
             'src/inference/engine.py', 'src/transformations/core.py', 'requirements.lock.txt']
    return {'config': config, 'input_hashes': {p: sha256(root / p) for p in paths},
            'python': sys.version, 'dependencies': {p: importlib.metadata.version(p)
            for p in ('sympy', 'latex2sympy2_extended', 'antlr4-python3-runtime')}}


def references(root: Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(root / config['gold_path'])
    require(len({r['id'] for r in rows}) == len(rows), 'Referencias duplicadas')
    return {row['id']: row for row in rows}


def evaluate_run(source: Path, output: Path, gold: dict[str, dict[str, Any]],
                 config: dict[str, Any], provenance: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
    require(source.is_file(), 'No existe la base de inferencias')
    before = sha256(source)
    reader = sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        with reader:
            reader.execute('BEGIN')
            run = {k: json.loads(v) for k, v in reader.execute('SELECT key,value FROM run')}
            tasks = reader.execute('SELECT variant_id,payload,status,result FROM tasks ORDER BY rowid').fetchall()
    finally:
        reader.close()
    require(before == sha256(source), 'La inferencia cambió durante la lectura')
    require(tasks and len({r[0] for r in tasks}) == len(tasks), 'Plan vacío o duplicado')
    prepared = []
    for variant, payload, status, result in tasks:
        task = json.loads(payload)
        require(task['variant_id'] == variant and task['id'] in gold
                and task['dataset'] in {'GSM8K', 'MATH'} and task['condition'] in {'O', 'B', 'N', 'D', 'W'}
                and type(task['applicable']) is bool, 'Tarea incompatible con referencias')
        require(task['id'].startswith(task['dataset']+'-'), 'Dataset e identificador inconsistentes')
        require(status in {'generated', 'failed', 'started', 'context_overflow', 'pending', 'not_applicable'},
                'Estado de inferencia desconocido')
        decoded = json.loads(result) if result is not None else None
        require((status != 'not_applicable') == task['applicable'], 'Aplicabilidad inconsistente')
        if status == 'generated':
            require(isinstance(decoded, dict) and decoded.get('status') == status
                    and isinstance(decoded.get('text'), str), 'Generación dañada')
        prepared.append((variant, task, status, decoded))
    bound = {**provenance, 'source_sha256': before, 'source_run_id': run['run_id']}
    signature = canonical(bound)
    database = output / 'evaluation.sqlite3'
    if resume:
        require(database.is_file(), 'No existe evaluación para reanudar')
    else:
        output.mkdir(parents=True, exist_ok=False)
    connection = sqlite3.connect(database, timeout=0)
    try:
        connection.execute('PRAGMA synchronous=FULL')
        if not resume:
            with connection:
                connection.execute('CREATE TABLE run (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
                connection.execute('CREATE TABLE evaluations (variant_id TEXT PRIMARY KEY,result TEXT NOT NULL)')
                connection.executemany('INSERT INTO run VALUES (?,?)', [
                    ('run_id', canonical(f'evaluation-{uuid4()}')), ('identity', signature),
                    ('created_at', canonical(utc_now()))])
        saved = dict(connection.execute('SELECT key,value FROM run'))
        require(saved['identity'] == signature, 'Fuente, referencias, reglas o entorno distintos; no reanudar')
        existing = {row[0] for row in connection.execute('SELECT variant_id FROM evaluations')}
        require(existing <= {row[0] for row in prepared}, 'Evaluación con variantes ajenas al plan')
        for variant, task, status, generated in prepared:
            if variant in existing:
                continue
            if status == 'not_applicable':
                verdict = {'classification': None, 'reason': 'not_applicable'}
            elif status != 'generated':
                verdict = {'classification': 'indeterminate', 'reason': 'inference_'+status}
            else:
                verdict = evaluate(task['dataset'], generated['text'], gold[task['id']], config)
            row = {**{k: task[k] for k in ('id', 'dataset', 'condition', 'applicable')},
                   'source_status': status, 'evaluated_at': utc_now(),
                   'reached_token_limit': generated.get('reached_token_limit')
                   if generated else None, **verdict}
            with connection:
                connection.execute('INSERT INTO evaluations VALUES (?,?)', (variant, canonical(row)))
            print(canonical({'event': 'evaluation_saved', 'variant_id': variant,
                             'classification': verdict['classification']}), file=sys.stderr, flush=True)
        counts = Counter(json.loads(row[0])['classification'] or 'not_applicable'
                         for row in connection.execute('SELECT result FROM evaluations'))
        unresolved = counts.get('indeterminate', 0)
        source_complete = all(row[2] in {'generated', 'not_applicable'} for row in prepared)
        require(before == sha256(source), 'La fuente cambió; conservar resultados y revisar antes de reanudar')
        return {'run_id': json.loads(saved['run_id']),
                'status': 'completed' if source_complete else 'partial_source',
                'resolution_status': 'has_indeterminate' if unresolved else 'resolved',
                'counts': dict(counts), 'database': str(database), 'source_unchanged': before == sha256(source),
                'notice': 'completed describe procesamiento; no certifica corrección matemática de todos los casos'}
    finally:
        connection.close()


def check(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    verify_artifacts(root, conditions_dir(root))
    gold = references(root, config)
    controls = []
    for problem_id, reference in gold.items():
        dataset = problem_id.split('-')[0]
        result = evaluate(dataset, '<final>'+reference['answer']+'</final>', reference, config)
        controls.append({'id': problem_id, **result})
    success = all(r['classification'] == 'correct' for r in controls)
    return {'status': 'checked' if success else 'check_failed',
            **identity(root, config), 'reference_roundtrip_controls': controls,
            'study_model_generations': 0, 'scientific_accuracy': None, 'protocol_frozen': False,
            'notice': 'Controles de compatibilidad con referencias; no validación independiente ni inferencia del piloto'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--evaluate', action='store_true')
    parser.add_argument('--input-db', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    try:
        require(sys.version_info[:2] == (3, 11), 'Usar Python 3.11 del proyecto')
        config = load_config(ROOT)
        if args.check:
            require(not (args.input_db or args.output_dir or args.resume), '--check no acepta opciones de ejecución')
            result = check(ROOT, config)
            if args.report:
                write_json(ROOT / args.report, result)
        else:
            require(args.input_db and args.output_dir and args.report is None,
                    '--evaluate requiere --input-db y --output-dir; no acepta --report')
            verify_artifacts(ROOT, conditions_dir(ROOT))
            with exclusive_lock(ROOT / 'experiments/.inference.lock'):
                result = evaluate_run(ROOT / args.input_db, ROOT / args.output_dir,
                                      references(ROOT, config), config, identity(ROOT, config), resume=args.resume)
        print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
        return 0 if result['status'] in {'checked', 'completed'} else 2
    except KeyboardInterrupt:
        print(canonical({'status': 'interrupted', 'notice': 'Evaluaciones confirmadas conservadas'}), file=sys.stderr)
        return 130
    except Exception as exc:
        print(canonical({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
