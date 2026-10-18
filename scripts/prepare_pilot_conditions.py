"""Prepara o verifica las cinco condiciones del piloto; no ejecuta inferencia."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, TextIO
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.transformations import CONDITIONS, build_conditions  # noqa: E402
from diagnose import read_jsonl, sha256  # noqa: E402
from validate_pilot_review import inspect_review, require  # noqa: E402

INPUTS, AUDITS = 'condition_inputs.jsonl', 'transformation_audit.jsonl'


def load_inputs(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    """Revalida las anotaciones del piloto y fija hashes de datos, política y código utilizado."""
    config_path = root / 'configs/transformations.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    require(config['schema_version'] == 1
            and config['implementation_version'] == 'reviewed-spans-v1'
            and config['human_audit_complete'] is False and config['protocol_frozen'] is False,
            'Configuración de transformación incompatible')
    current = inspect_review(root)
    policy_path = root / config['review_policy']
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    env = json.loads((root / 'configs/environment.json').read_text(encoding='utf-8'))
    prompts = read_jsonl(root / env['data_paths']['pilot'] / 'pilot_40_prompts.jsonl')
    reviews = {r['id']: r for r in read_jsonl(root / policy['review_path'])}
    hashes = {k.replace('\\', '/'): v for k, v in
              (current['original_hashes'] | current['review_hashes']).items()}
    for relative in ('configs/transformations.json', 'configs/environment.json',
                     'src/transformations/core.py',
                     'src/transformations/__init__.py', 'scripts/prepare_pilot_conditions.py',
                     'scripts/validate_pilot_review.py', 'scripts/diagnose.py'):
        hashes[relative] = sha256(root / relative)
    return prompts, reviews, hashes


def records_for_problem(
    prompt: dict[str, Any], review: dict[str, Any], run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Selecciona explícitamente campos de entrada; nunca copia gold ni soluciones."""
    require(prompt['id'] == review['id'] and prompt['dataset'] == review['dataset'],
            'Anotación vinculada a otro problema')
    inputs, audits = [], []
    for result in build_conditions(prompt['problem'], review):
        identity = {'run_id': run_id, 'variant_id': f'{prompt["id"]}:{result["condition"]}',
                    'id': prompt['id'], 'dataset': prompt['dataset'], 'condition': result['condition']}
        inputs.append({**identity, 'problem': result['problem']})
        audits.append({**identity, **{k: v for k, v in result.items() if k not in {'problem', 'condition'}},
                       'source_problem_sha256': review['problem_sha256'],
                       'output_problem_sha256': hashlib.sha256(result['problem'].encode('utf-8')).hexdigest(),
                       'source_issue': review['source_issue'], 'human_review_status': 'pending'})
    return inputs, audits


def verify_rows(
    prompts: list[dict[str, Any]], reviews: dict[str, dict[str, Any]], run_id: str,
    inputs: list[dict[str, Any]], audits: list[dict[str, Any]],
) -> dict[str, Any]:
    require(len(inputs) == len(audits) == len(prompts)*5 == 200, 'Lote incompleto: deben existir 200 registros')
    expected_inputs, expected_audits = [], []
    for prompt in prompts:
        new_inputs, new_audits = records_for_problem(prompt, reviews[prompt['id']], run_id)
        expected_inputs.extend(new_inputs)
        expected_audits.extend(new_audits)
    require(inputs == expected_inputs, 'Textos, IDs, campos u orden de condiciones alterados')
    require(audits == expected_audits, 'Auditoría no corresponde a los textos verificados')
    counts: dict[str, Any] = {}
    for dataset in ('GSM8K', 'MATH'):
        rows = [a for a in audits if a['dataset'] == dataset]
        counts[dataset] = {
            'problems': len(rows)//5,
            'applicable': {c: sum(a['applicable'] for a in rows if a['condition'] == c) for c in CONDITIONS},
            'non_applicable': dict(Counter(a['non_applicable_reason'] for a in rows if not a['applicable'])),
        }
    return {'problems': 40, 'records': 200, 'datasets': counts,
            'numeric_span_checks': sum(len(a['edits']) for a in audits if a['condition'] in {'B', 'N', 'D'}),
            'doubled_prose_spaces': sum(len(a['edits']) for a in audits if a['condition'] == 'W'),
            'human_reviews_completed': 0, 'rejected_problems': 0,
            'scope': 'value_recovery_and_untouched_context; not_human_semantic_certification'}


def write_line(stream: TextIO, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def write_json(path: Path, record: dict[str, Any]) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def prepare(root: Path, output: Path) -> dict[str, Any]:
    prompts, reviews, hashes = load_inputs(root)
    run_id = f'conditions-{uuid4()}'
    # Reserva atómica del directorio; nunca reutilizar una salida, ni siquiera parcial.
    output.mkdir(parents=True, exist_ok=False)
    ident = None
    try:
        with (output / INPUTS).open('x', encoding='utf-8', newline='\n') as input_stream, \
                (output / AUDITS).open('x', encoding='utf-8', newline='\n') as audit_stream:
            for prompt in prompts:
                ident = prompt['id']
                inputs, audits = records_for_problem(prompt, reviews[ident], run_id)
                for entry, audit in zip(inputs, audits, strict=True):
                    write_line(input_stream, entry)
                    write_line(audit_stream, audit)
        summary = verify_rows(prompts, reviews, run_id, read_jsonl(output / INPUTS), read_jsonl(output / AUDITS))
        require(load_inputs(root)[2] == hashes, 'Las entradas cambiaron durante la preparación')
        manifest = {'schema_version': 1, 'run_id': run_id,
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                    'status': 'completed_with_pending_human_audit',
                    'validation': 'passed', 'original_files_unchanged': True,
                    'input_hashes': hashes,
                    'output_hashes': {name: sha256(output / name) for name in (INPUTS, AUDITS)},
                    'summary': summary, 'protocol_frozen': False, 'experiment_ready': False,
                    'model_generations': 0}
        write_json(output / 'manifest.json', manifest)
        return manifest
    except Exception as exc:
        write_json(output / 'failure.json', {'run_id': run_id, 'status': 'failed',
                                            'problem_id': ident, 'error': f'{type(exc).__name__}: {exc}'})
        raise


def verify_artifacts(root: Path, output: Path) -> dict[str, Any]:
    require(not (output / 'failure.json').exists(), 'Lote marcado como fallido')
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    require(manifest['schema_version'] == 1
            and manifest['status'] == 'completed_with_pending_human_audit'
            and manifest['validation'] == 'passed' and manifest['protocol_frozen'] is False
            and manifest['experiment_ready'] is False and manifest['model_generations'] == 0,
            'Manifiesto incompleto o de alcance incompatible')
    prompts, reviews, hashes = load_inputs(root)
    require(manifest['input_hashes'] == hashes, 'Cambió una entrada, configuración o implementación')
    require(manifest['output_hashes'] == {name: sha256(output / name) for name in (INPUTS, AUDITS)},
            'Hash de salida alterado')
    summary = verify_rows(prompts, reviews, manifest['run_id'],
                          read_jsonl(output / INPUTS), read_jsonl(output / AUDITS))
    require(summary == manifest['summary'], 'Resumen distinto de los archivos reales')
    return {'run_id': manifest['run_id'], 'status': 'verified', 'summary': summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output-dir', type=Path, help='Directorio NUEVO para preparar el lote')
    group.add_argument('--verify-dir', type=Path, help='Verifica un lote sin escribir ni sobrescribir')
    args = parser.parse_args()
    try:
        if args.verify_dir is not None:
            result = verify_artifacts(ROOT, ROOT / args.verify_dir)
        else:
            result = prepare(ROOT, ROOT / args.output_dir)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'},
                         ensure_ascii=True), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
