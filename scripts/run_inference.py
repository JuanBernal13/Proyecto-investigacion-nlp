"""Comprueba el motor o ejecuta UN modelo; sin descargas salvo opción explícita."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import TransformersBackend, execute, exclusive_lock, resolve_device  # noqa: E402
from src.transformations.core import require  # noqa: E402
from prepare_pilot_conditions import read_jsonl, sha256, verify_artifacts, write_json  # noqa: E402


def load_config(root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    config = json.loads((root / 'configs/inference.json').read_text(encoding='utf-8'))
    env = json.loads((root / 'configs/environment.json').read_text(encoding='utf-8'))
    require(config['schema_version'] == 1 and config['protocol_frozen'] is False,
            'Configuración incompatible; este motor corresponde al piloto sin congelar')
    require(config['do_sample'] is False and config['batch_size'] == 1
            and config['quantization'] is None and config['attention_implementation'] == 'eager'
            and config['cuda_dtype'] == 'float16' and config['cpu_dtype'] == 'float32',
            'Backend incompatible con los parámetros solicitados')
    require(type(config['max_new_tokens']) is int and type(config['max_context_tokens']) is int
            and 0 < config['max_new_tokens'] < config['max_context_tokens']
            and type(config['seed']) is int and 0 <= config['seed'] < 2**32
            and isinstance(config['prompt'], str) and '<final>' in config['prompt']
            and '</final>' in config['prompt'], 'Límites, semilla o prompt inválidos')
    require({m['id'] for m in env['models']} ==
            {'Qwen/Qwen2.5-1.5B-Instruct', 'HuggingFaceTB/SmolLM2-1.7B-Instruct'}
            and all(re.fullmatch('[0-9a-f]{40}', m['revision']) for m in env['models']),
            'Modelos sin revisiones exactas válidas')
    resolve_device(config['device'])
    return config, env['models']


def prepare_plan(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    directory = root / config['conditions_dir']
    verify_artifacts(root, directory)
    inputs = read_jsonl(directory / 'condition_inputs.jsonl')
    audit = {a['variant_id']: a for a in read_jsonl(directory / 'transformation_audit.jsonl')}
    tasks = [{key: row[key] for key in ('variant_id', 'id', 'dataset', 'condition', 'problem')} |
             {'applicable': audit[row['variant_id']]['applicable'],
              'baseline': audit[row['variant_id']]['baseline'],
              'non_applicable_reason': audit[row['variant_id']]['non_applicable_reason']}
             for row in inputs]
    paths = [directory / name for name in ('manifest.json', 'condition_inputs.jsonl', 'transformation_audit.jsonl')]
    paths += [root / name for name in (
        'configs/inference.json', 'configs/environment.json', 'scripts/run_inference.py',
        'requirements.txt', 'requirements.lock.txt',
        'src/inference/__init__.py', 'src/inference/engine.py', 'src/inference/runtime.py')]
    return tasks, {str(p.relative_to(root)).replace('\\', '/'): sha256(p) for p in paths}


def environment_info(device: str) -> dict[str, Any]:
    import torch
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'device': resolve_device(device), 'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'cuda_runtime': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'torch_build': torch.__config__.show(),
            'dependencies': dict(sorted((d.metadata['Name'].lower(), d.version)
                                        for d in importlib.metadata.distributions() if d.metadata['Name']))}


def cached_model(model: dict[str, str]) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        snapshot = Path(snapshot_download(model['id'], revision=model['revision'], local_files_only=True))
    except LocalEntryNotFoundError:
        return {**model, 'snapshot_cached': False, 'safetensors_complete': False}
    index = snapshot / 'model.safetensors.index.json'
    weights = [snapshot / 'model.safetensors']
    if index.is_file():
        weights = [snapshot / name for name in set(json.loads(index.read_text())['weight_map'].values())]
    return {**model, 'snapshot_cached': True, 'safetensors_complete': bool(weights) and all(p.is_file() for p in weights)}


def log(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=True, allow_nan=False), file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true', help='Solo inspección; sin red ni generaciones')
    action.add_argument('--execute', action='store_true', help='Ejecutar inferencia cuando el protocolo esté aprobado')
    parser.add_argument('--model', choices=['Qwen/Qwen2.5-1.5B-Instruct', 'HuggingFaceTB/SmolLM2-1.7B-Instruct'])
    parser.add_argument('--output-dir', type=Path, help='Directorio nuevo o, con --resume, ejecución existente')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int, default=1, help='Máximo de tareas nuevas en esta invocación; por defecto 1')
    parser.add_argument('--allow-download', action='store_true', help='Permite descargar pesos de varios GB; por defecto offline')
    parser.add_argument('--report', type=Path, help='Guardar inspección en un archivo NUEVO')
    args = parser.parse_args()
    try:
        require(sys.version_info[:2] == (3, 11), 'Usar el entorno Python 3.11 del proyecto')
        config, models = load_config(ROOT)
        tasks, hashes = prepare_plan(ROOT, config)
        environment = environment_info(config['device'])
        if args.check:
            require(not (args.model or args.output_dir or args.resume or args.allow_download),
                    '--check no acepta opciones de ejecución')
            result = {'status': 'checked', 'environment': environment,
                      'variants': len(tasks), 'applicable': sum(t['applicable'] for t in tasks),
                      'models': [cached_model(m) for m in models], 'input_hashes': hashes,
                      'study_model_generations': 0, 'study_models_validated': False,
                      'protocol_frozen': False, 'experiment_ready': False}
            if args.report:
                write_json(ROOT / args.report, result)
        else:
            require(args.model is not None and args.output_dir is not None and args.report is None,
                    '--execute requiere --model y --output-dir; no acepta --report')
            require(args.limit > 0, '--limit debe ser positivo')
            model = next(m for m in models if m['id'] == args.model)
            identity = {'model': model, 'config': config, 'input_hashes': hashes, 'environment': environment}
            # Bloqueo compartido por todas las ejecuciones de este proyecto: un modelo a la vez.
            with exclusive_lock(ROOT / 'experiments/.inference.lock'):
                if args.allow_download:
                    log({'event': 'download_enabled', 'model': model['id'],
                         'notice': 'Se permite descargar pesos de varios gigabytes'})
                result = execute(ROOT / args.output_dir, identity, tasks,
                                 lambda: TransformersBackend(model, config, allow_download=args.allow_download),
                                 resume=args.resume, limit=args.limit, emit=log)
        print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
        return 0 if result['status'] in {'checked', 'completed'} else 2
    except KeyboardInterrupt:
        log({'status': 'interrupted', 'notice': 'Resultados confirmados conservados; no repetir intentos started'})
        return 130
    except Exception as exc:
        log({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
