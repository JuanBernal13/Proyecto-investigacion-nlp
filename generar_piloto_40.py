#!/usr/bin/env python3
"""Obtiene 40 problemas reales para un piloto MATH/GSM8K SIN usar splits test.

Ejecutar: python generar_piloto_40.py
Solo necesita Python 3.9+ e internet; sin paquetes de terceros.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

SEED = 2026
BASE = Path(__file__).resolve().parent
ORIG = BASE / 'fuentes'
OUT = BASE / 'salida'
ORIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

SOURCES = {
    'GSM8K_train': {
        'urls': [
            'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl',
            'https://github.com/openai/grade-school-math/raw/refs/heads/master/grade_school_math/data/train.jsonl',
        ],
        'sha256': '17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465',
        'size': 7473,
    },
    'MATH_train': {
        'urls': [
            'https://huggingface.co/datasets/Mai0313/prm800k/resolve/main/prm800k/math_splits/train.jsonl?download=true',
            'https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/math_splits/train.jsonl',
            'https://github.com/openai/prm800k/raw/refs/heads/main/prm800k/math_splits/train.jsonl',
        ],
        'sha256': '90d96daeac3fe343ebb1e22ce93dd99690f75983e957f88de42f87cffe1e8076',
        'size': 12000,
    },
}


def get_source(name: str):
    cfg = SOURCES[name]
    target = ORIG / (name + '.jsonl')
    if target.exists():
        data = target.read_bytes()
        if hashlib.sha256(data).hexdigest() == cfg['sha256']:
            return read_jsonl(target)
        print('Advertencia: archivo local incorrecto, se volverá a descargar:', target.name)
    errors = []
    for url in cfg['urls']:
        try:
            print('Descargando', name, 'desde', url.split('?')[0], flush=True)
            request = urllib.request.Request(url, headers={'User-Agent': 'math-tokenization-pilot/1.0'})
            with urllib.request.urlopen(request, timeout=100) as r:
                raw = r.read()
            actual = hashlib.sha256(raw).hexdigest()
            if actual != cfg['sha256']:
                raise ValueError(
                    f'SHA256 incorrecto (recibido {actual[:16]}...). '
                    'Posiblemente descargó puntero Git LFS en lugar de datos reales.'
                )
            target.write_bytes(raw)
            return read_jsonl(target)
        except Exception as exc:
            errors.append(f'{url}: {exc}')
    raise RuntimeError('No se pudo descargar ' + name + '.\n' + '\n'.join(errors) +
                       '\nDescarga manualmente el train.jsonl original y guárdalo como ' + str(target))


def read_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def math_level(item):
    v = str(item.get('level', ''))
    m = re.search(r'[1-5]', v)
    return int(m.group()) if m else None


def math_problem(item):
    return item.get('problem', '')


def valid_math(row):
    p = math_problem(row)
    # Para el piloto, excluir geometría con imagen ausente y enunciados sin respuesta.
    return bool(p.strip() and str(row.get('answer', '')).strip() and
                '[asy]' not in p.lower() and '[/asy]' not in p.lower())


def sample_math(data):
    rng = random.Random(SEED)
    usable = [(i, r) for i, r in enumerate(data) if valid_math(r)]
    picked = []
    ids = set()
    used_subject = set()
    text_seen = set()

    def choose(candidates):
        valid = [(idx, row) for idx, row in candidates
                 if idx not in ids and math_problem(row) not in text_seen]
        if not valid:
            return False
        idx, item = rng.choice(valid)
        picked.append((idx, item))
        ids.add(idx)
        used_subject.add(str(item.get('subject', '')))
        text_seen.add(math_problem(item))
        return True

    # Cubrir dificultades 1–5 y priorizar variedad de materias.
    for difficulty in range(1, 6):
        candidates = [(i, r) for i, r in usable if math_level(r) == difficulty]
        new_subjects = [(i, r) for i, r in candidates if str(r.get('subject', '')) not in used_subject]
        if not choose(new_subjects if new_subjects else candidates):
            raise RuntimeError(f'No existe muestra MATH disponible para nivel {difficulty}')

    # Una representación de cada materia, hasta llenar los 10.
    for subject in sorted({str(r.get('subject', '')) for _, r in usable}):
        if len(picked) >= 10:
            break
        if subject and subject not in used_subject:
            choose([(i, r) for i, r in usable if str(r.get('subject', '')) == subject])

    # Resto: privilegiar LaTeX fraccional, números largos y niveles variados.
    while len(picked) < 10:
        options = [(i, r) for i, r in usable
                   if i not in ids and math_problem(r) not in text_seen and
                   ('\\frac' in math_problem(r) or re.search(r'\d{4,}', math_problem(r)))]
        if not choose(options if options else usable):
            raise RuntimeError('Faltan problemas MATH válidos')
    return picked


def sample_gsm(data):
    rng = random.Random(SEED + 1)
    all_records = [(i, r) for i, r in enumerate(data)
                   if r.get('question', '').strip() and r.get('answer', '').strip()]
    # 10 problemas con cadenas de al menos cuatro dígitos para comprobar el detector.
    # Atención: la detección es solo superficial; NO autoriza transformación automática.
    numeric = [(i, r) for i, r in all_records if re.search(r'(?<!\d)\d{4,}(?!\d)', r['question'])]
    selected = rng.sample(numeric, min(10, len(numeric)))
    selected_ids = {i for i, _ in selected}
    remain = [(i, r) for i, r in all_records if i not in selected_ids]
    selected += rng.sample(remain, 30 - len(selected))
    return selected


def main():
    gsm = get_source('GSM8K_train')
    math = get_source('MATH_train')
    assert len(gsm) == SOURCES['GSM8K_train']['size'], f'GSM8K: {len(gsm)}'
    assert len(math) == SOURCES['MATH_train']['size'], f'MATH: {len(math)}'

    selected_gsm = sample_gsm(gsm)
    selected_math = sample_math(math)
    prompts, gold = [], []

    for source_index, record in selected_gsm:
        id_ = f'GSM8K-train-{source_index:05d}'
        parts = record['answer'].rsplit('####', 1)
        if len(parts) != 2 or not parts[1].strip():
            raise RuntimeError(f'GSM8K sin referencia final: {id_}')
        prompts.append({
            'id': id_, 'dataset': 'GSM8K', 'split': 'train', 'source_index': source_index,
            'problem': record['question'],
            'has_long_digits': bool(re.search(r'(?<!\d)\d{4,}(?!\d)', record['question'])),
        })
        gold.append({'id': id_, 'answer': parts[1].strip(), 'solution': record['answer']})

    for source_index, record in selected_math:
        id_ = f'MATH-train-{source_index:05d}'
        prompts.append({
            'id': id_, 'dataset': 'MATH', 'split': 'train', 'source_index': source_index,
            'original_unique_id': record.get('unique_id'),
            'subject': record.get('subject'), 'level': math_level(record),
            'problem': record['problem'],
            'has_long_digits': bool(re.search(r'(?<!\d)\d{4,}(?!\d)', record['problem'])),
        })
        gold.append({'id': id_, 'answer': str(record['answer']), 'solution': record['solution']})

    assert len(prompts) == 40 and len(gold) == 40
    assert len({p['id'] for p in prompts}) == 40
    assert Counter(p['dataset'] for p in prompts) == {'GSM8K': 30, 'MATH': 10}
    assert all(p['split'] == 'train' for p in prompts)

    write_jsonl(OUT / 'pilot_40_prompts.jsonl', prompts)
    write_jsonl(OUT / 'pilot_40_gold.jsonl', gold)
    with (OUT / 'pilot_40_indice.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'dataset', 'split', 'source_index',
                         'original_unique_id', 'subject', 'level', 'has_long_digits'],
                         extrasaction='ignore')
        writer.writeheader()
        writer.writerows(prompts)
    with (OUT / 'pilot_40_lectura.md').open('w', encoding='utf-8') as f:
        f.write('# Piloto: 40 problemas originales\n\n')
        f.write('**Solo enunciados. Las soluciones están en pilot_40_gold.jsonl.**\n\n')
        for n, p in enumerate(prompts, 1):
            f.write(f"## {n}. {p['id']} ({p['dataset']})\n\n{p['problem']}\n\n")
    info = {
        'seed': SEED,
        'sample': {'GSM8K': 30, 'MATH': 10},
        'restrictions': 'Solo train. Se excluyen MATH con Asymptote. Gold nunca va al prompt.',
        'sources': {name: {'sha256': cfg['sha256'], 'download_url': cfg['urls'][0]}
                    for name, cfg in SOURCES.items()},
        'selected_ids': [p['id'] for p in prompts],
        'math_distribution': {'subjects': dict(Counter(p.get('subject', '') for p in prompts if p['dataset']=='MATH')),
                              'levels': dict(Counter(str(p.get('level')) for p in prompts if p['dataset']=='MATH'))},
    }
    (OUT / 'pilot_40_procedencia.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nLISTO: archivos en', OUT)
    print('40 problemas auténticos: 30 GSM8K, 10 MATH; ninguno es del test.')
    print('Enunciados para enviar al LLM: pilot_40_prompts.jsonl')
    print('Respuestas y soluciones, solo para evaluar: pilot_40_gold.jsonl')
    print('Versión legible: pilot_40_lectura.md')
    print('CSV e información de reproducibilidad: pilot_40_indice.csv / pilot_40_procedencia.json')


if __name__ == '__main__':
    main()
