"""Valida el registro de revisión del piloto; no transforma ni genera respuestas."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from diagnose import ROOT, inspect_data, read_jsonl, sha256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def covered(start: int, end: int, spans: list[dict[str, Any]]) -> bool:
    """Exige protección de toda la expresión por al menos un intervalo."""
    return any(s['start'] <= start and end <= s['end'] for s in spans)


def validate_review(
    prompts: list[dict[str, Any]], gold: list[dict[str, Any]],
    reviews: list[dict[str, Any]], audits: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Comprueba anotaciones explícitas; no demuestra equivalencia semántica."""
    ids = {p['id'] for p in prompts}
    require(len(prompts) == len(ids) == 40, 'Se requieren 40 enunciados únicos')
    for label, rows in (('review', reviews), ('audit', audits), ('gold', gold)):
        require(len(rows) == 40 and {r['id'] for r in rows} == ids,
                f'IDs incompletos o duplicados: {label}')
    require(policy['schema_version'] == 1,
            'Versión de política inválida')
    require(policy['human_audit_complete'] is False and policy['protocol_frozen'] is False,
            'Esta revisión no certifica auditoría humana ni protocolo congelado')
    by_id = {p['id']: p for p in prompts}
    gold_by_id = {g['id']: g for g in gold}
    review_by_id = {r['id']: r for r in reviews}
    summary: dict[str, Any] = {'datasets': {}, 'exclusion_reasons': {}, 'source_issues': []}
    exclusions: Counter[str] = Counter()
    required_keys = {
        'schema_version', 'review_id', 'id', 'dataset', 'problem_sha256',
        'review_kind', 'human_review_status', 'source_issue', 'numeric_decisions',
        'w_protected_spans', 'w_candidate_space_offsets', 'contrast_candidates', 'notes',
    }
    for review in reviews:
        ident = review['id']
        require(set(review) == required_keys, f'Campos de revisión inesperados: {ident}')
        prompt, text = by_id[ident], by_id[ident]['problem']
        require(review['schema_version'] == 1 and review['review_id'] == policy['review_id'],
                f'Versión de revisión inconsistente: {ident}')
        require(review['dataset'] == prompt['dataset'], f'Dataset inconsistente: {ident}')
        require(review['problem_sha256'] == hashlib.sha256(text.encode('utf-8')).hexdigest(),
                f'Texto cambiado desde revisión: {ident}')
        require(review['review_kind'] == 'ai_assisted_case_review'
                and review['human_review_status'] == 'pending', f'Autoría incorrecta: {ident}')
        require(review['source_issue'] in {'none_identified', 'ambiguous_wording'},
                f'Estado del enunciado desconocido: {ident}')
        require(bool(review['notes'].strip()), f'Falta justificación: {ident}')
        decisions = review['numeric_decisions']
        expected = [(m.start(), m.end(), m.group()) for m in re.finditer(r'[0-9]+', text)]
        actual = [(d['start'], d['end'], d['text']) for d in decisions]
        require(expected == actual, f'Inventario de dígitos incompleto o alterado: {ident}')
        approved = []
        for decision in decisions:
            base = {'start', 'end', 'text', 'decision', 'reason'}
            if decision['decision'] == 'approve':
                require(set(decision) == base | {'value'}
                        and decision['reason'] == 'unambiguous_whole_quantity'
                        and decision['value'] == str(int(decision['text']))
                        and not (len(decision['text']) > 1 and decision['text'].startswith('0')),
                        f'Valor aprobado inconsistente: {ident}')
                approved.append(decision)
            else:
                require(decision['decision'] == 'protect' and set(decision) == base
                        and decision['reason'] in policy['allowed_exclusion_reasons'],
                        f'Exclusión desconocida: {ident}')
                exclusions[decision['reason']] += 1
        spans = review['w_protected_spans']
        for span in spans:
            require(set(span) == {'start', 'end', 'text', 'reason'}
                    and type(span['start']) is int and type(span['end']) is int
                    and 0 <= span['start'] < span['end'] <= len(text)
                    and text[span['start']:span['end']] == span['text']
                    and span['reason'] in policy['allowed_protection_reasons'],
                    f'Protección W inválida: {ident}')
        for start, end, _ in expected:
            require(covered(start, end, spans), f'Dígitos sin protección W: {ident}')
        # Reglas de cobertura para las notaciones presentes, no un parser LaTeX general.
        patterns = [r'[0-9]+ [0-9]+/[0-9]+']
        if prompt['dataset'] == 'MATH':
            patterns += [r'\$[^$]*\$', r'\\\[.*?\\\]',
                         r'\\begin\{tabular\}.*?\\end\{tabular\}']
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.DOTALL):
                require(covered(match.start(), match.end(), spans),
                        f'Expresión sin protección W completa: {ident}')
        spaces = [m.start() for m in re.finditer(r'(?<=\S) (?=\S)', text)
                  if not covered(m.start(), m.end(), spans)]
        require(review['w_candidate_space_offsets'] == spaces,
                f'Espacios W inconsistentes: {ident}')
        contrasts = {'B_vs_O': bool(approved),
                     'N_vs_B': any(len(d['text']) >= 4 for d in approved),
                     'D_vs_B': any(len(d['text']) >= 2 for d in approved),
                     'W_vs_O': bool(spaces)}
        require(review['contrast_candidates'] == contrasts
                and all(type(v) is bool for v in review['contrast_candidates'].values()),
                f'Contrastes candidatos inconsistentes: {ident}')
        counts = summary['datasets'].setdefault(prompt['dataset'], {
            'problems': 0, 'digit_runs': 0, 'approved_spans': 0,
            'contrast_candidates': dict.fromkeys(contrasts, 0),
        })
        counts['problems'] += 1
        counts['digit_runs'] += len(decisions)
        counts['approved_spans'] += len(approved)
        for contrast, applies in contrasts.items():
            counts['contrast_candidates'][contrast] += int(applies)
        if review['source_issue'] != 'none_identified':
            summary['source_issues'].append({'id': ident, 'issue': review['source_issue']})
    for audit in audits:
        ident = audit['id']
        require(set(audit) == {'schema_version', 'id', 'review_kind', 'human_review_status',
                              'status', 'expected_answer', 'reasoning', 'method'},
                f'Campos de auditoría inesperados: {ident}')
        require(audit['schema_version'] == 1 and audit['review_kind'] == 'ai_assisted_case_review'
                and audit['human_review_status'] == 'pending'
                and audit['method'] == 'case_specific_reasoning_not_general_math_verifier',
                f'Alcance de auditoría incorrecto: {ident}')
        require(audit['expected_answer'] == gold_by_id[ident]['answer']
                and bool(audit['reasoning'].strip()), f'Referencia auditada inconsistente: {ident}')
        expected_status = ('interpretation_dependent'
                           if review_by_id[ident]['source_issue'] == 'ambiguous_wording'
                           else 'consistent_by_case_reasoning')
        require(audit['status'] == expected_status, f'Ambigüedad no reflejada en auditoría: {ident}')
    summary['exclusion_reasons'] = dict(sorted(exclusions.items()))
    summary.update(problem_count=40, human_reviews_completed=0,
                   reference_statuses=dict(Counter(a['status'] for a in audits)),
                   transformed_value_preservation='pending_transformation_check',
                   tokenizer_context_fit='not_checked_no_model_tokenizers',
                   independent_mathematical_verification='not_claimed',
                   semantic_scope='explicit_case_annotations_not_general_detector')
    return summary


def inspect_review(root: Path) -> dict[str, Any]:
    config = json.loads((root / 'configs/environment.json').read_text(encoding='utf-8'))
    policy_path = root / 'configs/pilot_review_policy.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    data = inspect_data(root, config)
    pilot = root / config['data_paths']['pilot']
    paths = [policy_path, root / policy['review_path'], root / policy['reference_audit_path']]
    summary = validate_review(read_jsonl(pilot / 'pilot_40_prompts.jsonl'),
                              read_jsonl(pilot / 'pilot_40_gold.jsonl'),
                              read_jsonl(paths[1]), read_jsonl(paths[2]), policy)
    return {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'completed_with_pending_human_audit',
            'structural_validation': 'ok', 'original_files_unchanged': True,
            'original_hashes': data['hashes'],
            'review_hashes': {p.relative_to(root).as_posix(): sha256(p) for p in paths},
            'heldout_overlap': data['heldout_overlap'], 'summary': summary,
            'protocol_frozen': False, 'experiment_ready': False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='JSON nuevo; nunca sobrescribe')
    args = parser.parse_args()
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError(f'El reporte ya existe: {args.output}')
        result = inspect_review(ROOT)
        if args.output is not None:
            # Creación exclusiva: también protege ante otro proceso concurrente.
            with args.output.open('x', encoding='utf-8') as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'failed',
                          'error': f'{type(exc).__name__}: {exc}'}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
