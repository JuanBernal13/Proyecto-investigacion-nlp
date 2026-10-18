"""Reglas conservadoras: un fallo del verificador nunca demuestra una respuesta incorrecta."""
from __future__ import annotations

from fractions import Fraction
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


class ExtractionError(ValueError):
    """No existe una respuesta final inequívoca bajo las reglas configuradas."""


def boxed_blocks(text: str) -> list[tuple[int, int, str]]:
    """Bloques exteriores completos; las llaves escapadas no cambian profundidad."""
    answers: list[tuple[int, int, str]] = []
    consumed = 0
    for match in re.finditer(r'\\boxed\s*\{', text):
        if match.start() < consumed:
            continue
        depth, start, cursor = 1, match.end(), match.end()
        while cursor < len(text) and depth:
            if text[cursor] == '\\':
                cursor += 2
                continue
            if text[cursor] == '{':
                depth += 1
            elif text[cursor] == '}':
                depth -= 1
            cursor += 1
        if depth == 0:
            answers.append((match.start(), cursor, text[start:cursor-1].strip()))
            consumed = cursor
    return answers


def extract_boxed(text: str) -> str:
    """Último bloque exterior completo con llaves anidadas."""
    answers = boxed_blocks(text)
    if not answers or not answers[-1][2]:
        raise ExtractionError('no_complete_boxed')
    return answers[-1][2]


def extract_prediction(text: str, dataset: str) -> tuple[str, str]:
    """GSM8K exige un delimitador único; MATH toma el último completo (propuesta 5.6)."""
    if '<final>' in text or '</final>' in text:
        if dataset == 'MATH':
            matches = list(re.finditer(r'<final>((?:(?!<final>|</final>).)*)</final>', text, re.DOTALL))
            if matches and matches[-1][1].strip():
                return matches[-1][1].strip(), 'final_tag'
            return extract_boxed(text), 'boxed_fallback'
        if text.count('<final>') != 1 or text.count('</final>') != 1:
            raise ExtractionError('multiple_or_incomplete_final')
        match = re.search(r'<final>(.*?)</final>', text, re.DOTALL)
        if match is None or not match[1].strip():
            raise ExtractionError('empty_or_malformed_final')
        return match[1].strip(), 'final_tag'
    if dataset == 'MATH':
        return extract_boxed(text), 'boxed_fallback'
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    match = re.fullmatch(r'Final answer\s*(?:is\s*:?\s*|:\s*)(.+)', lines[-1], re.IGNORECASE) if lines else None
    if match is None:
        raise ExtractionError('no_final_answer')
    return match[1].strip(), 'final_line_fallback'


def unwrap_math(text: str) -> str:
    value = text.strip().replace('\N{MINUS SIGN}', '-')
    for left, right in (('$$', '$$'), ('$', '$'), (r'\(', r'\)'), (r'\[', r'\]')):
        if value.startswith(left) and value.endswith(right) and len(value) > len(left)+len(right):
            value = value[len(left):-len(right)].strip()
            break
    if re.match(r'\\boxed\s*\{', value):
        # El bloque debe abarcar la respuesta entera, sin texto residual.
        blocks = boxed_blocks(value)
        if len(blocks) != 1 or blocks[0][:2] != (0, len(value)):
            raise ExtractionError('boxed_has_extra_content')
        value = blocks[0][2]
    return value


def numeric_value(text: str) -> Fraction:
    """Decimal/entero/racional exacto; comas solo en grupos de miles válidos."""
    value = unwrap_math(text)
    atom = r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)'
    if not re.fullmatch(rf'{atom}(?:\s*/\s*{atom})?', value, flags=re.ASCII):
        raise ExtractionError('unsupported_numeric_answer')
    parts = [Fraction(part.strip().replace(',', '')) for part in value.split('/')]
    try:
        return parts[0] if len(parts) == 1 else parts[0] / parts[1]
    except ZeroDivisionError as exc:
        raise ExtractionError('zero_denominator') from exc


def indeterminate(reason: str, **details: Any) -> dict[str, Any]:
    return {'classification': 'indeterminate', 'reason': reason, **details}


def symbolic_compare(gold: str, prediction: str, timeout: float) -> dict[str, Any]:
    """El plazo incluye importación, parsing y simplificación; run mata y espera al hijo."""
    try:
        completed = subprocess.run(
            [sys.executable, '-m', 'src.evaluation.worker'],
            input=json.dumps({'gold': gold, 'prediction': prediction}),
            capture_output=True, text=True, encoding='utf-8', timeout=timeout,
            cwd=Path(__file__).resolve().parents[2], check=False,
        )
        if completed.returncode:
            return indeterminate('worker_failure', worker_returncode=completed.returncode)
        result = json.loads(completed.stdout)
        if result.get('classification') not in {'correct', 'incorrect', 'indeterminate'}:
            return indeterminate('invalid_worker_result')
        return result
    except subprocess.TimeoutExpired:
        return indeterminate('verification_timeout')
    except (OSError, ValueError) as exc:
        return indeterminate('verifier_error', error=f'{type(exc).__name__}: {exc}')


def evaluate(dataset: str, response: str, reference: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Evalúa una salida; las referencias se reciben únicamente en este módulo."""
    if dataset not in {'GSM8K', 'MATH'}:
        raise ValueError('Dataset no admitido')
    if not isinstance(response, str) or len(response) > config['max_response_chars']:
        return indeterminate('response_size_or_type')
    try:
        prediction, extraction = extract_prediction(response, dataset)
    except ExtractionError as exc:
        return indeterminate(str(exc), extraction=None)
    details = {'prediction': prediction, 'extraction': extraction,
               'final_open_count': response.count('<final>'),
               'fallback_used': extraction != 'final_tag'}
    if len(prediction) > config['max_answer_chars']:
        return indeterminate('answer_too_long', **details)
    try:
        if dataset == 'GSM8K':
            solution = reference['solution']
            if solution.count('####') != 1:
                return indeterminate('invalid_reference_marker', **details)
            gold = numeric_value(solution.split('####')[1].strip())
            if gold != numeric_value(reference['answer']):
                return indeterminate('reference_disagreement', **details)
            try:
                predicted = numeric_value(prediction)
            except (ExtractionError, ValueError, ZeroDivisionError) as exc:
                return indeterminate('prediction_numeric_parse', error=str(exc), **details)
            return {'classification': 'correct' if gold == predicted else 'incorrect',
                    'reason': 'exact_rational', 'reference_normalized': str(gold),
                    'prediction_normalized': str(predicted), **details}
        gold = reference['answer']
        if not isinstance(gold, str) or not gold.strip() or len(gold) > config['max_answer_chars']:
            return indeterminate('invalid_reference_answer', **details)
        result = symbolic_compare(gold, prediction, config['timeout_seconds'])
        return {**result, **details}
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return indeterminate('reference_error', error=f'{type(exc).__name__}: {exc}', **details)
