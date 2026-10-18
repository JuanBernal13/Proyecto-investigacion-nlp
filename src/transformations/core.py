"""Cinco condiciones del protocolo; no selecciona números automáticamente."""
from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

Condition = Literal['O', 'B', 'N', 'D', 'W']
CONDITIONS: tuple[Condition, ...] = ('O', 'B', 'N', 'D', 'W')
BASELINES = {'O': 'O', 'B': 'O', 'N': 'B', 'D': 'B', 'W': 'O'}
OPEN, CLOSE = '<num>', '</num>'
PAYLOADS = {'B': r'(?:0|[1-9][0-9]*)',
            'N': r'(?:0|[1-9][0-9]{0,2}(?:,[0-9]{3})*)',
            'D': r'[0-9](?: [0-9])*'}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_integer(payload: str, condition: str) -> str:
    """Lee solo la gramática del brazo y devuelve dígitos canónicos sin redondeo."""
    require(condition in PAYLOADS, f'Brazo numérico desconocido: {condition}')
    require(re.fullmatch(PAYLOADS[condition], payload) is not None,
            f'Formato numérico inválido para {condition}: {payload!r}')
    digits = payload.replace(',', '').replace(' ', '')
    require(len(digits) == 1 or not digits.startswith('0'), 'Ceros iniciales ambiguos')
    return digits


def _format(digits: str, condition: Condition) -> str:
    if condition == 'N':
        first = len(digits) % 3 or 3
        return ','.join([digits[:first]] + [digits[i:i+3] for i in range(first, len(digits), 3)])
    if condition == 'D':
        return ' '.join(digits)
    return digits


def _covered(start: int, end: int, spans: list[dict[str, Any]]) -> bool:
    return any(s['start'] <= start and end <= s['end'] for s in spans)


def _notation(text: str, dataset: str) -> list[tuple[int, int, str]]:
    # shortcut: gramáticas presentes en el piloto y código común; no parser LaTeX.
    # Ampliar con reglas revisadas antes de aceptar nuevos corpus o sintaxis.
    patterns = [r'```.*?```', r'`[^`\n]*`', r'\\\[.*?\\\]', r'\\\(.*?\\\)',
                r'\\begin\{tabular\}.*?\\end\{tabular\}']
    if dataset == 'MATH':
        patterns.append(r'\$[^$]*\$')
    return [(m.start(), m.end(), m.group()) for p in patterns
            for m in re.finditer(p, text, flags=re.DOTALL)]


def _validate_binding(text: str, review: dict[str, Any]) -> list[dict[str, Any]]:
    require(review['problem_sha256'] == hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'El enunciado no coincide con el hash revisado')
    require(re.search(r'</?num\b', text, flags=re.IGNORECASE) is None,
            'El original ya contiene delimitadores num ambiguos')
    decisions = review['numeric_decisions']
    expected = [(m.start(), m.end(), m.group()) for m in re.finditer(r'[0-9]+', text)]
    require([(d['start'], d['end'], d['text']) for d in decisions] == expected,
            'Inventario de números incompleto, desordenado o con offsets alterados')
    approved = []
    for span in decisions:
        require(span['decision'] in {'approve', 'protect'}, 'Decisión numérica desconocida')
        if span['decision'] != 'approve':
            continue
        start, end = span['start'], span['end']
        require(span['value'] == read_integer(span['text'], 'B'), 'Valor fuente inconsistente')
        # Barreras locales adicionales: una anotación no debe partir un literal compuesto.
        left, right = text[:start], text[end:]
        require(not (left and (left[-1].isalnum() or left[-1] in '_\\'))
                and not (right and (right[0].isalnum() or right[0] == '_')),
                'Entero embebido en identificador o unidad sin separación')
        require(not re.search(r'[0-9][.,/-]$|(?:[/+^-]|\^\s*\{)\s*$', left)
                and not re.match(r'\s*(?:[.,/-][0-9]|[/^])', right),
                'El intervalo divide decimal, fracción, fecha, signo o exponente')
        approved.append(span)
    protected = review['w_protected_spans']
    for span in protected:
        require(type(span['start']) is int and type(span['end']) is int
                and 0 <= span['start'] < span['end'] <= len(text)
                and span['text'] == text[span['start']:span['end']], 'Protección W inválida')
    for start, end, _ in expected:
        require(_covered(start, end, protected), 'Número sin protección W')
    for start, end, notation in _notation(text, review['dataset']):
        require(_covered(start, end, protected), 'Notación o código sin protección W completa')
        for span in approved:
            if start < span['end'] and span['start'] < end:
                require(notation.strip('$').strip() == span['text'] and notation.startswith('$'),
                        'Intervalo numérico aprobado dentro de código o notación compuesta')
    for match in re.finditer(r'[0-9]+ [0-9]+/[0-9]+', text):
        require(_covered(match.start(), match.end(), protected), 'Fracción mixta sin protección W')
    spaces = [m.start() for m in re.finditer(r'(?<=\S) (?=\S)', text)
              if not _covered(m.start(), m.end(), protected)]
    require(review['w_candidate_space_offsets'] == spaces, 'Espacios de prosa no corresponden a revisión')
    return approved


def _render(text: str, condition: Condition, spans: list[dict[str, Any]], spaces: list[int]) -> str:
    if condition == 'O':
        return text
    replacements = ([(p, p+1, '  ') for p in spaces] if condition == 'W' else
                    [(s['start'], s['end'], OPEN+_format(s['text'], condition)+CLOSE) for s in spans])
    chunks, cursor = [], 0
    for start, end, replacement in replacements:
        chunks.extend((text[cursor:start], replacement))
        cursor = end
    return ''.join(chunks) + text[cursor:]


def validate_condition(
    original: str, transformed: str, condition: Condition, review: dict[str, Any],
) -> dict[str, Any]:
    """Valida por lectura y comparación de segmentos, sin llamar al renderizador."""
    require(condition in CONDITIONS, f'Condición desconocida: {condition}')
    approved = _validate_binding(original, review)
    edits: list[dict[str, Any]] = []
    original_cursor, output_cursor = 0, 0
    if condition == 'O':
        require(transformed == original, 'O debe ser idéntico al original')
    elif condition == 'W':
        for pos in review['w_candidate_space_offsets']:
            prefix = original[original_cursor:pos]
            target = output_cursor + len(prefix)
            require(transformed[output_cursor:target] == prefix
                    and transformed[target:target+2] == '  ', 'W alteró texto o un espacio no autorizado')
            edits.append({'source_start': pos, 'source_end': pos+1, 'source_text': ' ',
                          'output_start': target, 'output_end': target+2, 'output_text': '  '})
            original_cursor, output_cursor = pos+1, target+2
        require(transformed[output_cursor:] == original[original_cursor:], 'W alteró contexto exterior')
    else:
        matches = list(re.finditer(r'<num>(.*?)</num>', transformed, flags=re.DOTALL))
        require(len(matches) == len(approved), 'Cantidad de etiquetas distinta de spans revisados')
        for span, match in zip(approved, matches, strict=True):
            require(transformed[output_cursor:match.start()] == original[original_cursor:span['start']],
                    'Cambió texto fuera de los números aprobados')
            recovered = read_integer(match.group(1), condition)
            require(recovered == span['value'], 'Cambió valor u orden de los números')
            edits.append({'source_start': span['start'], 'source_end': span['end'],
                          'source_text': span['text'], 'value_before': span['value'],
                          'output_start': match.start(), 'output_end': match.end(),
                          'output_text': match.group(), 'payload_start': match.start(1),
                          'payload_end': match.end(1), 'value_after': recovered})
            original_cursor, output_cursor = span['end'], match.end()
        require(transformed[output_cursor:] == original[original_cursor:], 'Cambió el contexto final')
    return {'condition': condition, 'value_and_context_checks': 'passed', 'edits': edits}


def build_conditions(original: str, review: dict[str, Any]) -> list[dict[str, Any]]:
    """Genera y valida todas las condiciones; rechaza el problema entero ante un fallo."""
    approved = _validate_binding(original, review)
    texts = {c: _render(original, c, approved, review['w_candidate_space_offsets']) for c in CONDITIONS}
    result = []
    for condition in CONDITIONS:
        validation = validate_condition(original, texts[condition], condition, review)
        baseline = BASELINES[condition]
        applies = condition == 'O' or texts[condition] != texts[baseline]
        reason = None
        if not applies:
            reason = ('no_approved_integer' if not approved and condition in {'B', 'N', 'D'} else
                      {'B': 'no_change_vs_O', 'N': 'no_grouping_change_vs_B',
                       'D': 'no_digit_spacing_change_vs_B', 'W': 'no_prose_space_change_vs_O'}[condition])
        result.append({'condition': condition, 'problem': texts[condition],
                       'baseline': baseline, 'applicable': applies, 'non_applicable_reason': reason,
                       'changed_from_original': texts[condition] != original, **validation})
    # La comprobación por condición ya compara cada intervalo con el original común.
    source_spans = [[(e['source_start'], e['source_end'], e['value_before']) for e in row['edits']]
                    for row in result if row['condition'] in {'B', 'N', 'D'}]
    require(source_spans[0] == source_spans[1] == source_spans[2], 'B/N/D no usan los mismos spans')
    actual = {f'{r["condition"]}_vs_{r["baseline"]}': r['applicable']
              for r in result if r['condition'] != 'O'}
    require(actual == review['contrast_candidates'], 'Aplicabilidad real distinta de la revisión')
    return result
