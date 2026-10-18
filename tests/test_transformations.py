"""Gramáticas, conservación exacta y barreras de seguridad de transformaciones."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import re
from typing import Any

import pytest

from src.transformations import build_conditions, read_integer, validate_condition


def annotation(text: str, approved_indices: tuple[int, ...] = (0,),
               protected: tuple[str, ...] = (), dataset: str = 'GSM8K') -> dict[str, Any]:
    """Anotación explícita de fixtures; no detector de candidatos para producción."""
    decisions, protections = [], []
    for index, match in enumerate(re.finditer(r'[0-9]+', text)):
        span = {'start': match.start(), 'end': match.end(), 'text': match.group()}
        decisions.append({**span, 'decision': 'approve' if index in approved_indices else 'protect',
                          **({'value': match.group()} if index in approved_indices else {})})
        protections.append(span)
    for literal in protected:
        start = text.index(literal)
        protections.append({'start': start, 'end': start+len(literal), 'text': literal})
    spaces = [m.start() for m in re.finditer(r'(?<=\S) (?=\S)', text)
              if not any(s['start'] <= m.start() < s['end'] for s in protections)]
    approved = [d for d in decisions if d['decision'] == 'approve']
    return {'problem_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'dataset': dataset, 'numeric_decisions': decisions, 'w_protected_spans': protections,
            'w_candidate_space_offsets': spaces,
            'contrast_candidates': {'B_vs_O': bool(approved),
                                    'N_vs_B': any(len(s['text']) >= 4 for s in approved),
                                    'D_vs_B': any(len(s['text']) >= 2 for s in approved),
                                    'W_vs_O': bool(spaces)}}


def test_five_conditions_exact_example_and_units() -> None:
    text = 'Buy $12345 and 8 kg.'
    rows = {r['condition']: r for r in build_conditions(text, annotation(text, (0, 1), ('8 kg',)))}
    assert {c: r['problem'] for c, r in rows.items()} == {
        'O': text, 'B': 'Buy $<num>12345</num> and <num>8</num> kg.',
        'N': 'Buy $<num>12,345</num> and <num>8</num> kg.',
        'D': 'Buy $<num>1 2 3 4 5</num> and <num>8</num> kg.',
        'W': 'Buy  $12345  and  8 kg.'}
    assert all(r['applicable'] for r in rows.values())


@pytest.mark.parametrize(('payload', 'condition', 'value'), [
    ('0', 'B', '0'), ('1,234', 'N', '1234'), ('123,456,789', 'N', '123456789'),
    ('123', 'N', '123'), ('0', 'N', '0'), ('1 2 3 4', 'D', '1234'), ('0', 'D', '0'),
])
def test_exact_reader(payload: str, condition: str, value: str) -> None:
    assert read_integer(payload, condition) == value


@pytest.mark.parametrize(('payload', 'condition'), [
    ('12,34', 'N'), ('1234', 'N'), ('1,,234', 'N'), ('01', 'B'), ('0 1', 'D'),
    ('1  2', 'D'), ('1\t2', 'D'), ('-12', 'B'), ('+12', 'B'), ('1.2', 'B'),
    ('1/2', 'B'), ('١٢', 'B'), (' 12', 'B'), ('12\n', 'B'), ('12', 'O'),
])
def test_reader_rejects_ambiguous_formats(payload: str, condition: str) -> None:
    with pytest.raises(ValueError):
        read_integer(payload, condition)


def test_unchanged_conditions_are_not_applicable() -> None:
    text = 'There are 6 boxes.'
    rows = {r['condition']: r for r in build_conditions(text, annotation(text))}
    assert rows['B']['applicable']
    assert rows['N']['problem'] == rows['B']['problem'] == rows['D']['problem']
    assert rows['N']['non_applicable_reason'] == 'no_grouping_change_vs_B'
    assert rows['D']['non_applicable_reason'] == 'no_digit_spacing_change_vs_B'
    empty = {r['condition']: r for r in build_conditions('word', annotation('word', ()))}
    assert empty['O']['applicable'] and all(not empty[c]['applicable'] for c in ('B', 'N', 'D', 'W'))


def test_unicode_offsets_repeated_values_and_very_long_integer() -> None:
    text = 'José has 1000 and 1000.'
    rows = build_conditions(text, annotation(text, (0, 1)))
    numeric = next(r for r in rows if r['condition'] == 'N')
    assert numeric['problem'] == 'José has <num>1,000</num> and <num>1,000</num>.'
    for edit in numeric['edits']:
        assert text[edit['source_start']:edit['source_end']] == '1000'
        assert numeric['problem'][edit['payload_start']:edit['payload_end']] == '1,000'
    long_text = 'Count ' + '1'*5000
    assert len(build_conditions(long_text, annotation(long_text))) == 5


def test_protected_context_whitespace_latex_and_code_are_untouched() -> None:
    text = 'Use 3000 items in 2020 at 3.50; 2 1/2 kg.  Keep\n\t`x = 12` and $x^2 + 1$.'
    ann = annotation(text, (0,), ('2 1/2 kg', '`x = 12`', '$x^2 + 1$'), dataset='MATH')
    rows = {r['condition']: r for r in build_conditions(text, ann)}
    assert rows['N']['problem'] == text.replace('3000', '<num>3,000</num>')
    for literal in ('2 1/2 kg', '  Keep\n\t', '`x = 12`', '$x^2 + 1$'):
        assert literal in rows['W']['problem']


@pytest.mark.parametrize('text', ['Price 3.50', 'Fraction 1/2', 'Name x12', 'Code A_12',
                                 'Power 2^3', 'Date 2020-01-01', 'Signed -12', 'Grouped 32,000'])
def test_reject_split_or_embedded_numeric_literals(text: str) -> None:
    with pytest.raises(ValueError):
        build_conditions(text, annotation(text))


@pytest.mark.parametrize('damage', ['value', 'order', 'context', 'missing_tag', 'extra_tag',
                                   'wrong_format', 'w_context', 'w_math', 'original'])
def test_validator_rejects_corrupted_transformed_text(damage: str) -> None:
    text = 'Use 12345 and 67; keep $x + 2$.'
    ann = annotation(text, (0, 1), ('$x + 2$',), dataset='MATH')
    good = {r['condition']: r['problem'] for r in build_conditions(text, ann)}
    condition, altered = 'N', good['N']
    if damage == 'value':
        altered = altered.replace('12,345', '12,346')
    elif damage == 'order':
        altered = altered.replace('<num>12,345</num> and <num>67</num>', '<num>67</num> and <num>12,345</num>')
    elif damage == 'context':
        altered = altered.replace('$x + 2$', '$x + 3$')
    elif damage == 'missing_tag':
        altered = altered.replace('</num>', '', 1)
    elif damage == 'extra_tag':
        altered += '<num>1</num>'
    elif damage == 'wrong_format':
        altered = altered.replace('12,345', '1,2345')
    elif damage in {'w_context', 'w_math'}:
        condition = 'W'
        altered = good['W'].replace('keep', 'drop') if damage == 'w_context' else good['W'].replace('x +', 'x  +')
    else:
        condition, altered = 'O', text + ' '
    with pytest.raises(ValueError):
        validate_condition(text, altered, condition, ann)


@pytest.mark.parametrize('damage', ['hash', 'offset', 'duplicate', 'w_protection', 'w_offset', 'tags'])
def test_reject_invalid_source_annotation(damage: str) -> None:
    text = 'Use 12345 and $x + 2$.'
    ann = annotation(text, (0,), ('$x + 2$',), dataset='MATH')
    if damage == 'hash':
        ann['problem_sha256'] = '0'*64
    elif damage == 'offset':
        ann['numeric_decisions'][0]['start'] += 1
    elif damage == 'duplicate':
        ann['numeric_decisions'].append(deepcopy(ann['numeric_decisions'][0]))
    elif damage == 'w_protection':
        ann['w_protected_spans'].pop()
    elif damage == 'w_offset':
        ann['w_candidate_space_offsets'].append(text.index('x +')+1)
    else:
        text = '<num>12345</num>'
        ann = annotation(text)
    with pytest.raises(ValueError):
        build_conditions(text, ann)
