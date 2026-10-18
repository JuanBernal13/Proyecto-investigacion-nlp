"""Casos controlados; ninguna salida pertenece a los modelos del estudio."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from src.evaluation.core import ExtractionError, evaluate, extract_boxed, extract_prediction, numeric_value
from src.evaluation.worker import compare_objects, parse_math

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs/evaluation.json').read_text())


@pytest.mark.parametrize(('value', 'expected'), [
    ('−12,345.50', Fraction(-24691, 2)), ('+0.5', Fraction(1, 2)),
    ('1 / 3', Fraction(1, 3)), ('$1,000$', Fraction(1000)),
    (r'\(0.125\)', Fraction(1, 8)), (r'\boxed{42}', Fraction(42)),
    (r'\boxed{ 42 }', Fraction(42)),
])
def test_exact_numbers(value: str, expected: Fraction) -> None:
    assert numeric_value(value) == expected


@pytest.mark.parametrize('value', ['1,00', '12,34,567', '1 2', '42 dollars', '50%',
                                 '1/0', '1e3', 'nan', '__import__("os")', '1.2.3'])
def test_reject_ambiguous_numbers(value: str) -> None:
    with pytest.raises((ExtractionError, ValueError)):
        numeric_value(value)


def test_nested_and_last_complete_box() -> None:
    assert extract_boxed(r'\boxed{0} then \boxed{\frac{1}{\sqrt{2}}}') == r'\frac{1}{\sqrt{2}}'
    assert extract_boxed(r'\boxed{\{1,2\}}') == r'\{1,2\}'
    assert extract_boxed(r'\boxed{2} then \boxed{broken') == '2'
    with pytest.raises(ExtractionError):
        extract_boxed(r'\boxed{\frac{1}{2}')


def test_extraction_priority_and_fallback() -> None:
    assert extract_prediction(r'\boxed{1} <final>2</final>', 'MATH') == ('2', 'final_tag')
    assert extract_prediction('calculated 99\nFinal answer is: 2', 'GSM8K')[1] == 'final_line_fallback'
    assert extract_prediction(r'We get \boxed{\frac{1}{2}}', 'MATH')[1] == 'boxed_fallback'
    assert extract_prediction('<final>1</final><final>2</final>', 'MATH') == ('2', 'final_tag')
    assert extract_boxed(r'\boxed{\frac{\boxed{1}}{2}}') == r'\frac{\boxed{1}}{2}'
    assert extract_prediction('Final answer is: 2', 'GSM8K')[0] == '2'
    for text in ('We get 42.', '<final>42', '<final></final>', '<final>1</final><final>2</final>'):
        with pytest.raises(ExtractionError):
            extract_prediction(text, 'GSM8K')


@pytest.mark.parametrize(('response', 'classification', 'reason'), [
    ('<final>1,000.0</final>', 'correct', 'exact_rational'),
    ('<final>999</final>', 'incorrect', 'exact_rational'),
    ('The answer involves 1000', 'indeterminate', 'no_final_answer'),
    ('<final>1000 dollars</final>', 'indeterminate', 'prediction_numeric_parse'),
    ('<final>1,00</final>', 'indeterminate', 'prediction_numeric_parse'),
    ('<final>1000.000000000000001</final>', 'incorrect', 'exact_rational'),
])
def test_gsm_classification(response: str, classification: str, reason: str) -> None:
    result = evaluate('GSM8K', response, {'answer': '1000', 'solution': 'reason\n#### 1,000'}, CONFIG)
    assert (result['classification'], result['reason']) == (classification, reason)


def test_bad_reference_and_limits() -> None:
    for reference in ({}, {'answer': '1', 'solution': 'no marker'}, {'answer': '2', 'solution': '#### 1'}):
        assert evaluate('GSM8K', '<final>1</final>', reference, CONFIG)['classification'] == 'indeterminate'
    assert evaluate('MATH', 'x'*100001, {'answer': '1'}, CONFIG)['reason'] == 'response_size_or_type'
    assert evaluate('MATH', '<final>'+'1'*2049+'</final>', {'answer': '1'}, CONFIG)['reason'] == 'answer_too_long'
    with pytest.raises(ValueError):
        evaluate('other', '', {}, CONFIG)


@pytest.mark.parametrize(('gold', 'prediction', 'classification'), [
    (r'\frac{1}{2}', '0.5', 'correct'),
    (r'\frac{1}{3}', '0.3333333333333333', 'incorrect'),
    ('0.5', '0.500000000001', 'incorrect'),
    (r'\sqrt{8}', r'2\sqrt{2}', 'correct'),
    ('(x+1)^2', 'x^2+2x+1', 'correct'),
    ('x+1', 'x+2', 'incorrect'),
    ('X+1', 'x+1', 'incorrect'),
    (r'\sin(x)^2+\cos(x)^2', '1', 'correct'),
    (r'\{1,2\}', r'\{2,1\}', 'correct'),
    (r'\{1,2\}', r'\{1,3\}', 'incorrect'),
    (r'\{1\}', '1', 'incorrect'),
    (r'[1,2]', r'[1,2)', 'incorrect'),
    (r'\emptyset', r'\emptyset', 'correct'),
    (r'\frac{3}{4}', r'\frac{1}{2}+0.25', 'correct'),
])
def test_exact_symbolic_rules(gold: str, prediction: str, classification: str) -> None:
    result = compare_objects(parse_math(gold), parse_math(prediction))
    assert result['classification'] == classification


@pytest.mark.parametrize('text', [r'\text{two}', '(1,2)', '1 garbage', '1;2',
                                r'\frac{1}{2}junk', r'\unknown{1}', '1=2'])
def test_no_partial_or_ambiguous_math(text: str) -> None:
    with pytest.raises(Exception):
        parse_math(text)


def test_real_isolated_worker() -> None:
    result = evaluate('MATH', r'<final>2\sqrt{2}</final>', {'answer': r'\sqrt{8}'}, CONFIG)
    assert result['classification'] == 'correct'
    result = evaluate('MATH', '<final>unparseable</final>', {'answer': '2'}, CONFIG)
    assert result['classification'] == 'indeterminate' and result['reason'] == 'prediction_math_parse'
    result = evaluate('MATH', '<final>2</final>', {'answer': r'\unknown{2}'}, CONFIG)
    assert result['reason'] == 'reference_math_parse'


def test_real_timeout_kills_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.evaluation import core
    real_run = subprocess.run
    marker = tmp_path / 'should_not_exist'
    code = 'import time,pathlib; time.sleep(2); pathlib.Path('+repr(str(marker))+').touch()'

    def slow_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return real_run([sys.executable, '-c', code], **kwargs)

    monkeypatch.setattr(core.subprocess, 'run', slow_run)
    start = time.perf_counter()
    result = evaluate('MATH', '<final>1</final>', {'answer': '1'}, {**CONFIG, 'timeout_seconds': 0.15})
    assert result['reason'] == 'verification_timeout'
    assert time.perf_counter()-start < 2
    time.sleep(2.1)
    assert not marker.exists()


def test_worker_crash_and_corrupt_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.evaluation import core
    for completed, reason in [(subprocess.CompletedProcess([], 1, '', ''), 'worker_failure'),
                              (subprocess.CompletedProcess([], 0, 'broken', ''), 'verifier_error'),
                              (subprocess.CompletedProcess([], 0, '{}', ''), 'invalid_worker_result')]:
        monkeypatch.setattr(core.subprocess, 'run', lambda *a, **k: completed)
        assert evaluate('MATH', '<final>1</final>', {'answer': '1'}, CONFIG)['reason'] == reason


def test_unresolved_symbolic_comparison() -> None:
    import sympy as sp
    x = sp.Symbol('x')
    assert compare_objects(sp.sin(x), x)['classification'] == 'indeterminate'
    assert compare_objects(sp.zoo, sp.Integer(1))['classification'] == 'indeterminate'
    assert compare_objects(parse_math(r'\frac{x}{x}'), sp.Integer(1))['reason'] == 'domain_assumptions_required'


def test_prose_is_not_silently_discarded() -> None:
    assert evaluate('GSM8K', '42\nFinal answer: 1000',
                    {'answer': '1000', 'solution': '#### 1000'}, CONFIG)['fallback_used'] is True
    result = evaluate('MATH', r'<final>\boxed{1} or \boxed{2}</final>', {'answer': '2'}, CONFIG)
    assert result['classification'] == 'indeterminate'
