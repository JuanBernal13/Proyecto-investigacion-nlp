"""Proceso desechable: parser LaTeX completo y pruebas simbólicas exactas, sin eval."""
from __future__ import annotations

import json
import re
import sys
from typing import Any

from src.evaluation.core import ExtractionError, indeterminate, numeric_value, unwrap_math


def parse_math(text: str) -> Any:
    import sympy as sp
    from latex2sympy2_extended import latex2sympy
    from latex2sympy2_extended.latex2sympy2 import ConversionConfig, _Latex2Sympy

    value = unwrap_math(text)
    try:
        rational = numeric_value(value)
        return sp.Rational(rational.numerator, rational.denominator)
    except (ValueError, ZeroDivisionError):
        pass
    allowed = {'frac', 'dfrac', 'tfrac', 'sqrt', 'sin', 'cos', 'tan', 'log', 'ln',
               'pi', 'infty', 'cdot', 'times', 'left', 'right', 'emptyset',
               'cup', 'cap', 'langle', 'rangle'}
    if any(command not in allowed for command in re.findall(r'\\([A-Za-z]+)', value)):
        raise ExtractionError('unsupported_latex_command')
    residue = re.sub(r'\\[A-Za-z]+', '', value)
    if re.search(r'[A-Za-z]{2,}', residue) or re.search(r'[^A-Za-z0-9\s+\-*/^{}()[\],.\\<>|]', residue):
        raise ExtractionError('unsupported_math_syntax')
    if re.fullmatch(r'\([^()]*,[^()]*\)', value):
        raise ExtractionError('ambiguous_tuple_or_interval')
    # Decimales se convierten antes del parser para evitar redondeo a Float.
    def exact_decimal(match: re.Match[str]) -> str:
        number = numeric_value(match[0])
        return rf'\frac{{{number.numerator}}}{{{number.denominator}}}'
    value = re.sub(r'(?<![\d.])(?:\d+\.\d+|\.\d+)(?![\d.])', exact_decimal, value)
    conversion = ConversionConfig(lowercase_symbols=False, interpret_simple_eq_as_assignment=False,
                                  interpret_contains_as_eq=False)
    # API privada fijada en lock: exige EOF, pues aceptar un prefijo pierde significado.
    parser = _Latex2Sympy(config=conversion).create_parser(value)
    parser.math()
    if parser.getCurrentToken().type != -1:
        raise ExtractionError('unconsumed_latex')
    result = latex2sympy(value, normalization_config=None, conversion_config=conversion)
    if isinstance(result, sp.Basic) and result.has(sp.Float):
        raise ExtractionError('inexact_parser_result')
    return result


def compare_objects(gold: Any, prediction: Any) -> dict[str, Any]:
    import sympy as sp

    def verdict(equal: bool, reason: str) -> dict[str, Any]:
        return {'classification': 'correct' if equal else 'incorrect', 'reason': reason}

    if isinstance(gold, sp.Set) or isinstance(prediction, sp.Set):
        if not isinstance(gold, sp.Set) or not isinstance(prediction, sp.Set):
            return verdict(False, 'set_scalar_type_mismatch')
        difference = sp.simplify(sp.SymmetricDifference(gold, prediction))
        if difference.is_empty is not None:
            return verdict(bool(difference.is_empty), 'exact_set_comparison')
        return indeterminate('set_equality_unresolved')
    if not isinstance(gold, sp.Expr) or not isinstance(prediction, sp.Expr):
        return indeterminate('unsupported_mathematical_type')
    if gold.has(sp.zoo, sp.nan) or prediction.has(sp.zoo, sp.nan):
        return indeterminate('undefined_expression')
    if any(power.base.free_symbols and power.exp.is_negative is True
           for expression in (gold, prediction) for power in expression.atoms(sp.Pow)):
        return indeterminate('domain_assumptions_required')
    if gold == prediction:
        return verdict(True, 'exact_symbolic_identity')
    difference = sp.simplify(gold - prediction)
    if difference == 0:
        return verdict(True, 'exact_symbolic_simplification')
    if not difference.free_symbols and difference.is_zero is False:
        return verdict(False, 'exact_constant_difference')
    if difference.free_symbols and difference.is_polynomial(*difference.free_symbols):
        return verdict(False, 'nonzero_polynomial')
    return indeterminate('symbolic_equality_unresolved')


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        try:
            gold = parse_math(payload['gold'])
        except Exception as exc:
            result = indeterminate('reference_math_parse', error=f'{type(exc).__name__}: {exc}')
        else:
            try:
                prediction = parse_math(payload['prediction'])
            except Exception as exc:
                result = indeterminate('prediction_math_parse', error=f'{type(exc).__name__}: {exc}')
            else:
                result = compare_objects(gold, prediction)
    except Exception as exc:
        result = indeterminate('verifier_error', error=f'{type(exc).__name__}: {exc}')
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))


if __name__ == '__main__':
    main()
