"""Cálculos específicos de estos 40 casos; no es el verificador matemático.

Las fórmulas representan la lectura documentada de cada enunciado, incluidas
las suposiciones señaladas en la auditoría. No certifican ausencia de ambigüedad.
"""
from __future__ import annotations

from fractions import Fraction as F
import json
from math import cos, gcd, isclose, isqrt, pi, sin, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def references() -> dict[str, str]:
    rows = [json.loads(line) for line in
            (ROOT / 'salida/pilot_40_gold.jsonl').read_text(encoding='utf-8').splitlines()]
    return {r['id']: r['answer'] for r in rows}


def test_gsm8k_case_specific_calculations() -> None:
    # Índice de fuente -> cálculo bajo la interpretación documentada.
    calculations = {
        1044: 5*3*4000, 4043: 1000 + 2*1000*F(3, 2),
        732: (5-1)*1000*8, 93: 1000*8 + 2000*3,
        5544: F(6500-150, 50)*72, 2756: 23+5-2+10-6,
        6649: 5350*F(5, 2), 3515: 2400*3, 5344: 3*4000,
        7455: 15*300 + 2*250 + 2000-4000,
        6609: 5*3*F(80, 100), 1506: 24-2*5-5-4,
        3270: 12*3+3*5+4*2, 1478: 750-200-250,
        2852: F(2*50+8*25, 3*25), 4882: 480*F(3, 8),
        1469: 28-F(28, 4), 2269: 40+80+120,
        2369: 500+(20-14)*50,
        5634: 32000*F(20, 100)*F(6, 100) + 15000*F(40, 100)*F(12, 100),
        5000: (5+F(350, 100)+4*2+F(350, 100))*F(9, 10),
        628: F(75, 1)/F(45, 3), 5245: 6*12*(F(150, 100)-F(25, 100))/2,
        5415: max(F(50, 25), F(99, 33)), 2802: (4*15+8*10)*F(8, 10),
        2478: 6+9+17, 548: (10000+3*10000)*F(5, 100)*7,
        2089: F(1200, 1)/F(5, 6)*F(1, 6), 6162: F(80, 20+40*F(1, 2))/2,
        6784: 2*F(120, 6),
    }
    gold = references()
    assert len(calculations) == 30
    for index, result in calculations.items():
        assert F(result) == F(gold[f'GSM8K-train-{index:05d}'])
    # El caso ambiguo se comprueba solo bajo la lectura de 4 s por caja de Tom.
    assert calculations[3515] == 1800*4


def test_math_case_specific_calculations() -> None:
    gold = references()
    results: dict[int, str] = {}
    product = F(1)
    for k in range(2, 101):
        product *= F(k-1, k)
    results[1558] = rf'\frac{{{product.numerator}}}{{{product.denominator}}}'
    assert 2**3 == F(1, 8)**-1
    results[4775] = str(4+2)
    assert all(43 % divisor for divisor in range(2, isqrt(43)+1))
    assert all(int('43'*length) % 43 == 0 and int('43'*length) > 43
               for length in (2, 3))
    results[9539] = '1'
    results[11911] = str(next(x for x in range(47) if 35*x % 47 == 1))
    edges = [(n//10, n % 10) for n in range(10, 100) if n % 19 == 0 or n % 31 == 0]
    reachable = {1}
    for _ in range(2001):
        reachable = {b for a, b in edges if a in reachable}
    results[6217] = str(max(reachable))
    width, height = F(192, 5), F(144, 5)
    assert (width-30)**2 + height**2 == 900
    assert width**2 + (height-40)**2 == 1600
    perimeter = 2*(width+height)
    results[9891] = str(perimeter.numerator+perimeter.denominator)
    assert 6*24 == 12**2
    cosine = F(3, 12)
    results[2161] = rf'\frac{{{cosine.numerator}}}{{{cosine.denominator}}}'
    assert gcd(54317, 80000) == 1
    denominator, powers = 80000, []
    for prime in (2, 5):
        power = 0
        while denominator % prime == 0:
            denominator //= prime
            power += 1
        powers.append(power)
    assert denominator == 1
    results[3841] = str(max(powers))
    results[10633] = str(F(24*4, 8))
    # Cota analítica: LHS <= sqrt(2), por tanto n <= 8.
    # Comprobación numérica del conjunto finito; no prueba simbólica general.
    residuals = {n: abs(sin(pi/(2*n))+cos(pi/(2*n))-sqrt(n)/2) for n in range(1, 9)}
    solutions = [n for n, distance in residuals.items() if isclose(distance, 0, abs_tol=1e-12)]
    assert solutions == [6] and all(d > 0.01 for n, d in residuals.items() if n != 6)
    results[11018] = str(solutions[0])
    assert len(results) == 10
    for index, result in results.items():
        assert result == gold[f'MATH-train-{index:05d}']
