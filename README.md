# Tokenización de texto matemático y desempeño de LLMs

Investigación sobre cómo la presentación de problemas matemáticos, su tokenización y el desempeño de modelos de lenguaje con pesos congelados se relacionan.

**Estado:** el entorno, la validación del piloto, las transformaciones, el motor de inferencia y el verificador están implementados y probados. **No existe ninguna respuesta generada por los modelos del estudio ni métricas de exactitud.** El protocolo no está congelado (`protocol_frozen=false`).

## Objetivo

> ¿Cómo se relacionan las características de tokenización de los enunciados matemáticos con los errores de un LLM, y cuánto cambia su exactitud al modificar de forma controlada la presentación del mismo problema?

Se miden segmentación de números, consumo de tokens, exactitud, sensibilidad al formato, tipos de error, latencia y memoria GPU, comparando dos modelos preentrenados sin fine-tuning ni APIs comerciales.

Las transformaciones también alteran la familiaridad de la notación: **el diseño no identifica un efecto causal exclusivo del tokenizador**.

## Datos

| Conjunto | Desarrollo | Piloto | Evaluación prevista |
|---|---:|---:|---:|
| GSM8K | 7.473 | 30 | 400 |
| MATH (PRM800K) | 12.000 | 10 | 200 |
| **Total** | **19.473** | **40** | **600** |

El piloto usa semillas 2026 (MATH) y 2027 (GSM8K) y sobrerrepresenta números largos: no es una muestra poblacional ni tiene pesos de inclusión. GSM8K test y MATH-500 no están en el repositorio; el cruce de solapamientos está pendiente.

## Condiciones experimentales

| ID | Cambio | Ejemplo |
|---|---|---|
| O | Original | `12345` |
| B | Entero delimitado | `<num>12345</num>` |
| N | Comas de miles | `<num>12,345</num>` |
| D | Dígitos separados | `<num>1 2 3 4 5</num>` |
| W | Espacios dobles en prosa autorizada | fuera de expresiones protegidas |

B, N y D usan los mismos números y delimitadores. W no se aplica sobre B/N/D. No se transforman decimales, fechas, fracciones, exponentes, identificadores ni notación ambigua. Los offsets son caracteres Unicode de Python, `[inicio, fin)`.

| Dataset | O | B | N | D | W |
|---|---:|---:|---:|---:|---:|
| GSM8K | 30 | 30 | 11 | 29 | 30 |
| MATH | 10 | 2 | 1 | 2 | 10 |
| **Total aplicable** | **40** | **32** | **12** | **31** | **40** |

Se conservan 200 variantes: 155 aplicables y 45 no aplicables, que no cuentan como aciertos ni errores. El piloto requiere hasta 310 generaciones entre los dos modelos. Los controles recuperaron los 94 valores aprobados en B/N/D y 1.416 espacios duplicados en W; son conteos de transformación, no de desempeño.

## Modelos

| Modelo | Revisión | Estado |
|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | pesos no descargados |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `31b70e2e869a7173562077fd711b654946d38674` | pesos no descargados |

Configuración en `configs/inference.json`: pesos congelados, `do_sample=false`, lote de uno, `max_new_tokens=1024`, contexto de `4096`, plantilla de chat oficial, `float16` en GPU y `float32` en CPU, atención `eager`, un modelo a la vez y sin truncamiento silencioso.

## Verificación

| Estado | Significado |
|---|---|
| `correct` | Coincidencia bajo las reglas documentadas. |
| `incorrect` | Discrepancia matemática determinada. |
| `indeterminate` | Extracción fallida, sintaxis no soportada, timeout u otro impedimento. |

**GSM8K:** referencia del marcador `####`, comparación de racionales exactos sin tolerancia; la predicción requiere `<final>...</final>`.
**MATH:** último `<final>...</final>` completo o último `\boxed{...}` completo, con SymPy, parser LaTeX y plazo externo de 10 segundos. No cubre toda la sintaxis MATH.

Inferencia y evaluación usan diarios SQLite separados, escritura incremental y reanudación con identidad verificada. No se sobrescriben resultados ni se repiten intentos inciertos. Un fallo de extracción no demuestra un error de razonamiento.

## Qué existe

- `fuentes/`: 7.473 problemas GSM8K y 12.000 MATH de desarrollo.
- `salida/`: 40 enunciados del piloto, 40 referencias separadas, índice y procedencia.
- `data/processed/pilot_40_review_v1.jsonl`: anotación de 192 secuencias de dígitos y espacios.
- `data/processed/pilot_40_reference_audit_v1.jsonl`: auditoría de las 40 referencias.
- `data/processed/conditions_v1/`: 200 variantes sin referencias, auditoría y manifiesto con hashes.
- `src/transformations/`, `src/inference/`, `src/evaluation/`: implementación.
- `scripts/`: diagnóstico, validación, preparación de condiciones, inferencia y evaluación.
- `tests/`: 163 pruebas registradas.
- `configs/`: cinco archivos de configuración.

## Qué falta

1. Auditoría humana de 50 casos de desarrollo, exigida por la propuesta; hoy no existe.
2. Adjudicar `GSM8K-train-03515`, ambiguo entre segundos por caja o por pareja.
3. Ampliar cobertura MATH para N/B y D/B antes de generar.
4. Cruzar solapamientos con GSM8K test y MATH-500.
5. Descargar y validar los pesos en la VRAM disponible, un modelo a la vez.
6. Medir tokenización y ajuste al contexto con los tokenizadores oficiales.
7. Ejecutar el piloto y medir latencia, VRAM, tokens, cortes de salida y fallos.
8. Auditar respuestas reales con el verificador.
9. Documentar ajustes y congelar el protocolo.
10. Seleccionar 600 problemas y registrar probabilidades de inclusión.
11. Ejecutar el experimento final, hasta 6.000 generaciones.
12. Análisis estadístico, revisión de errores e informe reproducible.

## Cómo ejecutar

Requiere Python 3.11. El lock fija Windows x64 con PyTorch CUDA 12.6; en otra plataforma, instalar `requirements.txt` con la rueda de PyTorch correspondiente al hardware.

```bash
python3.11 -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.lock.txt
pip check
```

Comprobaciones que no descargan modelos ni generan respuestas del estudio:

```bash
python scripts/diagnose.py
python scripts/validate_pilot_review.py
python scripts/prepare_pilot_conditions.py --verify-dir data/processed/conditions_v1
python scripts/run_inference.py --check
python scripts/evaluate_results.py --check
python -m pytest tests -q
```

`--check` del evaluador contrasta las 40 referencias consigo mismas: detecta incompatibilidades, no mide exactitud. Para guardar otra comprobación, usar un nombre de salida nuevo; la evidencia existente no se sobrescribe.

Con pesos descargados y protocolo aprobado:

```bash
python scripts/run_inference.py --execute --model Qwen/Qwen2.5-1.5B-Instruct --output-dir experiments/qwen_pilot_v1 --limit 1
python scripts/run_inference.py --execute --model Qwen/Qwen2.5-1.5B-Instruct --output-dir experiments/qwen_pilot_v1 --resume --limit 200
python scripts/evaluate_results.py --evaluate --input-db experiments/qwen_pilot_v1/results.sqlite3 --output-dir results/qwen_pilot_eval_v1
```

SmolLM2 usa otro directorio. Sin pesos locales los comandos fallan de forma explícita; `--allow-download` habilita varios gigabytes de descarga.

**No ejecutar `generar_piloto_40.py` sobre la muestra existente:** puede sobrescribirla.

## Estructura

```text
configs/        configuración de entorno, revisión, transformaciones, inferencia y evaluación
fuentes/        datos públicos de desarrollo
salida/         piloto original, con referencias separadas
data/processed/ anotaciones y variantes generadas
src/            transformaciones, inferencia y verificación
scripts/        comandos ejecutables
tests/          pruebas unitarias y de integración
```

## Límites

- El diseño no aísla el tokenizador de la familiaridad de la notación.
- El piloto está enriquecido en números largos; no permite estimar exactitud poblacional sin ponderación.
- La cobertura MATH es pequeña: N/B tiene un solo caso.
- El verificador no cubre toda la sintaxis MATH ni prueba todos los pasos de una solución.
- No hay tablas de exactitud, intervalos ni conclusiones empíricas. Una diferencia no significativa no demuestra equivalencia.

Los artefactos incluyen hashes de datos y componentes. `.gitattributes` preserva los bytes originales para no invalidar esos controles.
