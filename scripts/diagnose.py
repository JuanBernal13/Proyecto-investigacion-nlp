"""Inspección local del entorno: no descarga datos ni ejecuta modelos."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "torch", "transformers", "datasets", "tokenizers", "tiktoken", "pandas",
    "numpy", "sympy", "math-verify", "scipy", "statsmodels", "matplotlib", "pytest",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_pilot(
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    sources: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Comprueba integridad y procedencia; no certifica transformabilidad semántica."""
    if len(prompts) != 40 or len(gold) != 40:
        raise ValueError("Se requieren 40 enunciados y 40 referencias")
    ids = [row["id"] for row in prompts]
    gold_ids = [row["id"] for row in gold]
    if len(set(ids)) != 40 or len(set(gold_ids)) != 40 or set(ids) != set(gold_ids):
        raise ValueError("IDs duplicados o correspondencia prompt/gold incompleta")
    if len({row["problem"] for row in prompts}) != 40:
        raise ValueError("Enunciados duplicados")
    counts = Counter(row["dataset"] for row in prompts)
    if counts != {"GSM8K": 30, "MATH": 10}:
        raise ValueError("Composición del piloto distinta de 30 GSM8K + 10 MATH")
    gold_by_id = {row["id"]: row for row in gold}
    for row in prompts:
        dataset, index = row["dataset"], row["source_index"]
        if row["split"] != "train":
            raise ValueError(f"Partición ajena a desarrollo: {row['id']}")
        allowed = {"id", "dataset", "split", "source_index", "problem", "has_long_digits"}
        if dataset == "MATH":
            allowed |= {"original_unique_id", "subject", "level"}
        if set(row) != allowed:
            raise ValueError(f"Campos inesperados o ausentes en prompt: {row['id']}")
        if type(index) is not int or not 0 <= index < len(sources[dataset]):
            raise ValueError(f"Índice fuera de fuente: {row['id']}")
        if row["id"] != f"{dataset}-train-{index:05d}":
            raise ValueError(f"ID no corresponde al índice: {row['id']}")
        original = sources[dataset][index]
        reference = gold_by_id[row["id"]]
        problem = original["question" if dataset == "GSM8K" else "problem"]
        if not problem.strip() or row["problem"] != problem:
            raise ValueError(f"Enunciado vacío o alterado: {row['id']}")
        if dataset == "GSM8K":
            parts = original["answer"].rsplit("####", 1)
            if len(parts) != 2:
                raise ValueError(f"Referencia GSM8K sin delimitador: {row['id']}")
            answer, solution = parts[1].strip(), original["answer"]
        else:
            answer, solution = str(original["answer"]), original["solution"]
            expected_metadata = {
                "original_unique_id": original["unique_id"],
                "subject": original["subject"], "level": int(original["level"]),
            }
            if any(row[key] != value for key, value in expected_metadata.items()):
                raise ValueError(f"Metadatos MATH alterados: {row['id']}")
            if "[asy]" in problem.lower() or "[/asy]" in problem.lower():
                raise ValueError(f"Asymptote excluido por el piloto existente: {row['id']}")
        if not answer.strip() or reference["answer"] != answer or reference["solution"] != solution:
            raise ValueError(f"Referencia vacía o alterada: {row['id']}")
        if row["has_long_digits"] != bool(re.search(r"(?<!\d)\d{4,}(?!\d)", problem)):
            raise ValueError(f"Indicador numérico inconsistente: {row['id']}")
    return {
        "status": "ok", "count": len(prompts), "datasets": dict(counts),
        "long_digit_candidates": dict(Counter(p["dataset"] for p in prompts if p["has_long_digits"])),
        "math_levels": dict(Counter(str(p["level"]) for p in prompts if p["dataset"] == "MATH")),
        "math_subjects": dict(Counter(p["subject"] for p in prompts if p["dataset"] == "MATH")),
        "heldout_overlap": "not_checked_no_local_test_files",
        "semantic_validation": "pending_review_and_transformations",
    }


def inspect_data(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    raw, pilot = (root / config["data_paths"][key] for key in ("raw", "pilot"))
    sources = {name: read_jsonl(raw / f"{name}_train.jsonl") for name in ("GSM8K", "MATH")}
    for name, expected_count in (("GSM8K", 7473), ("MATH", 12000)):
        if len(sources[name]) != expected_count:
            raise ValueError(f"Conteo fuente inesperado: {name}")
        if sha256(raw / f"{name}_train.jsonl") != config["source_sha256"][name]:
            raise ValueError(f"Hash fuente inesperado: {name}")
    prompts = read_jsonl(pilot / "pilot_40_prompts.jsonl")
    result = validate_pilot(prompts, read_jsonl(pilot / "pilot_40_gold.jsonl"), sources)
    provenance = json.loads((pilot / "pilot_40_procedencia.json").read_text(encoding="utf-8"))
    if provenance["selected_ids"] != [p["id"] for p in prompts] or provenance["seed"] != config["seed"]:
        raise ValueError("Procedencia o semilla no corresponde al piloto")
    for name, expected in config["source_sha256"].items():
        if provenance["sources"][f"{name}_train"]["sha256"] != expected:
            raise ValueError(f"Hash de procedencia inconsistente: {name}")
    with (pilot / "pilot_40_indice.csv").open(encoding="utf-8-sig", newline="") as stream:
        index = list(csv.DictReader(stream))
    for p, entry in zip(prompts, index, strict=True):
        for key, value in entry.items():
            if value != str(p.get(key) if p.get(key) is not None else ""):
                raise ValueError(f"Índice CSV inconsistente: {p['id']}/{key}")
    result["source_counts"] = {name: len(rows) for name, rows in sources.items()}
    result["hashes"] = {str(p.relative_to(root)): sha256(p) for directory in (raw, pilot) for p in sorted(directory.iterdir()) if p.is_file()}
    return result


def inspect_packages() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in PACKAGES:
        try:
            version = importlib.metadata.version(name)
            importlib.import_module(name.replace("-", "_"))
            result[name] = {"status": "ok", "version": version}
        except Exception as exc:
            result[name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return result


def inspect_gpu() -> dict[str, Any]:
    try:
        import torch
        result: dict[str, Any] = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
        if result["cuda_available"]:
            device = torch.device("cuda:0")
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(device)
            value = (torch.tensor([2.0], device=device) * 3).item()
            torch.cuda.synchronize(device)
            if value != 6.0:
                raise RuntimeError("Operación CUDA de comprobación incorrecta")
            free, total = torch.cuda.mem_get_info(device)
            result.update(name=torch.cuda.get_device_name(device), total_bytes=total,
                          free_bytes=free, tensor_smoke_test="ok",
                          smoke_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        return result
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def inspect_library_smoke() -> dict[str, Any]:
    """Prueba solo constantes confiables; no valida el futuro verificador."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import statsmodels.api
        from math_verify import parse, verify

        gold = parse(r"$\frac{1}{2}$", parsing_timeout=None)
        prediction = parse("$0.5$", parsing_timeout=None)
        if not gold or not prediction or not verify(gold, prediction, timeout_seconds=None):
            raise RuntimeError("Falló equivalencia de constantes 1/2 y 0.5")
        return {
            "status": "ok", "transformers_auto_imports": "ok", "statsmodels_api": "ok",
            "math_verify_fixed_constants": "ok_without_internal_timeout",
            "internal_timeout_validated": False,
            "limitation": "Windows: timeout interno de math-verify 0.8.0 falló por función local no serializable. El verificador requiere aislamiento externo con plazo o entorno Linux; no usar sin plazo sobre salidas de modelos.",
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def inspect_model_cache(config: dict[str, Any]) -> dict[str, Any]:
    hf_home = Path(os.environ.get("HF_HOME", str(Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface")))
    roots = {hf_home / "hub"}
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        if os.environ.get(key):
            roots.add(Path(os.environ[key]))
    result: dict[str, Any] = {"searched_roots": sorted(str(p) for p in roots)}
    for model in config["models"]:
        paths = [root / ("models--" + model["id"].replace("/", "--")) / "snapshots" / model["revision"] for root in roots]
        result[model["id"]] = {
            "revision": model["revision"],
            "candidate_snapshot_paths": [str(p) for p in paths],
            "weights_present": any((p / "model.safetensors").is_file() for p in paths),
            "tokenizer_present": any((p / "tokenizer.json").is_file() for p in paths),
            "scope": "standard_and_configured_HF_caches_only; no_integrity_or_load_test",
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="JSON nuevo; nunca sobrescribe")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error(f"El archivo ya existe; elija otro nombre: {args.output}")
    config = json.loads((ROOT / "configs/environment.json").read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "python_311": sys.version_info[:2] == (3, 11),
    }
    try:
        report["data"] = inspect_data(ROOT, config)
    except Exception as exc:
        report["data"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    report["packages"] = inspect_packages()
    report["library_smoke"] = inspect_library_smoke()
    report["gpu"] = inspect_gpu()
    report["models"] = inspect_model_cache(config)
    try:
        process = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=60)
        report["pip_check"] = {"returncode": process.returncode, "output": process.stdout + process.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["pip_check"] = {"returncode": -1, "output": str(exc)}
    report["environment_ready"] = bool(report["python_311"] and all(p["status"] == "ok" for p in report["packages"].values()) and report["library_smoke"]["status"] == "ok" and report["gpu"].get("tensor_smoke_test") == "ok" and report["pip_check"]["returncode"] == 0)
    report["experiment_ready"] = False
    report["status"] = "inspection_completed_with_pending_work" if report["data"].get("status") == "ok" and report["environment_ready"] else "inspection_with_blockers"
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    print(payload)
    return 0 if report["data"].get("status") == "ok" and report["environment_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
