"""Integridad de datos reales y rechazo de daños sin modificar archivos del piloto."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("diagnose", ROOT / "scripts/diagnose.py")
assert SPEC is not None and SPEC.loader is not None
diagnose = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnose)


@pytest.fixture(scope="module")
def data() -> tuple:
    return (
        diagnose.read_jsonl(ROOT / "salida/pilot_40_prompts.jsonl"),
        diagnose.read_jsonl(ROOT / "salida/pilot_40_gold.jsonl"),
        {name: diagnose.read_jsonl(ROOT / f"fuentes/{name}_train.jsonl") for name in ("GSM8K", "MATH")},
    )


def test_real_files_integrity() -> None:
    config = json.loads((ROOT / "configs/environment.json").read_text(encoding="utf-8"))
    assert diagnose.inspect_data(ROOT, config)["status"] == "ok"


def test_cli_preserves_existing_report(tmp_path: Path) -> None:
    report = tmp_path / "existing.json"
    report.write_text("existing result", encoding="utf-8")
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts/diagnose.py"), "--output", str(report)],
        capture_output=True, text=True, timeout=10,
    )
    assert process.returncode != 0
    assert report.read_text(encoding="utf-8") == "existing result"


@pytest.mark.parametrize("damage", ["duplicate", "gold", "leak", "split", "index", "metadata"])
def test_reject_corruption(data: tuple, damage: str) -> None:
    prompts, gold, sources = deepcopy(data)
    if damage == "duplicate":
        prompts[1]["id"] = prompts[0]["id"]
    elif damage == "gold":
        gold[0]["answer"] = "not the reference"
    elif damage == "leak":
        prompts[0]["solution"] = gold[0]["solution"]
    elif damage == "split":
        prompts[0]["split"] = "test"
    elif damage == "index":
        prompts[0]["source_index"] = -1
    else:
        prompts[30]["level"] = 99
    with pytest.raises(ValueError):
        diagnose.validate_pilot(prompts, gold, sources)
