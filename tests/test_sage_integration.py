"""Static, launcher, and optional real SageMath integration checks."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from integrations.sage_runner import build_sage_command

ROOT = Path(__file__).resolve().parents[1]


def test_sage_crosscheck_script_exists():
    assert (ROOT / "integrations/sage_crosscheck.sage").is_file()


def test_estimator_script_explicit_path_and_parameters():
    source = (ROOT / "integrations/estimate_lwe.sage").read_text()
    assert "--estimator-path" in source
    assert "ND.DiscreteGaussian(args.sigma)" in source
    assert "log2_rop" in source


def test_runner_uses_sage_c_and_preserves_arguments():
    command = build_sage_command("crosscheck", ["--output", "some path.json"])
    assert command[:2] == ["sage", "-c"]
    assert "runpy.run_path" in command[2]
    assert "'some path.json'" in command[2]
    assert "--output" in command[2]


def test_runner_rejects_unrecognized_command():
    with pytest.raises(ValueError, match="unknown SageMath integration"):
        build_sage_command("wrong")


@pytest.mark.skipif(shutil.which("sage") is None, reason="SageMath not installed")
def test_sage_crosscheck(tmp_path):
    target = tmp_path / "crosscheck.json"
    completed = subprocess.run(
        [sys.executable, "-m", "integrations.sage_runner", "crosscheck",
         "--output", str(target)],
        cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(target.read_text())
    assert result["passed"] is True
    assert result["absolute_determinant"] > 0
