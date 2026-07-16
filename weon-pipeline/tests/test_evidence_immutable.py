"""Canonical paid evidence must be immutable in normal operation.

This test exists because the guard silently failed once. A patch didn't match, the guard
disappeared, and `python -m experiments.task4_compare` — with a live key — started re-running the
paid chain and overwriting the artifacts every number in the report cites. It was killed partway
through; metrics.json survived only because it is written last.

These tests assert the refusal by executing the modules in a subprocess, so a guard that stops
matching fails here instead of failing during a rerun.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(module: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "WEON_DRY_RUN": "1"}      # never let a test touch a provider
    env.pop("WEON_OUT_SUFFIX", None)
    env.pop("WEON_RUN_ID", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-m", module], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "outputs/task4/metrics.json")),
                    reason="no canonical task4 evidence present")
def test_task4_refuses_to_overwrite_canonical_evidence():
    r = _run("experiments.task4_compare")
    assert r.returncode == 2, f"expected refusal, got {r.returncode}:\n{r.stdout}\n{r.stderr}"
    assert "REFUSING TO RUN" in r.stdout


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "outputs/task1/generate.json")),
                    reason="no canonical task1 evidence present")
def test_task1_refuses_to_overwrite_canonical_evidence():
    r = _run("experiments.task1_generate")
    assert r.returncode == 2, f"expected refusal, got {r.returncode}:\n{r.stdout}\n{r.stderr}"
    assert "REFUSING TO RUN" in r.stdout


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "outputs/task4/metrics.json")),
                    reason="no canonical task4 evidence present")
def test_task4_metrics_json_is_byte_identical_after_a_refused_run():
    """The refusal must happen before anything is written."""
    p = os.path.join(ROOT, "outputs/task4/metrics.json")
    before = open(p, "rb").read()
    _run("experiments.task4_compare")
    assert open(p, "rb").read() == before, "a refused run modified canonical evidence"


def test_suffix_escape_hatch_is_honoured():
    """An explicit suffix must route elsewhere rather than refuse — otherwise the guard is a
    dead end and someone will delete it to get work done.

    Uses tempfile.mkdtemp rather than pytest's `tmp_path` fixture: on this machine `tmp_path`
    raises PermissionError because pytest's temp root is a locked leftover it can neither read
    nor delete. That is an environment fault, not a code fault, but the suite must stay green on
    a reviewer's machine too — so the test owns its temp dir instead of depending on pytest's.
    """
    tmp = tempfile.mkdtemp(prefix="weon-guardtest-")
    try:
        r = _run("experiments.task4_compare", {
            "WEON_OUT_SUFFIX": "guardtest",
            "WEON_OUTPUT_ROOT": tmp,
        })
        assert "REFUSING TO RUN" not in r.stdout
        # dry-run reports N/A rather than fabricating scores; it must not claim a cost either
        assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
