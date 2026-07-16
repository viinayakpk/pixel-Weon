"""Build and verify the curated interview submission ZIP.

The working repository intentionally contains exploratory artifacts. This release uses an explicit
allow-list so stale experiments, logs, caches and credentials cannot enter the submission.
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
REPO = PROJECT / "weon-pipeline"
DEST = PROJECT / "weon-submission.zip"
TEMP = PROJECT / "weon-submission.zip.tmp"


def add(matches: set[Path], *patterns: str) -> None:
    for pattern in patterns:
        found = list(REPO.glob(pattern))
        if not found:
            raise FileNotFoundError(f"release pattern matched nothing: {pattern}")
        matches.update(p for p in found if p.is_file())


selected: set[Path] = {PROJECT / "REPORT.md"}
add(
    selected,
    ".env.example",
    ".gitignore",
    "README.md",
    "requirements.txt",
    "requirements.pinned.txt",
    "requirements-paid.txt",
    "run.py",
    "cases/*.py",
    "pipeline/*.py",
    "experiments/*.py",
    "tests/*.py",
    "test_data/*.jpg",
    "outputs/eval_receipt.json",
    "outputs/actual_cost.json",
    "outputs/human_eval/Q*.png",
    "outputs/human_eval/_aggregate.json",
    "outputs/human_eval/_distorted_reference.png",
    "outputs/human_eval/alice.json",
)

# Task 1: paid sources, local conditions, inventory, retained probes and author sanity check.
add(
    selected,
    "outputs/task1/FINDINGS.md",
    "outputs/task1/generate.json",
    "outputs/task1/p1_plain.png",
    "outputs/task1/p2_constrained.png",
    "outputs/task1/logo_reference.png",
    "outputs/task1/logo_zoom.png",
    "outputs/task1/p2_target_grid.png",
    "outputs/task1/asset_arigato.png",
    "outputs/task1/cond_A_model_only.png",
    "outputs/task1/cond_A_model_only_rectified.png",
    "outputs/task1/cond_B0_cleared_substrate.png",
    "outputs/task1/cond_B_hard_alpha_graft.png",
    "outputs/task1/cond_B_hard_alpha_graft_rectified.png",
    "outputs/task1/cond_C_linear_relit_graft.png",
    "outputs/task1/cond_C_linear_relit_graft_rectified.png",
    "outputs/task1/task1_compare.json",
    "outputs/task1/rectified_compare.png",
    "outputs/task1/identity_manifest.json",
    "outputs/task1/identity_manifest_grid.png",
    "outputs/task1/manifest_*.png",
    "outputs/task1/iou_reference_sensitivity.json",
    "outputs/task1/material_probe.json",
    "outputs/task1/material_grid.png",
    "outputs/task1/relief_probe.json",
    "outputs/task1/relief_compare.png",
    "outputs/task1/judge_calibration.json",
    "outputs/task1/prompt_ceiling.json",
    "outputs/task1/ceiling_s2_attributes.png",
    "outputs/task1/ceiling_s3_negatives.png",
)

# Task 4 main run and derived evidence.
add(
    selected,
    "outputs/task4/FINDINGS.md",
    "outputs/task4/THREE_ZONE_FINDINGS.md",
    "outputs/task4/step0_original.png",
    "outputs/task4/label_reference.png",
    "outputs/task4/intent_check.png",
    "outputs/task4/naive_step*.png",
    "outputs/task4/ledger_step*.png",
    "outputs/task4/metrics.json",
    "outputs/task4/common_union_recompute.json",
    "outputs/task4/resample_control.json",
    "outputs/task4/three_zone_ablation.json",
    "outputs/task4/zone_compare.png",
    "outputs/task4/turn4_artifact.png",
    "outputs/task4/task4_grid.png",
    "outputs/task4/task4_label_grid.png",
    "outputs/task4/task4_curve.png",
    "outputs/task4/task4_table.md",
    "outputs/task4/model_comparison_4x.json",
    "outputs/task4/model_comparison_4x.png",
)

# Cross-editor source frames. Logs are deliberately not selected.
for run_dir in ("task4_nano", "task4_g31", "task4_gpt54"):
    add(selected, f"outputs/{run_dir}/metrics.json", f"outputs/{run_dir}/*.png")


def archive_name(path: Path) -> str:
    if path == PROJECT / "REPORT.md":
        return "REPORT.md"
    return "weon-pipeline/" + path.relative_to(REPO).as_posix()


files = sorted(((archive_name(p), p) for p in selected), key=lambda x: x[0])
names = {name for name, _ in files}

forbidden = ("/.git/", "__pycache__", ".pytest_cache", "_archive", "guardtest")
for name, path in files:
    probe = "/" + name.lower()
    if any(token in probe for token in forbidden):
        raise RuntimeError(f"forbidden release path: {name}")
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        raise RuntimeError(f"forbidden release credential file: {name}")
    if path.suffix.lower() in {".py", ".md", ".txt", ".json", ".example", ".gitignore"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"sk-or-v1-[A-Za-z0-9_-]{20,}", text):
            raise RuntimeError(f"possible OpenRouter secret in {name}")
        for match in re.finditer(
            r"(?im)^\s*(OPENROUTER_API_KEY|FAL_KEY|FAL_API_KEY)\s*=\s*([^\s#]+)", text
        ):
            if match.group(2).strip() not in {"", "your-key-here", "YOUR_KEY_HERE"}:
                raise RuntimeError(f"possible credential assignment in {name}")

# Every local report link must be shipped.
report = (PROJECT / "REPORT.md").read_text(encoding="utf-8")
for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", report):
    if re.match(r"https?://", target):
        continue
    clean = target.split("#", 1)[0]
    if clean not in names:
        raise RuntimeError(f"report link is absent from archive: {clean}")

manifest_lines = []
for name, path in files:
    manifest_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}")
manifest = "# SHA-256 for every shipped source/evidence file (excluding this manifest)\n" + "\n".join(manifest_lines) + "\n"

if TEMP.exists():
    TEMP.unlink()
with zipfile.ZipFile(TEMP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for name, path in files:
        zf.write(path, name)
    zf.writestr("MANIFEST.sha256", manifest)

with zipfile.ZipFile(TEMP, "r") as zf:
    bad = zf.testzip()
    if bad:
        raise RuntimeError(f"ZIP CRC failure: {bad}")
    archived = set(zf.namelist())
    if archived != names | {"MANIFEST.sha256"}:
        raise RuntimeError("ZIP contents differ from release allow-list")

TEMP.replace(DEST)
print(f"wrote {DEST}")
print(f"files: {len(files) + 1}")
print(f"size: {DEST.stat().st_size / 1024 / 1024:.1f} MiB")
