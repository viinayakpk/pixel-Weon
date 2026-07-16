"""Weon pipeline CLI — constraint-preserving image editing.

Every command here corresponds to something that was actually run and measured. Commands for
designs that were superseded have been archived to legacy/ rather than left broken.

  python run.py test              run the test suite (offline, free)
  python run.py control           resampling control: is the degradation ours or the model's?
                                  (offline, free)
  python run.py task4             regenerate the 5-edit comparison; refuses to overwrite the
                                  shipped canonical evidence unless WEON_OUT_SUFFIX is set
  python run.py evidence          build grid/curve/table from outputs/task4/
  python run.py eval A B x0 y0 x1 y1
                                  honest metrics between two images, measured inside and
                                  outside a DECLARED box (never a post-hoc diff mask)

Credentials live in .env. WEON_DRY_RUN=1 forces offline mode even if a key is present. Dry runs
report N/A rather than fabricating perfect scores.
"""
from __future__ import annotations
import json
import subprocess
import sys

import numpy as np

from pipeline import metrics
from pipeline.metrics import load_rgb


def cmd_test() -> None:
    sys.exit(subprocess.call([
        sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"
    ]))


def cmd_control() -> None:
    from experiments import resample_control
    resample_control.main()


def cmd_task4() -> None:
    from experiments import task4_compare
    task4_compare.main()


def cmd_evidence() -> None:
    from experiments import make_evidence
    make_evidence.main()


def cmd_eval(a_path: str, b_path: str, box: tuple[int, int, int, int] | None) -> None:
    """Compare two images. If a box is declared, report inside vs outside separately.

    The box must be supplied by the caller. The previous version derived the region from the
    difference between the two images, which let the output decide which of its own changes
    counted as 'the edit' — a circular measurement.
    """
    a, b = load_rgb(a_path), load_rgb(b_path)
    out: dict = {"note": "measured over the whole image" if box is None
                 else f"box declared by caller: {box}"}
    if box is None:
        out["whole_image"] = metrics.preservation(a, b, None)
    else:
        m = np.zeros(a.shape[:2], np.uint8)
        x0, y0, x1, y1 = box
        m[y0:y1, x0:x1] = 1
        out["inside_box"] = metrics.preservation(a, b, m)
        out["outside_box"] = metrics.preservation(a, b, 1 - m)
    out["color_delta_e"] = metrics.color_delta_e(a, b)
    print(json.dumps(out, indent=2, default=str))


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "test":
        cmd_test()
    elif cmd == "control":
        cmd_control()
    elif cmd == "task4":
        cmd_task4()
    elif cmd == "evidence":
        cmd_evidence()
    elif cmd == "eval":
        if len(sys.argv) < 4:
            print("usage: run.py eval A B [x0 y0 x1 y1]")
            return
        box = tuple(int(v) for v in sys.argv[4:8]) if len(sys.argv) >= 8 else None
        cmd_eval(sys.argv[2], sys.argv[3], box)  # type: ignore[arg-type]
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
