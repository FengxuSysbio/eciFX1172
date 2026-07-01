from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT = r"""
import sys
import numpy as np
np.object = object
from memote.suite.cli.reports import report
report.main(args=sys.argv[1:], prog_name="memote report", standalone_mode=True)
"""


def run_memote(args, cwd: Path, log_path: Path) -> int:
    cmd = [sys.executable, "-c", SCRIPT, *args]
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    log_path.write_text(
        "COMMAND: " + " ".join(cmd) + "\n\nSTDOUT:\n" + proc.stdout + "\n\nSTDERR:\n" + proc.stderr,
        encoding="utf-8",
    )
    return proc.returncode


def main():
    root = Path.cwd()
    project = next(p for p in root.iterdir() if p.is_dir() and (p / "results" / "ec_iFX1172_final_calibrated").exists())
    base = project / "results" / "ec_iFX1172_final_calibrated"
    out = base / "memote_comparison"
    out.mkdir(parents=True, exist_ok=True)

    original = root / "iFX1172.xml"
    ecmodel = base / "formats" / "eciFX1172.xml"
    tmp = Path.home() / "memote_ec_compare_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    tmp_original = tmp / "iFX1172.xml"
    tmp_ecmodel = tmp / "eciFX1172.xml"
    shutil.copy2(original, tmp_original)
    shutil.copy2(ecmodel, tmp_ecmodel)
    jobs = [
        (
            "iFX1172",
            ["snapshot", "--skip", "test_consistency", "--filename", str(tmp / "iFX1172_memote_skip_consistency.html"), str(tmp_original)],
        ),
        (
            "eciFX1172",
            ["snapshot", "--skip", "test_consistency", "--filename", str(tmp / "eciFX1172_memote_skip_consistency.html"), str(tmp_ecmodel)],
        ),
        (
            "diff",
            ["diff", "--skip", "test_consistency", "--filename", str(tmp / "iFX1172_vs_eciFX1172_memote_diff_skip_consistency.html"), str(tmp_original), str(tmp_ecmodel)],
        ),
    ]
    results = {}
    for name, args in jobs:
        rc = run_memote(args, root, out / f"{name}_memote_run.log")
        results[name] = rc
        print(name, rc)
    for html in tmp.glob("*.html"):
        shutil.copy2(html, out / html.name)
    (out / "run_status.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
