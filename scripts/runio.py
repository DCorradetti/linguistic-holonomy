"""
runio.py -- immutable run directories and manifests.
=======================================================================
Article  : Article_LLW
Plan ref : Plan_v1.md, Reproducibility Plan; CYCLE.md Sec. 8
Version  : v1
Purpose  : Every script that supports a claim in the manuscript writes a
           fresh timestamped directory under `7. Results/Article_LLW/`
           containing output.json, log.txt and manifest.json.  Nothing
           under `7. Results/` is ever edited by hand afterwards.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Public-repository layout.  The research workspace used numbered
# directories; this release keeps the scripts in ``scripts/`` and writes
# fresh, immutable runs below ``results/runs/``.
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "runs"


def _pkg_versions() -> dict:
    out = {}
    for name in ("numpy", "scipy", "torch", "transformers"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = "absent"
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Run:
    """A single immutable run directory."""

    def __init__(self, script: str, version: str = "v1"):
        stamp = time.strftime("%Y%m%dT%H%M%S")
        self.dir = RESULTS / f"run_{script}_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.script = script
        self.version = version
        self.t0 = time.time()
        self._log = open(self.dir / "log.txt", "w", encoding="utf-8")
        self.log(f"run {script} {version} started {stamp}")

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def write_json(self, name: str, obj: Any) -> Path:
        path = self.dir / name
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False, default=float)
        return path

    def finish(self, conclusions: dict, limitations: list[str],
               inputs: dict | None = None, command: str | None = None) -> None:
        outputs = {p.name: file_sha256(p) for p in sorted(self.dir.glob("*.json"))}
        manifest = {
            "script": self.script,
            "script_version": self.version,
            "script_sha256": file_sha256(Path(__file__).parent / f"{self.script}.py")
            if (Path(__file__).parent / f"{self.script}.py").exists() else None,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_seconds": round(time.time() - self.t0, 1),
            "command": command or " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "packages": _pkg_versions(),
            },
            "inputs": inputs or {},
            "output_sha256": outputs,
            "conclusions": conclusions,
            "limitations": limitations,
        }
        with open(self.dir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False, default=float)
        self.log(f"run complete -> {self.dir.name}")
        self._log.close()
