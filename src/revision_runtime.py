"""Run a project script using an available Python and the recorded dependencies.

The local venv executable may refer to an unavailable base Python. This launcher
does not modify that environment: it exposes its installed packages to a usable
Python 3.12 runtime and runs the requested project script normally.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")
packages = ROOT / ".venv" / "Lib" / "site-packages"
if packages.is_dir():
    sys.path.insert(0, str(packages))
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python revision_runtime.py SCRIPT [ARGUMENTS]")
    target = Path(sys.argv[1]).resolve()
    if not target.is_file() or ROOT not in target.parents:
        raise SystemExit("The target must be an existing script in this project.")
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")
