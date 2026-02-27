"""Bootstrap shim for local execution.

This helper loads runtime config and launches `launcher.py`.
"""

import pathlib
import subprocess
import sys

from ouroboros.config import load_runtime_config


cfg = load_runtime_config()
cfg.export_env()

repo_dir = pathlib.Path(cfg.ouroboros_repo_dir).expanduser().resolve()
launcher_path = repo_dir / "launcher.py"
assert launcher_path.exists(), f"Missing launcher: {launcher_path}"
subprocess.run([sys.executable, str(launcher_path)], cwd=str(repo_dir), check=True)
