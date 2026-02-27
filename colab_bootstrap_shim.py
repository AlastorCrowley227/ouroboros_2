"""Bootstrap shim for local execution.

This helper prepares environment variables and launches `colab_launcher.py`
(which now supports local-only runtime).
"""

import os
import pathlib
import subprocess
import sys
from typing import Optional


def get_secret(name: str, required: bool = False) -> Optional[str]:
    v = os.environ.get(name)
    if required:
        assert v is not None and str(v).strip() != "", f"Missing required secret: {name}"
    return v


def export_secret_to_env(name: str, required: bool = False) -> Optional[str]:
    val = get_secret(name, required=required)
    if val is not None and str(val).strip() != "":
        os.environ[name] = str(val)
    return val


for _name in ("OPENROUTER_API_KEY", "TELEGRAM_BOT_TOKEN", "TOTAL_BUDGET", "GITHUB_TOKEN", "GITHUB_USER", "GITHUB_REPO"):
    export_secret_to_env(_name, required=True)

for _name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    export_secret_to_env(_name, required=False)

os.environ.setdefault("OUROBOROS_WORKER_START_METHOD", "fork")
os.environ.setdefault("OUROBOROS_DIAG_HEARTBEAT_SEC", "30")
os.environ.setdefault("OUROBOROS_DIAG_SLOW_CYCLE_SEC", "20")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OUROBOROS_HOME", str(pathlib.Path.home() / ".ouroboros"))
os.environ.setdefault("OUROBOROS_REPO_DIR", str(pathlib.Path(__file__).resolve().parent))

repo_dir = pathlib.Path(os.environ["OUROBOROS_REPO_DIR"]).expanduser().resolve()
launcher_path = repo_dir / "colab_launcher.py"
assert launcher_path.exists(), f"Missing launcher: {launcher_path}"
subprocess.run([sys.executable, str(launcher_path)], cwd=str(repo_dir), check=True)
