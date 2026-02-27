import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG_PATH = Path("ouroboros.config.json")


@dataclass(frozen=True)
class RuntimeConfig:
    telegram_bot_token: str
    vcs_platform: str = "github"
    github_token: str = ""
    github_user: str = ""
    github_repo: str = ""
    gitea_base_url: str = ""
    git_remote_url: str = ""
    anthropic_api_key: str = ""
    total_budget: float = 0.0
    ouroboros_home: str = str(Path.home() / ".ouroboros")
    ouroboros_repo_dir: str = ""
    max_workers: int = 5
    model: str = "qwen2.5:14b"
    model_code: str = "qwen2.5:14b"
    model_light: str = "google/gemini-3-pro-preview"
    max_rounds: int = 200
    soft_timeout_sec: int = 600
    hard_timeout_sec: int = 1800
    diag_heartbeat_sec: int = 30
    diag_slow_cycle_sec: int = 20
    worker_start_method: str = "fork"

    def export_env(self) -> None:
        os.environ["TELEGRAM_BOT_TOKEN"] = self.telegram_bot_token
        os.environ["OUROBOROS_VCS_PLATFORM"] = self.vcs_platform
        os.environ["GITHUB_TOKEN"] = self.github_token
        os.environ["GITHUB_USER"] = self.github_user
        os.environ["GITHUB_REPO"] = self.github_repo
        if self.gitea_base_url:
            os.environ["GITEA_BASE_URL"] = self.gitea_base_url
        if self.git_remote_url:
            os.environ["GIT_REMOTE_URL"] = self.git_remote_url
        os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key
        os.environ["TOTAL_BUDGET"] = str(self.total_budget)
        os.environ["OUROBOROS_HOME"] = self.ouroboros_home
        os.environ["OUROBOROS_REPO_DIR"] = self.ouroboros_repo_dir
        os.environ["OUROBOROS_MAX_WORKERS"] = str(self.max_workers)
        os.environ["OUROBOROS_MODEL"] = self.model
        os.environ["OUROBOROS_MODEL_CODE"] = self.model_code
        os.environ["OUROBOROS_MODEL_LIGHT"] = self.model_light
        os.environ["OUROBOROS_MAX_ROUNDS"] = str(self.max_rounds)
        os.environ["OUROBOROS_SOFT_TIMEOUT_SEC"] = str(self.soft_timeout_sec)
        os.environ["OUROBOROS_HARD_TIMEOUT_SEC"] = str(self.hard_timeout_sec)
        os.environ["OUROBOROS_DIAG_HEARTBEAT_SEC"] = str(self.diag_heartbeat_sec)
        os.environ["OUROBOROS_DIAG_SLOW_CYCLE_SEC"] = str(self.diag_slow_cycle_sec)
        os.environ["OUROBOROS_WORKER_START_METHOD"] = self.worker_start_method


def _required(data: Dict[str, Any], key: str) -> str:
    val = str(data.get(key, "")).strip()
    if not val:
        raise AssertionError(f"Missing required config key: {key}")
    return val


def _int(data: Dict[str, Any], key: str, default: int, minimum: int = 0) -> int:
    try:
        val = int(data.get(key, default))
    except Exception:
        val = default
    return max(minimum, val)


def _float(data: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(data.get(key, default))
    except Exception:
        return default


def load_runtime_config(path: Optional[str] = None) -> RuntimeConfig:
    cfg_path = Path(path or os.environ.get("OUROBOROS_CONFIG", str(DEFAULT_CONFIG_PATH))).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    data = json.loads(cfg_path.read_text(encoding="utf-8"))

    repo_dir = str(Path(data.get("ouroboros_repo_dir") or Path.cwd()).expanduser().resolve())
    home_dir = str(Path(data.get("ouroboros_home") or (Path.home() / ".ouroboros")).expanduser().resolve())

    vcs_platform = str(data.get("vcs_platform", "github")).strip().lower() or "github"
    if vcs_platform not in {"github", "gitea", "git"}:
        raise AssertionError("vcs_platform must be one of: github, gitea, git")

    github_token = str(data.get("github_token", "")).strip()
    github_user = str(data.get("github_user", "")).strip()
    github_repo = str(data.get("github_repo", "")).strip()
    gitea_base_url = str(data.get("gitea_base_url", "")).strip()
    git_remote_url = str(data.get("git_remote_url", "")).strip()

    if vcs_platform == "github":
        if not github_token:
            github_token = _required(data, "github_token")
        github_user = github_user or _required(data, "github_user")
        github_repo = github_repo or _required(data, "github_repo")
    elif vcs_platform == "gitea":
        github_user = github_user or _required(data, "github_user")
        github_repo = github_repo or _required(data, "github_repo")
        if not (gitea_base_url or git_remote_url):
            raise AssertionError("For vcs_platform=gitea, set gitea_base_url or git_remote_url")

    return RuntimeConfig(
        telegram_bot_token=_required(data, "telegram_bot_token"),
        vcs_platform=vcs_platform,
        github_token=github_token,
        github_user=github_user,
        github_repo=github_repo,
        gitea_base_url=gitea_base_url,
        git_remote_url=git_remote_url,
        anthropic_api_key=str(data.get("anthropic_api_key", "")),
        total_budget=_float(data, "total_budget", 0.0),
        ouroboros_home=home_dir,
        ouroboros_repo_dir=repo_dir,
        max_workers=_int(data, "max_workers", 5, minimum=1),
        model=str(data.get("model", "qwen2.5:14b")),
        model_code=str(data.get("model_code", "qwen2.5:14b")),
        model_light=str(data.get("model_light", "google/gemini-3-pro-preview")),
        max_rounds=_int(data, "max_rounds", 200, minimum=1),
        soft_timeout_sec=_int(data, "soft_timeout_sec", 600, minimum=60),
        hard_timeout_sec=_int(data, "hard_timeout_sec", 1800, minimum=120),
        diag_heartbeat_sec=_int(data, "diag_heartbeat_sec", 30, minimum=0),
        diag_slow_cycle_sec=_int(data, "diag_slow_cycle_sec", 20, minimum=0),
        worker_start_method=str(data.get("worker_start_method", "fork")).strip() or "fork",
    )
