"""
Supervisor — State management.

Persistent local state: load, save, atomic writes, file locks.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import pathlib
import time
import uuid
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level config (set via init())
# ---------------------------------------------------------------------------
DRIVE_ROOT: pathlib.Path = pathlib.Path.home() / ".ouroboros"
STATE_PATH: pathlib.Path = DRIVE_ROOT / "state" / "state.json"
STATE_LAST_GOOD_PATH: pathlib.Path = DRIVE_ROOT / "state" / "state.last_good.json"
STATE_LOCK_PATH: pathlib.Path = DRIVE_ROOT / "locks" / "state.lock"
QUEUE_SNAPSHOT_PATH: pathlib.Path = DRIVE_ROOT / "state" / "queue_snapshot.json"


def init(drive_root: pathlib.Path, total_budget_limit: float = 0.0) -> None:
    global DRIVE_ROOT, STATE_PATH, STATE_LAST_GOOD_PATH, STATE_LOCK_PATH, QUEUE_SNAPSHOT_PATH
    DRIVE_ROOT = drive_root
    STATE_PATH = drive_root / "state" / "state.json"
    STATE_LAST_GOOD_PATH = drive_root / "state" / "state.last_good.json"
    STATE_LOCK_PATH = drive_root / "locks" / "state.lock"
    QUEUE_SNAPSHOT_PATH = drive_root / "state" / "queue_snapshot.json"
    set_budget_limit(total_budget_limit)


# ---------------------------------------------------------------------------
# Atomic file operations
# ---------------------------------------------------------------------------

def atomic_write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        data = content.encode("utf-8")
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(str(tmp), str(path))


def json_load_file(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        log.debug(f"Failed to load JSON from {path}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# File locks
# ---------------------------------------------------------------------------

def acquire_file_lock(lock_path: pathlib.Path, timeout_sec: float = 4.0,
                      stale_sec: float = 90.0) -> Optional[int]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    while (time.time() - started) < timeout_sec:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(fd, f"pid={os.getpid()} ts={datetime.datetime.now(datetime.timezone.utc).isoformat()}\n".encode("utf-8"))
            except Exception:
                log.debug(f"Failed to write lock metadata to {lock_path}", exc_info=True)
                pass
            return fd
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > stale_sec:
                    lock_path.unlink()
                    continue
            except Exception:
                log.debug(f"Failed to check/remove stale lock at {lock_path}", exc_info=True)
                pass
            time.sleep(0.05)
        except Exception:
            log.warning(f"Failed to acquire lock at {lock_path}", exc_info=True)
            break
    return None


def release_file_lock(lock_path: pathlib.Path, lock_fd: Optional[int]) -> None:
    if lock_fd is None:
        return
    try:
        os.close(lock_fd)
    except Exception:
        log.debug(f"Failed to close lock fd {lock_fd} for {lock_path}", exc_info=True)
        pass
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        log.debug(f"Failed to unlink lock file {lock_path}", exc_info=True)
        pass


# Re-export append_jsonl from ouroboros.utils (single source of truth)
from ouroboros.utils import append_jsonl  # noqa: F401


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

def ensure_state_defaults(st: Dict[str, Any]) -> Dict[str, Any]:
    st.setdefault("created_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
    st.setdefault("owner_id", None)
    st.setdefault("owner_chat_id", None)
    st.setdefault("tg_offset", 0)
    st.setdefault("session_id", uuid.uuid4().hex)
    st.setdefault("current_branch", None)
    st.setdefault("current_sha", None)
    st.setdefault("last_owner_message_at", "")
    st.setdefault("last_evolution_task_at", "")
    st.setdefault("evolution_mode_enabled", False)
    st.setdefault("evolution_cycle", 0)
    st.setdefault("evolution_consecutive_failures", 0)
    for legacy_key in ("approvals", "idle_cursor", "idle_stats", "last_idle_task_at",
                        "last_auto_review_at", "last_review_task_id", "session_daily_snapshot"):
        st.pop(legacy_key, None)
    return st


def default_state_dict() -> Dict[str, Any]:
    """Create a fresh state dict. Single source of truth: ensure_state_defaults."""
    return ensure_state_defaults({})


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def _load_state_unlocked() -> Dict[str, Any]:
    """Load state without acquiring lock. Caller must hold STATE_LOCK."""
    recovered = False
    st_obj = json_load_file(STATE_PATH)
    if st_obj is None:
        st_obj = json_load_file(STATE_LAST_GOOD_PATH)
        recovered = st_obj is not None

    if st_obj is None:
        st = ensure_state_defaults(default_state_dict())
        _save_state_unlocked(st)
        return st

    st = ensure_state_defaults(st_obj)
    if recovered:
        _save_state_unlocked(st)
    return st


def _save_state_unlocked(st: Dict[str, Any]) -> None:
    """Save state without acquiring lock. Caller must hold STATE_LOCK."""
    st = ensure_state_defaults(st)
    payload = json.dumps(st, ensure_ascii=False, indent=2)
    atomic_write_text(STATE_PATH, payload)
    atomic_write_text(STATE_LAST_GOOD_PATH, payload)


def load_state() -> Dict[str, Any]:
    lock_fd = acquire_file_lock(STATE_LOCK_PATH)
    try:
        return _load_state_unlocked()
    finally:
        release_file_lock(STATE_LOCK_PATH, lock_fd)


def save_state(st: Dict[str, Any]) -> None:
    lock_fd = acquire_file_lock(STATE_LOCK_PATH)
    try:
        _save_state_unlocked(st)
    finally:
        release_file_lock(STATE_LOCK_PATH, lock_fd)


def init_state() -> Dict[str, Any]:
    """Initialize state at session start."""
    lock_fd = acquire_file_lock(STATE_LOCK_PATH)
    try:
        st = _load_state_unlocked()
        _save_state_unlocked(st)
        return st
    finally:
        release_file_lock(STATE_LOCK_PATH, lock_fd)

# ---------------------------------------------------------------------------
# Usage tracking (local-only; budget disabled)
# ---------------------------------------------------------------------------

TOTAL_BUDGET_LIMIT: float = 0.0
EVOLUTION_BUDGET_RESERVE: float = 0.0


def set_budget_limit(limit: float) -> None:
    _ = limit


def budget_remaining(st: Dict[str, Any]) -> float:
    _ = st
    return float("inf")


def budget_pct(st: Dict[str, Any]) -> float:
    _ = st
    return 0.0


def update_budget_from_usage(usage: Dict[str, Any]) -> None:
    _ = usage


# ---------------------------------------------------------------------------
# Budget breakdown by category
# ---------------------------------------------------------------------------

def budget_breakdown(st: Dict[str, Any]) -> Dict[str, float]:
    _ = st
    return {}


def model_breakdown(st: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Calculate budget breakdown by model from events.jsonl.

    Returns dict like:
    {
        "qwen2.5:14b": {"calls": 120, "prompt_tokens": 50000, "completion_tokens": 3000},
    }
    """
    events_path = DRIVE_ROOT / "logs" / "events.jsonl"
    if not events_path.exists():
        return {}

    breakdown: Dict[str, Dict[str, float]] = {}
    try:
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if event.get("type") != "llm_usage":
                        continue

                    model = event.get("model") or "unknown"
                    if not model:
                        model = "unknown"

                    # Get cost
                    cost = 0.0
                    if "cost" in event:
                        cost = float(event.get("cost", 0))
                    elif "usage" in event and isinstance(event["usage"], dict):
                        cost = float(event["usage"].get("cost", 0))

                    # Get tokens
                    prompt_tokens = int(event.get("prompt_tokens", 0) or 0)
                    completion_tokens = int(event.get("completion_tokens", 0) or 0)
                    cached_tokens = int(event.get("cached_tokens", 0) or 0)

                    if model not in breakdown:
                        breakdown[model] = {"cost": 0.0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}

                    breakdown[model]["cost"] += cost
                    breakdown[model]["calls"] += 1
                    breakdown[model]["prompt_tokens"] += prompt_tokens
                    breakdown[model]["completion_tokens"] += completion_tokens
                    breakdown[model]["cached_tokens"] += cached_tokens

                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
    except Exception:
        log.warning("Failed to calculate model breakdown", exc_info=True)

    return breakdown


def per_task_cost_summary(max_tasks: int = 10, tail_bytes: int = 512_000) -> List[Dict[str, Any]]:
    _ = max_tasks
    _ = tail_bytes
    return []

# ---------------------------------------------------------------------------
# Status text (moved from workers.py)
# ---------------------------------------------------------------------------

def status_text(workers_dict: Dict[int, Any], pending_list: list, running_dict: Dict[str, Dict[str, Any]],
                soft_timeout_sec: int, hard_timeout_sec: int) -> str:
    """Build status text from worker and queue state."""
    st = load_state()
    now = time.time()
    lines = []
    lines.append(f"owner_id: {st.get('owner_id')}")
    lines.append(f"session_id: {st.get('session_id')}")
    lines.append(f"version: {st.get('current_branch')}@{(st.get('current_sha') or '')[:8]}")
    busy_count = sum(1 for w in workers_dict.values() if getattr(w, 'busy_task_id', None) is not None)
    lines.append(f"workers: {len(workers_dict)} (busy: {busy_count})")
    lines.append(f"pending: {len(pending_list)}")
    lines.append(f"running: {len(running_dict)}")
    if pending_list:
        preview = []
        for t in pending_list[:10]:
            preview.append(
                f"{t.get('id')}:{t.get('type')}:pr{t.get('priority')}:a{int(t.get('_attempt') or 1)}")
        lines.append("pending_queue: " + ", ".join(preview))
    if running_dict:
        lines.append("running_ids: " + ", ".join(list(running_dict.keys())[:10]))
    busy = [f"{getattr(w, 'wid', '?')}:{getattr(w, 'busy_task_id', '?')}"
            for w in workers_dict.values() if getattr(w, 'busy_task_id', None)]
    if busy:
        lines.append("busy: " + ", ".join(busy))
    if running_dict:
        details = []
        for task_id, meta in list(running_dict.items())[:10]:
            task = meta.get("task") if isinstance(meta, dict) else {}
            started = float(meta.get("started_at") or 0.0) if isinstance(meta, dict) else 0.0
            hb = float(meta.get("last_heartbeat_at") or 0.0) if isinstance(meta, dict) else 0.0
            runtime_sec = int(max(0.0, now - started)) if started > 0 else 0
            hb_lag_sec = int(max(0.0, now - hb)) if hb > 0 else -1
            details.append(
                f"{task_id}:type={task.get('type')} pr={task.get('priority')} "
                f"attempt={meta.get('attempt')} runtime={runtime_sec}s hb_lag={hb_lag_sec}s")
        if details:
            lines.append("running_details:")
            lines.extend([f"  - {d}" for d in details])
    if running_dict and busy_count == 0:
        lines.append("queue_warning: running>0 while busy=0")
    # Model/token breakdown
    models = model_breakdown(st)
    if models:
        sorted_models = sorted(models.items(), key=lambda x: x[1]["calls"], reverse=True)
        lines.append("model_breakdown:")
        for model_name, stats in sorted_models:
            if stats["calls"] > 0:
                calls = int(stats["calls"])
                pt = int(stats["prompt_tokens"])
                ct = int(stats["completion_tokens"])
                lines.append(f"  {model_name}: {calls} calls, {pt:,}p/{ct:,}c tok")

    lines.append(
        "evolution: "
        + f"enabled={int(bool(st.get('evolution_mode_enabled')))}, "
        + f"cycle={int(st.get('evolution_cycle') or 0)}")
    lines.append(f"last_owner_message_at: {st.get('last_owner_message_at') or '-'}")
    lines.append(f"timeouts: soft={soft_timeout_sec}s, hard={hard_timeout_sec}s")
    return "\n".join(lines)


def rotate_chat_log_if_needed(drive_root: pathlib.Path, max_bytes: int = 800_000) -> None:
    """Rotate chat log if it exceeds max_bytes."""
    chat = drive_root / "logs" / "chat.jsonl"
    if not chat.exists():
        return
    if chat.stat().st_size < max_bytes:
        return
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_path = drive_root / "archive" / f"chat_{ts}.jsonl"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(chat.read_bytes())
    chat.write_text("", encoding="utf-8")
