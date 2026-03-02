"""
Supervisor — файловые блокировки для синхронизации Git операций.
"""

import os
import time
import pathlib
import logging
from typing import Optional

log = logging.getLogger(__name__)

DRIVE_ROOT: Optional[pathlib.Path] = None


def init(drive_root: pathlib.Path) -> None:
    global DRIVE_ROOT
    DRIVE_ROOT = drive_root


def acquire_git_lock(timeout_sec: int = 120) -> pathlib.Path:
    """Получить эксклюзивную блокировку для Git операций."""
    if DRIVE_ROOT is None:
        raise RuntimeError("lock module not initialized")
    lock_dir = DRIVE_ROOT / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "git.lock"
    stale_sec = 600
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if lock_path.exists():
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > stale_sec:
                    lock_path.unlink()
                    continue
            except (FileNotFoundError, OSError):
                pass
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(fd, f"locked_at={time.time()}\n".encode("utf-8"))
            finally:
                os.close(fd)
            return lock_path
        except FileExistsError:
            time.sleep(0.5)
    raise TimeoutError(f"Git lock not acquired within {timeout_sec}s: {lock_path}")


def release_git_lock(lock_path: pathlib.Path) -> None:
    """Освободить блокировку."""
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
