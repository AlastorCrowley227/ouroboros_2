import json

import pytest

from ouroboros.config import load_runtime_config


def _write_cfg(tmp_path, data):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_load_runtime_config_git_mode_allows_empty_github_fields(tmp_path):
    cfg = _write_cfg(tmp_path, {
        "telegram_bot_token": "tg",
        "vcs_platform": "git",
        "total_budget": 1,
    })
    rc = load_runtime_config(str(cfg))
    assert rc.vcs_platform == "git"
    assert rc.github_token == ""


def test_load_runtime_config_gitea_requires_endpoint(tmp_path):
    cfg = _write_cfg(tmp_path, {
        "telegram_bot_token": "tg",
        "vcs_platform": "gitea",
        "github_user": "u",
        "github_repo": "r",
        "total_budget": 1,
    })
    with pytest.raises(AssertionError):
        load_runtime_config(str(cfg))
