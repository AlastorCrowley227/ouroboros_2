"""Issue tools for GitHub/Gitea, with graceful fallback for plain git."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

import requests

from ouroboros.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)


def _vcs_platform() -> str:
    return str(os.environ.get("OUROBOROS_VCS_PLATFORM", "github")).strip().lower() or "github"


def _gh_cmd(args: List[str], ctx: ToolContext, timeout: int = 30, input_data: Optional[str] = None) -> str:
    cmd = ["gh"] + args
    try:
        res = subprocess.run(
            cmd,
            cwd=str(ctx.repo_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            input=input_data,
        )
        if res.returncode != 0:
            err = (res.stderr or "").strip()
            return f"⚠️ GH_ERROR: {err.split(chr(10))[0][:200]}"
        return res.stdout.strip()
    except FileNotFoundError:
        return "⚠️ GH_ERROR: `gh` CLI not found."
    except subprocess.TimeoutExpired:
        return f"⚠️ GH_TIMEOUT: exceeded {timeout}s."
    except Exception as e:
        return f"⚠️ GH_ERROR: {e}"


def _repo_owner_name() -> tuple[str, str]:
    return os.environ.get("GITHUB_USER", "").strip(), os.environ.get("GITHUB_REPO", "").strip()


def _gitea_base_url() -> str:
    return str(os.environ.get("GITEA_BASE_URL", "")).strip().rstrip("/")


def _gitea_headers() -> Dict[str, str]:
    token = str(os.environ.get("GITHUB_TOKEN", "")).strip()
    h = {"Accept": "application/json"}
    if token:
        h["Authorization"] = f"token {token}"
    return h


def _gitea_request(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> tuple[int, Any]:
    base = _gitea_base_url()
    if not base:
        return 0, "GITEA_BASE_URL is not set"
    url = f"{base}/api/v1{path}"
    try:
        r = requests.request(method, url, headers=_gitea_headers(), json=payload, timeout=30)
    except Exception as e:
        return 0, str(e)
    try:
        body = r.json()
    except Exception:
        body = r.text
    return r.status_code, body


def _format_issue_list(issues: List[Dict[str, Any]], state: str) -> str:
    if not issues:
        return f"No {state} issues found."
    lines = [f"**{len(issues)} {state} issue(s):**\n"]
    for issue in issues:
        labels = issue.get("labels") or []
        labels_str = ", ".join((l.get("name") if isinstance(l, dict) else str(l)) for l in labels)
        author_obj = issue.get("author") or issue.get("user") or {}
        author = author_obj.get("login") or author_obj.get("username") or "unknown"
        number = issue.get("number")
        title = issue.get("title", "")
        lines.append(f"- **#{number}** {title} (by @{author}{', labels: ' + labels_str if labels_str else ''})")
        body = str(issue.get("body") or "").strip()
        if body:
            lines.append(f"  > {body[:200]}{'...' if len(body) > 200 else ''}")
    return "\n".join(lines)


def _list_issues(ctx: ToolContext, state: str = "open", labels: str = "", limit: int = 20) -> str:
    platform = _vcs_platform()
    if platform == "git":
        return "⚠️ Issue tracker disabled for vcs_platform=git (local-only repo mode)."
    if platform == "gitea":
        owner, repo = _repo_owner_name()
        params = [f"state={state}", f"limit={min(limit, 50)}"]
        if labels:
            params.append(f"labels={labels}")
        code, body = _gitea_request("GET", f"/repos/{owner}/{repo}/issues?{'&'.join(params)}")
        if code < 200 or code >= 300:
            return f"⚠️ GITEA_ERROR: {body}"
        return _format_issue_list(body if isinstance(body, list) else [], state)

    args = [
        "issue", "list",
        "--state", state,
        "--limit", str(min(limit, 50)),
        "--json", "number,title,body,labels,createdAt,author,assignees,state",
    ]
    if labels:
        args.extend(["--label", labels])
    raw = _gh_cmd(args, ctx)
    if raw.startswith("⚠️"):
        return raw
    try:
        return _format_issue_list(json.loads(raw), state)
    except json.JSONDecodeError:
        return f"⚠️ Failed to parse issues JSON: {raw[:500]}"


def _get_issue(ctx: ToolContext, number: int) -> str:
    if number <= 0:
        return "⚠️ issue number must be positive"
    platform = _vcs_platform()
    if platform == "git":
        return "⚠️ Issue tracker disabled for vcs_platform=git (local-only repo mode)."

    if platform == "gitea":
        owner, repo = _repo_owner_name()
        code, issue = _gitea_request("GET", f"/repos/{owner}/{repo}/issues/{number}")
        if code < 200 or code >= 300:
            return f"⚠️ GITEA_ERROR: {issue}"
        code_c, comments = _gitea_request("GET", f"/repos/{owner}/{repo}/issues/{number}/comments")
        if code_c < 200 or code_c >= 300:
            comments = []
    else:
        raw = _gh_cmd([
            "issue", "view", str(number),
            "--json", "number,title,body,labels,createdAt,author,assignees,state,comments",
        ], ctx)
        if raw.startswith("⚠️"):
            return raw
        try:
            issue = json.loads(raw)
            comments = issue.get("comments", [])
        except json.JSONDecodeError:
            return f"⚠️ Failed to parse issue JSON: {raw[:500]}"

    labels = issue.get("labels") or []
    labels_str = ", ".join((l.get("name") if isinstance(l, dict) else str(l)) for l in labels)
    author_obj = issue.get("author") or issue.get("user") or {}
    author = author_obj.get("login") or author_obj.get("username") or "unknown"

    lines = [
        f"## Issue #{issue.get('number')}: {issue.get('title', '')}",
        f"**State:** {issue.get('state', 'unknown')}  |  **Author:** @{author}",
    ]
    if labels_str:
        lines.append(f"**Labels:** {labels_str}")
    body = str(issue.get("body") or "").strip()
    if body:
        lines.append(f"\n**Body:**\n{body[:3000]}")

    if comments:
        lines.append(f"\n**Comments ({len(comments)}):**")
        for c in comments[:10]:
            c_author_obj = c.get("author") or c.get("user") or {}
            c_author = c_author_obj.get("login") or c_author_obj.get("username") or "unknown"
            c_body = str(c.get("body") or "").strip()[:500]
            lines.append(f"\n@{c_author}:\n{c_body}")
    return "\n".join(lines)


def _comment_on_issue(ctx: ToolContext, number: int, body: str) -> str:
    if number <= 0:
        return "⚠️ issue number must be positive"
    if not body or not body.strip():
        return "⚠️ Comment body cannot be empty."
    platform = _vcs_platform()
    if platform == "git":
        return "⚠️ Issue tracker disabled for vcs_platform=git (local-only repo mode)."
    if platform == "gitea":
        owner, repo = _repo_owner_name()
        code, resp = _gitea_request("POST", f"/repos/{owner}/{repo}/issues/{number}/comments", {"body": body})
        if code < 200 or code >= 300:
            return f"⚠️ GITEA_ERROR: {resp}"
        return f"✅ Comment added to issue #{number}."

    raw = _gh_cmd(["issue", "comment", str(number), "--body-file", "-"], ctx, input_data=body)
    if raw.startswith("⚠️"):
        return raw
    return f"✅ Comment added to issue #{number}."


def _close_issue(ctx: ToolContext, number: int, comment: str = "") -> str:
    if number <= 0:
        return "⚠️ issue number must be positive"
    if comment and comment.strip():
        result = _comment_on_issue(ctx, number, comment)
        if result.startswith("⚠️"):
            return result
    platform = _vcs_platform()
    if platform == "git":
        return "⚠️ Issue tracker disabled for vcs_platform=git (local-only repo mode)."
    if platform == "gitea":
        owner, repo = _repo_owner_name()
        code, resp = _gitea_request("PATCH", f"/repos/{owner}/{repo}/issues/{number}", {"state": "closed"})
        if code < 200 or code >= 300:
            return f"⚠️ GITEA_ERROR: {resp}"
        return f"✅ Issue #{number} closed."

    raw = _gh_cmd(["issue", "close", str(number)], ctx)
    if raw.startswith("⚠️"):
        return raw
    return f"✅ Issue #{number} closed."


def _create_issue(ctx: ToolContext, title: str, body: str = "", labels: str = "") -> str:
    if not title or not title.strip():
        return "⚠️ Issue title cannot be empty."
    platform = _vcs_platform()
    if platform == "git":
        return "⚠️ Issue tracker disabled for vcs_platform=git (local-only repo mode)."

    if platform == "gitea":
        owner, repo = _repo_owner_name()
        payload: Dict[str, Any] = {"title": title}
        if body:
            payload["body"] = body
        if labels:
            payload["labels"] = [l.strip() for l in labels.split(",") if l.strip()]
        code, resp = _gitea_request("POST", f"/repos/{owner}/{repo}/issues", payload)
        if code < 200 or code >= 300:
            return f"⚠️ GITEA_ERROR: {resp}"
        num = (resp or {}).get("number") if isinstance(resp, dict) else "?"
        return f"✅ Issue created: #{num}"

    args = ["issue", "create", f"--title={title}"]
    raw = _gh_cmd(args + (["--body-file=-"] if body else []), ctx, input_data=body if body else None)
    if labels and not raw.startswith("⚠️"):
        import re
        match = re.search(r'/issues/(\d+)', raw)
        if match:
            _gh_cmd(["issue", "edit", str(int(match.group(1))), f"--add-label={labels}"], ctx)
    if raw.startswith("⚠️"):
        return raw
    return f"✅ Issue created: {raw}"


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("list_github_issues", {
            "name": "list_github_issues",
            "description": "List issues from configured platform (GitHub or Gitea).",
            "parameters": {"type": "object", "properties": {
                "state": {"type": "string", "default": "open", "enum": ["open", "closed", "all"]},
                "labels": {"type": "string", "default": ""},
                "limit": {"type": "integer", "default": 20},
            }, "required": []},
        }, _list_issues),
        ToolEntry("get_github_issue", {
            "name": "get_github_issue",
            "description": "Get issue details from configured platform.",
            "parameters": {"type": "object", "properties": {
                "number": {"type": "integer"},
            }, "required": ["number"]},
        }, _get_issue),
        ToolEntry("comment_on_issue", {
            "name": "comment_on_issue",
            "description": "Add a comment to an issue.",
            "parameters": {"type": "object", "properties": {
                "number": {"type": "integer"},
                "body": {"type": "string"},
            }, "required": ["number", "body"]},
        }, _comment_on_issue),
        ToolEntry("close_github_issue", {
            "name": "close_github_issue",
            "description": "Close an issue on configured platform.",
            "parameters": {"type": "object", "properties": {
                "number": {"type": "integer"},
                "comment": {"type": "string", "default": ""},
            }, "required": ["number"]},
        }, _close_issue),
        ToolEntry("create_github_issue", {
            "name": "create_github_issue",
            "description": "Create an issue on configured platform.",
            "parameters": {"type": "object", "properties": {
                "title": {"type": "string"},
                "body": {"type": "string", "default": ""},
                "labels": {"type": "string", "default": ""},
            }, "required": ["title"]},
        }, _create_issue),
    ]
