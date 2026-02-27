"""Search tool module intentionally disabled.

Ouroboros runs in local-only mode without web-search capabilities.
"""

from __future__ import annotations

from typing import List

from ouroboros.tools.registry import ToolEntry


def get_tools() -> List[ToolEntry]:
    """No search tools are exposed in local-only mode."""
    return []
