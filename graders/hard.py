"""Hard task grader."""

from __future__ import annotations

from typing import Any

from graders.common import extract_total_score


class HardGrader:
    def grade(self, env: Any, *args: Any, **kwargs: Any) -> float:
        return extract_total_score(env, *args, **kwargs)
