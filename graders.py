from typing import Any


def _normalize_reward(reward: Any) -> float:
    try:
        value = float(reward)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _task_matches(state: dict[str, Any] | None, task_id: str) -> bool:
    if not isinstance(state, dict):
        return True
    for key in ("task_id", "task", "name", "id"):
        value = state.get(key)
        if isinstance(value, str) and value == task_id:
            return True
    return False


def grade_easy(state: dict[str, Any] | None = None, reward: Any = 0.0) -> float:
    return _normalize_reward(reward if _task_matches(state, "easy") else 0.0)


def grade_medium(state: dict[str, Any] | None = None, reward: Any = 0.0) -> float:
    return _normalize_reward(reward if _task_matches(state, "medium") else 0.0)


def grade_hard(state: dict[str, Any] | None = None, reward: Any = 0.0) -> float:
    return _normalize_reward(reward if _task_matches(state, "hard") else 0.0)


GRADERS = {
    "rag_rl_easy": grade_easy,
    "rag_rl_medium": grade_medium,
    "rag_rl_hard": grade_hard,
}

TASK_GRADER_PAIRS = [
    ("rag_rl_easy", grade_easy),
    ("rag_rl_medium", grade_medium),
    ("rag_rl_hard", grade_hard),
]

__all__ = [
    "grade_easy",
    "grade_medium",
    "grade_hard",
    "GRADERS",
    "TASK_GRADER_PAIRS",
]
