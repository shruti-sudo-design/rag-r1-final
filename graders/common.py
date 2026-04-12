"""Utilities for task graders."""

from __future__ import annotations

from typing import Any


def clamp_score(score: float) -> float:
    """Clamp validator-facing scores away from exact 0.0 / 1.0 edges."""
    return round(max(0.01, min(0.99, float(score))), 4)


def extract_total_score(env: Any, *args: Any, **kwargs: Any) -> float:
    """Best-effort extraction of a total reward from common OpenEnv call shapes."""
    candidates = []

    if kwargs:
        candidates.extend(
            [
                kwargs.get("score"),
                kwargs.get("reward"),
                kwargs.get("total"),
                (kwargs.get("reward") or {}).get("total")
                if isinstance(kwargs.get("reward"), dict)
                else None,
                (kwargs.get("result") or {}).get("reward", {}).get("total")
                if isinstance(kwargs.get("result"), dict)
                else None,
            ]
        )

    if args:
        candidates.extend(args)

    if env is not None:
        for attr in ("last_reward", "reward", "score", "last_score"):
            value = getattr(env, attr, None)
            if isinstance(value, dict):
                candidates.append(value.get("total"))
            else:
                candidates.append(value)

        get_state = getattr(env, "get_state", None)
        if callable(get_state):
            try:
                state = get_state()
            except Exception:
                state = None
            if isinstance(state, dict):
                reward = state.get("reward")
                if isinstance(reward, dict):
                    candidates.append(reward.get("total"))
                candidates.append(state.get("score"))

    for value in candidates:
        if isinstance(value, (int, float)):
            return clamp_score(float(value))
        if isinstance(value, dict):
            nested = value.get("total")
            if isinstance(nested, (int, float)):
                return clamp_score(float(nested))

    return 0.01

