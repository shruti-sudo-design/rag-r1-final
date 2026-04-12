TASKS = [
    {
        "id": "rag_rl_easy",
        "task_id": "easy",
        "name": "rag-rl-easy",
        "difficulty": "easy",
        "description": "Redundancy-heavy evidence selection with a small budget",
        "max_steps": 2,
        "reset_params": {"task": "easy"},
        "action_schema": {"selected_chunk_indices": "list[int]"},
        "grader": "graders:grade_easy",
        "graders": ["graders:grade_easy"],
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "rag_rl_medium",
        "task_id": "medium",
        "name": "rag-rl-medium",
        "difficulty": "medium",
        "description": "Stale docs, contradictions, and adversarial summaries under moderate budget pressure",
        "max_steps": 3,
        "reset_params": {"task": "medium"},
        "action_schema": {"selected_chunk_indices": "list[int]"},
        "grader": "graders:grade_medium",
        "graders": ["graders:grade_medium"],
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "rag_rl_hard",
        "task_id": "hard",
        "name": "rag-rl-hard",
        "difficulty": "hard",
        "description": "Tight-budget multi-hop evidence composition with partial-support distractors",
        "max_steps": 4,
        "reset_params": {"task": "hard"},
        "action_schema": {"selected_chunk_indices": "list[int]"},
        "grader": "graders:grade_hard",
        "graders": ["graders:grade_hard"],
        "reward_range": [0.0, 1.0],
    },
]

TASK_ID_TO_INDEX = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
}

TASK_GRADER_PAIRS = [
    ("rag_rl_easy", "graders:grade_easy"),
    ("rag_rl_medium", "graders:grade_medium"),
    ("rag_rl_hard", "graders:grade_hard"),
]

__all__ = ["TASKS", "TASK_ID_TO_INDEX", "TASK_GRADER_PAIRS"]
