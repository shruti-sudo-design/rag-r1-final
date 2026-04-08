"""OpenEnv server entry point.

This module exposes the FastAPI `app` object from main.py and provides
a `main()` function that is wired to the `[project.scripts]` entry point
in pyproject.toml so OpenEnv can start the server with `uv run server`.
"""

import uvicorn
from main import app  # noqa: F401  re-exported for OpenEnv discovery

__all__ = ["app"]


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
