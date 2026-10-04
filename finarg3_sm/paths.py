"""Project paths, independent of the working directory or workflow location."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def required_input_path(variable):
    """Read an explicitly configured input path; never guess another source."""
    value = os.environ.get(variable)
    if not value or not value.strip():
        raise SystemExit(f"Set {variable} to the organizer-provided input path first.")
    path = Path(value).expanduser()
    if not path.exists():
        raise SystemExit(f"{variable} does not exist: {path}")
    return path
