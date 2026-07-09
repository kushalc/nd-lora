"""Git repository utilities."""

import subprocess
from pathlib import Path


def check_git_repo_clean() -> None:
    """Check if git repository is clean before experiment."""
    project_root = Path.cwd()
    result = subprocess.run(
        ["/usr/bin/git", "status", "--porcelain"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True
    )
    dirty_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
    assert len(dirty_files) == 0, f"Git repo must be clean before experiment: {dirty_files}"

