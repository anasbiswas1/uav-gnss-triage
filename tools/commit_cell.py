"""Single commit step for this repo. Run from the scratch console notebook only:

    subprocess.run(["python", "tools/commit_cell.py", "message"])

Restores git identity and credentials from the Drive project folder, strips all notebook outputs,
refuses to stage data or credential files, then commits and pushes to origin/main.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path

DRIVE_PROJECT_ROOT = Path("/content/drive/MyDrive/UAV_GNSS_Research")
REPO_ROOT = Path(os.environ.get("UAV_GNSS_REPO_ROOT", str(DRIVE_PROJECT_ROOT / "uav-gnss-triage")))
GIT_NAME = "Md Anas Biswas"
GIT_EMAIL = "anasbiswas@gmail.com"
FORBIDDEN_NAMES = {".git-credentials", ".gitconfig", "kaggle.json", ".env"}
FORBIDDEN_PREFIXES = ("data/", "external/", ".secrets/")
MAX_FILE_MB = 50


def run(args, check=True):
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"{' '.join(args)}\n{r.stdout}{r.stderr}")
    return r.stdout


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        raise SystemExit('usage: python tools/commit_cell.py "commit message"')
    message = sys.argv[1].strip()

    for dotfile in (".gitconfig", ".git-credentials"):
        source = DRIVE_PROJECT_ROOT / dotfile
        if source.exists():
            shutil.copy(source, Path.home() / dotfile)
    credentials = Path.home() / ".git-credentials"
    if credentials.exists():
        os.chmod(credentials, 0o600)

    os.chdir(REPO_ROOT)
    run(["git", "config", "user.name", GIT_NAME])
    run(["git", "config", "user.email", GIT_EMAIL])
    run(["git", "config", "credential.helper", "store"])

    for notebook in sorted(glob(str(REPO_ROOT / "notebooks" / "*.ipynb"))):
        run(["jupyter", "nbconvert", "--ClearOutputPreprocessor.enabled=True", "--inplace", notebook])

    run(["git", "add", "-A"])
    staged = [l for l in run(["git", "diff", "--cached", "--name-only"]).splitlines() if l.strip()]
    bad = [p for p in staged if Path(p).name in FORBIDDEN_NAMES or p.startswith(FORBIDDEN_PREFIXES)]
    big = [p for p in staged if Path(p).exists() and Path(p).stat().st_size > MAX_FILE_MB * 1024 * 1024]
    if bad or big:
        run(["git", "reset", "-q"], check=False)
        raise SystemExit(f"refused: forbidden={bad} oversized={big}")
    if not staged:
        print("nothing to commit")
        print(run(["git", "log", "--oneline", "-1"]).strip())
        return

    commit = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True)
    print(commit.stdout or commit.stderr)
    push = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
    print(push.stdout or push.stderr)
    print(run(["git", "log", "--oneline", "-1"]).strip())
    print(run(["git", "status", "--short"]).strip() or "working tree clean")


if __name__ == "__main__":
    main()
