"""Canonical paths for the uav-gnss-triage project (repo lives inside the Drive project folder)."""
from pathlib import Path

PROJECT = "UAV_GNSS"
REPO_NAME = "uav-gnss-triage"
GH_USER = "anasbiswas1"
GIT_NAME = "Md Anas Biswas"
GIT_EMAIL = "anasbiswas@gmail.com"

DRIVE_MOUNT = Path("/content/drive")
DRIVE_ROOT = DRIVE_MOUNT / "MyDrive" / f"{PROJECT}_Research"
REPO_DIR = DRIVE_ROOT / REPO_NAME

SRC = REPO_DIR / "src"
DATA = REPO_DIR / "data"            # gitignored
SIH_RUNS = DATA / "sih"             # one folder per generation run (manifest.jsonl + ULogs)
FEATURES = DATA / "features"        # one folder per feature run (windows.csv, flights.csv)
PX4_CACHE = DATA / "px4_cache"      # px4_autopilot.tgz
WHELAN_ZIP = DATA / "uav_attack" / "UAVAttackData.zip"
REPORTS = REPO_DIR / "reports"      # committed result tables, one folder per model run
FIGURES = REPO_DIR / "figures"
PX4_SRC = Path("/content/PX4-Autopilot")   # ephemeral build tree, restored from PX4_CACHE
