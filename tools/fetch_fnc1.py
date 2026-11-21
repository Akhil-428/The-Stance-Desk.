"""Download the four FNC-1 CSV files and verify them against pinned SHA-256 hashes.

The dataset is not redistributed in this repository. This script fetches exactly
the four data files from the official FakeNewsChallenge/fnc-1 repository at the
commit recorded in ``config.FNC1_COMMIT`` and checks each against
``data/raw/fnc1/SHA256SUMS.txt``. Nothing from that repository is executed.

Usage:
    python tools/fetch_fnc1.py            # download missing files, verify all
    python tools/fetch_fnc1.py --verify   # verify only, no network access
    python tools/fetch_fnc1.py --base-url file:///some/local/mirror   # testing

Files that already exist and match their hash are left untouched. A file whose
hash does not match is never kept.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

FILES = ("train_bodies.csv", "train_stances.csv", "competition_test_bodies.csv", "competition_test_stances.csv")
DEFAULT_BASE = f"https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/{config.FNC1_COMMIT}"


def read_expected(path: Path) -> dict[str, str]:
    expected = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            digest, name = line.split(None, 1)
            expected[name.lstrip("*")] = digest.lower()
    missing = set(FILES) - set(expected)
    if missing:
        raise SystemExit(f"{path} has no hash for: {sorted(missing)}")
    return expected


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="only verify existing files; never download")
    ap.add_argument("--base-url", default=DEFAULT_BASE, help="where to fetch files from (default: pinned upstream commit)")
    ap.add_argument("--dest", type=Path, default=config.FNC1_DIR)
    args = ap.parse_args()

    dest: Path = args.dest
    expected = read_expected(config.FNC1_DIR / "SHA256SUMS.txt")
    dest.mkdir(parents=True, exist_ok=True)
    failed = False
    for name in FILES:
        target = dest / name
        if target.exists() and sha256_bytes(target.read_bytes()) == expected[name]:
            print(f"ok       {name}")
            continue
        if args.verify:
            print(f"MISSING  {name}" if not target.exists() else f"MISMATCH {name}")
            failed = True
            continue
        url = f"{args.base_url.rstrip('/')}/{name}"
        print(f"fetching {url}")
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed, pinned https URL)
            data = resp.read()
        if sha256_bytes(data) != expected[name]:
            print(f"MISMATCH {name}: downloaded file does not match the pinned SHA-256; not saved.")
            failed = True
            continue
        target.write_bytes(data)
        print(f"saved    {target} ({len(data):,} bytes)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
