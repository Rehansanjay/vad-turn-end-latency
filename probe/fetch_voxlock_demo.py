"""Fetch VoxLock's published before/after demo pairs.

The audio belongs to the VoxLock project and is not redistributed here. This pulls it
straight from the upstream repository so the measurement can be reproduced without this
repo carrying someone else's data.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/arbazsiddiqui/voxlock/HEAD/demo/audio"
CLIPS = [
    "bystander_000_off.mp3",
    "bystander_000_on.mp3",
    "tail_024_off.mp3",
    "tail_024_on.mp3",
    "tail_028_off.mp3",
    "tail_028_on.mp3",
]


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "audio/voxlock")
    out.mkdir(parents=True, exist_ok=True)
    for name in CLIPS:
        dest = out / name
        urllib.request.urlretrieve(f"{BASE}/{name}", dest)  # noqa: S310
        print(f"{dest}  {dest.stat().st_size} bytes")


if __name__ == "__main__":
    main()
