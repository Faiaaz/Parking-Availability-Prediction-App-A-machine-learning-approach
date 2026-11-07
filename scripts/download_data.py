"""Download public labelled parking datasets."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

BIRMINGHAM_ZIP = (
    "https://archive.ics.uci.edu/static/public/482/parking+birmingham.zip"
)
MELBOURNE_CSV = (
    "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/"
    "on-street-parking-bay-sensors/exports/csv?limit=-1"
)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  → {dest}")
    urlretrieve(url, dest)
    print(f"  saved ({dest.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["birmingham", "melbourne", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.dataset in {"birmingham", "all"}:
        zpath = RAW / "parking_birmingham.zip"
        download(BIRMINGHAM_ZIP, zpath)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(RAW)
        print(f"Birmingham CSV → {RAW / 'dataset.csv'}")

    if args.dataset in {"melbourne", "all"}:
        download(MELBOURNE_CSV, RAW / "melbourne_bay_sensors.csv")
        print("Melbourne live bay sensors saved (status snapshot for map view).")


if __name__ == "__main__":
    main()
