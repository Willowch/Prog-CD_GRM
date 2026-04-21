from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT_DIR / "data" / "Sports_and_Outdoors" / "reviews_Sports_and_Outdoors_5.json.gz"
DST_PATH = ROOT_DIR / "data" / "Sports_and_Outdoors" / "Sports_and_Outdoors.inter"

HEADER = [
    "user_id:token",
    "item_id:token",
    "rating:float",
    "timestamp:float",
]


def parse_record(obj: dict) -> tuple[str, str, float, float] | None:
    user_id = obj.get("reviewerID")
    item_id = obj.get("asin")
    timestamp = obj.get("unixReviewTime")
    rating = obj.get("overall", 0.0)

    if not user_id or not item_id or timestamp is None:
        return None

    try:
        return (
            str(user_id),
            str(item_id),
            float(rating),
            float(timestamp),
        )
    except (TypeError, ValueError):
        return None


def convert_json_gz_to_inter(src_path: Path, dst_path: Path) -> None:
    if not src_path.exists():
        raise FileNotFoundError(f"Input file not found: {src_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    written_lines = 0
    skipped_lines = 0
    bad_json_lines = 0

    records = []

    with gzip.open(src_path, "rt", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            total_lines += 1
            line = line.strip()

            if not line:
                skipped_lines += 1
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_json_lines += 1
                skipped_lines += 1
                continue

            record = parse_record(obj)
            if record is None:
                skipped_lines += 1
                continue

            records.append(record)
            written_lines += 1

            if line_no % 100000 == 0:
                print(f"[Progress] processed={line_no}, written={written_lines}, skipped={skipped_lines}")

    records.sort(key=lambda row: (row[0], row[3], row[1]))

    with open(dst_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(records)

    print("=" * 60)
    print("Conversion finished.")
    print(f"Input : {src_path}")
    print(f"Output: {dst_path}")
    print(f"Total lines : {total_lines}")
    print(f"Written     : {written_lines}")
    print(f"Skipped     : {skipped_lines}")
    print(f"Bad JSON    : {bad_json_lines}")
    print("=" * 60)


def main() -> None:
    convert_json_gz_to_inter(SRC_PATH, DST_PATH)


if __name__ == "__main__":
    main()
