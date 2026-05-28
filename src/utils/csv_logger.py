from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


class CsvEventLogger:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_file()

    def _init_file(self):
        file_exists = self.output_path.exists()
        with self.output_path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            if not file_exists:
                writer.writerow(["timestamp", "direction", "tracker_id", "cx", "cy"])

    def log(self, direction: str, tracker_id: int, centroid: tuple[int, int]):
        with self.output_path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                direction,
                tracker_id,
                centroid[0],
                centroid[1],
            ])
