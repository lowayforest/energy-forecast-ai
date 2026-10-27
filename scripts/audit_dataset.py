"""Summarize station coverage in the HKUST public rooftop PV archive."""

from __future__ import annotations

import csv
import io
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def main(archive: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        stations = [
            name
            for name in source.namelist()
            if "/Site level dataset/" in name and name.endswith(".csv")
        ]
        summaries = []
        for name in stations:
            daily_rows: dict[str, int] = defaultdict(int)
            daily_energy: dict[str, float] = defaultdict(float)
            missing = 0
            nonzero = 0
            first = last = None
            with io.TextIOWrapper(source.open(name), encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                for row in reader:
                    try:
                        stamp = datetime.fromisoformat(row["Time"])
                        energy = float(row["generation(kWh)"])
                    except (KeyError, TypeError, ValueError):
                        missing += 1
                        continue
                    first = stamp if first is None else min(first, stamp)
                    last = stamp if last is None else max(last, stamp)
                    daily_rows[stamp.date().isoformat()] += 1
                    daily_energy[stamp.date().isoformat()] += energy
                    nonzero += energy > 0
            if first is None:
                continue
            summaries.append(
                (
                    sum(count >= 90 for count in daily_rows.values()),
                    name.rsplit("/", 1)[-1],
                    first.date().isoformat(),
                    last.date().isoformat(),
                    len(daily_rows),
                    missing,
                    nonzero,
                    round(max(daily_energy.values()), 1),
                )
            )
    print("full_days,station,first,last,days,invalid_rows,nonzero_intervals,max_daily_kwh")
    for summary in sorted(summaries, reverse=True):
        print(",".join(map(str, summary)))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/hkust_rooftop_pv_2021_2023.zip"))
