"""Audit a folder of race-lap CSV files exported from FastF1 by someone else.

Used to decide whether third-party copies of the timing data are fit to use (they were not; see
docs/EXTERNAL_SOURCES.md). The script reports, for every `*_R_laps.csv` file under a folder:
laps and drivers, whether lap times are present, laps missing inside a driver's race, the compounds
used, and whether the file is a byte-for-byte copy of another race.

Run:  python scripts/audit_external_laps.py <folder>
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd


def audit(folder: Path) -> pd.DataFrame:
    rows, seen = [], {}
    for path in sorted(folder.rglob("*_R_laps.csv")):
        laps = pd.read_csv(path, low_memory=False)
        digest = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 (identity check, not security)
        missing = sum(
            int(g["LapNumber"].max() - g["LapNumber"].min() + 1 - g["LapNumber"].nunique())
            for _, g in laps.groupby("Driver")
        )
        rows.append(
            {
                "file": str(path.relative_to(folder)),
                "laps": len(laps),
                "drivers": laps["Driver"].nunique(),
                "race_laps": int(laps["LapNumber"].max()),
                "lap_times_present": int(laps["LapTime"].notna().sum()) if "LapTime" in laps else 0,
                "laps_missing_inside_races": missing,
                "drivers_starting_lap_1": int((laps["LapNumber"] == 1).sum()),
                "compounds": ",".join(sorted(laps["Compound"].dropna().astype(str).unique())),
                "same_as": seen.get(digest, ""),
            }
        )
        seen.setdefault(digest, str(path.relative_to(folder)))
    return pd.DataFrame(rows)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    table = audit(Path(sys.argv[1]))
    if table.empty:
        raise SystemExit("No *_R_laps.csv files found.")
    pd.set_option("display.width", 220, "display.max_rows", 500)
    print(table.to_string(index=False))
    standard = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
    odd = table[~table["compounds"].apply(lambda c: set(c.split(",")) <= standard)]
    print(f"\nRace files: {len(table)}; laps: {int(table['laps'].sum()):,}")
    print(f"Files with no lap times at all: {int((table['lap_times_present'] == 0).sum())}")
    print(f"Files that copy another race: {int((table['same_as'] != '').sum())}")
    gaps = int((table["laps_missing_inside_races"] > 0).sum())
    truncated = int((table["drivers_starting_lap_1"] < 15).sum())
    print(f"Files with laps missing inside a driver's race: {gaps}")
    print(f"Files where fewer than 15 drivers have a first lap: {truncated}")
    print(f"Files with non-standard compound names: {len(odd)}")


if __name__ == "__main__":
    main()
