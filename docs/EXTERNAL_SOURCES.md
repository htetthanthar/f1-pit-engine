# External sources reviewed

Six public sources were reviewed on 4 October 2026 for ideas and for data. Race data comes from FastF1
through `python -m f1pit.ingest`. One further source, the TracingInsights archive, is used as a second
route to the same FastF1 lap tables for races FastF1 cannot deliver (see the last section). No data
from the other sources is used.

| Source | What it is | Data | Used here |
| --- | --- | --- | --- |
| [FastF1](https://github.com/theOehrly/Fast-F1) | Python library for Formula 1 timing data (MIT licence; unofficial) | Formula 1 live-timing feed; Jolpica-F1 | Yes: the data source |
| [TracingInsights archive](https://github.com/TracingInsights-Archive) | One GitHub repository per season with FastF1's lap table exported to JSON for every driver and session (MIT licence) | FastF1 exports (2022 to 2026 checked) | Yes: second source, only for races FastF1 returns nothing for |
| [F1 Vision](https://github.com/ArsalanKaleem/F1-Vision) | Flutter app: live timing, telemetry, analytics, race replay (MIT licence) | OpenF1 (2023 onwards), Jolpica-F1 | No data. Reference for the replay idea |
| [f1-telemetry](https://github.com/ArslanKamchybekov/f1-telemetry) | C++17 multi-threaded race simulator at 50 Hz (no licence stated) | Synthetic | No |
| [F1 tire degradation strategy predictor](https://github.com/sidgaikwad07/F1_tire_degradation_strategy_predictor) | Multi-task neural network for tyre degradation (no licence stated) | FastF1 exports, 2022 to 2025 | No: the stored data has defects (below) |
| [Predicting Formula 1 Tire Degradation](https://schilamkur.github.io/Predict-Tire-Deg/) | Tree-based regression of laps before degradation; best MAE 1.70 laps | FastF1, 3,375 stints | No: stint summaries, not laps |

## Why the stored lap files were rejected

The `sidgaikwad07` repository stores one lap file per session. Its 76 race files (82,290 laps) were
audited with `scripts/audit_external_laps.py`:

| Check | Result |
| --- | --- |
| Race files with no lap times at all | 76 of 76 |
| Race files that are a copy of another race | 1 (2024 Great Britain is identical to 2024 Austria) |
| Race files with laps missing inside a driver's race | 12 |
| Race files where most drivers have no first lap | 1 (2025 Miami) |
| Race files with non-standard compound names | 9 (all of 2025, relabelled C1 to C3 by a fixed rule) |
| Seasons covered | 2022 (21 races), 2023 (22), 2024 (24, one duplicated), 2025 (9); no 2026 |

The repository also contains scripts that generate artificial 2024 and 2025 lap data, and its training
pipeline loads the generated 2024 file in preference to the real laps.

## Why the stint file was rejected

The Chilamkur project stores one row per stint. This project needs one row per lap. Its model also
splits stints at random, and some inputs use the end of the stint or the target itself, so its
reported error is not a forecast that could be made during a race.

## Reproduce the audit

```bash
git clone --depth 1 https://github.com/sidgaikwad07/F1_tire_degradation_strategy_predictor ext
python scripts/audit_external_laps.py ext
```

These repositories have no licence file, so their contents are not copied here.

## Why the TracingInsights archive is accepted as a second source

The Formula 1 timing service refuses many requests from cloud computers, so on Google Colab FastF1 often
returns no laps. The archive holds exports of the same FastF1 lap table (lap time, session time, pit in
and out times, compound, tyre life, stint, position, track status, accuracy flags and the weather at
each lap) and is served by GitHub, which Colab can reach. `f1pit.archive` converts a race to this
project's lap format and marks it `Source = "tracinginsights"`.

Checks made on 4 October 2026, on all 96 races the archive holds for 2022 to 2026 (105,375 laps):

| Check | Result |
| --- | --- |
| Races available | 2022: 22 of 22, 2023: 22 of 22, 2024: 24 of 24, 2025: 24 of 24, 2026: rounds 1 to 4 only |
| Agreement with a direct FastF1 download | 2022 rounds 1 to 4: identical counts after cleaning (3 dry races, 2,982 laps, 99 pit laps, 21 drivers) and the same fuel slope (-0.080 s per lap) as the FastF1 files downloaded for this project |
| Duplicate laps, or laps missing inside a driver's race | none |
| Session time going backwards | none |
| Lap time equal to the difference of session times | median difference 0.000 s; 0.04% of laps differ by more than 0.5 s |
| Missing lap times | 0.15% of laps (0.8% in the worst race), as in FastF1 |
| Races that are copies of another race | none |

Limits to state in the report:

- The archive is a third party's export, not the timing service itself. The agreement check above
  covers four races; it is evidence of equivalence, not proof for every race.
- It has no scheduled race distance. For archive races `ScheduledLaps` is empty and the cleaning step
  uses the laps actually run, which differs only for races stopped early.
- It held only the first four races of 2026 when checked, so the 2026 extra test is small unless the
  rest is downloaded with FastF1 on a home or university connection.
- `results/report/data_facts.md` states how many races came from each source.
