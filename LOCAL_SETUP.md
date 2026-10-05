# Run the project on your own computer

This guide takes you from the zip file to the final results and the dashboard. Every command was run
before this guide was written. Commands are given for **Windows (PowerShell)** and for **macOS / Linux**;
where only one block is shown, it is the same on both.

## 1. What you need

| Item | Requirement |
| --- | --- |
| Python | 3.10 or newer (3.11 or 3.12 recommended). Check with `python --version` |
| Disk space | About 3 GB (PyTorch is the largest part) |
| Internet | Needed for installing packages and for the first race download |
| GPU | Not needed. Full training took about 25 minutes on a 2-core CPU |
| Git, Docker | Optional (Docker only for section 9) |

On Windows, install Python from python.org and tick **"Add python.exe to PATH"** during installation.

## 2. Unzip and open a terminal in the project folder

Unzip `f1-pit-engine.zip`, then open a terminal **inside the `f1-pit-engine` folder** (the folder that
contains `pyproject.toml`).

- Windows: open the folder in File Explorer, click the address bar, type `powershell`, press Enter.
- macOS / Linux: `cd path/to/f1-pit-engine`

## 3. Create a virtual environment and install

Windows (PowerShell):

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,dl,app]"
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,dl,app]"
```

Notes:

- Your prompt should now start with `(.venv)`. Activate the environment again each time you open a new
  terminal (the `Activate.ps1` or `source` line).
- The `torch --index-url ...` line installs the small CPU build. If it fails (some networks block that
  site), skip it; the next line installs the standard build from PyPI, which is larger but works.
- If you have an NVIDIA GPU and want to use it, skip the `torch` line and install the CUDA build from
  pytorch.org before the last line.

## 4. Run the tests

```bash
python -m pytest
```

Expected: **91 passed** in about three minutes. The tests need no internet and no real race data; they
use practice data generated in memory. If this passes, the installation is correct.

Useful variations:

```bash
python -m pytest tests/test_clean.py            # one file
python -m pytest -k pit                         # only tests with "pit" in the name
python -m pytest -x                             # stop at the first failure
ruff check .                                    # code style check
ruff format --check .
```

## 5. Trial run on real data (about 10 minutes)

A trial run uses the first four races of each season, to check everything works before the long run.

1. Open `configs/default.yaml` in a text editor and change `quick: false` to `quick: true`. Save.
2. Run the steps in order:

```bash
python -m f1pit.ingest
python -m f1pit.clean
python -m f1pit.features
python -m f1pit.stage1
python -m f1pit.stage2
```

What each step does and what to expect:

| Step | What it does | Output | Time (trial) |
| --- | --- | --- | --- |
| `ingest` | Downloads race laps. Tries FastF1 first; if FastF1 returns nothing, reads the same lap tables from the TracingInsights archive on GitHub | `data/races/*.parquet`; prints `OK` per race and how many races came from each source | 1 min |
| `clean` | Removes wet races, labels pit laps, flags rule-forced races, stops if a race file is a copy of another | `data/processed/laps_clean.parquet` | seconds |
| `features` | Fuel correction, degradation, model inputs | `data/processed/features.parquet` | seconds |
| `stage1` | Tyre degradation models, laps-to-threshold forecast | `data/processed/features_stage1.parquet`, `results/stage1_metrics.json` | under 1 min |
| `stage2` | Trains the seven pit-decision experiments with three seeds each; prints the validation table and H1/H2 | `results/stage2_metrics.json`, `results/models/` | 5 to 10 min |

A trial run cannot be used for conclusions, and the final test refuses to run on it.

## 6. Full run and the final test

1. Change `quick: true` back to `quick: false` in `configs/default.yaml`.
2. Run:

```bash
python -m f1pit.ingest
python -m f1pit.clean
python -m f1pit.features
python -m f1pit.stage1
python -m f1pit.stage2
python -m f1pit.evaluate
python -m f1pit.report
```

| Step | Time (full, 2-core CPU) |
| --- | --- |
| `ingest` | 2 to 5 min the first time; seconds afterwards (saved races are reused) |
| `clean`, `features`, `stage1` | under 1 min together |
| `stage2` | about 25 min (the published Bi-LSTM replica takes most of it) |
| `evaluate` | about 5 min |
| `report` | seconds |

Where the results are:

| File | Content |
| --- | --- |
| `results/report/data_facts.md` | Table 1 and the facts of the run. Copy numbers into your report from here |
| `results/report/results_tables.md` | Tables 2 to 9: Stage 1, validation, final test, hypotheses, calibration, explanations |
| `results/figures/*.png` | The charts |
| `results/final_model_table.csv`, `results/final_hypothesis_table.csv` | The same numbers as CSV |
| `results/final_evaluation_log.json` | One entry per evaluation of the test seasons |

**The final test is meant to be run once.** Running `evaluate` again on the same models only shows the
saved results. If you retrain and evaluate again, the log records it and the report must say why.

## 7. Race poster

```bash
python -m f1pit.race_poster --list
python -m f1pit.race_poster --race 2025_10
```

The picture is written to `results/figures/race_poster_2025_10.png`.

## 8. Dashboard

```bash
python -m f1pit.dashboard
streamlit run app/streamlit_app.py
```

The first command exports your final results to `app/data/`; the second opens the dashboard at
<http://localhost:8501>. Press Ctrl+C in the terminal to stop it. Without the first command the dashboard
shows the practice data in `app/demo_data/` with a banner saying so.

## 9. Docker (optional)

```bash
docker build -t f1pit-tests .
docker run --rm f1pit-tests                         # runs the tests inside a container

docker build -f Dockerfile.dashboard -t f1pit-dashboard .
docker run --rm -p 8501:8501 f1pit-dashboard        # dashboard at http://localhost:8501
```

## 10. Shortcuts with make (macOS / Linux)

`make install`, `make test`, `make lint`, `make pipeline`, `make evaluate`, `make report`,
`make poster RACE=2025_10`, `make dashboard-data`, `make dashboard`. Windows has no `make` by default; use
the `python -m ...` commands above.

## 11. If something goes wrong

| Message | Cause and fix |
| --- | --- |
| `python` is not recognised | Python is not on PATH. Reinstall with "Add python.exe to PATH", or use `py` instead of `python` on Windows |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same window, then activate again |
| `No module named f1pit` | The environment is not active, or `pip install -e ".[dev,dl,app]"` was not run in this folder |
| `No module named torch` | Run `pip install torch` |
| `MISS 2026 round ..` during `ingest` | Neither source has that race yet. The pipeline carries on without it |
| `ingest stopped: 3 races in a row returned no data from any source` | No internet, or GitHub is unreachable. Check the connection and run `ingest` again |
| `CopiedRaceError` during `clean` | Two race files hold the same laps. Delete the copies from `data/races/` and run `ingest` again |
| `No validation laps` in `stage1` | The validation season has no races in `data/races/`. Run `ingest` again and read its output |
| `These models were trained in quick mode` in `evaluate` | Set `quick: false`, run the pipeline again from `ingest`, then `evaluate` |
| Old numbers after changing the code | Run the steps again from `clean`; each step overwrites its own output |

To start completely fresh, delete the `data/processed` and `results` folders (keep `data/races` to avoid
downloading again).
