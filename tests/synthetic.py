"""Fake FastF1 objects that return race data in FastF1's exact shape, for offline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENTS = [
    "Bahrain Grand Prix",
    "Saudi Arabian Grand Prix",
    "Australian Grand Prix",
    "Qatar Grand Prix",
    "Japanese Grand Prix",
    "Monaco Grand Prix",
]
TRUE_FUEL_SLOPE = -0.06  # seconds per lap built into the fake lap times
DEG_RATE = {"SOFT": 0.09, "MEDIUM": 0.06, "HARD": 0.04, "INTERMEDIATE": 0.05}
STINT_TARGET = {"SOFT": 18, "MEDIUM": 26, "HARD": 34, "INTERMEDIATE": 20}
N_LAPS = 55
N_DRIVERS = 20
RETIRING_DRIVER = "D19"
RETIRE_LAP = 45


class FakeLaps(pd.DataFrame):
    _metadata = ["_weather"]

    @property
    def _constructor(self):
        return FakeLaps

    def get_weather_data(self):
        return self._weather


def make_race(year: int, rnd: int, wet: bool = False) -> FakeLaps:
    rng = np.random.default_rng(year * 100 + rnd)
    sc_laps = set(range(20, 23)) if rnd % 2 else set()
    rows = []
    for d_i in range(N_DRIVERS):
        drv = f"D{d_i:02d}"
        comps = ["MEDIUM", "HARD"] if d_i % 3 else ["SOFT", "HARD", "MEDIUM"]
        if wet:
            comps = ["INTERMEDIATE", "MEDIUM"]
        stint, comp_i, life, t = 1, 0, 0, 0.0
        for lap in range(1, N_LAPS + 1):
            comp = comps[comp_i]
            life += 1
            out_lap = life == 1 and stint > 1
            sc = lap in sc_laps
            target = STINT_TARGET[comp] + int(rng.integers(-3, 4))
            pit = (life >= target or (sc and life > 10)) and comp_i < len(comps) - 1 and lap < N_LAPS - 3
            retire = drv == RETIRING_DRIVER and lap == RETIRE_LAP
            lt = float(90 + 0.05 * d_i + TRUE_FUEL_SLOPE * lap + DEG_RATE[comp] * life + rng.normal(0, 0.25))
            lt += 25 * sc + 3 * (pit or retire) + 20 * out_lap
            t = float(t + lt)
            rows.append(
                dict(
                    Driver=drv,
                    Team=f"T{d_i // 2}",
                    LapNumber=float(lap),
                    Stint=float(stint),
                    Compound=comp,
                    TyreLife=float(life),
                    FreshTyre=True if stint > 1 else None,
                    TrackStatus="4" if sc else "1",
                    IsAccurate=not (sc or pit or retire or out_lap),
                    Deleted=None if lap % 7 else False,
                    FastF1Generated=False,
                    LapTime=lt,
                    Time=t,
                    PitInTime=t if (pit or retire) else np.nan,
                    PitOutTime=t - lt + 2 if out_lap else np.nan,
                )
            )
            if retire:
                break
            if pit:
                stint, comp_i, life = stint + 1, comp_i + 1, 0
    df = pd.DataFrame(rows)
    for col in ["LapTime", "Time", "PitInTime", "PitOutTime"]:  # seconds -> Timedelta, as FastF1 gives
        df[col] = pd.to_timedelta(df[col], unit="s")
    df["Position"] = df.groupby("LapNumber")["Time"].rank(method="first").astype(float)
    laps = FakeLaps(df)
    laps._weather = pd.DataFrame(
        {
            "Time": df["Time"],
            "AirTemp": 25.0,
            "TrackTemp": 38.0 + rng.normal(0, 1, len(df)),
            "Humidity": 50.0,
            "Pressure": 1010.0,
            "Rainfall": wet,
            "WindDirection": 0,
            "WindSpeed": 1.0,
        }
    )
    return laps


class FakeSession:
    def __init__(self, year, rnd, wet=False, broken=False):
        self.year, self.rnd, self.wet, self.broken = year, rnd, wet, broken
        self.event = {"EventName": EVENTS[(rnd - 1) % len(EVENTS)]}
        self._laps = None
        self.total_laps = N_LAPS  # scheduled distance, as FastF1's Session.total_laps

    def load(self, **kwargs):
        if not self.broken:
            self._laps = make_race(self.year, self.rnd, self.wet)

    @property
    def laps(self):
        if self._laps is None:
            raise RuntimeError("The data you are trying to access has not been loaded yet.")
        return self._laps


class FakeFastF1:
    """Stands in for the `fastf1` module."""

    def __init__(self, rounds=6, wet_races=(), broken_races=(), future_rounds=0):
        self.rounds, self.future = rounds, future_rounds
        self.wet, self.broken = set(wet_races), set(broken_races)
        self.download_calls = 0

        class _Cache:
            @staticmethod
            def enable_cache(path):
                return None

        self.Cache = _Cache

    def get_event_schedule(self, year, include_testing=True):
        total = self.rounds + self.future
        past = pd.Timestamp("2020-01-01")
        future = pd.Timestamp("2099-01-01")
        dates = [past] * self.rounds + [future] * self.future
        return pd.DataFrame({"RoundNumber": range(1, total + 1), "EventDate": dates})

    def get_session(self, year, rnd, kind):
        self.download_calls += 1
        return FakeSession(year, rnd, wet=(year, rnd) in self.wet, broken=(year, rnd) in self.broken)
