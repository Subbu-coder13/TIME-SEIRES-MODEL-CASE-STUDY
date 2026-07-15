"""
Shared utilities for German electricity demand forecasting (Colab-friendly).

Import in each notebook:
    import sys
    sys.path.insert(0, "/content/drive/MyDrive/Assignment/colab")
    from utils import *
"""

from __future__ import annotations

import gc
import json
import warnings
from pathlib import Path
from typing import Any

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TEST_WEEKS = 104
SEASONALITY = 52
TEST_HOURS = TEST_WEEKS * 7 * 24  # 17,520 hours = 2 years

# Colab users: update this to their Drive folder path.
DRIVE_ROOT = Path("/content/drive/MyDrive/Assignment")
DATA_CSV = "time_series_60min_singleindex.csv"
LOAD_COLUMN = "DE_load_actual_entsoe_transparency"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def get_paths(root: Path | str | None = None) -> dict[str, Path]:
    """Return standard output directories, creating them if needed."""
    root = Path(root) if root is not None else DRIVE_ROOT
    paths = {
        "root": root,
        "data": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "figures": root / "outputs" / "figures",
        "metrics": root / "outputs" / "metrics",
        "forecasts": root / "outputs" / "forecasts",
        "checkpoints": root / "outputs" / "checkpoints",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def to_naive_index(obj: pd.Series | pd.DataFrame | pd.DatetimeIndex):
    """Convert timezone-aware datetime index to naive UTC (avoids tz vs naive errors)."""
    if isinstance(obj, pd.DatetimeIndex):
        idx = obj
        if getattr(idx, "tz", None) is not None:
            return idx.tz_convert("UTC").tz_localize(None)
        return idx

    out = obj.copy()
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out


def load_hourly_load(
    csv_path: Path | str,
    start_date: str = "2015-01-01",
) -> pd.Series:
    """Load German hourly electricity load (MW) from OPSD CSV."""
    csv_path = Path(csv_path)
    df = pd.read_csv(
        csv_path,
        usecols=["utc_timestamp", LOAD_COLUMN],
        parse_dates=["utc_timestamp"],
    )
    df = df.rename(columns={
        "utc_timestamp": "date",
        LOAD_COLUMN: "load_mw",
    })
    df = df.set_index("date").sort_index()
    # Force timezone-naive UTC so temperature/holiday joins work
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    load = df["load_mw"].astype(float)
    load = load[load.notna()]
    load = load[start_date:]
    load.name = "load_mw"
    return load


def aggregate_load(
    hourly: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Return daily mean (GW) and weekly mean (GW) series."""
    hourly = to_naive_index(hourly)

    daily = (hourly.resample("D").mean() / 1000.0).dropna()
    daily.name = "load_gw"

    weekly = (hourly.resample("W").mean() / 1000.0)
    weekly = weekly.asfreq("W")
    weekly = weekly.interpolate("time")
    weekly.name = "load_gw"
    return to_naive_index(daily), to_naive_index(weekly)


def train_test_split_series(
    series: pd.Series,
    test_size: int,
) -> tuple[pd.Series, pd.Series]:
    """Chronological split: last `test_size` observations are test."""
    if test_size <= 0 or test_size >= len(series):
        raise ValueError(f"Invalid test_size={test_size} for length={len(series)}")
    return series.iloc[:-test_size].copy(), series.iloc[-test_size:].copy()


# ---------------------------------------------------------------------------
# Temperature (Open-Meteo)
# ---------------------------------------------------------------------------

def fetch_temperature_daily(
    start_date: str,
    end_date: str,
    latitude: float = 52.52,
    longitude: float = 13.41,
) -> pd.DataFrame:
    """Download daily mean temperature for Berlin from Open-Meteo archive."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Berlin",
    }
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()["daily"]
    temp = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "temperature_2m_mean": data["temperature_2m_mean"],
    }).set_index("date")
    return temp


def make_weekly_temperature(
    temp_daily: pd.DataFrame,
    weekly_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Aggregate daily temperature to weekly features aligned to load index."""
    weekly_index = to_naive_index(weekly_index)
    temp_daily = to_naive_index(temp_daily)
    temp = temp_daily["temperature_2m_mean"].astype(float)

    # Resample first, then align to the electricity weekly index
    weekly_mean = temp.resample("W").mean()
    weekly_min = temp.resample("W").min()
    weekly_max = temp.resample("W").max()

    base_heat, base_cool = 15.5, 22.0
    hdd = np.maximum(base_heat - temp, 0).resample("W").sum()
    cdd = np.maximum(temp - base_cool, 0).resample("W").sum()

    temp_weekly = pd.DataFrame(index=weekly_index)
    temp_weekly["temp_mean"] = weekly_mean.reindex(weekly_index)
    temp_weekly["temp_min"] = weekly_min.reindex(weekly_index)
    temp_weekly["temp_max"] = weekly_max.reindex(weekly_index)
    temp_weekly["heating_degree"] = hdd.reindex(weekly_index)
    temp_weekly["cooling_degree"] = cdd.reindex(weekly_index)
    return temp_weekly


# ---------------------------------------------------------------------------
# Holiday features
# ---------------------------------------------------------------------------

def make_weekly_holiday_features(
    weekly_index: pd.DatetimeIndex,
    country: str = "DE",
) -> pd.DataFrame:
    """Count German public holidays per week."""
    weekly_index = to_naive_index(weekly_index)
    de_holidays = holidays.country_holidays(country, years=range(2015, 2021))
    holiday_dates = pd.to_datetime(list(de_holidays.keys()))

    weekly = pd.DataFrame(index=weekly_index)
    weekly["holiday_days"] = 0
    weekly["has_holiday"] = 0

    for idx in weekly_index:
        # Compare as naive timestamps
        week_start = pd.Timestamp(idx) - pd.Timedelta(days=6)
        week_end = pd.Timestamp(idx)
        if week_start.tz is not None:
            week_start = week_start.tz_localize(None)
        if week_end.tz is not None:
            week_end = week_end.tz_localize(None)
        in_week = holiday_dates[(holiday_dates >= week_start) & (holiday_dates <= week_end)]
        weekly.loc[idx, "holiday_days"] = len(in_week)
        weekly.loc[idx, "has_holiday"] = int(len(in_week) > 0)

    return weekly


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mase(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_train: pd.Series,
    seasonality: int = SEASONALITY,
) -> float:
    """Mean Absolute Scaled Error using seasonal naive in-sample scale."""
    naive_errors = np.abs(
        y_train.iloc[seasonality:].values - y_train.iloc[:-seasonality].values
    )
    scale = naive_errors.mean()
    if scale == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def evaluate_forecast(
    name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_train: pd.Series,
    seasonality: int = SEASONALITY,
) -> dict[str, Any]:
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred, index=y_true.index).astype(float)
    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, seasonality=seasonality),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def save_metrics(metrics: list[dict], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(metrics).sort_values("MASE").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Benchmark models (no leakage — training data only)
# ---------------------------------------------------------------------------

def mean_forecast(train: pd.Series, horizon: int, index: pd.Index) -> pd.Series:
    return pd.Series(train.mean(), index=index[:horizon])


def naive_forecast(train: pd.Series, horizon: int, index: pd.Index) -> pd.Series:
    return pd.Series(train.iloc[-1], index=index[:horizon])


def seasonal_naive_forecast(
    train: pd.Series,
    horizon: int,
    index: pd.Index,
    seasonality: int = SEASONALITY,
) -> pd.Series:
    """Multi-step seasonal naive using calendar lag (52 weeks), recursive when needed."""
    values = []
    combined = train.copy()

    for date in index[:horizon]:
        lag_date = date - pd.DateOffset(weeks=seasonality)
        if lag_date in combined.index:
            val = float(combined.loc[lag_date])
        elif lag_date in train.index:
            val = float(train.loc[lag_date])
        else:
            val = float(train.iloc[-seasonality])

        values.append(val)
        combined = pd.concat([combined, pd.Series([val], index=[date])])

    return pd.Series(values, index=index[:horizon])


def drift_forecast(train: pd.Series, horizon: int, index: pd.Index) -> pd.Series:
    slope = (train.iloc[-1] - train.iloc[0]) / max(len(train) - 1, 1)
    values = train.iloc[-1] + slope * np.arange(1, horizon + 1)
    return pd.Series(values, index=index[:horizon])


def run_benchmarks(
    train: pd.Series,
    test: pd.Series,
) -> dict[str, pd.Series]:
    h = len(test)
    return {
        "mean": mean_forecast(train, h, test.index),
        "naive": naive_forecast(train, h, test.index),
        "seasonal_naive": seasonal_naive_forecast(train, h, test.index),
        "drift": drift_forecast(train, h, test.index),
    }


# ---------------------------------------------------------------------------
# SARIMA / SARIMAX
# ---------------------------------------------------------------------------

def sarima_grid_search(
    train: pd.Series,
    exog_train: pd.DataFrame | None = None,
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, SEASONALITY),
    p_range: range = range(0, 7),
    d_range: range = range(0, 3),
    q_range: range = range(0, 7),
    checkpoint_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Loop all (p,d,q) combinations and select best AIC on training data.
    Saves incremental results to checkpoint_path to survive interruptions.
    """
    results = []
    best_aic = np.inf
    best_params: dict = {}

    combos = [(p, d, q) for p in p_range for d in d_range for q in q_range]
    for p, d, q in tqdm(combos, desc="SARIMA grid search"):
        try:
            model = SARIMAX(
                train,
                exog=exog_train,
                order=(p, d, q),
                seasonal_order=seasonal_order,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=200)
            aic = fit.aic
            results.append({
                "p": p, "d": d, "q": q,
                "AIC": aic,
                "seasonal_order": str(seasonal_order),
                "converged": True,
            })
            if aic < best_aic:
                best_aic = aic
                best_params = {
                    "order": (p, d, q),
                    "seasonal_order": seasonal_order,
                    "AIC": aic,
                }
        except Exception as exc:
            results.append({
                "p": p, "d": d, "q": q,
                "AIC": np.nan,
                "seasonal_order": str(seasonal_order),
                "converged": False,
                "error": str(exc)[:120],
            })

        if checkpoint_path and len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)

    results_df = pd.DataFrame(results)
    if checkpoint_path:
        results_df.to_csv(checkpoint_path, index=False)

    return results_df, best_params


def fit_sarimax(
    train: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    exog_train: pd.DataFrame | None = None,
):
    model = SARIMAX(
        train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=300)


def forecast_sarimax(
    fit,
    steps: int,
    index: pd.Index,
    exog_test: pd.DataFrame | None = None,
    alphas: list[float] | None = None,
) -> dict[str, pd.Series | pd.DataFrame]:
    """Return mean forecast and optional confidence intervals."""
    if alphas is None:
        alphas = [0.05, 0.20]

    fc = fit.get_forecast(steps=steps, exog=exog_test)
    out: dict[str, pd.Series | pd.DataFrame] = {
        "mean": fc.predicted_mean.copy(),
    }
    out["mean"].index = index[:steps]

    for alpha in alphas:
        ci = fc.conf_int(alpha=alpha)
        ci.index = index[:steps]
        pct = int((1 - alpha) * 100)
        out[f"ci_{pct}"] = ci

    return out


# ---------------------------------------------------------------------------
# Feature-based ML (recursive forecast — no leakage)
# ---------------------------------------------------------------------------

def make_supervised_features(
    series: pd.Series,
    exog: pd.DataFrame | None = None,
    max_lag: int = 52,
) -> pd.DataFrame:
    """Build supervised table; lags/rolling use only past target values."""
    df = pd.DataFrame({"load_gw": series})

    for lag in [1, 2, 4, 8, 13, 26, 52]:
        if lag <= max_lag:
            df[f"lag_{lag}"] = df["load_gw"].shift(lag)

    shifted = df["load_gw"].shift(1)
    df["roll_mean_4"] = shifted.rolling(4).mean()
    df["roll_mean_13"] = shifted.rolling(13).mean()
    df["roll_mean_52"] = shifted.rolling(52).mean()

    week = df.index.isocalendar().week.astype(int)
    df["week_of_year"] = week
    df["month"] = df.index.month
    df["year"] = df.index.year
    for k in range(1, 4):
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * week / 52)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * week / 52)

    if exog is not None:
        df = df.join(exog, how="left")

    return df.dropna()


def build_single_feature_row(
    history: pd.Series,
    date: pd.Timestamp,
    exog_row: pd.Series | None = None,
) -> pd.DataFrame:
    """Build one feature row for recursive forecasting (no future load used)."""
    row: dict[str, float] = {}
    for lag in [1, 2, 4, 8, 13, 26, 52]:
        row[f"lag_{lag}"] = float(
            history.iloc[-lag] if len(history) >= lag else history.iloc[-1]
        )

    tail = history.iloc[-52:] if len(history) >= 52 else history
    row["roll_mean_4"] = float(tail.iloc[-4:].mean())
    row["roll_mean_13"] = float(tail.iloc[-13:].mean())
    row["roll_mean_52"] = float(tail.mean())

    week = int(date.isocalendar().week)
    row["week_of_year"] = week
    row["month"] = date.month
    row["year"] = date.year
    for k in range(1, 4):
        row[f"sin_{k}"] = float(np.sin(2 * np.pi * k * week / 52))
        row[f"cos_{k}"] = float(np.cos(2 * np.pi * k * week / 52))

    if exog_row is not None:
        for col in exog_row.index:
            row[col] = float(exog_row[col])

    return pd.DataFrame([row])


def recursive_ml_forecast(
    model,
    history: pd.Series,
    exog_full: pd.DataFrame | None,
    test_index: pd.DatetimeIndex,
    feature_cols: list[str],
) -> pd.Series:
    """
    Multi-step forecast: each step appends prediction (not actual) to history.
    Exogenous variables for future steps must be known (temperature, holidays).
    """
    preds = []
    extended = history.copy()

    for date in tqdm(test_index, desc="Recursive ML forecast"):
        exog_row = exog_full.loc[date] if exog_full is not None and date in exog_full.index else None
        X_row = build_single_feature_row(extended, date, exog_row)
        X_row = X_row[feature_cols]
        y_hat = float(model.predict(X_row)[0])
        preds.append(y_hat)
        extended = pd.concat([extended, pd.Series([y_hat], index=[date])])

    return pd.Series(preds, index=test_index)


def fit_feature_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------

def run_stationarity_tests(series: pd.Series, name: str = "series") -> pd.DataFrame:
    """ADF and KPSS tests on a series."""
    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(series.dropna(), autolag="AIC")
    kpss_stat, kpss_p, _, kpss_crit = kpss(series.dropna(), regression="c", nlags="auto")

    return pd.DataFrame([
        {
            "series": name,
            "test": "ADF",
            "statistic": adf_stat,
            "p_value": adf_p,
            "crit_5pct": adf_crit["5%"],
            "stationary_at_5pct": adf_p < 0.05,
        },
        {
            "series": name,
            "test": "KPSS",
            "statistic": kpss_stat,
            "p_value": kpss_p,
            "crit_5pct": kpss_crit["5%"],
            "stationary_at_5pct": kpss_p >= 0.05,
        },
    ])


def plot_and_save(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def free_memory() -> None:
    """Release memory between heavy Colab sections."""
    gc.collect()


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_processed_weekly(paths: dict[str, Path]) -> pd.DataFrame:
    """Load weekly processed dataset saved by notebook 01."""
    path = paths["processed"] / "weekly_features.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df


def build_weekly_feature_table(
    weekly: pd.Series,
    paths: dict[str, Path],
) -> pd.DataFrame:
    """Fetch temperature + holidays and merge into modelling table."""
    weekly = to_naive_index(weekly)
    start = str(weekly.index.min().date())
    end = str(weekly.index.max().date())

    temp_daily = fetch_temperature_daily(start, end)
    temp_daily = to_naive_index(temp_daily)
    temp_weekly = make_weekly_temperature(temp_daily, weekly.index)
    holiday_weekly = make_weekly_holiday_features(weekly.index)

    feature_df = pd.DataFrame({"load_gw": weekly})
    feature_df = feature_df.join(temp_weekly).join(holiday_weekly)
    feature_df = feature_df.interpolate("time").dropna()

    feature_df.index.name = "date"
    feature_df.to_csv(paths["processed"] / "weekly_features.csv")
    return feature_df
