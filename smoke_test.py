"""Quick local smoke test — verifies core pipeline without full grid search."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "colab"))

from utils import (
    aggregate_load,
    evaluate_forecast,
    load_hourly_load,
    make_supervised_features,
    run_benchmarks,
    train_test_split_series,
    TEST_WEEKS,
)

CSV = ROOT / "time_series_60min_singleindex.csv"


def main():
    print("Loading data...")
    hourly = load_hourly_load(CSV)
    _, weekly = aggregate_load(hourly)
    train, test = train_test_split_series(weekly, TEST_WEEKS)

    print(f"Weekly: {len(weekly)} | Train: {len(train)} | Test: {len(test)}")

    benchmarks = run_benchmarks(train, test)
    metrics = [evaluate_forecast(n, test, p, train) for n, p in benchmarks.items()]
    df = pd.DataFrame(metrics).sort_values("MASE")
    print("\nBenchmark metrics:")
    print(df.round(3).to_string(index=False))

    supervised = make_supervised_features(weekly)
    assert supervised.isna().sum().sum() == 0
    print(f"\nSupervised table: {supervised.shape} — OK")

    # Quick SARIMA fit (single config, not full grid)
    from utils import fit_sarimax, forecast_sarimax
    fit = fit_sarimax(train, (1, 1, 1), (1, 1, 1, 52))
    fc = forecast_sarimax(fit, len(test), test.index)
    m = evaluate_forecast("sarima_quick", test, fc["mean"], train)
    print(f"\nQuick SARIMA MASE: {m['MASE']:.3f}")

    print("\nSmoke test passed")


if __name__ == "__main__":
    main()
