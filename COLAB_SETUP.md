# Colab Setup Guide

## Quick start (Google Colab)

1. Upload your `Assignment` folder to Google Drive:
   ```
   MyDrive/Assignment/
   ├── colab/
   │   ├── utils.py
   │   ├── lstm_utils.py
   │   ├── 01_EDA_and_Benchmarks.ipynb
   │   ├── 02_SARIMA_GridSearch.ipynb
   │   ├── 03_SARIMAX_and_ML.ipynb
   │   └── 04_LSTM_Hourly.ipynb
   └── time_series_60min_singleindex.csv
   ```

2. Open notebooks **in order** in Colab (File → Open notebook → Drive).

3. If your folder path differs, change `PROJECT_ROOT` in each notebook:
   ```python
   PROJECT_ROOT = "/content/drive/MyDrive/Assignment"
   ```

## Run order & runtime

| Notebook | Parts | Runtime | Notes |
|----------|-------|---------|-------|
| `01_EDA_and_Benchmarks` | 1–2 | ~15 min | Fetches temperature from API |
| `02_SARIMA_GridSearch` | 3 | ~45–90 min | Checkpoints every 10 models; skips if already done |
| `03_SARIMAX_and_ML` | 4–5 | ~20 min | Recursive ML forecast |
| `04_LSTM_Hourly` | 6 | ~30–60 min | Checkpoints every 2000 hours |

## Outputs (saved to Drive)

```
outputs/
├── figures/     # All plots for your report
├── metrics/     # CSV comparison tables
├── forecasts/   # Model predictions
└── checkpoints/ # Resume files (grid search, LSTM)
```

## Crash recovery

- **Notebook 02:** Re-run — it detects `sarima_best_params.json` and skips grid search.
- **Notebook 04:** Re-run — it resumes from `lstm_partial_preds.npy` if present.

## Local testing (optional)

```bash
pip install -r requirements.txt
python colab/smoke_test.py
```

## Report & GitHub

- Write the report yourself (assignment penalises AI-generated text).
- Use saved figures and metrics CSVs from `outputs/`.
- Push to GitHub when ready.
