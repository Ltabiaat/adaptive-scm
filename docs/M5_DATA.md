# M5 dataset — download guide

This project uses the **M5 Forecasting — Accuracy** competition dataset (Walmart hierarchical sales, 2011-2016). It is too large for git (~450 MB compressed, ~2 GB extracted) and is gitignored. Every contributor downloads it locally.

---

## Option 1: Kaggle CLI (recommended)

### 1. Kaggle account + API token

1. Create a free Kaggle account at <https://www.kaggle.com> if you don't have one.
2. Go to the competition page: <https://www.kaggle.com/competitions/m5-forecasting-accuracy>.
3. Click **"Join Competition"** (or "Late Submission") and accept the competition rules. You won't be submitting — this just gates the data download.
4. Go to **Account → Settings → API → Create New Token**. A `kaggle.json` file downloads.
5. Move it to the expected location:

```bash
# macOS / Linux
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

```powershell
# Windows PowerShell
New-Item -ItemType Directory -Force -Path $HOME\.kaggle
Move-Item $HOME\Downloads\kaggle.json $HOME\.kaggle\
```

### 2. Install the CLI and download

```bash
pip install kaggle

# From the repo root
kaggle competitions download -c m5-forecasting-accuracy -p data/raw

cd data/raw
unzip m5-forecasting-accuracy.zip
rm m5-forecasting-accuracy.zip
cd ../..
```

You should now have these files in `data/raw/`:

```
calendar.csv                  # ~103 KB    — date metadata, events, SNAP
sales_train_evaluation.csv    # ~120 MB    — daily sales, 30,490 series × 1,941 days
sales_train_validation.csv    # ~115 MB    — same as eval, one less month (we use eval)
sell_prices.csv               # ~205 MB    — weekly prices per item × store
sample_submission.csv         # ~5 MB      — not used in this project
```

The PRD's data pipeline (Feature 1) reads `sales_train_evaluation.csv`, `calendar.csv`, and `sell_prices.csv`. The other two files are not required but harmless to keep.

---

## Option 2: Manual browser download

1. Go to <https://www.kaggle.com/competitions/m5-forecasting-accuracy/data>.
2. Accept the rules.
3. Click **"Download All"** at the bottom of the data tab.
4. Extract the zip into `data/raw/` so the layout matches Option 1.

---

## Option 3: Mirror (use only if Kaggle is unavailable)

The M5 organizers also published the data through the University of Nicosia / Makridakis open-source repository. URLs change occasionally; search "M5 forecasting dataset open access" if Kaggle is unreachable. Verify file hashes if downloading from a mirror.

---

## Verifying the download

After extraction, sanity-check from the repo root:

```bash
python - <<'EOF'
import pandas as pd
from pathlib import Path

raw = Path("data/raw")

cal = pd.read_csv(raw / "calendar.csv")
prices = pd.read_csv(raw / "sell_prices.csv")
sales = pd.read_csv(raw / "sales_train_evaluation.csv")

print(f"calendar:                {len(cal):>8,} rows")
print(f"sell_prices:             {len(prices):>8,} rows")
print(f"sales_train_evaluation:  {len(sales):>8,} rows  ({sales.shape[1]} cols)")

# Spot-check the product the PRD uses by default
target = sales[(sales["item_id"] == "FOODS_3_090") & (sales["store_id"] == "CA_1")]
print(f"\nFOODS_3_090 @ CA_1:      {len(target)} row(s)  — expect exactly 1")
EOF
```

Expected output (numbers are approximate but rows must match exactly):

```
calendar:                   1,969 rows
sell_prices:            6,841,121 rows
sales_train_evaluation:    30,490 rows  (1,947 cols)

FOODS_3_090 @ CA_1:      1 row(s)  — expect exactly 1
```

If `FOODS_3_090 @ CA_1` returns 0 rows, you have `sales_train_validation.csv` instead of `sales_train_evaluation.csv`. The PRD uses the evaluation set.

---

## Disk space note

Once Phase 1's preprocessor runs, you'll also have `data/processed/FOODS_3_090_CA_1.parquet` (~50 KB). The raw CSVs can be moved to cold storage if disk is tight — preprocessing only needs them once.

---

## License

The M5 dataset is released by Walmart under the competition's terms. See the Kaggle competition page for the full license. The data is not redistributed through this repository.
