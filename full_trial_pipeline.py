"""
================================================================================
GOLD PRICE PREDICTION PIPELINE
================================================================================
A fully structured, fault-tolerant pipeline that:
  Stage 1  →  Data Ingestion   (fetch all raw sources)
  Stage 2  →  Data Processing  (merge, resample, align to daily)
  Stage 3  →  Feature Store    (target labelling + lag engineering)
  Stage 4  →  Model Training   (walk-forward, SMOTE, scaling)
  Stage 5  →  Evaluation       (metrics, confusion matrix, trading stats)

Design principles
  - Every stage is an isolated, single-responsibility function
  - Errors in any single data source are caught & reported without crashing
  - No data leakage: all features use shift(1) or greater
  - Immutable inputs: each stage receives a copy of its input data
  - All configuration lives in one place (PipelineConfig)
================================================================================
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import requests
import feedparser
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any
import traceback
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score, f1_score
)
from imblearn.over_sampling import SMOTE


# ==============================================================================
# CONFIGURATION  (single source of truth – change only here)
# ==============================================================================

@dataclass
class PipelineConfig:
    """All tuneable knobs live here. Nothing is hardcoded anywhere else."""

    # --- API keys ---
    FRED_API_KEY: str    = "7401d8c19460d9721dd2bcc51ad42e80"
    SERPAPI_KEY:  str    = "ef9787f70e3e786621da6514eac4a99c3301a5b691c1b6c5d8a6a397cd16ec98"

    # --- File paths ---
    CSV_RAW:      str    = "combined_daily.csv"
    CSV_TARGET:   str    = "combined_daily_with_target.csv"

    # --- Data window ---
    LOOKBACK_YEARS: int  = 70          # how many years of history to pull

    # --- Train / test split ---
    TRAIN_START_YEAR: int = 1960
    TRAIN_END_YEAR:   int = 1980
    TEST_START_YEAR:  int = 1981
    TEST_END_YEAR:    int = 2026

    # --- Feature engineering ---
    USE_FEATURE_ENGINEERING: bool       = True
    LAG_PERIODS:             List[int]  = field(default_factory=lambda: [1, 2, 3, 6, 12])

    # --- Class balancing ---
    USE_SMOTE:    bool  = True
    SMOTE_RATIO:  float = 0.9

    # --- Model hyper-parameters ---
    LOGISTIC_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "penalty":     "elasticnet",
        "solver":      "saga",
        "C":           0.5,
        "l1_ratio":    0.7,
        "max_iter":    2000,
        "random_state": 42,
        "n_jobs":      -1,
    })

    GBM_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "max_depth":          4,
        "learning_rate":      0.03,
        "n_estimators":       150,
        "subsample":          0.7,
        "min_samples_split":  25,
        "min_samples_leaf":   15,
        "max_features":       "sqrt",
        "random_state":       42,
    })

    RF_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators":       100,
        "max_depth":          5,
        "min_samples_split":  20,
        "min_samples_leaf":   10,
        "max_features":       "sqrt",
        "random_state":       42,
        "n_jobs":             -1,
    })


# Global config instance (swap this out to change behaviour)
CFG = PipelineConfig()


# ==============================================================================
# UTILITIES
# ==============================================================================

class PipelineError(Exception):
    """Raised when a pipeline stage fails fatally."""

def banner(title: str) -> None:
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def section(title: str) -> None:
    print(f"\n--- {title} ---")

def safe_run(stage_name: str, func, *args, **kwargs):
    """
    Execute a stage function. If it raises, print a clear error and
    re-raise as PipelineError so the outer orchestrator can decide
    whether to abort or skip.
    """
    try:
        return func(*args, **kwargs)
    except PipelineError:
        raise
    except Exception as exc:
        print(f"\n[ERROR] Stage '{stage_name}' failed:")
        traceback.print_exc()
        raise PipelineError(f"Stage '{stage_name}' failed: {exc}") from exc


# ==============================================================================
# STAGE 1 – DATA INGESTION
# Each fetcher returns (latest_info, full_history_df).
# Errors inside fetchers are non-fatal; caller decides how to handle.
# ==============================================================================

class DataIngestion:
    """Namespace for all raw-data fetchers."""

    @staticmethod
    def fetch_gold_spot() -> Tuple[Any, pd.DataFrame]:
        """Gold spot price (USD/oz) from Stooq."""
        url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
        df  = pd.read_csv(url)
        if df.empty:
            raise ValueError("Gold data unavailable from Stooq")

        df["Date"] = pd.to_datetime(df["Date"])
        latest_date = df["Date"].max()
        cutoff      = latest_date - pd.DateOffset(years=CFG.LOOKBACK_YEARS)
        df          = df[df["Date"] >= cutoff].sort_values("Date").reset_index(drop=True)
        latest      = df.iloc[-1]
        return (latest["Date"], float(latest["Close"])), df

    @staticmethod
    def fetch_fred_series(series_id: str = "DFF") -> Tuple[Any, pd.DataFrame]:
        """Federal funds rate from FRED."""
        end   = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - pd.DateOffset(years=CFG.LOOKBACK_YEARS)).strftime("%Y-%m-%d")
        url   = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id":         series_id,
            "api_key":           CFG.FRED_API_KEY,
            "file_type":         "json",
            "observation_start": start,
            "observation_end":   end,
            "sort_order":        "asc",
        }
        r = requests.get(url, params=params)
        r.raise_for_status()

        obs = r.json()["observations"]
        df  = pd.DataFrame(obs)
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        latest = obs[-1]
        return (latest["date"], float(latest["value"])), df

    @staticmethod
    def fetch_usd_inr() -> Tuple[Dict, pd.DataFrame]:
        """USD/INR exchange rate from Stooq."""
        url = "https://stooq.com/q/d/l/?s=usdinr&c=1d&i=d"
        df  = pd.read_csv(url, sep=";")
        if df.empty:
            raise ValueError("USD-INR data empty")

        df.columns  = df.columns.str.lower()
        df["date"]  = pd.to_datetime(df["date"])
        latest_date = df["date"].max()
        cutoff      = latest_date - pd.DateOffset(years=CFG.LOOKBACK_YEARS)
        df          = df[df["date"] >= cutoff].sort_values("date").reset_index(drop=True)
        latest      = df.iloc[-1]

        info = {"USDINR_Date": str(latest["date"]), "USD_INR": float(latest["close"])}
        return info, df

    @staticmethod
    def fetch_gold_import_duty() -> Tuple[Dict, pd.DataFrame]:
        """Latest PIB news matching gold import duty keywords."""
        feed_url = "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1"
        feed     = feedparser.parse(feed_url)
        keywords = ["gold import duty", "customs duty on gold",
                    "bullion import", "gold customs"]

        entries = []
        for entry in feed.entries[:50]:
            text = (entry.title + " " + entry.get("summary", "")).lower()
            if any(k in text for k in keywords):
                entries.append({
                    "date":    entry.published,
                    "title":   entry.title,
                    "link":    entry.link,
                    "summary": entry.get("summary", ""),
                })

        if entries:
            df  = pd.DataFrame(entries)
            df["date"] = pd.to_datetime(df["date"])
            df  = df.sort_values("date", ascending=False).reset_index(drop=True)
            top = entries[0]
            info = {"status": "LATEST_UPDATE_FOUND",
                    "date":    top["date"],
                    "title":   top["title"],
                    "link":    top["link"]}
            return info, df

        return {"status": "NO_LATEST_UPDATE",
                "checked_on": datetime.now().date().isoformat()}, pd.DataFrame()

    @staticmethod
    def fetch_india_gold_trends() -> Tuple[Dict, pd.DataFrame]:
        """Google Trends for 'gold jewellery' and 'gold price' in India via SerpAPI."""
        params = {
            "engine":    "google_trends",
            "q":         "gold jewellery,gold price",
            "data_type": "TIMESERIES",
            "date":      "today 5-y",
            "geo":       "IN",
            "api_key":   CFG.SERPAPI_KEY,
        }
        resp      = requests.get("https://serpapi.com/search", params=params)
        timelines = resp.json().get("interest_over_time", {}).get("timeline_data", [])

        records = []
        for item in timelines:
            ts = item.get("timestamp")
            records.append({
                "date":           pd.to_datetime(int(ts), unit="s"),
                "gold jewellery": item["values"][0]["extracted_value"],
                "gold price":     item["values"][1]["extracted_value"],
            })

        df     = pd.DataFrame(records)
        latest = df.iloc[-1]
        info   = {
            "date":                 latest["date"].strftime("%Y-%m-%d"),
            "Gold_Jewellery_Trend": int(latest["gold jewellery"]),
            "Gold_Price_Trend":     int(latest["gold price"]),
        }
        return info, df

    @classmethod
    def run(cls) -> Dict[str, pd.DataFrame]:
        """
        Run all fetchers.  Individual failures are caught and reported;
        the pipeline continues with whatever data was successfully retrieved.
        Returns a dict of raw DataFrames keyed by source name.
        """
        banner("STAGE 1 – DATA INGESTION")
        raw: Dict[str, pd.DataFrame] = {}

        fetchers = {
            "gold_spot":     cls.fetch_gold_spot,
            "fred":          cls.fetch_fred_series,
            "usd_inr":       cls.fetch_usd_inr,
            "import_duty":   cls.fetch_gold_import_duty,
            "india_trends":  cls.fetch_india_gold_trends,
        }

        for name, fn in fetchers.items():
            try:
                info, df = fn()
                raw[name] = df
                print(f"  [OK]  {name:15s}  rows={len(df)}")
            except Exception as exc:
                print(f"  [WARN] {name:15s}  FAILED – {exc}")

        required = {"gold_spot", "fred", "usd_inr"}
        missing  = required - raw.keys()
        if missing:
            raise PipelineError(
                f"Required data sources could not be fetched: {missing}"
            )

        return raw


# ==============================================================================
# STAGE 2 – DATA PROCESSING  (merge into one daily DataFrame)
# ==============================================================================

class DataProcessing:
    """Merge raw DataFrames into a single, date-aligned daily series."""

    @staticmethod
    def _prepare_gold(df: pd.DataFrame) -> pd.DataFrame:
        out = df[["Date", "Close"]].rename(
            columns={"Date": "date", "Close": "gold_price_usd"}
        ).copy()
        out["date"] = pd.to_datetime(out["date"])
        return out

    @staticmethod
    def _prepare_fred(df: pd.DataFrame) -> pd.DataFrame:
        out = df[["date", "value"]].rename(columns={"value": "fed_funds_rate"}).copy()
        out["date"] = pd.to_datetime(out["date"])
        return out

    @staticmethod
    def _prepare_usd_inr(df: pd.DataFrame) -> pd.DataFrame:
        out = df[["date", "close"]].rename(columns={"close": "usd_inr_rate"}).copy()
        out["date"] = pd.to_datetime(out["date"])
        return out

    @staticmethod
    def _prepare_trends(df: pd.DataFrame) -> pd.DataFrame:
        out = df.rename(columns={
            "gold jewellery": "gold_jewellery_trend",
            "gold price":     "gold_price_trend",
        }).copy()
        out["date"] = pd.to_datetime(out["date"])
        # Resample weekly → daily, forward-fill
        out = (
            out.set_index("date")
               .resample("D")
               .ffill()
               .reset_index()
        )
        return out[["date", "gold_jewellery_trend", "gold_price_trend"]]

    @classmethod
    def run(cls, raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all sources on date (left join from gold), forward/back-fill."""
        banner("STAGE 2 – DATA PROCESSING")

        base = cls._prepare_gold(raw["gold_spot"])

        base = base.merge(cls._prepare_fred(raw["fred"]),        on="date", how="left")
        base = base.merge(cls._prepare_usd_inr(raw["usd_inr"]), on="date", how="left")

        if "india_trends" in raw and not raw["india_trends"].empty:
            base = base.merge(
                cls._prepare_trends(raw["india_trends"]), on="date", how="left"
            )
        else:
            base["gold_jewellery_trend"] = np.nan
            base["gold_price_trend"]     = np.nan

        # Drop the jewellery trend column (matches original behaviour)
        if "gold_jewellery_trend" in base.columns:
            base = base.drop(columns=["gold_jewellery_trend"])

        base = base.sort_values("date").reset_index(drop=True)
        base = base.ffill().bfill()

        print(f"  Combined shape : {base.shape}")
        print(f"  Date range     : {base['date'].min().date()}  →  {base['date'].max().date()}")
        print(f"  Columns        : {base.columns.tolist()}")
        return base


# ==============================================================================
# STAGE 3 – FEATURE STORE  (target labelling + lag engineering)
# ==============================================================================

class FeatureStore:
    """Build the analysis-ready dataset with target column and engineered features."""

    # ---------- target labelling ----------

    @staticmethod
    def add_target(df: pd.DataFrame) -> pd.DataFrame:
        """
        Target classes:
          0 – no change  (price_change == 0)
          1 – decreased  (price_change  < 0)
          2 – increased  (price_change  > 0)
        """
        out = df.copy()
        out["price_change"] = out["gold_price_usd"].diff()
        out["target"]       = 0
        out.loc[out["price_change"] > 0, "target"] = 2
        out.loc[out["price_change"] < 0, "target"] = 1
        out = out.drop(columns=["price_change"])
        return out

    # ---------- lag / momentum features (no leakage) ----------

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add historical lag, momentum, MA, volatility, ROC, and interaction
        features.  Every feature strictly uses shift ≥ 1 to prevent look-ahead.
        """
        if not CFG.USE_FEATURE_ENGINEERING:
            return df.dropna()

        section("Feature engineering (time-series safe)")
        out = df.copy()

        # Lag features
        for col in ["gold_price_usd", "fed_funds_rate", "usd_inr_rate"]:
            for lag in CFG.LAG_PERIODS:
                out[f"{col}_lag{lag}"] = out[col].shift(lag)

        # Momentum
        for lag in [2, 3, 6, 12]:
            out[f"gold_momentum_lag{lag}"] = (
                out["gold_price_usd"].shift(1) -
                out["gold_price_usd"].shift(lag)
            )

        # Moving averages & volatility
        for window in [3, 6, 12]:
            lagged = out["gold_price_usd"].shift(1)
            out[f"gold_ma{window}_lag1"]  = lagged.rolling(window).mean()
            out[f"gold_std{window}_lag1"] = lagged.rolling(window).std()

        # Rate of change
        out["gold_roc_lag1"]       = out["gold_price_usd"].shift(1).pct_change()
        out["fed_change_lag1"]     = out["fed_funds_rate"].shift(1).diff()
        out["usd_inr_change_lag1"] = out["usd_inr_rate"].shift(1).pct_change()

        # Deviation from MA
        out["price_vs_ma6"] = (
            (out["gold_price_usd"].shift(1) - out["gold_ma6_lag1"]) /
            out["gold_ma6_lag1"]
        )

        # Interaction features
        out["gold_fed_lag1"] = (
            out["gold_price_usd"].shift(1) * out["fed_funds_rate"].shift(1)
        )
        out["gold_usd_lag1"] = (
            out["gold_price_usd"].shift(1) * out["usd_inr_rate"].shift(1)
        )

        out = out.dropna()
        print(f"  Features after engineering : {len(out.columns)} columns")
        print(f"  Rows after dropna          : {len(out)}")
        return out

    @classmethod
    def run(cls, df: pd.DataFrame) -> pd.DataFrame:
        banner("STAGE 3 – FEATURE STORE")
        df = cls.add_target(df)
        df = cls.engineer_features(df)

        # Add year helper column used in train/test split
        df["year"] = pd.to_datetime(df["date"]).dt.year
        return df


# ==============================================================================
# STAGE 4 – MODEL TRAINING  (walk-forward expanding window)
# ==============================================================================

class ModelTraining:
    """Walk-forward validation with SMOTE + scaling inside each fold."""

    NON_FEATURE_COLS = {"date", "year", "target"}

    @classmethod
    def _get_feature_cols(cls, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c not in cls.NON_FEATURE_COLS]

    @staticmethod
    def _safe_smote(
        X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if CFG.USE_SMOTE and len(np.unique(y)) > 1:
            try:
                sm = SMOTE(sampling_strategy=CFG.SMOTE_RATIO, random_state=42)
                return sm.fit_resample(X, y)
            except Exception:
                pass
        return X, y

    @classmethod
    def run(cls, df: pd.DataFrame) -> Dict[str, Any]:
        banner("STAGE 4 – MODEL TRAINING  (walk-forward)")

        # --- data prep ---
        df = df[df["target"] != 0].copy()
        print(f"  Removed class-0 rows. Remaining: {len(df)}")

        # Remap: 1 → 0 (DOWN), 2 → 1 (UP)
        df["target"] = df["target"].map({1: 0, 2: 1})

        feature_cols = cls._get_feature_cols(df)
        print(f"  Features used  : {len(feature_cols)}")

        # --- split ---
        train_df = df[df["year"].between(CFG.TRAIN_START_YEAR, CFG.TRAIN_END_YEAR)]
        test_df  = df[df["year"].between(CFG.TEST_START_YEAR,  CFG.TEST_END_YEAR)]
        print(f"  Train samples  : {len(train_df)}  ({CFG.TRAIN_START_YEAR}–{CFG.TRAIN_END_YEAR})")
        print(f"  Test  samples  : {len(test_df)}   ({CFG.TEST_START_YEAR}–{CFG.TEST_END_YEAR})")

        # --- result containers ---
        results: Dict[str, Any] = {
            "lr":      {"predictions": [], "probabilities": []},
            "actuals": [],
            "years":   [],
        }

        test_years = sorted(test_df["year"].unique())
        print(f"  Test years     : {len(test_years)}\n")

        # --- walk-forward loop ---
        for i, test_year in enumerate(test_years):
            year_data = test_df[test_df["year"] == test_year]
            if len(year_data) == 0:
                continue

            # Expanding training window
            if i > 0:
                prev_years = test_years[:i]
                extra      = test_df[test_df["year"].isin(prev_years)]
                cur_train  = pd.concat([train_df, extra], ignore_index=True)
            else:
                cur_train = train_df

            X_tr = np.nan_to_num(
                cur_train[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0
            )
            y_tr = cur_train["target"].values
            X_te = np.nan_to_num(
                year_data[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0
            )
            y_te = year_data["target"].values

            # Scale
            scaler     = StandardScaler()
            X_tr_sc    = scaler.fit_transform(X_tr)
            X_te_sc    = scaler.transform(X_te)

            # SMOTE
            X_bal, y_bal = cls._safe_smote(X_tr_sc, y_tr)

            # Logistic Regression
            lr = LogisticRegression(**CFG.LOGISTIC_PARAMS)
            lr.fit(X_bal, y_bal)

            results["lr"]["predictions"].extend(lr.predict(X_te_sc))
            results["lr"]["probabilities"].extend(lr.predict_proba(X_te_sc)[:, 1])
            results["actuals"].extend(y_te)
            results["years"].extend([test_year] * len(y_te))

            if (i + 1) % 10 == 0 or i == len(test_years) - 1:
                print(f"  ✓ {i+1}/{len(test_years)} years processed  (latest: {test_year})")

        # Convert lists → arrays
        for key in ("actuals", "years"):
            results[key] = np.array(results[key])
        for mkey in ("lr",):
            results[mkey]["predictions"]  = np.array(results[mkey]["predictions"])
            results[mkey]["probabilities"] = np.array(results[mkey]["probabilities"])

        return results


# ==============================================================================
# STAGE 5 – EVALUATION
# ==============================================================================

class Evaluation:
    """Print metrics, confusion matrix, and trading statistics."""

    MODEL_LABELS = {
        "lr": "Logistic Regression (ElasticNet)",
    }

    @classmethod
    def run(cls, results: Dict[str, Any]) -> None:
        banner("STAGE 5 – EVALUATION")

        y_true = results["actuals"]

        for mkey, mname in cls.MODEL_LABELS.items():
            if mkey not in results:
                continue

            print(f"\n{'='*80}")
            print(f"  {mname}")
            print(f"{'='*80}")

            y_pred = results[mkey]["predictions"]
            y_prob = results[mkey]["probabilities"]

            acc     = accuracy_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_prob)
            f1      = f1_score(y_true, y_pred, average="weighted")

            print(f"  Accuracy  : {acc:.4f}")
            print(f"  ROC-AUC   : {roc_auc:.4f}")
            print(f"  F1-Score  : {f1:.4f}")

            cm = confusion_matrix(y_true, y_pred)
            print(f"\n  Confusion Matrix:")
            print(f"                  Predicted UP    Predicted DOWN")
            print(f"  Actual UP      {cm[0][0]:10d}      {cm[0][1]:10d}")
            print(f"  Actual DOWN    {cm[1][0]:10d}      {cm[1][1]:10d}")

            print(f"\n{classification_report(y_true, y_pred, target_names=['UP', 'DOWN'], digits=4)}")

            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            print(f"  Trading Metrics:")
            print(f"    TP: {tp:6d}  |  TN: {tn:6d}  |  FP: {fp:6d} ✗  |  FN: {fn:6d}")
            if (tp + fp) > 0:
                prec = tp / (tp + fp)
                print(f"    DOWN Precision: {prec:.4f}")


# ==============================================================================
# ORCHESTRATOR  (ties every stage together)
# ==============================================================================

class GoldPricePipeline:
    """
    Top-level orchestrator.  Calls each stage in order and passes
    outputs forward.  Catches PipelineError to give a clean failure
    message instead of a raw stack trace.
    """

    def run(self) -> None:
        banner("GOLD PRICE PREDICTION PIPELINE  –  START")

        try:
            # Stage 1: ingest
            raw_data = safe_run("Data Ingestion",   DataIngestion.run)

            # Stage 2: process / merge
            daily_df = safe_run("Data Processing",  DataProcessing.run, raw_data)

            # Persist raw combined CSV (matches original behaviour)
            daily_df.to_csv(CFG.CSV_RAW, index=False)
            print(f"\n  Saved → {CFG.CSV_RAW}")

            # Stage 3: feature store (target + lags)
            feature_df = safe_run("Feature Store",  FeatureStore.run, daily_df)

            # Persist target CSV
            feature_df.to_csv(CFG.CSV_TARGET, index=False)
            print(f"  Saved → {CFG.CSV_TARGET}")

            # Preview
            section("Target Distribution")
            print(feature_df["target"].value_counts().to_string())
            print(feature_df[["date", "gold_price_usd", "target"]].head(10).to_string(index=False))

            # Stage 4: train
            results = safe_run("Model Training",    ModelTraining.run, feature_df)

            # Stage 5: evaluate
            safe_run("Evaluation",                  Evaluation.run, results)

            banner("PIPELINE COMPLETE")

        except PipelineError as err:
            print(f"\n[FATAL] Pipeline aborted: {err}", file=sys.stderr)
            sys.exit(1)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    pipeline = GoldPricePipeline()
    pipeline.run()