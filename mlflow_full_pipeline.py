"""
================================================================================
GOLD PRICE PREDICTION PIPELINE  –  MLflow / DagsHub Edition
================================================================================
All original functionality is preserved.
MLflow tracking is layered on top via a dedicated MLflowTracker helper class.

What gets logged per experiment run:
  • Parameters  : all PipelineConfig fields (model hyper-params, lag periods, …)
  • Metrics     : Accuracy, ROC-AUC, F1, TP/TN/FP/FN, DOWN Precision
  • Artifacts   : combined_daily.csv, combined_daily_with_target.csv,
                  confusion_matrix.json, classification_report.json
  • Tags        : model_type, pipeline_version, run context
================================================================================
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
import os
import json
import pandas as pd
import numpy as np
import requests
import feedparser
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, List, Any
import traceback
import sys

import mlflow
import mlflow.sklearn
import dagshub

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

    # --- DagsHub / MLflow ---
    DAGSHUB_REPO_OWNER: str = "garvitwork"
    DAGSHUB_REPO_NAME:  str = "gold_backend"

    # --- File paths ---
    CSV_RAW:      str    = "combined_daily.csv"
    CSV_TARGET:   str    = "combined_daily_with_target.csv"

    # --- Data window ---
    LOOKBACK_YEARS: int  = 70

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


# Global config instance
CFG = PipelineConfig()


# ==============================================================================
# MLFLOW / DAGSHUB TRACKER  (new – does not touch existing logic)
# ==============================================================================

class MLflowTracker:
    """
    Thin wrapper around MLflow that:
      1. Initialises DagsHub remote tracking once at startup.
      2. Provides helpers to log params, metrics, artifacts and models.
      3. Is injected into each stage so stage code stays clean.

    Usage
    -----
    tracker = MLflowTracker()
    with tracker.start_run("gold_prediction"):
        tracker.log_config(CFG)
        ...
        tracker.log_metrics({"accuracy": 0.91, ...})
        tracker.log_artifact("combined_daily.csv")
        tracker.log_sklearn_model(lr_model, "logistic_regression")
    """

    EXPERIMENT_NAME = "gold_price_prediction"
    PIPELINE_VERSION = "1.0.0"

    def __init__(self):
        self._active_run = None
        self._setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self):
        """Initialise DagsHub and point MLflow at the remote tracking server."""
        try:
            dagshub.init(
                repo_owner=CFG.DAGSHUB_REPO_OWNER,
                repo_name=CFG.DAGSHUB_REPO_NAME,
                mlflow=True,          # auto-sets MLFLOW_TRACKING_URI
            )
            print(f"  [MLflow] DagsHub initialised → "
                  f"https://dagshub.com/{CFG.DAGSHUB_REPO_OWNER}/{CFG.DAGSHUB_REPO_NAME}.mlflow")
        except Exception as exc:
            print(f"  [MLflow][WARN] DagsHub init failed ({exc}). "
                  f"Falling back to local tracking.")
            mlflow.set_tracking_uri("mlruns")

        mlflow.set_experiment(self.EXPERIMENT_NAME)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, run_name: str = None):
        """Context manager: wraps the pipeline in a single MLflow run."""
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._active_run = mlflow.start_run(run_name=run_name)
        mlflow.set_tags({
            "pipeline_version": self.PIPELINE_VERSION,
            "model_type":       "LogisticRegression_ElasticNet",
            "smote_enabled":    str(CFG.USE_SMOTE),
            "feature_eng":      str(CFG.USE_FEATURE_ENGINEERING),
        })
        print(f"  [MLflow] Run started  → {self._active_run.info.run_id}")
        return self._active_run

    def end_run(self):
        mlflow.end_run()
        print(f"  [MLflow] Run ended    → status logged to DagsHub")

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_config(self, cfg: PipelineConfig):
        """Log all scalar PipelineConfig fields as MLflow parameters."""
        params = {
            # data
            "lookback_years":       cfg.LOOKBACK_YEARS,
            "train_start_year":     cfg.TRAIN_START_YEAR,
            "train_end_year":       cfg.TRAIN_END_YEAR,
            "test_start_year":      cfg.TEST_START_YEAR,
            "test_end_year":        cfg.TEST_END_YEAR,
            # feature eng
            "use_feature_engineering": cfg.USE_FEATURE_ENGINEERING,
            "lag_periods":          str(cfg.LAG_PERIODS),
            # class balancing
            "use_smote":            cfg.USE_SMOTE,
            "smote_ratio":          cfg.SMOTE_RATIO,
            # logistic regression
            "lr_penalty":           cfg.LOGISTIC_PARAMS["penalty"],
            "lr_solver":            cfg.LOGISTIC_PARAMS["solver"],
            "lr_C":                 cfg.LOGISTIC_PARAMS["C"],
            "lr_l1_ratio":          cfg.LOGISTIC_PARAMS["l1_ratio"],
            "lr_max_iter":          cfg.LOGISTIC_PARAMS["max_iter"],
        }
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]):
        """Log a flat dict of metric name → float value."""
        mlflow.log_metrics(metrics)

    def log_artifact(self, filepath: str):
        """Log a local file as an MLflow artifact (best-effort)."""
        if os.path.exists(filepath):
            mlflow.log_artifact(filepath)
        else:
            print(f"  [MLflow][WARN] Artifact not found, skipping: {filepath}")

    def log_json_artifact(self, data: Any, filename: str):
        """Serialize data to JSON and log as artifact."""
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        mlflow.log_artifact(filename)
        os.remove(filename)   # clean up local temp file

    def log_sklearn_model(self, model, artifact_path: str):
        """Log a fitted sklearn model so it can be loaded later."""
        mlflow.sklearn.log_model(model, artifact_path)
        print(f"  [MLflow] Model logged → {artifact_path}")

    def log_data_info(self, df: pd.DataFrame, label: str):
        """Log basic dataset statistics as metrics."""
        mlflow.log_metrics({
            f"{label}_rows":    len(df),
            f"{label}_columns": len(df.columns),
        })


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
    try:
        return func(*args, **kwargs)
    except PipelineError:
        raise
    except Exception as exc:
        print(f"\n[ERROR] Stage '{stage_name}' failed:")
        traceback.print_exc()
        raise PipelineError(f"Stage '{stage_name}' failed: {exc}") from exc


# ==============================================================================
# STAGE 1 – DATA INGESTION  (unchanged)
# ==============================================================================

class DataIngestion:

    @staticmethod
    def fetch_gold_spot() -> Tuple[Any, pd.DataFrame]:
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
    def run(cls, tracker: MLflowTracker = None) -> Dict[str, pd.DataFrame]:
        banner("STAGE 1 – DATA INGESTION")
        raw: Dict[str, pd.DataFrame] = {}

        fetchers = {
            "gold_spot":     cls.fetch_gold_spot,
            "fred":          cls.fetch_fred_series,
            "usd_inr":       cls.fetch_usd_inr,
            "import_duty":   cls.fetch_gold_import_duty,
            "india_trends":  cls.fetch_india_gold_trends,
        }

        source_statuses = {}
        for name, fn in fetchers.items():
            try:
                info, df = fn()
                raw[name] = df
                print(f"  [OK]  {name:15s}  rows={len(df)}")
                source_statuses[f"source_{name}_rows"] = len(df)
            except Exception as exc:
                print(f"  [WARN] {name:15s}  FAILED – {exc}")
                source_statuses[f"source_{name}_rows"] = -1   # -1 = failed

        # Log ingestion stats to MLflow
        if tracker:
            mlflow.log_metrics(source_statuses)

        required = {"gold_spot", "fred", "usd_inr"}
        missing  = required - raw.keys()
        if missing:
            raise PipelineError(
                f"Required data sources could not be fetched: {missing}"
            )

        return raw


# ==============================================================================
# STAGE 2 – DATA PROCESSING  (unchanged, tracker injection only)
# ==============================================================================

class DataProcessing:

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
        out = (
            out.set_index("date")
               .resample("D")
               .ffill()
               .reset_index()
        )
        return out[["date", "gold_jewellery_trend", "gold_price_trend"]]

    @classmethod
    def run(cls, raw: Dict[str, pd.DataFrame],
            tracker: MLflowTracker = None) -> pd.DataFrame:
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

        if "gold_jewellery_trend" in base.columns:
            base = base.drop(columns=["gold_jewellery_trend"])

        base = base.sort_values("date").reset_index(drop=True)
        base = base.ffill().bfill()

        print(f"  Combined shape : {base.shape}")
        print(f"  Date range     : {base['date'].min().date()}  →  {base['date'].max().date()}")
        print(f"  Columns        : {base.columns.tolist()}")

        if tracker:
            tracker.log_data_info(base, "processed_daily")
            mlflow.log_params({
                "data_date_min": str(base["date"].min().date()),
                "data_date_max": str(base["date"].max().date()),
            })

        return base


# ==============================================================================
# STAGE 3 – FEATURE STORE  (unchanged, tracker injection only)
# ==============================================================================

class FeatureStore:

    @staticmethod
    def add_target(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["price_change"] = out["gold_price_usd"].diff()
        out["target"]       = 0
        out.loc[out["price_change"] > 0, "target"] = 2
        out.loc[out["price_change"] < 0, "target"] = 1
        out = out.drop(columns=["price_change"])
        return out

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        if not CFG.USE_FEATURE_ENGINEERING:
            return df.dropna()

        section("Feature engineering (time-series safe)")
        out = df.copy()

        for col in ["gold_price_usd", "fed_funds_rate", "usd_inr_rate"]:
            for lag in CFG.LAG_PERIODS:
                out[f"{col}_lag{lag}"] = out[col].shift(lag)

        for lag in [2, 3, 6, 12]:
            out[f"gold_momentum_lag{lag}"] = (
                out["gold_price_usd"].shift(1) -
                out["gold_price_usd"].shift(lag)
            )

        for window in [3, 6, 12]:
            lagged = out["gold_price_usd"].shift(1)
            out[f"gold_ma{window}_lag1"]  = lagged.rolling(window).mean()
            out[f"gold_std{window}_lag1"] = lagged.rolling(window).std()

        out["gold_roc_lag1"]       = out["gold_price_usd"].shift(1).pct_change()
        out["fed_change_lag1"]     = out["fed_funds_rate"].shift(1).diff()
        out["usd_inr_change_lag1"] = out["usd_inr_rate"].shift(1).pct_change()

        out["price_vs_ma6"] = (
            (out["gold_price_usd"].shift(1) - out["gold_ma6_lag1"]) /
            out["gold_ma6_lag1"]
        )

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
    def run(cls, df: pd.DataFrame,
            tracker: MLflowTracker = None) -> pd.DataFrame:
        banner("STAGE 3 – FEATURE STORE")
        df = cls.add_target(df)
        df = cls.engineer_features(df)
        df["year"] = pd.to_datetime(df["date"]).dt.year

        if tracker:
            tracker.log_data_info(df, "feature_store")
            mlflow.log_metric("feature_count", len(df.columns))
            # Log target class distribution
            vc = df["target"].value_counts().to_dict()
            mlflow.log_metrics({
                "target_class_0_count": vc.get(0, 0),
                "target_class_1_count": vc.get(1, 0),
                "target_class_2_count": vc.get(2, 0),
            })

        return df


# ==============================================================================
# STAGE 4 – MODEL TRAINING  (MLflow model logging added)
# ==============================================================================

class ModelTraining:

    NON_FEATURE_COLS = {"date", "year", "target"}

    @classmethod
    def _get_feature_cols(cls, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c not in cls.NON_FEATURE_COLS]

    @staticmethod
    def _safe_smote(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if CFG.USE_SMOTE and len(np.unique(y)) > 1:
            try:
                sm = SMOTE(sampling_strategy=CFG.SMOTE_RATIO, random_state=42)
                return sm.fit_resample(X, y)
            except Exception:
                pass
        return X, y

    @classmethod
    def run(cls, df: pd.DataFrame,
            tracker: MLflowTracker = None) -> Dict[str, Any]:
        banner("STAGE 4 – MODEL TRAINING  (walk-forward)")

        df = df[df["target"] != 0].copy()
        print(f"  Removed class-0 rows. Remaining: {len(df)}")
        df["target"] = df["target"].map({1: 0, 2: 1})

        feature_cols = cls._get_feature_cols(df)
        print(f"  Features used  : {len(feature_cols)}")

        train_df = df[df["year"].between(CFG.TRAIN_START_YEAR, CFG.TRAIN_END_YEAR)]
        test_df  = df[df["year"].between(CFG.TEST_START_YEAR,  CFG.TEST_END_YEAR)]
        print(f"  Train samples  : {len(train_df)}  ({CFG.TRAIN_START_YEAR}–{CFG.TRAIN_END_YEAR})")
        print(f"  Test  samples  : {len(test_df)}   ({CFG.TEST_START_YEAR}–{CFG.TEST_END_YEAR})")

        if tracker:
            mlflow.log_metrics({
                "train_samples": len(train_df),
                "test_samples":  len(test_df),
                "feature_count": len(feature_cols),
            })

        results: Dict[str, Any] = {
            "lr":      {"predictions": [], "probabilities": []},
            "actuals": [],
            "years":   [],
            "last_lr_model": None,   # will hold the last fitted LR (for logging)
        }

        test_years = sorted(test_df["year"].unique())
        print(f"  Test years     : {len(test_years)}\n")

        for i, test_year in enumerate(test_years):
            year_data = test_df[test_df["year"] == test_year]
            if len(year_data) == 0:
                continue

            if i > 0:
                prev_years = test_years[:i]
                extra      = test_df[test_df["year"].isin(prev_years)]
                cur_train  = pd.concat([train_df, extra], ignore_index=True)
            else:
                cur_train = train_df

            X_tr = np.nan_to_num(cur_train[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
            y_tr = cur_train["target"].values
            X_te = np.nan_to_num(year_data[feature_cols].values,  nan=0.0, posinf=0.0, neginf=0.0)
            y_te = year_data["target"].values

            scaler     = StandardScaler()
            X_tr_sc    = scaler.fit_transform(X_tr)
            X_te_sc    = scaler.transform(X_te)

            X_bal, y_bal = cls._safe_smote(X_tr_sc, y_tr)

            lr = LogisticRegression(**CFG.LOGISTIC_PARAMS)
            lr.fit(X_bal, y_bal)

            results["lr"]["predictions"].extend(lr.predict(X_te_sc))
            results["lr"]["probabilities"].extend(lr.predict_proba(X_te_sc)[:, 1])
            results["actuals"].extend(y_te)
            results["years"].extend([test_year] * len(y_te))
            results["last_lr_model"] = Pipeline([("scaler", scaler), ("model", lr)])

            if (i + 1) % 10 == 0 or i == len(test_years) - 1:
                print(f"  ✓ {i+1}/{len(test_years)} years processed  (latest: {test_year})")

        for key in ("actuals", "years"):
            results[key] = np.array(results[key])
        for mkey in ("lr",):
            results[mkey]["predictions"]  = np.array(results[mkey]["predictions"])
            results[mkey]["probabilities"] = np.array(results[mkey]["probabilities"])

        # Log last-fold model to MLflow
        if tracker and results["last_lr_model"] is not None:
            tracker.log_sklearn_model(results["last_lr_model"], "logistic_regression_last_fold")

        return results


# ==============================================================================
# STAGE 5 – EVALUATION  (metrics pushed to MLflow)
# ==============================================================================

class Evaluation:

    MODEL_LABELS = {
        "lr": "Logistic Regression (ElasticNet)",
    }

    @classmethod
    def run(cls, results: Dict[str, Any],
            tracker: MLflowTracker = None) -> None:
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

            report = classification_report(
                y_true, y_pred, target_names=["UP", "DOWN"], digits=4
            )
            print(f"\n{report}")

            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            print(f"  Trading Metrics:")
            print(f"    TP: {tp:6d}  |  TN: {tn:6d}  |  FP: {fp:6d} ✗  |  FN: {fn:6d}")

            down_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            if (tp + fp) > 0:
                print(f"    DOWN Precision: {down_precision:.4f}")

            # ── MLflow logging ──────────────────────────────────────────────
            if tracker:
                prefix = mkey  # "lr"
                tracker.log_metrics({
                    f"{prefix}_accuracy":       acc,
                    f"{prefix}_roc_auc":        roc_auc,
                    f"{prefix}_f1_weighted":    f1,
                    f"{prefix}_tp":             int(tp),
                    f"{prefix}_tn":             int(tn),
                    f"{prefix}_fp":             int(fp),
                    f"{prefix}_fn":             int(fn),
                    f"{prefix}_down_precision": down_precision,
                })

                # Confusion matrix as JSON artifact
                tracker.log_json_artifact(
                    {
                        "model":           mname,
                        "confusion_matrix": cm.tolist(),
                        "labels":          ["UP", "DOWN"],
                    },
                    f"{prefix}_confusion_matrix.json",
                )

                # Classification report as JSON artifact
                from sklearn.metrics import classification_report as cr
                report_dict = cr(
                    y_true, y_pred, target_names=["UP", "DOWN"],
                    digits=4, output_dict=True
                )
                tracker.log_json_artifact(report_dict, f"{prefix}_classification_report.json")


# ==============================================================================
# ORCHESTRATOR  (MLflow run wraps the whole pipeline)
# ==============================================================================

class GoldPricePipeline:

    def run(self) -> None:
        banner("GOLD PRICE PREDICTION PIPELINE  –  START")

        # ── Initialise MLflow tracker ────────────────────────────────────────
        tracker = MLflowTracker()

        try:
            with tracker.start_run(
                run_name=f"gold_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                # Log all config params up-front
                tracker.log_config(CFG)

                # Stage 1
                raw_data = safe_run(
                    "Data Ingestion", DataIngestion.run, tracker
                )

                # Stage 2
                daily_df = safe_run(
                    "Data Processing", DataProcessing.run, raw_data, tracker
                )

                daily_df.to_csv(CFG.CSV_RAW, index=False)
                print(f"\n  Saved → {CFG.CSV_RAW}")
                tracker.log_artifact(CFG.CSV_RAW)

                # Stage 3
                feature_df = safe_run(
                    "Feature Store", FeatureStore.run, daily_df, tracker
                )

                feature_df.to_csv(CFG.CSV_TARGET, index=False)
                print(f"  Saved → {CFG.CSV_TARGET}")
                tracker.log_artifact(CFG.CSV_TARGET)

                # Preview
                section("Target Distribution")
                print(feature_df["target"].value_counts().to_string())
                print(feature_df[["date", "gold_price_usd", "target"]].head(10).to_string(index=False))

                # Stage 4
                results = safe_run(
                    "Model Training", ModelTraining.run, feature_df, tracker
                )

                # Stage 5
                safe_run(
                    "Evaluation", Evaluation.run, results, tracker
                )

                banner("PIPELINE COMPLETE")
                # MLflow run ends automatically when `with` block exits

        except PipelineError as err:
            mlflow.set_tag("pipeline_status", "FAILED")
            mlflow.log_param("failure_reason", str(err))
            tracker.end_run()
            print(f"\n[FATAL] Pipeline aborted: {err}", file=sys.stderr)
            sys.exit(1)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    pipeline = GoldPricePipeline()
    pipeline.run()