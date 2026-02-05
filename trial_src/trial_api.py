"""
GOLD PRICE PREDICTION API - FastAPI Implementation
===================================================
Complete FastAPI wrapper for the gold price prediction pipeline
Retains all existing functionalities with RESTful endpoints

Version: 1.0 - Production Ready
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum
import pandas as pd
import numpy as np
import requests
import feedparser
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings
import traceback
import os
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Gold Price Prediction API",
    description="Complete production-ready API for gold price prediction with ML ensemble models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION (Same as original)
# ============================================================================

FRED_API_KEY = "7401d8c19460d9721dd2bcc51ad42e80"
SERPAPI_KEY = "ef9787f70e3e786621da6514eac4a99c3301a5b691c1b6c5d8a6a397cd16ec98"

# Training Configuration
TRAIN_START_YEAR = 1960
TRAIN_END_YEAR = 1980
TEST_START_YEAR = 1981

# Feature Engineering
LAG_PERIODS = [3, 6, 12, 20]

# Model Configuration
USE_SMOTE = False
SMOTE_RATIO = 0.9

# Model Parameters
LOGISTIC_PARAMS = {
    'penalty': 'elasticnet',
    'solver': 'saga',
    'C': 0.3,
    'l1_ratio': 0.8,
    'max_iter': 2000,
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced'
}

GBM_PARAMS = {
    'max_depth': 3,
    'learning_rate': 0.02,
    'n_estimators': 100,
    'subsample': 0.6,
    'min_samples_split': 40,
    'min_samples_leaf': 20,
    'max_features': 'sqrt',
    'random_state': 42
}

RF_PARAMS = {
    'n_estimators': 80,
    'max_depth': 4,
    'min_samples_split': 30,
    'min_samples_leaf': 15,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced'
}

# Global variables to store trained model
TRAINED_MODEL = None
TRAINED_SCALER = None
FEATURE_COLUMNS = None
ENGINEERED_DATA = None

# ============================================================================
# PYDANTIC MODELS FOR API
# ============================================================================

class PredictionMovement(str, Enum):
    DOWN = "DOWN"
    UP = "UP"

class DayPrediction(BaseModel):
    date: str
    day: str
    predicted_movement: PredictionMovement
    confidence: float
    prob_down: float
    prob_up: float
    estimated_price: float
    price_change: float

class ModelMetrics(BaseModel):
    accuracy: float
    roc_auc: float
    f1_score: float
    up_precision: float
    down_precision: float
    up_recall: float
    down_recall: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

class PipelineConfig(BaseModel):
    train_start_year: int = Field(default=1960)
    train_end_year: int = Field(default=1980)
    test_start_year: int = Field(default=1981)
    use_smote: bool = Field(default=False)
    lag_periods: List[int] = Field(default=[3, 6, 12, 20])

class PipelineStatus(BaseModel):
    status: str
    message: str
    timestamp: str
    data_points: Optional[int] = None
    features_created: Optional[int] = None

class DatasetInfo(BaseModel):
    filename: str
    rows: int
    columns: int
    date_range_start: str
    date_range_end: str
    columns_list: List[str]

# ============================================================================
# DATA FETCHING FUNCTIONS (Same as original)
# ============================================================================

def fetch_gold_spot():
    """Fetch gold spot price from Stooq"""
    url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
    df = pd.read_csv(url)
    
    if df.empty:
        raise Exception("Gold data unavailable from Stooq")
    
    df['Date'] = pd.to_datetime(df['Date'])
    latest_date = df['Date'].max()
    seventy_years_ago = latest_date - pd.DateOffset(years=70)
    df_filtered = df[df['Date'] >= seventy_years_ago].copy()
    df_filtered = df_filtered.sort_values('Date').reset_index(drop=True)
    
    return df_filtered

def fetch_fred_data(series_id='DFF'):
    """Fetch Federal Funds Rate from FRED"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.DateOffset(years=70)).strftime('%Y-%m-%d')
    
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
        "sort_order": "asc"
    }
    
    r = requests.get(url, params=params)
    r.raise_for_status()
    observations = r.json()["observations"]
    
    df = pd.DataFrame(observations)
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    return df

def fetch_usd_inr():
    """Fetch USD-INR exchange rate"""
    url = "https://stooq.com/q/d/l/?s=usdinr&c=1d&i=d"
    df = pd.read_csv(url, sep=';')
    
    if df.empty:
        raise Exception("USD-INR data empty")
    
    df.columns = df.columns.str.lower()
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    seventy_years_ago = latest_date - pd.DateOffset(years=70)
    df_filtered = df[df['date'] >= seventy_years_ago].copy()
    df_filtered = df_filtered.sort_values('date').reset_index(drop=True)
    
    return df_filtered

def fetch_gold_trends():
    """Fetch India gold demand trends from Google Trends"""
    params = {
        "engine": "google_trends",
        "q": "gold jewellery,gold price",
        "data_type": "TIMESERIES",
        "date": "today 5-y",
        "geo": "IN",
        "api_key": SERPAPI_KEY
    }
    
    response = requests.get("https://serpapi.com/search", params=params)
    data = response.json()
    
    timelines = data.get('interest_over_time', {}).get('timeline_data', [])
    
    records = []
    for item in timelines:
        timestamp = item.get('timestamp')
        records.append({
            'date': pd.to_datetime(int(timestamp), unit='s'),
            'gold jewellery': item['values'][0]['extracted_value'],
            'gold price': item['values'][1]['extracted_value']
        })
    
    df = pd.DataFrame(records)
    return df

def create_combined_daily_dataframe():
    """Fetch all data sources and combine into single daily DataFrame"""
    gold_df = fetch_gold_spot()
    fred_df = fetch_fred_data()
    usdinr_df = fetch_usd_inr()
    trends_df = fetch_gold_trends()
    
    # Prepare Gold data
    gold_daily = gold_df.copy()
    gold_daily = gold_daily.rename(columns={'Date': 'date', 'Close': 'gold_price_usd'})
    gold_daily['date'] = pd.to_datetime(gold_daily['date'])
    gold_daily = gold_daily[['date', 'gold_price_usd']]
    
    # Prepare FRED data
    fred_daily = fred_df.copy()
    fred_daily['date'] = pd.to_datetime(fred_daily['date'])
    fred_daily = fred_daily.rename(columns={'value': 'fed_funds_rate'})
    fred_daily = fred_daily[['date', 'fed_funds_rate']]
    
    # Prepare USD-INR data
    usdinr_daily = usdinr_df.copy()
    usdinr_daily['date'] = pd.to_datetime(usdinr_daily['date'])
    usdinr_daily = usdinr_daily.rename(columns={'close': 'usd_inr_rate'})
    usdinr_daily = usdinr_daily[['date', 'usd_inr_rate']]
    
    # Prepare Trends data
    trends_daily = trends_df.copy()
    trends_daily['date'] = pd.to_datetime(trends_daily['date'])
    trends_daily = trends_daily.rename(columns={
        'gold jewellery': 'gold_jewellery_trend',
        'gold price': 'gold_price_trend'
    })
    trends_daily = trends_daily.set_index('date')
    trends_daily = trends_daily.resample('D').ffill().reset_index()
    
    # Merge all dataframes
    combined_df = gold_daily.copy()
    combined_df = combined_df.merge(fred_daily, on='date', how='left')
    combined_df = combined_df.merge(usdinr_daily, on='date', how='left')
    combined_df = combined_df.merge(trends_daily, on='date', how='left')
    
    # Sort and fill missing values
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    combined_df = combined_df.ffill().bfill()
    
    return combined_df

def add_target_classes(df):
    """Add target column with classes: 0=no change, 1=decrease, 2=increase"""
    df = df.copy()
    df['target'] = 0
    
    df['price_change'] = df['gold_price_usd'].diff()
    
    df.loc[df['price_change'] > 0, 'target'] = 2
    df.loc[df['price_change'] < 0, 'target'] = 1
    df.loc[df['price_change'] == 0, 'target'] = 0
    
    df = df.drop('price_change', axis=1)
    
    return df

# ============================================================================
# FEATURE ENGINEERING (Same as original)
# ============================================================================

def create_features_no_leakage(df):
    """Create time-series safe features using ONLY historical data"""
    df_features = df.copy()
    
    # Lag features
    for col in ['gold_price_usd', 'fed_funds_rate', 'usd_inr_rate']:
        for lag in LAG_PERIODS:
            df_features[f'{col}_lag{lag}'] = df_features[col].shift(lag)
    
    # Momentum
    for lag in [6, 12, 20]:
        df_features[f'gold_momentum_{lag}'] = (
            df_features['gold_price_usd'].shift(3) - 
            df_features['gold_price_usd'].shift(lag)
        )
    
    # Moving averages
    for window in [6, 12, 20]:
        df_features[f'gold_ma{window}_lag3'] = (
            df_features['gold_price_usd'].shift(3).rolling(window).mean()
        )
        
        df_features[f'gold_std{window}_lag3'] = (
            df_features['gold_price_usd'].shift(3).rolling(window).std()
        )
    
    # Rate of change
    df_features['gold_roc_lag3'] = df_features['gold_price_usd'].shift(3).pct_change()
    df_features['fed_change_lag3'] = df_features['fed_funds_rate'].shift(3).diff()
    df_features['usd_inr_change_lag3'] = df_features['usd_inr_rate'].shift(3).pct_change()
    
    # Trend
    df_features['price_trend_12'] = (
        df_features['gold_price_usd'].shift(3).rolling(12).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0, raw=False
        )
    )
    
    # Volatility
    df_features['volatility_20'] = (
        df_features['gold_price_usd'].shift(3).rolling(20).std() / 
        df_features['gold_price_usd'].shift(3).rolling(20).mean()
    )
    
    df_features = df_features.dropna()
    
    return df_features

# ============================================================================
# MODEL TRAINING (Same as original)
# ============================================================================

def train_models(df_engineered):
    """Train models using expanding window walk-forward validation"""
    df_engineered['year'] = pd.to_datetime(df_engineered['date']).dt.year
    
    df_model = df_engineered[df_engineered['target'] != 0].copy()
    df_model['target'] = df_model['target'].map({1: 0, 2: 1})
    
    feature_cols = [col for col in df_model.columns 
                    if col not in ['date', 'year', 'target']]
    
    current_year = datetime.now().year
    train_data = df_model[df_model['year'].between(TRAIN_START_YEAR, TRAIN_END_YEAR)].copy()
    test_data = df_model[df_model['year'].between(TEST_START_YEAR, current_year)].copy()
    
    results = {
        'ensemble': {'predictions': [], 'probabilities': []},
        'actuals': [],
        'years': [],
        'dates': []
    }
    
    test_years = sorted(test_data['year'].unique())
    
    for i, test_year in enumerate(test_years):
        test_year_data = test_data[test_data['year'] == test_year]
        
        if len(test_year_data) == 0:
            continue
        
        if i > 0:
            prev_test_years = test_years[:i]
            additional_train = test_data[test_data['year'].isin(prev_test_years)]
            current_train = pd.concat([train_data, additional_train], ignore_index=True)
        else:
            current_train = train_data
        
        X_train = current_train[feature_cols].values
        y_train = current_train['target'].values
        X_test = test_year_data[feature_cols].values
        y_test = test_year_data['target'].values
        
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if USE_SMOTE and len(np.unique(y_train)) > 1:
            try:
                smote = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            except:
                X_train_balanced, y_train_balanced = X_train_scaled, y_train
        else:
            X_train_balanced, y_train_balanced = X_train_scaled, y_train
        
        lr_model = LogisticRegression(**LOGISTIC_PARAMS)
        lr_model.fit(X_train_balanced, y_train_balanced)
        
        gbm_model = GradientBoostingClassifier(**GBM_PARAMS)
        gbm_model.fit(X_train_balanced, y_train_balanced)
        
        rf_model = RandomForestClassifier(**RF_PARAMS)
        rf_model.fit(X_train_balanced, y_train_balanced)
        
        ensemble = VotingClassifier(
            estimators=[('lr', lr_model), ('gbm', gbm_model), ('rf', rf_model)],
            voting='soft',
            weights=[2.0, 1.0, 1.0]
        )
        ensemble.fit(X_train_balanced, y_train_balanced)
        
        if test_year == test_years[-1]:
            final_scaler = scaler
            final_ensemble = ensemble
            final_feature_cols = feature_cols
        
        results['ensemble']['predictions'].extend(ensemble.predict(X_test_scaled))
        results['ensemble']['probabilities'].extend(ensemble.predict_proba(X_test_scaled)[:, 1])
        results['actuals'].extend(y_test)
        results['years'].extend([test_year] * len(y_test))
        results['dates'].extend(test_year_data['date'].tolist())
    
    for key in results:
        if key == 'ensemble':
            results[key]['predictions'] = np.array(results[key]['predictions'])
            results[key]['probabilities'] = np.array(results[key]['probabilities'])
        elif key not in ['dates']:
            results[key] = np.array(results[key])
    
    return results, final_scaler, final_ensemble, final_feature_cols, df_model

# ============================================================================
# 7-DAY PREDICTIONS (Same as original)
# ============================================================================

def create_forecast_features(df, required_features):
    """Create features for forecasting"""
    try:
        df_feat = df.copy()
        
        for col in ['gold_price_usd', 'fed_funds_rate', 'usd_inr_rate']:
            for lag in LAG_PERIODS:
                df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)
        
        for lag in [6, 12, 20]:
            df_feat[f'gold_momentum_{lag}'] = (
                df_feat['gold_price_usd'].shift(3) - 
                df_feat['gold_price_usd'].shift(lag)
            )
        
        for window in [6, 12, 20]:
            df_feat[f'gold_ma{window}_lag3'] = (
                df_feat['gold_price_usd'].shift(3).rolling(window).mean()
            )
            df_feat[f'gold_std{window}_lag3'] = (
                df_feat['gold_price_usd'].shift(3).rolling(window).std()
            )
        
        df_feat['gold_roc_lag3'] = df_feat['gold_price_usd'].shift(3).pct_change()
        df_feat['fed_change_lag3'] = df_feat['fed_funds_rate'].shift(3).diff()
        df_feat['usd_inr_change_lag3'] = df_feat['usd_inr_rate'].shift(3).pct_change()
        
        df_feat['price_trend_12'] = (
            df_feat['gold_price_usd'].shift(3).rolling(12).apply(
                lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0, raw=False
            )
        )
        
        df_feat['volatility_20'] = (
            df_feat['gold_price_usd'].shift(3).rolling(20).std() / 
            df_feat['gold_price_usd'].shift(3).rolling(20).mean()
        )
        
        return df_feat.dropna()
    except:
        return None

def generate_7day_predictions(df_engineered, scaler, ensemble, feature_cols):
    """Generate realistic 7-day predictions using Monte Carlo simulation"""
    history = df_engineered.tail(100).copy()
    last_date = pd.to_datetime(history['date'].iloc[-1])
    last_gold_price = history['gold_price_usd'].iloc[-1]
    
    recent_prices = history['gold_price_usd'].tail(60)
    daily_returns = recent_prices.pct_change().dropna()
    
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    
    predictions = []
    n_simulations = 10
    
    for day in range(1, 8):
        pred_date = last_date + pd.Timedelta(days=day)
        
        sim_predictions = []
        sim_probabilities = []
        sim_prices = []
        
        for sim in range(n_simulations):
            current_history = history.tail(50).copy()
            
            current_price = last_gold_price
            for d in range(1, day + 1):
                random_return = np.random.normal(mean_return, std_return)
                current_price = current_price * (1 + random_return)
            
            future_row = current_history.iloc[-1].copy()
            future_row['gold_price_usd'] = current_price
            future_row['date'] = pred_date
            
            temp_history = pd.concat([current_history, pd.DataFrame([future_row])], ignore_index=True)
            
            temp_with_features = create_forecast_features(temp_history, feature_cols)
            
            if temp_with_features is None or len(temp_with_features) == 0:
                continue
            
            last_row = temp_with_features.iloc[-1]
            feature_values = []
            for col in feature_cols:
                if col in last_row.index:
                    feature_values.append(last_row[col])
                else:
                    feature_values.append(0.0)
            
            X_pred = np.array([feature_values])
            X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)
            X_pred_scaled = scaler.transform(X_pred)
            
            pred_class = ensemble.predict(X_pred_scaled)[0]
            pred_proba = ensemble.predict_proba(X_pred_scaled)[0]
            
            sim_predictions.append(pred_class)
            sim_probabilities.append(pred_proba)
            sim_prices.append(current_price)
        
        if len(sim_predictions) > 0:
            unique, counts = np.unique(sim_predictions, return_counts=True)
            final_pred = unique[np.argmax(counts)]
            
            avg_proba = np.mean(sim_probabilities, axis=0)
            median_price = np.median(sim_prices)
            
            movement = "DOWN" if final_pred == 0 else "UP"
            confidence = avg_proba[final_pred] * 100
            
            predictions.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'day': f"Day +{day}",
                'predicted_movement': movement,
                'confidence': float(confidence),
                'prob_down': float(avg_proba[0] * 100),
                'prob_up': float(avg_proba[1] * 100),
                'estimated_price': float(median_price),
                'price_change': float(median_price - last_gold_price)
            })
    
    return predictions

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def calculate_metrics(results):
    """Calculate comprehensive model metrics"""
    y_true = results['actuals']
    y_pred = results['ensemble']['predictions']
    y_prob = results['ensemble']['probabilities']
    
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    up_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    down_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    up_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    down_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': float(acc),
        'roc_auc': float(roc_auc),
        'f1_score': float(f1),
        'up_precision': float(up_precision),
        'down_precision': float(down_precision),
        'up_recall': float(up_recall),
        'down_recall': float(down_recall),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'confusion_matrix': cm.tolist()
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """API Health Check"""
    return {
        "status": "running",
        "api": "Gold Price Prediction API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "model_trained": TRAINED_MODEL is not None
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": {
            "trained": TRAINED_MODEL is not None,
            "scaler_available": TRAINED_SCALER is not None,
            "features_available": FEATURE_COLUMNS is not None
        },
        "configuration": {
            "train_start_year": TRAIN_START_YEAR,
            "train_end_year": TRAIN_END_YEAR,
            "test_start_year": TEST_START_YEAR,
            "lag_periods": LAG_PERIODS,
            "use_smote": USE_SMOTE
        }
    }

@app.post("/pipeline/run", tags=["Pipeline"])
async def run_complete_pipeline(background_tasks: BackgroundTasks):
    """
    Execute complete end-to-end pipeline
    - Fetches data from all sources
    - Engineers features
    - Trains models
    - Generates predictions
    - Saves all outputs
    """
    try:
        global TRAINED_MODEL, TRAINED_SCALER, FEATURE_COLUMNS, ENGINEERED_DATA
        
        # Step 1: Fetch data
        df_combined = create_combined_daily_dataframe()
        
        # Step 2: Add targets
        df_with_target = add_target_classes(df_combined)
        
        # Step 3: Engineer features
        df_engineered = create_features_no_leakage(df_with_target)
        
        # Step 4: Train models
        results, scaler, ensemble, feature_cols, df_model = train_models(df_engineered)
        
        # Store trained model globally
        TRAINED_MODEL = ensemble
        TRAINED_SCALER = scaler
        FEATURE_COLUMNS = feature_cols
        ENGINEERED_DATA = df_engineered
        
        # Step 5: Generate 7-day predictions
        predictions_7day = generate_7day_predictions(df_engineered, scaler, ensemble, feature_cols)
        
        # Step 6: Calculate metrics
        metrics = calculate_metrics(results)
        
        # Step 7: Save all datasets
        df_combined.to_csv('01_combined_daily.csv', index=False)
        df_with_target.to_csv('02_combined_with_target.csv', index=False)
        df_engineered.to_csv('03_engineered_features.csv', index=False)
        
        hist_predictions = pd.DataFrame({
            'date': results['dates'],
            'year': results['years'],
            'actual': results['actuals'],
            'predicted': results['ensemble']['predictions'],
            'probability_up': results['ensemble']['probabilities']
        })
        hist_predictions['actual_label'] = hist_predictions['actual'].map({0: 'DOWN', 1: 'UP'})
        hist_predictions['predicted_label'] = hist_predictions['predicted'].map({0: 'DOWN', 1: 'UP'})
        hist_predictions['correct'] = (hist_predictions['actual'] == hist_predictions['predicted']).astype(int)
        hist_predictions.to_csv('04_historical_predictions.csv', index=False)
        
        pd.DataFrame(predictions_7day).to_csv('05_next_7days_predictions.csv', index=False)
        
        summary = {
            'model': 'Ensemble (LR + GBM + RF)',
            'accuracy': metrics['accuracy'],
            'roc_auc': metrics['roc_auc'],
            'f1_score': metrics['f1_score'],
            'up_precision': metrics['up_precision'],
            'down_precision': metrics['down_precision'],
            'training_period': f'{TRAIN_START_YEAR}-{TRAIN_END_YEAR}',
            'test_samples': len(results['actuals']),
            'features_used': len(feature_cols),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        pd.DataFrame([summary]).to_csv('06_model_summary.csv', index=False)
        
        return {
            "status": "success",
            "message": "Pipeline executed successfully",
            "timestamp": datetime.now().isoformat(),
            "data_summary": {
                "total_records": len(df_combined),
                "engineered_records": len(df_engineered),
                "features_created": len(feature_cols),
                "date_range": {
                    "start": df_combined['date'].min().strftime('%Y-%m-%d'),
                    "end": df_combined['date'].max().strftime('%Y-%m-%d')
                }
            },
            "model_metrics": metrics,
            "predictions_7day": predictions_7day,
            "files_saved": [
                "01_combined_daily.csv",
                "02_combined_with_target.csv",
                "03_engineered_features.csv",
                "04_historical_predictions.csv",
                "05_next_7days_predictions.csv",
                "06_model_summary.csv"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}\n{traceback.format_exc()}")

@app.get("/data/fetch", tags=["Data"])
async def fetch_all_data():
    """Fetch data from all sources without training"""
    try:
        df_combined = create_combined_daily_dataframe()
        
        return {
            "status": "success",
            "data_summary": {
                "total_records": len(df_combined),
                "columns": df_combined.columns.tolist(),
                "date_range": {
                    "start": df_combined['date'].min().strftime('%Y-%m-%d'),
                    "end": df_combined['date'].max().strftime('%Y-%m-%d')
                }
            },
            "sample_data": df_combined.tail(10).to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {str(e)}")

@app.get("/data/gold-spot", tags=["Data"])
async def get_gold_spot():
    """Fetch only gold spot prices"""
    try:
        df = fetch_gold_spot()
        return {
            "status": "success",
            "records": len(df),
            "data": df.tail(30).to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch gold spot: {str(e)}")

@app.get("/data/fed-rate", tags=["Data"])
async def get_fed_rate():
    """Fetch Federal Funds Rate"""
    try:
        df = fetch_fred_data()
        return {
            "status": "success",
            "records": len(df),
            "data": df.tail(30).to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Fed rate: {str(e)}")

@app.get("/data/usd-inr", tags=["Data"])
async def get_usd_inr():
    """Fetch USD-INR exchange rate"""
    try:
        df = fetch_usd_inr()
        return {
            "status": "success",
            "records": len(df),
            "data": df.tail(30).to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch USD-INR: {str(e)}")

@app.get("/data/trends", tags=["Data"])
async def get_gold_trends():
    """Fetch gold trends from Google Trends"""
    try:
        df = fetch_gold_trends()
        return {
            "status": "success",
            "records": len(df),
            "data": df.tail(30).to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch trends: {str(e)}")

@app.post("/model/train", tags=["Model"])
async def train_model():
    """Train the model without generating predictions"""
    try:
        global TRAINED_MODEL, TRAINED_SCALER, FEATURE_COLUMNS, ENGINEERED_DATA
        
        df_combined = create_combined_daily_dataframe()
        df_with_target = add_target_classes(df_combined)
        df_engineered = create_features_no_leakage(df_with_target)
        
        results, scaler, ensemble, feature_cols, df_model = train_models(df_engineered)
        
        TRAINED_MODEL = ensemble
        TRAINED_SCALER = scaler
        FEATURE_COLUMNS = feature_cols
        ENGINEERED_DATA = df_engineered
        
        metrics = calculate_metrics(results)
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "features_used": len(feature_cols),
            "training_samples": len(df_model)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@app.get("/model/metrics", tags=["Model"])
async def get_model_metrics():
    """Get trained model metrics"""
    if TRAINED_MODEL is None:
        raise HTTPException(status_code=400, detail="Model not trained. Please run /pipeline/run or /model/train first.")
    
    try:
        df_combined = create_combined_daily_dataframe()
        df_with_target = add_target_classes(df_combined)
        df_engineered = create_features_no_leakage(df_with_target)
        
        results, _, _, _, _ = train_models(df_engineered)
        metrics = calculate_metrics(results)
        
        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate metrics: {str(e)}")

@app.post("/predict/7days", tags=["Predictions"])
async def predict_7_days():
    """Generate 7-day predictions"""
    if TRAINED_MODEL is None or TRAINED_SCALER is None or FEATURE_COLUMNS is None:
        raise HTTPException(status_code=400, detail="Model not trained. Please run /pipeline/run or /model/train first.")
    
    try:
        if ENGINEERED_DATA is None:
            df_combined = create_combined_daily_dataframe()
            df_with_target = add_target_classes(df_combined)
            df_engineered = create_features_no_leakage(df_with_target)
        else:
            df_engineered = ENGINEERED_DATA
        
        predictions = generate_7day_predictions(df_engineered, TRAINED_SCALER, TRAINED_MODEL, FEATURE_COLUMNS)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/files/list", tags=["Files"])
async def list_output_files():
    """List all generated CSV files"""
    files = []
    for filename in ['01_combined_daily.csv', '02_combined_with_target.csv', 
                     '03_engineered_features.csv', '04_historical_predictions.csv',
                     '05_next_7days_predictions.csv', '06_model_summary.csv']:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            files.append({
                "filename": filename,
                "exists": True,
                "rows": len(df),
                "columns": len(df.columns),
                "size_kb": round(os.path.getsize(filename) / 1024, 2)
            })
        else:
            files.append({
                "filename": filename,
                "exists": False
            })
    
    return {
        "status": "success",
        "files": files
    }

@app.get("/files/download/{filename}", tags=["Files"])
async def download_file(filename: str):
    """Download a specific CSV file"""
    allowed_files = [
        '01_combined_daily.csv',
        '02_combined_with_target.csv',
        '03_engineered_features.csv',
        '04_historical_predictions.csv',
        '05_next_7days_predictions.csv',
        '06_model_summary.csv'
    ]
    
    if filename not in allowed_files:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="File not found. Please run the pipeline first.")
    
    return FileResponse(
        path=filename,
        filename=filename,
        media_type='text/csv'
    )

@app.get("/files/preview/{filename}", tags=["Files"])
async def preview_file(filename: str, rows: int = Query(default=10, ge=1, le=100)):
    """Preview contents of a CSV file"""
    allowed_files = [
        '01_combined_daily.csv',
        '02_combined_with_target.csv',
        '03_engineered_features.csv',
        '04_historical_predictions.csv',
        '05_next_7days_predictions.csv',
        '06_model_summary.csv'
    ]
    
    if filename not in allowed_files:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="File not found. Please run the pipeline first.")
    
    try:
        df = pd.read_csv(filename)
        return {
            "status": "success",
            "filename": filename,
            "total_rows": len(df),
            "columns": df.columns.tolist(),
            "preview": df.head(rows).to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")

@app.get("/config", tags=["Configuration"])
async def get_configuration():
    """Get current pipeline configuration"""
    return {
        "training": {
            "start_year": TRAIN_START_YEAR,
            "end_year": TRAIN_END_YEAR,
            "test_start_year": TEST_START_YEAR
        },
        "features": {
            "lag_periods": LAG_PERIODS,
            "use_smote": USE_SMOTE,
            "smote_ratio": SMOTE_RATIO
        },
        "model_params": {
            "logistic_regression": LOGISTIC_PARAMS,
            "gradient_boosting": GBM_PARAMS,
            "random_forest": RF_PARAMS
        }
    }

@app.get("/status", tags=["Status"])
async def get_pipeline_status():
    """Get current pipeline status"""
    return {
        "model_trained": TRAINED_MODEL is not None,
        "scaler_available": TRAINED_SCALER is not None,
        "features_available": FEATURE_COLUMNS is not None,
        "data_available": ENGINEERED_DATA is not None,
        "num_features": len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 0,
        "data_records": len(ENGINEERED_DATA) if ENGINEERED_DATA is not None else 0,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("GOLD PRICE PREDICTION API - Starting Server")
    print("="*80)
    print(f"API Documentation: http://localhost:8000/docs")
    print(f"Alternative Docs: http://localhost:8000/redoc")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)