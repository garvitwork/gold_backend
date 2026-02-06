"""
FastAPI Application for Gold Price Prediction Pipeline
=======================================================
Complete API implementation with all pipeline functionalities:
- Data fetching from multiple sources
- Feature engineering and model training
- 7-day predictions with Monte Carlo simulation
- Historical data exports
- Health checks and status monitoring

Deploy on Render.com
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import requests
import feedparser
from datetime import datetime, date
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings
import os
import json
from pathlib import Path
import asyncio
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
FRED_API_KEY = os.getenv("FRED_API_KEY", "7401d8c19460d9721dd2bcc51ad42e80")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "ef9787f70e3e786621da6514eac4a99c3301a5b691c1b6c5d8a6a397cd16ec98")

TRAIN_START_YEAR = 1960
TRAIN_END_YEAR = 1980
TEST_START_YEAR = 1981
LAG_PERIODS = [3, 6, 12, 20]
USE_SMOTE = False
SMOTE_RATIO = 0.9

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
trained_model = None
trained_scaler = None
trained_features = None
last_training_time = None
df_engineered_cache = None

# Create data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Gold Price Prediction API",
    description="Complete pipeline for gold price prediction with 7-day forecasts",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_trained: bool
    last_training: Optional[str]

class PredictionResponse(BaseModel):
    date: str
    day: str
    predicted_movement: str
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
    training_samples: int
    test_samples: int

class TrainingStatus(BaseModel):
    status: str
    message: str
    started_at: Optional[str]
    completed_at: Optional[str]
    metrics: Optional[ModelMetrics]

# ============================================================================
# DATA FETCHING FUNCTIONS
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
    """Fetch all data sources and combine"""
    gold_df = fetch_gold_spot()
    fred_df = fetch_fred_data()
    usdinr_df = fetch_usd_inr()
    trends_df = fetch_gold_trends()
    
    # Prepare datasets
    gold_daily = gold_df.rename(columns={'Date': 'date', 'Close': 'gold_price_usd'})[['date', 'gold_price_usd']]
    gold_daily['date'] = pd.to_datetime(gold_daily['date'])
    
    fred_daily = fred_df.rename(columns={'value': 'fed_funds_rate'})[['date', 'fed_funds_rate']]
    fred_daily['date'] = pd.to_datetime(fred_daily['date'])
    
    usdinr_daily = usdinr_df.rename(columns={'close': 'usd_inr_rate'})[['date', 'usd_inr_rate']]
    usdinr_daily['date'] = pd.to_datetime(usdinr_daily['date'])
    
    trends_daily = trends_df.rename(columns={
        'gold jewellery': 'gold_jewellery_trend',
        'gold price': 'gold_price_trend'
    })
    trends_daily['date'] = pd.to_datetime(trends_daily['date'])
    trends_daily = trends_daily.set_index('date').resample('D').ffill().reset_index()
    
    # Merge
    combined_df = gold_daily.merge(fred_daily, on='date', how='left')
    combined_df = combined_df.merge(usdinr_daily, on='date', how='left')
    combined_df = combined_df.merge(trends_daily, on='date', how='left')
    
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    combined_df = combined_df.ffill().bfill()
    
    return combined_df

def add_target_classes(df):
    """Add target column with classes"""
    df = df.copy()
    df['target'] = 0
    df['price_change'] = df['gold_price_usd'].diff()
    
    df.loc[df['price_change'] > 0, 'target'] = 2
    df.loc[df['price_change'] < 0, 'target'] = 1
    df.loc[df['price_change'] == 0, 'target'] = 0
    
    df = df.drop('price_change', axis=1)
    return df

def create_features_no_leakage(df):
    """Create time-series safe features"""
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

def train_models_pipeline(df_engineered):
    """Train models using walk-forward validation"""
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
    
    return results, final_scaler, final_ensemble, final_feature_cols

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

def generate_7day_predictions_monte_carlo(df_engineered, scaler, ensemble, feature_cols):
    """Generate 7-day predictions using Monte Carlo"""
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
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_trained": trained_model is not None,
        "last_training": last_training_time.isoformat() if last_training_time else None
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_trained": trained_model is not None,
        "last_training": last_training_time.isoformat() if last_training_time else None
    }

@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    """Train the model with full pipeline"""
    global trained_model, trained_scaler, trained_features, last_training_time, df_engineered_cache
    
    try:
        start_time = datetime.now()
        
        # Fetch data
        df_combined = create_combined_daily_dataframe()
        df_with_target = add_target_classes(df_combined)
        df_engineered = create_features_no_leakage(df_with_target)
        
        # Train
        results, scaler, ensemble, feature_cols = train_models_pipeline(df_engineered)
        
        # Calculate metrics
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
        
        # Store trained model
        trained_model = ensemble
        trained_scaler = scaler
        trained_features = feature_cols
        last_training_time = datetime.now()
        df_engineered_cache = df_engineered
        
        # Save datasets
        df_combined.to_csv(DATA_DIR / '01_combined_daily.csv', index=False)
        df_with_target.to_csv(DATA_DIR / '02_combined_with_target.csv', index=False)
        df_engineered.to_csv(DATA_DIR / '03_engineered_features.csv', index=False)
        
        end_time = datetime.now()
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "metrics": {
                "accuracy": float(acc),
                "roc_auc": float(roc_auc),
                "f1_score": float(f1),
                "up_precision": float(up_precision),
                "down_precision": float(down_precision),
                "training_samples": len(df_engineered[df_engineered['year'].between(TRAIN_START_YEAR, TRAIN_END_YEAR)]),
                "test_samples": len(y_true)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/predict", response_model=List[PredictionResponse])
async def get_predictions():
    """Get 7-day predictions"""
    global trained_model, trained_scaler, trained_features, df_engineered_cache
    
    if trained_model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Call /train first.")
    
    try:
        predictions = generate_7day_predictions_monte_carlo(
            df_engineered_cache, 
            trained_scaler, 
            trained_model, 
            trained_features
        )
        
        # Save predictions
        pd.DataFrame(predictions).to_csv(DATA_DIR / '05_next_7days_predictions.csv', index=False)
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics")
async def get_model_metrics():
    """Get current model performance metrics"""
    if trained_model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Call /train first.")
    
    try:
        # Read saved metrics if available
        if (DATA_DIR / '06_model_summary.csv').exists():
            summary = pd.read_csv(DATA_DIR / '06_model_summary.csv')
            return summary.to_dict('records')[0]
        else:
            return {"message": "Metrics not available. Train model first."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/latest")
async def get_latest_data():
    """Get latest gold price and market data"""
    try:
        df_combined = create_combined_daily_dataframe()
        latest = df_combined.iloc[-1]
        
        return {
            "date": latest['date'].strftime('%Y-%m-%d'),
            "gold_price_usd": float(latest['gold_price_usd']),
            "fed_funds_rate": float(latest['fed_funds_rate']),
            "usd_inr_rate": float(latest['usd_inr_rate']),
            "gold_jewellery_trend": int(latest['gold_jewellery_trend']),
            "gold_price_trend": int(latest['gold_price_trend'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated CSV files"""
    file_path = DATA_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='text/csv'
    )

@app.get("/files")
async def list_files():
    """List all available files"""
    files = []
    for file in DATA_DIR.glob("*.csv"):
        files.append({
            "filename": file.name,
            "size": file.stat().st_size,
            "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
        })
    return {"files": files}

@app.post("/pipeline/full")
async def run_full_pipeline():
    """Run complete pipeline: train + predict"""
    try:
        # Train model
        train_result = await train_model(BackgroundTasks())
        
        # Generate predictions
        predictions = await get_predictions()
        
        return {
            "training": train_result,
            "predictions": predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Get comprehensive system status"""
    return {
        "api_version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "model_status": {
            "trained": trained_model is not None,
            "last_training": last_training_time.isoformat() if last_training_time else None
        },
        "configuration": {
            "train_period": f"{TRAIN_START_YEAR}-{TRAIN_END_YEAR}",
            "test_start": TEST_START_YEAR,
            "lag_periods": LAG_PERIODS,
            "smote_enabled": USE_SMOTE
        },
        "data_sources": {
            "gold": "Stooq",
            "fed_rate": "FRED",
            "usd_inr": "Stooq",
            "trends": "Google Trends (SerpAPI)"
        }
    }

# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("="*80)
    print("Gold Price Prediction API - Starting Up")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Data Directory: {DATA_DIR.absolute()}")
    print("="*80)
    print("\nAPI Endpoints:")
    print("  GET  /          - Health check")
    print("  POST /train     - Train model")
    print("  GET  /predict   - Get 7-day predictions")
    print("  GET  /metrics   - Model performance")
    print("  GET  /data/latest - Latest market data")
    print("  POST /pipeline/full - Run complete pipeline")
    print("  GET  /files     - List available files")
    print("  GET  /download/{filename} - Download CSV")
    print("="*80)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)