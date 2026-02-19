# ⬡ AURUM — Gold Price Direction Intelligence

> **An end-to-end machine learning system that predicts the directional movement of gold prices using 70 years of financial data, deployed as a production FastAPI service with a live web frontend.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-DagsHub-orange)](https://dagshub.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ElasticNet-red)](https://scikit-learn.org)

---

##  What Is This & Why Does It Matter

Gold is not just a commodity — it is the primary savings instrument for over **300 million Indian households**, particularly in rural and semi-urban areas where access to formal banking and equity markets is limited. For most of these families, gold is their emergency fund, their daughter's wedding fund, and their retirement plan — all in one.

Yet **no accessible, free, data-driven tool** exists to help ordinary people understand where gold prices are heading. Professional traders use Bloomberg terminals. Retail investors use gut instinct and WhatsApp forwards.

**AURUM bridges that gap.**

By combining international gold spot prices, US Federal Reserve interest rate data, USD/INR exchange dynamics, and Indian consumer sentiment signals, AURUM's model learns the macroeconomic patterns that drive gold's directional movement — and surfaces them in a clean, readable interface that anyone can use.

This is not financial advice. It is financial education and decision-support, built with the rigor of production ML and the accessibility of a simple website.

---

##  How It Works — The ML Pipeline

The entire system is built around a **walk-forward cross-validation** approach to avoid data leakage — the model is never evaluated on data it could have seen during training.

### Data Sources (Stage 1 — Ingestion)
| Source | Signal | Provider |
|--------|--------|----------|
| Gold Spot Price (XAU/USD) | Primary price signal | Stooq |
| Federal Funds Rate | US monetary policy tightening/easing | FRED (St. Louis Fed) |
| USD/INR Exchange Rate | Currency pressure on Indian gold prices | Stooq |
| India Gold Import Duty | Policy shock detection | PIB RSS Feed |
| Google Trends — India | Consumer demand sentiment | SerpAPI |

### Feature Engineering (Stage 3 — Feature Store)
37 features are engineered in a **time-series safe** manner — every feature uses only past data (`.shift(1)` minimum) so no future information leaks into the model:

- **Lag features** (1, 2, 3, 6, 12 months) for price, fed rate, and USD/INR
- **Momentum indicators** — price change over 2, 3, 6, 12 month windows
- **Moving averages & volatility** — rolling mean and std over 3, 6, 12 months
- **Rate of change** — percentage change for gold, fed rate, and INR
- **Deviation from MA** — how far current price is from its 6-month average
- **Interaction features** — gold × fed rate, gold × INR (cross-asset signals)

### Model (Stage 4 — Training)
- **Algorithm**: Logistic Regression with ElasticNet regularisation (`l1_ratio=0.7`, `C=0.5`)
- **Solver**: SAGA (handles ElasticNet, scales to large datasets)
- **Class Balancing**: SMOTE (Synthetic Minority Oversampling) at 0.9 ratio
- **Evaluation**: Walk-forward validation — train on 1960–1980, test year-by-year from 1981–2026
- **Pipeline**: `sklearn.pipeline.Pipeline(StandardScaler → LogisticRegression)` — scaler is saved with the model, preventing inference-time scaling errors

### Experiment Tracking
Every training run is logged to **DagsHub MLflow** with:
- All hyperparameters and config
- Accuracy, ROC-AUC, F1, TP/TN/FP/FN, DOWN Precision
- Confusion matrix and classification report as JSON artifacts
- The trained model registered in the Model Registry with `@production` alias

---

##  Backend — FastAPI Inference Service

The API follows an **inference-first architecture** — the model is loaded once at startup from the DagsHub Model Registry and held in memory. Prediction requests never trigger retraining.

### Architecture
```
Startup → Load @production model from DagsHub Registry → Keep in memory
                                                              │
POST /predict → Fetch live data (Stage 1+2) → Build features → model.predict() → Log to MLflow
POST /retrain → Full pipeline (all 5 stages) → Register new version → Hot-swap in memory
```

### Key Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check + model info |
| `GET` | `/model/info` | Registry metadata, alias, version |
| `POST` | `/predict` | 7-day directional predictions (~5–10s) |
| `GET` | `/result` | Last prediction (no data fetch) |
| `GET` | `/history` | Past MLflow inference/retrain runs |
| `POST` | `/retrain` | Background retrain + auto-register |
| `GET` | `/retrain/status` | Retrain progress |
| `POST` | `/model/reload` | Hot-swap to latest @production model |

### Tech Stack
```
FastAPI · Uvicorn · MLflow · DagsHub · scikit-learn · imbalanced-learn
pandas · numpy · requests · feedparser · dvc
```

### Running Locally
```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set DagsHub token
export DAGSHUB_USER_TOKEN=your_token_here

# 3. Start server
uvicorn mlflow_api:app --reload

# 4. Open interactive docs
http://localhost:8000/docs
```

### Deployment
Deployed on **Render.com** via `render.yaml`. On startup, the server:
1. Authenticates with DagsHub
2. Downloads the `@production` model from the registry
3. Loads it into memory
4. Begins serving predictions

> **Note for recruiters:** Render free tier spins down after 15 minutes of inactivity. Use the **"Wake Server"** button on the frontend — it will ping the API every 3 seconds until the server responds (typically 30–50 seconds).

---

##  Frontend — AURUM Web Interface

A standalone 3-file web app (`index.html`, `style.css`, `script.js`) — no framework, no build step, no dependencies to install.

### Features
- **Wake Server button** — handles Render cold starts gracefully for users
- **Live step-by-step status** — shows exactly what the API is doing while waiting
- **7-day prediction cards** — each day shows date, gold price in INR (10g, 24k), UP/DOWN direction, arrow, confidence percentage, and an animated confidence bar
- **Price in Indian Rupees** — converts international USD/oz spot price to INR/10g with GST, using live USD/INR rate from Frankfurter API
- **Price trend chart** — Chart.js line chart with color-coded data points (green=UP, red=DOWN)
- **MLflow run link** — every prediction links directly to its DagsHub experiment run for full transparency
- **Model info badge** — shows active model version and alias in the header

### Price Calculation
```
INR per 10g = (USD/oz ÷ 31.1035) × 10 × live_USD_INR × 1.03 (GST)
```
Note: This reflects the international spot price + GST. Indian MCX retail prices may differ slightly due to import duty already embedded in domestic rates and local premiums.

### Design
Dark luxury editorial aesthetic — built to feel like a professional financial terminal, not a toy project. Cormorant Garamond serif for display, JetBrains Mono for data. Gold accent palette on deep black.

---

##  Project Structure

```
Backend_Deploy/
├── mlflow_full_pipeline.py   # Full 5-stage ML pipeline (training)
├── mlflow_api.py             # FastAPI inference service
├── requirements.txt          # Python dependencies
├── render.yaml               # Render.com deployment config

Frontend_Deploy/
├── index.html                # Frontend — structure
├── style.css                 # Frontend — styling
└── script.js                 # Frontend — logic & API calls
```

---

##  Design Decisions Worth Noting

**Why Logistic Regression over neural networks?**
With 70 years of monthly-frequency financial data, deep learning would overfit severely. ElasticNet-regularised logistic regression gives interpretable, stable predictions and generalises well under walk-forward validation.

**Why walk-forward validation?**
Standard k-fold cross-validation on time series data causes data leakage — the model sees future information during training. Walk-forward replicates real-world deployment: train on past, test on next unseen period.

**Why Pipeline(scaler + model)?**
Saving only the model and applying the scaler separately during inference is a common production bug. Wrapping both in `sklearn.pipeline.Pipeline` ensures the exact same transformation is always applied — at training time and inference time.

**Why inference-first architecture?**
Running the full training pipeline on every API request would take 10–30 minutes. The registered model is loaded once at startup and held in memory — predictions take 5–10 seconds (mostly data fetching).

**Why MLflow + DagsHub?**
Complete experiment reproducibility. Every prediction, every retrain, every metric is logged. You can compare model versions, detect drift, and roll back — all from the DagsHub UI.

---

##  Societal Impact

| Who Benefits | How |
|---|---|
| **Rural households** | Data-driven signal before buying/selling gold jewellery |
| **Small jewellers** | Inventory timing informed by directional predictions |
| **First-generation investors** | Free, accessible alternative to expensive advisory services |
| **Students & researchers** | Open demonstration of production ML on real financial data |
| **Gold loan borrowers** | Better timing for pledging or redeeming gold assets |

India imports approximately **800–900 tonnes of gold per year** — the second largest in the world. Even marginal improvements in timing decisions, aggregated across millions of households, represent significant real-world economic value.

AURUM does not replace a financial advisor. It democratises access to the same kind of data-driven signals that institutional investors take for granted.

---

##  Disclaimer

This project is for **educational and research purposes only**. Predictions are directional signals based on historical patterns and do not constitute financial advice. Past model performance does not guarantee future accuracy. Always consult a qualified financial advisor before making investment decisions.

---

##  Author

**Garvit**
- DagsHub: [garvitwork/gold_backend](https://dagshub.com/garvitwork/gold_backend)
- MLflow Experiments: [View on DagsHub](https://dagshub.com/garvitwork/gold_backend.mlflow)

---

*Built with 70 years of data, production ML discipline, and the belief that financial intelligence should be accessible to everyone — not just those who can afford a Bloomberg terminal.*
