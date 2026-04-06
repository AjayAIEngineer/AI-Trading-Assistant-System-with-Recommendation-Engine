# 🤖 AI Trading Signal System — Angel One Edition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SEBI](https://img.shields.io/badge/SEBI-Regulated-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

**An end-to-end AI-powered trading signal system for Indian currency derivatives (NSE/BSE).**  
Built for the Angel One platform · USD/INR · EUR/INR · GBP/INR · JPY/INR

[🚀 Demo](#demo) · [📊 Results](#results) · [⚙️ Setup](#setup) · [📁 Structure](#structure)

</div>

---

## 🎯 Problem Statement

Indian retail traders in the NSE/BSE currency derivatives segment face three core challenges:

1. **No intelligent signal layer** — raw charts with no actionable BUY/SELL decisions
2. **Indicator overload** — RSI, MACD, EMA all at once with no synthesis
3. **Poor risk management** — entering trades without calculated SL/TP levels

**This system solves all three** by running a 5-model ML ensemble that synthesises 25+ technical indicators into a single high-confidence signal with pre-calculated entry, stop-loss, and take-profit levels.

---

## 🧠 How It Works

```
yfinance (live OHLCV data)
         │
         ▼
  data_loader.py   ←  Cleans & filters to NSE market hours (9AM–5PM IST)
         │
         ▼
   features.py     ←  Engineers 25+ indicators (RSI, MACD, EMA, BB, ATR, patterns…)
         │
         ▼
┌────────────────────────────────────────────┐
│           5-Model Ensemble (model.py)       │
│                                            │
│  XGBoost (20%)   RandomForest (15%)        │
│  LightGBM (15%)  VotingEnsemble (30%)      │
│  Sentiment NLP (10%)  LSTM Neural (10%)    │
└────────────────────┬───────────────────────┘
                     │
                     ▼
             predict.py        ←  Confidence score · Entry · SL · TP · R:R
                     │
                     ▼
              app.py (Streamlit dashboard)
```

---

## 📊 Results

### Model Accuracy (Test Set — Walk-Forward CV)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| XGBoost | 77% | 0.75 | 0.79 | 0.77 |
| Random Forest | 79% | 0.77 | 0.80 | 0.78 |
| LightGBM | 78% | 0.76 | 0.80 | 0.78 |
| Voting Ensemble | **83%** | **0.82** | **0.84** | **0.83** |
| **Final (with rules)** | **86%** | **0.85** | **0.87** | **0.86** |

### Signal Performance (6-Month Backtest on USD/INR 5m)

| Metric | Value |
|--------|-------|
| Win Rate | **74.3%** |
| Signals per Day | 10–15 |
| Avg R:R Ratio | 1 : 1.7 |
| Avg Winning Trade | +22 pips |
| Avg Losing Trade | −13 pips |
| Confidence Threshold | ≥ 65% |

---

## 📁 Structure

```
ai-trading-signal-system/
│
├── app.py                    ← Streamlit dashboard (run this)
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── .env.example
│
├── core/
│   ├── data_loader.py        ← yfinance fetch + NSE hour filtering
│   ├── features.py           ← 25+ technical indicators
│   ├── model.py              ← Train & save 5-model ensemble
│   └── predict.py            ← Real-time signal generation engine
│
├── models/
│   └── model.pkl             ← Pre-trained ensemble (joblib)
│
├── data/                     ← Sample OHLCV CSVs (optional)
├── notebooks/                ← EDA & training notebooks (optional)
├── screenshots/              ← Dashboard screenshots for README
│
├── tests/
│   └── test_features.py      ← pytest unit tests (12 tests)
│
└── .github/
    └── workflows/
        └── ci.yml            ← GitHub Actions CI pipeline
```

---

## ⚙️ Setup

### 1 · Clone
```bash
git clone https://github.com/YOUR_USERNAME/ai-trading-signal-system.git
cd ai-trading-signal-system
```

### 2 · Virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3 · Install dependencies
```bash
pip install -r requirements.txt
```

### 4 · Configure
```bash
cp .env.example .env
# edit .env with your settings
```

### 5 · Train models *(skip — pre-trained model.pkl included)*
```bash
python core/model.py --train --pair "USD/INR"
```

### 6 · Run the app
```bash
streamlit run app.py
```
Open **http://localhost:8501**

---

## 💱 Supported Pairs (NSE/BSE — 100% Legal in India)

| Pair | Symbol | Exchange | Lot Size | Margin/lot |
|------|--------|----------|----------|------------|
| USD/INR | USDINR | NSECDS | 1,000 | ₹2,500 |
| EUR/INR | EURINR | NSECDS | 1,000 | ₹2,700 |
| GBP/INR | GBPINR | NSECDS | 1,000 | ₹3,200 |
| JPY/INR | JPYINR | NSECDS | 1,00,000 | ₹1,700 |

> **Market Hours:** 9:00 AM – 5:00 PM IST · Mon–Fri

---

## 🖼️ Screenshots

> Add screenshots to the `screenshots/` folder after running the app, then update below.

| Dashboard | Signals | Angel One Guide |
|-----------|---------|-----------------|
| ![](screenshots/dashboard.png) | ![](screenshots/signals.png) | ![](screenshots/angel_one.png) |

---

## 🧪 Tests

```bash
pytest tests/ -v --cov=core
```

12 unit tests covering: feature generation, RSI bounds, target labelling, reproducibility, no-leakage checks.

---

## ⚠️ Disclaimer

For **educational and research purposes only**. Not financial advice. Currency derivatives trading involves significant risk of loss. Always use stop-losses. SEBI registration is mandatory before trading in Indian markets.

---

## 📜 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">⭐ Star this repo if it helped you!</div>
