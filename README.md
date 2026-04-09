# 🛒 Retail Demand Intelligence: Causal Inference & Forecasting 
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Statsmodels](https://img.shields.io/badge/inference-statsmodels-red.svg)](https://www.statsmodels.org/)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-orange.svg)](https://lightgbm.readthedocs.io/)
[![Prophet](https://img.shields.io/badge/forecasting-prophet-772d2d.svg)](https://facebook.github.io/prophet/)

**A Forensic Case Study on the Dominick’s Finer Foods Dataset**

This project demonstrates a production-grade analytical pipeline that combines **Quasi-Experimental Inference** (to measure truth) with **Machine Learning Forecasting** (to predict future states). It solves the complex retail problem of distinguishing between stable baseline demand and high-volatility promotional shocks.

---

## 📖 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Project Architecture](#-project-architecture)
3. [The Hybrid Methodology](#-the-hybrid-methodology)
4. [Key Insights & Diagnostics](#-key-insights--diagnostics)
5. [Getting Started](#-getting-started)

---

## 📊 Executive Summary
In a analysis of over 6 million rows of grocery transactional data, this project successfully isolated a standard **~1,900 unit lift per promotion** across the Dominick's chain. 

**The Major Technical Discovery**: During the 12-week forecasting holdout, the data entered a **Saturation Regime** (Promo Intensity > 95%). This structural shift rendered exogenous features redundant and proved why, in certain retail environments, **Naive Persistence models** outperform complex ML by capturing the local autoregressive trend.

---

## 🏗 Project Architecture
The project is built as a modular CLI application, ensuring strict separation of concerns and reproducibility.

```bash
├── notebooks/
│   ├── 01_Inference_EDA.ipynb             # Panel Variance & Distribution Proofs
│   └── 02_Forecasting_Diagnostic_EDA.ipynb # Persistence vs ML (Saturation Reveal)
├── src/
│   ├── data/                              # Preprocessing & Multi-threaded Loaders
│   ├── features/                          # Seasonal Lags ($Lag_{52}$) & Promo Dynamics
│   ├── regression/                        # High-Dim Fixed Effects (Absorbing LS)
│   ├── forecasting/                       # Benchmarking Engines (Prophet, LGBM, Naive)
│   └── pipelines/                         # Orchestrating Inference & Prediction
└── main.py                                # CLI Interface
```

---

## 🧪 The Hybrid Methodology

### Phase 1: Causal Inference (Phase 1)
Instead of a simple group comparison, we used **Absorbing Fixed Effects Regression** at the store-level.
- **Why?** It allowed us to "absorb" the unobserved heterogeneity of 80+ stores, isolating the pure causal lift of the promotion.
- **Result**: Proved a homogeneous lift of ~1,900 units, justifying a unified chain-wide promotional strategy.

### Phase 2: Demand Forecasting (Phase 2)
We benchmarked state-of-the-art ML models against simple persistence in a 12-week holdout showdown.
- **Why?** To determine if the noise added by promotions is predictable or if simple historical persistence is safer for inventory planning.
- **Validation**: Implemented a strict **Temporal Leakage Test** to ensure future information never contaminated historical lags ($Lag_1 \neq Lag_{0}$).

---

## 💡 Key Insights & Diagnostics

### 🎯 The "Aha!" Moment: Feature Degeneracy
Our forecasting diagnostic revealed that promotions aren't always a useful features. When a store is *permanently* on promotion (as happened in our holdout), the signal has zero variance.
> [!IMPORTANT]
> **Conclusion**: ML failure is often a data regime issue. In high-saturation environments, the project recommends pivoting to **hierarchical store-level forecasting** to reintroduce cross-sectional variance.

---

## 🏃 Getting Started

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Causal Inference (Phase 1)**:
   ```bash
   python main.py --task regression
   python main.py --task heterogeneity
   ```

3. **Execute Forecasting Showdown (Phase 2)**:
   ```bash
   python main.py --task forecasting
   ```

---

## 🎓 Note for Hiring Managers
> *"I built this project to prove that I don't just 'fit models'—I diagnose them. By benchmarking ML against a Naive baseline and identifying the Saturation Regime, I demonstrated an ability to map model performance back to the underlying retail physics. This project is a roadmap for moving from simple 'accuracy' to 'structural understanding.'"*
