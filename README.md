# Dominick's Finer Foods: Retail Demand Analysis
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-informational.svg)](https://www.statsmodels.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-informational.svg)](https://lightgbm.readthedocs.io/)
[![Prophet](https://img.shields.io/badge/Prophet-informational.svg)](https://facebook.github.io/prophet/)

This is a case study on modeling retail demand using the Dominick’s Finer Foods dataset. The goal was to combine econometric methods to measure promotional impact with a machine learning pipeline to forecast future sales.

---

## Project Overview
I used a dataset of ~6 million rows of grocery transactions to answer two main questions:
1. **Inference**: How much does a promotion actually increase sales after controlling for different store locations?
2. **Prediction**: Can we beat a simple baseline forecast during a period where promotions are constant?

The main finding was a stable **~1,900 unit lift per promotion** across the chain. However, during the forecasting holdout, the stores were on promotion over 95% of the time. This "promo saturation" meant that the simple historical persistence (the Naive model) performed just as well, if not better, than the ML models.

---

## How it's Structured
The code is modular and can be run via a CLI.

```bash
├── notebooks/
│   ├── 01_Inference_EDA.ipynb      # Analyzing store variance and distributions
│   └── 02_Forecasting_EDA.ipynb    # Visualizing the "promo saturation" problem
├── src/
│   ├── data/                       # Loaders and aggregators
│   ├── features/                   # Lags and seasonal engineering
│   ├── regression/                 # Fixed effects modeling
│   ├── forecasting/                # Baseline, Prophet, and LightGBM engines
│   └── pipelines/                  # Orchestration
└── main.py                         # CLI entry point
```

---

## Methodology

### Measuring Promo Lift (Phase 1)
Instead of just comparing averages, I used **Fixed Effects Regression** (specifically `AbsorbingLS`). This controls for the fact that every store has a different baseline volume. By "absorbing" these differences, the model isolates the actual marginal effect of the promotion itself.

### Forecasting (Phase 2)
I tested LightGBM and Prophet against a 12-week holdout. To keep the tests fair, I implemented a strict temporal leakage check. This ensured that no "future" information (like next week's promotion) was leaked into the historical features.

---

## Key Lessons
The most interesting part of this project wasn't the final accuracy score, but diagnosing why the ML models were hitting a ceiling. 

Because the holdout period had almost constant promotions, there was very little variance in my most predictive feature. When a feature flatlines like that, it's a reminder that even complex models like LightGBM become dependent on autoregressive patterns (simple persistence). For future work, breaking the forecasts down to the store level would likely reintroduce the variance needed for ML to "win" again.

---

## How to Run

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the regression analysis**:
   ```bash
   python main.py --task regression
   ```

3. **Run the forecast comparison**:
   ```bash
   python main.py --task forecasting
   ```

---

### Reflection
> I built this project to move past just "fitting models." The real challenge was in the data cleaning and the diagnostic work—identifying why a simple baseline was so hard to beat and using that to understand the underlying retail regime.
