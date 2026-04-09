# CASE STUDY: Retail Demand & Promo Saturation

## The Challenge
In retail demand forecasting, the "Cold Start" or "Saturation" problem can render advanced ML models useless. This case study demonstrates how to identify these regimes and pivot to econometric inference when predictive signal collapses.

## Phase 0: Data Engineering and Quality Audit
Before modeling, I implemented a robust **Data Integrity Audit**.
*   **Quality Filtering:** Leveraged the `OK` flag to prune suspect data collection periods.
*   **Pipeline Orchestration:** Built a modular CLI (`main.py --task prepare`) to cache cleaned metrics, ensuring raw data remains immutable (Production Best Practice).

## Phase 1: Causal Inference (Fixed Effects)
**Objective:** Measure exactly how many units a promotion adds, independent of which store it runs in.
*   **The Problem:** High-volume stores promo more often (Selection Bias).
*   **The Solution:** Fixed-Effects Regression. By absorbing store-level demand baselines, I isolated a stable, unbiased lift of ~1,900 units.

## Phase 2: Forecasting Failure Post-Mortem
**Objective:** Predict the next 12 weeks of chain-level demand.
*   **Observation:** Naive Persistence (MAE: 1,324) outperformed LightGBM (MAE: 1,410) and Prophet.
*   **Forensic Diagnosis:** The holdout period suffered from **Feature Degeneracy**. With 95% promotional intensity, the ML models lost the variance they needed to learn. 
*   **Technical Insight:** In saturated regimes, the series is dominated by **Autocorrelation**, which simple lags capture better than complex additive models.

## Conclusion and Strategic Pivot
*   **Final Result:** A high-fidelity analytical pipeline that distinguishes between signal and noise.
*   **Business Impact:** This project successfully identified that aggregate-level forecasting has hit a performance ceiling. The recommendation to stakeholders is a pivot to **Hierarchical Store-Level Forecasting**, where localized variance provides the signal required for Machine Learning ROI.
