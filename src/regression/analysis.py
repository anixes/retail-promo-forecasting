import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import logging

logger = logging.getLogger(__name__)

def run_fixed_effects_regression(df: pd.DataFrame, use_log: bool = False):
    """
    Executes a Fixed-Effects (FE) regression with clustered standard errors.
    
    This model isolates the promotional lift by controlling for store-specific 
    time-invariant factors and monthly seasonality. Standard errors are 
    clustered at the store level to account for serial correlation.
    
    Formula: Sales ~ Promo + Price + C(Store_ID) + C(Month)

    Parameters
    ----------
    df : pd.DataFrame
        Panel data at the Store-Week level.
    use_log : bool, optional
        Whether to use log-transformed Sales as the dependent variable. 
        Useful for measuring constant percentage effects (elasticity).

    Returns
    -------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted OLS model results.
    """
    logger.info(f"Running Fixed-Effects Regression (Log Specification: {use_log})")
    
    df = df.copy()
    target = "Sales"
    if use_log:
        df["log_sales"] = np.log1p(df["Sales"])
        target = "log_sales"

    # Ensure categorical types for Fixed Effects
    df["Store_ID"] = df["Store_ID"].astype("category")
    df["Month"] = df["Month"].astype("category")

    formula = f"{target} ~ Promo + Price + C(Store_ID) + C(Month)"
    
    model = smf.ols(formula=formula, data=df).fit(
        cov_type="cluster", 
        cov_kwds={"groups": df["Store_ID"]}
    )

    return model

def interpret_results(model):
    """
    Provides business-centric interpretation of regression coefficients.
    
    Parameters
    ----------
    model : statsmodels model
        The fitted regression results.
    """
    coef = model.params
    pvals = model.pvalues
    
    promo_effect = coef.get("Promo", None)
    promo_pval = pvals.get("Promo", None)
    price_effect = coef.get("Price", None)
    
    is_log = "log_sales" in str(model.model.endog_names)

    print("\n" + "="*50)
    print(f"      REGRESSION INTERPRETATION ({'LOG-LINEAR' if is_log else 'LINEAR'})")
    print("="*50)
    print(f"R-squared:        {model.rsquared:.4f}")
    print("-" * 50)

    if promo_effect is not None:
        print(f"Promo Coefficient: {promo_effect:.4f} (p={promo_pval:.4f})")
        
        if is_log:
            pct_change = (np.exp(promo_effect) - 1) * 100
            print(f"Interpretation:    Promotions drive a {pct_change:.1f}% average lift in sales.")
        else:
            print(f"Interpretation:    Promotions drive a {promo_effect:.2f} unit average lift per week.")
            
    if price_effect is not None:
        print(f"Price Elasticity:  {price_effect:.4f} (Sanity: {'VALID' if price_effect < 0 else 'CHECK'})")
    
    print("="*50 + "\n")

def run_heterogeneity_analysis(df: pd.DataFrame):
    """
    Analyzes Heterogeneous Treatment Effects (HTE) by interacting Promo with Store Volume.
    
    Identifies if promotional impact varies by store performance segments 
    (High vs. Low volume).

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.

    Returns
    -------
    model : statsmodels model
    threshold : float
        The median sales threshold used for segmentation.
    """
    logger.info("Conducting Heterogeneity Analysis (Interaction Model)...")
    df = df.copy()
    
    # Store Segmentation
    store_volumes = df.groupby("Store_ID")["Sales"].mean()
    threshold = store_volumes.median()
    
    df["Store_Segment"] = df["Store_ID"].map(
        lambda x: "High" if store_volumes[x] >= threshold else "Low"
    )
    
    formula = "Sales ~ Promo * C(Store_Segment) + Price + C(Store_ID) + C(Month)"
    
    model = smf.ols(formula=formula, data=df).fit(
        cov_type="cluster", 
        cov_kwds={"groups": df["Store_ID"]}
    )
    
    return model, threshold

def interpret_heterogeneity(model, threshold: float):
    """
    Interprets the interaction terms to identify segment-specific lift.
    """
    coef = model.params
    pvals = model.pvalues
    
    # Detect baseline/comparison groups
    interaction_term = "Promo:C(Store_Segment)[T.Low]"
    if interaction_term not in coef.index:
        interaction_term = "Promo:C(Store_Segment)[T.High]"
        baseline, comparison = "Low", "High"
    else:
        baseline, comparison = "High", "Low"

    promo_baseline = coef.get("Promo", 0)
    interaction_coef = coef.get(interaction_term, 0)
    interaction_pval = pvals.get(interaction_term, 1.0)
    
    print("\n" + "="*60)
    print("      HETEROGENEOUS TREATMENT EFFECTS (HTE)")
    print("="*60)
    print(f"Promo Effect ({baseline} Stores):    {promo_baseline:.2f}")
    print(f"Promo Effect ({comparison} Stores):    {promo_baseline + interaction_coef:.2f}")
    print(f"Difference (Interaction p):      {interaction_pval:.4f}")
    
    if interaction_pval < 0.05:
        print("Conclusion: Promotional lift significantly varies by store segment.")
    else:
        print("Conclusion: Promotional lift is robust across store segments.")
    print("="*60 + "\n")
