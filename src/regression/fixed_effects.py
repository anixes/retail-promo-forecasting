import pandas as pd
import statsmodels.formula.api as smf

def run_fixed_effects_regression(df: pd.DataFrame, use_log: bool = False):
    """
    Runs a fixed-effects regression with clustered standard errors.
    Formula: [Sales|Log_Sales] ~ Promo + Price + Store FE + Month FE

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed panel data (Store × Week level)
    use_log : bool
        If True, uses np.log1p(Sales) as the dependent variable for elasticity interpretation.

    Returns
    -------
    model : statsmodels RegressionResults
    """

    # Defensive checks
    target = "Sales"
    if use_log:
        import numpy as np
        df["log_sales"] = np.log1p(df["Sales"])
        target = "log_sales"

    required_cols = ["Sales", "Promo", "Price", "Store_ID", "Month"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Ensure categorical types for Fixed Effects (FE)
    df["Store_ID"] = df["Store_ID"].astype("category")
    df["Month"] = df["Month"].astype("category")

    # Fixed effects regression with Clustered Standard Errors
    # Clustering at Store_ID level corrects for within-store serial correlation
    formula = f"{target} ~ Promo + Price + C(Store_ID) + C(Month)"
    
    model = smf.ols(formula=formula, data=df).fit(
        cov_type="cluster", 
        cov_kwds={"groups": df["Store_ID"]}
    )

    return model

def interpret_results(model):
    """
    Prints key business-relevant interpretations with statistical rigor.
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

    # Core Diagnostics
    print(f"R-squared:        {model.rsquared:.4f}")
    print(f"Condition Number: {model.condition_number:.2e} (Note: High values common with many FEs)")
    print("-" * 50)

    if promo_effect is not None:
        print(f"Promo Coefficient: {promo_effect:.4f}")
        print(f"P-value (Clustered): {promo_pval:.4f}")

        if promo_pval < 0.05:
            print("Status:            Statistically significant result.")
        else:
            print("Status:            NOT statistically significant.")

        print("\nBusiness Narrative:")
        print("Within the same store, comparing promotional vs non-promotional weeks—")
        print("holding price and seasonality constant:")
        
        if is_log:
            # For log-linear, (exp(coef)-1)*100 is the % change
            import numpy as np
            pct_change = (np.exp(promo_effect) - 1) * 100
            print(f"- Promotions are associated with an average increase of {pct_change:.1f}% in sales.")
        else:
            print(f"- Promotions are associated with an average increase of {promo_effect:.2f} units in weekly sales.")
    
    if price_effect is not None:
        print(f"- Price Coefficient: {price_effect:.4f} (Directional Sanity Check: {'PASS' if price_effect < 0 else 'FAIL'})")
    
    print("\nMethodological Note (Inference):")
    print("- Inference uses Clustered Standard Errors (at Store level) to adjust for ")
    print("  within-store serial correlation.")
    print("- Caution: Promotions may be timed during high-demand periods; results ")
    print("  may reflect residual time-varying confounding.")
    print("="*50 + "\n")
