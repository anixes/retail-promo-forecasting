import pandas as pd
import statsmodels.formula.api as smf
import logging

logger = logging.getLogger(__name__)

def run_heterogeneity_analysis(df: pd.DataFrame):
    """
    Analyzes Heterogeneous Treatment Effects (HTE) by interacting Promo with Store Volume.
    
    This identifies *where* promotions work best (e.g., High vs Low volume stores).
    """
    df = df.copy()
    
    # 1. Store Segmentation
    # We define segments based on historical average volume
    store_volumes = df.groupby("Store_ID")["Sales"].mean()
    threshold = store_volumes.median()
    
    df["Store_Segment"] = df["Store_ID"].map(
        lambda x: "High" if store_volumes[x] >= threshold else "Low"
    )
    
    logger.info(f"Store Segmentation Threshold: {threshold:.2f} units. Segment counts:\n{df['Store_Segment'].value_counts()}")

    # 2. Interaction Model
    # Promo * C(Store_Segment) expands to: Promo + C(Store_Segment) + Promo:C(Store_Segment)
    # C(Store_ID) will absorb the main effect of Store_Segment (since it's time-invariant per store)
    formula = "Sales ~ Promo * C(Store_Segment) + Price + C(Store_ID) + C(Month)"
    
    logger.info("Fitting Heterogeneity Model with Interaction Term...")
    model = smf.ols(formula=formula, data=df).fit(
        cov_type="cluster", 
        cov_kwds={"groups": df["Store_ID"]}
    )
    
    return model, threshold

def interpret_heterogeneity(model, threshold: float):
    """
    Interprets the interaction effects between Promo and Store Segments.
    """
    coef = model.params
    pvals = model.pvalues
    
    # baseline segment is "High" or "Low" depending on alphabetic order (usually "High" is T.High)
    # If "Low" is baseline, Promo is the effect for Low stores.
    # We check the specific interaction term name in model.params.index
    
    interaction_term = "Promo:C(Store_Segment)[T.Low]"
    is_low_baseline = "Promo:C(Store_Segment)[T.High]" in coef.index
    if is_low_baseline:
        interaction_term = "Promo:C(Store_Segment)[T.High]"
        baseline = "Low"
        comparison = "High"
    else:
        baseline = "High"
        comparison = "Low"

    promo_baseline = coef.get("Promo", 0)
    interaction_coef = coef.get(interaction_term, 0)
    interaction_pval = pvals.get(interaction_term, 1.0)
    
    promo_comparison = promo_baseline + interaction_coef

    print("\n" + "="*60)
    print("      HETEROGENEOUS TREATMENT EFFECTS (HTE)")
    print("="*60)
    print(f"Clustered Standard Errors (Store level)")
    print(f"Store Segmentation Utility: Median Threshold = {threshold:.1f}")
    print("-" * 60)
    
    print(f"Promo Effect ({baseline} Volume Stores):    {promo_baseline:.2f}")
    print(f"Promo Effect ({comparison} Volume Stores):    {promo_comparison:.2f}")
    print(f"Difference (Interaction):              {interaction_coef:.2f}")
    print(f"Interaction P-Value:                   {interaction_pval:.4f}")
    
    print("\nBusiness Narrative:")
    if interaction_pval < 0.05:
        print(f"The promotional impact is SIGNIFCANTLY different in {comparison} volume stores.")
        winning_segment = comparison if interaction_coef > 0 else baseline
        print(f"- Recommendation: Prioritize trade spend in {winning_segment} volume locations.")
    else:
        print("The promotional impact is consistent across different store volumes.")
        print("- Recommendation: Unified promotional strategy is likely efficient.")
    
    print("-" * 60)
    print("Note: Main effect of Store_Segment is absorbed by Store Fixed Effects.")
    print("="*60 + "\n")
