from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.regression.analysis import (
    run_fixed_effects_regression, 
    interpret_results, 
    run_heterogeneity_analysis, 
    interpret_heterogeneity
)
import logging

logger = logging.getLogger(__name__)

def run_regression_pipeline(transactions_path: str, products_path: str, task: str = "regression"):
    """
    Executes the regression modeling pipeline.
    
    Tasks:
    - 'regression': Base FE model + Log-Linear robustness check.
    - 'heterogeneity': Interaction model by store volume.
    """
    print(f"Executing Task: {task}")
    
    # 1. Pipeline Foundation
    transactions, products = load_data(transactions_path, products_path)
    df = preprocess_data(transactions, products)

    if task == "regression":
        # Standard Linear FE
        print("Running Baseline Linear Fixed-Effects Model...")
        model_linear = run_fixed_effects_regression(df, use_log=False)
        interpret_results(model_linear)
        
        # Robustness Check (Log-Linear)
        print("\n" + "~"*30)
        print("Robustness Check: Log-Linear Specification")
        print("~"*30)
        model_log = run_fixed_effects_regression(df, use_log=True)
        interpret_results(model_log)

    elif task == "heterogeneity":
        print("Running Heterogeneity Analysis (Store Segmentation Interaction)...")
        model_hte, threshold = run_heterogeneity_analysis(df)
        interpret_heterogeneity(model_hte, threshold)
        
    else:
        raise ValueError(f"Unknown task: {task}")
