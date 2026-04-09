import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import TimeSeriesSplit

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.data.aggregator import aggregate_to_weekly_chain
from src.features.time_series import create_forecasting_features
from src.forecasting.baseline import predict_naive_baseline
from src.forecasting.prophet_model import train_predict_prophet
from src.forecasting.lgbm_model import train_predict_lgbm, plot_feature_importance
from src.evaluation.metrics import calculate_metrics

def run_forecasting_pipeline(transactions_path: str, products_path: str):
    """
    Executes the Upgraded Forecasting Showdown pipeline.
    """
    os.makedirs("reports/figures", exist_ok=True)
    
    # 1. Data Prep
    transactions, products = load_data(transactions_path, products_path)
    df_panel = preprocess_data(transactions, products)
    df_weekly = aggregate_to_weekly_chain(df_panel)
    
    # 2. Advanced Feature Engineering
    df_feat = create_forecasting_features(df_weekly)
    
    # Drop cold-start rows (up to 52 for the longest lag)
    initial_len = len(df_feat)
    df_feat = df_feat.dropna().reset_index(drop=True)
    print(f"Dropped {initial_len - len(df_feat)} cold-start rows.")
    
    # Leakage Test
    lag_1_vals = df_feat['Sales_Lag_1'].iloc[1:].values
    sales_vals = df_feat['Sales'].iloc[:-1].values
    assert np.allclose(lag_1_vals, sales_vals), "LEAKAGE DETECTED: Lag_1 mismatch"
    print("Leakage Tests: PASSED")
    
    # 3. Train/Test Split (12-Week Holdout)
    test_size = 12
    train_df = df_feat.iloc[:-test_size]
    test_df = df_feat.iloc[-test_size:]
    
    print(f"Dataset Split: Train ({len(train_df)} weeks), Test ({len(test_df)} weeks)")
    
    # 4. Modeling
    # Baseline
    preds_naive = predict_naive_baseline(train_df, test_df)
    
    # Prophet
    print("Training Prophet (Multiplicative + Custom Seasonality)...")
    preds_prophet = train_predict_prophet(train_df, test_df)
    
    # LightGBM (Log Scale + Optimized Params)
    lgbm_features = [
        'Price', 'Promo', 'Promo_Lag_1', 'Promo_Diff', 'Promo_Roll_4',
        'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_4', 'Sales_Lag_12',
        'Sales_Roll_4', 'Sales_Roll_8', 'Time_Idx',
        'Month_Sin', 'Month_Cos', 'Week_Sin', 'Week_Cos'
    ]
    # Check if we have Lag_52 (only active if data long enough)
    if 'Sales_Lag_52' in df_feat.columns:
        lgbm_features.append('Sales_Lag_52')
        
    print(f"Training LightGBM on Log Scale with {len(lgbm_features)} features...")
    preds_lgbm, lgbm_model = train_predict_lgbm(train_df, test_df, lgbm_features)
    
    # 5. Segmented Evaluation
    evaluation_results = []
    
    # Identify Promo vs Non-Promo regimes in test set
    promo_threshold = 0.3
    is_promo = test_df['Promo'] > promo_threshold
    
    model_predictions = {
        "Naive Baseline": preds_naive,
        "Facebook Prophet": preds_prophet,
        "LightGBM": preds_lgbm
    }
    
    for name, preds in model_predictions.items():
        # Overall metrics
        overall = calculate_metrics(test_df['Sales'], preds, name)
        
        # Segmented metrics
        promo_mae = np.mean(np.abs(test_df.loc[is_promo, 'Sales'] - preds[is_promo]))
        stable_mae = np.mean(np.abs(test_df.loc[~is_promo, 'Sales'] - preds[~is_promo]))
        
        # Diagnostic: Error Variance
        errors = test_df['Sales'].values - preds
        err_var_promo = np.var(errors[is_promo])
        err_var_stable = np.var(errors[~is_promo])
        
        overall.update({
            'MAE (Promo)': promo_mae,
            'MAE (Stable)': stable_mae,
            'ErrVar (Promo)': err_var_promo,
            'ErrVar (Stable)': err_var_stable
        })
        evaluation_results.append(overall)
        
    results_df = pd.DataFrame(evaluation_results)
    
    print("\n--- UPGRADED FORECASTING PERFORMANCE ---")
    display_cols = ['Model', 'MAE', 'MAE (Promo)', 'MAE (Stable)']
    print(results_df[display_cols].to_string(index=False))
    
    print("\n--- DIAGNOSTIC: ERROR VARIANCE ---")
    var_cols = ['Model', 'ErrVar (Promo)', 'ErrVar (Stable)']
    print(results_df[var_cols].to_string(index=False))
    
    # 6. Visualization
    plt.figure(figsize=(15, 7))
    plt.plot(test_df['Week'], test_df['Sales'], label='Actual Sales', color='black', linewidth=2, marker='o')
    plt.plot(test_df['Week'], preds_naive, label='Naive Baseline', linestyle='--', alpha=0.7)
    plt.plot(test_df['Week'], preds_prophet, label='Prophet', alpha=0.8)
    plt.plot(test_df['Week'], preds_lgbm, label='LightGBM (Log-Scale)', linewidth=2, color='red')
    
    plt.title('Upgraded Forecasting: Log-Scale & Seasonal Lags (12-Week Holdout)')
    plt.xlabel('Week Number')
    plt.ylabel('Total Chain Sales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/upgraded_showdown.png")
    plt.close()
    
    plot_feature_importance(lgbm_model, lgbm_features, save_path="reports/figures/upgraded_lgbm_importance.png")
    
    return results_df
