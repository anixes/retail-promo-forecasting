import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_predict_lgbm(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list):
    """
    Trains a LightGBM regressor on Log Scale with bias correction.
    
    Adjustments:
    - Target: log1p(Sales)
    - Inverse: expm1(pred + 0.5 * sigma^2)
    - Tuned for smoother function (lower num_leaves, lower lr)
    """
    X_train = train_df[features]
    y_train_raw = train_df['Sales']
    
    # 1. Log Transformation
    y_train_log = np.log1p(y_train_raw)
    X_test = test_df[features]
    
    # 2. Optimized Hyperparams for Log Scale (Conservative)
    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03, # Slower learning for smoother fit
        num_leaves=20,      # Reduced from 31 to prevent overfit to noise
        min_child_samples=20,
        importance_type='gain',
        random_state=42,
        verbosity=-1
    )
    
    # Fit
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_train, y_train_log)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # 3. Predict on Log Scale
    preds_log = model.predict(X_test)
    
    # 4. Bias Correction (using residual variance from training set)
    train_preds_log = model.predict(X_train)
    residuals_log = y_train_log - train_preds_log
    res_var = np.var(residuals_log)
    
    # 5. Inverse Transformation with Bias Correction
    # E[y] = exp(mu + 0.5 * sigma^2) - 1
    preds_final = np.expm1(preds_log + 0.5 * res_var)
    
    # Safety Check: Guarantee positive consumption
    preds_final = np.maximum(preds_final, 0)
    
    return preds_final, model

def plot_feature_importance(model, features: list, save_path: str = None):
    """
    Visualizes feature importance.
    """
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title('LightGBM Feature Importance (Gain) - Log Scale Model')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Feature importance saved to: {save_path}")
    
    plt.close()
    
    return importance_df
