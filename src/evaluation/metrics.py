import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred, model_name: str) -> dict:
    """
    Calculates MAE, RMSE, and MAPE metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE calculation (handles divide by zero)
    y_true_safe = np.where(y_true == 0, 1e-9, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    metrics = {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape
    }
    
    return metrics
