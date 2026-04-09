import numpy as np

def predict_naive_baseline(train_df, test_df):
    """
    Implements the Naive Baseline: y_t = y_{t-1}.
    Provides a minimum benchmark for ML models.
    """
    # The first prediction for test_df is the last observed value in train_df
    last_val = train_df['Sales'].iloc[-1]
    
    # Simple persistence: [last_val, test_1, test_2, ...]
    # For a fair 'next week' forecast, we use the shifted sequence
    preds = [last_val] + test_df['Sales'].iloc[:-1].tolist()
    
    return np.array(preds)
