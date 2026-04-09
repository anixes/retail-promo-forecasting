import pandas as pd
import numpy as np

def create_forecasting_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time-series features with strict temporal integrity.
    Includes seasonal lags, trend features, and promo dynamics.
    """
    df = df.copy()
    
    # Ensure temporal sort
    df = df.sort_values('Week').reset_index(drop=True)
    
    # 1. SALES LAGS
    # Short-term
    df['Sales_Lag_1'] = df['Sales'].shift(1)
    df['Sales_Lag_2'] = df['Sales'].shift(2)
    df['Sales_Lag_4'] = df['Sales'].shift(4)
    
    # Seasonal Lags (Only if sufficient history exists)
    history_len = len(df)
    if history_len > 13:
        df['Sales_Lag_12'] = df['Sales'].shift(12)
    if history_len > 53:
        df['Sales_Lag_52'] = df['Sales'].shift(52)
        
    # 2. ROLLING SALES
    df['Sales_Roll_4'] = df['Sales'].shift(1).rolling(window=4).mean()
    df['Sales_Roll_8'] = df['Sales'].shift(1).rolling(window=8).mean()
    
    # 3. PROMO DYNAMICS (High Signal)
    df['Promo_Lag_1'] = df['Promo'].shift(1)
    df['Promo_Diff'] = df['Promo'].diff()
    df['Promo_Roll_4'] = df['Promo'].shift(1).rolling(window=4).mean()
    
    # 4. TREND & SEASONALITY
    df['Time_Idx'] = np.arange(len(df))
    
    df['WeekIndex'] = df['Date'].dt.isocalendar().week
    df['MonthIndex'] = df['Date'].dt.month
    
    # Cyclic encoding
    df['Month_Sin'] = np.sin(2 * np.pi * df['MonthIndex'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['MonthIndex'] / 12)
    df['Week_Sin'] = np.sin(2 * np.pi * df['WeekIndex'] / 53)
    df['Week_Cos'] = np.cos(2 * np.pi * df['WeekIndex'] / 53)
    
    return df
