import pandas as pd
from prophet import Prophet

def train_predict_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Trains a Prophet model with custom seasonality and regressors.
    """
    m_df = train_df.rename(columns={'Date': 'ds', 'Sales': 'y'}).copy()
    
    # Initialize with yearly switched on
    model = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=False, 
        daily_seasonality=False,
        seasonality_mode='multiplicative' # Often better for retail spikes
    )
    
    # Add custom monthly seasonality
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Add US Holidays
    model.add_country_holidays(country_name='US')
    
    # Add Promo Intensity
    model.add_regressor('Promo')
    
    # Fit
    model.fit(m_df)
    
    # Predict
    future = test_df.rename(columns={'Date': 'ds'}).copy()
    forecast = model.predict(future)
    
    return forecast['yhat'].values
