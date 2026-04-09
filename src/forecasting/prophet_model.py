import pandas as pd
from prophet import Prophet
import logging

logger = logging.getLogger(__name__)

def train_predict_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Trains a Facebook Prophet model with multiplicative seasonality and custom regressors.
    
    This specification accounts for yearly trends, monthly cycles, and 
    US holiday shocks, which are critical for retail chain-level forecasting.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with 'Date' and 'Sales' columns.
    test_df : pd.DataFrame
        Test data with 'Date' for prediction.

    Returns
    -------
    predictions : np.ndarray
        Array of forecasted sales values ('yhat').
    """
    logger.info("Initializing Prophet model (Multiplicative mode)...")
    
    m_df = train_df.rename(columns={'Date': 'ds', 'Sales': 'y'}).copy()
    
    # Initialize with Multiplicative seasonality to handle volume scaling
    model = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=False, # Data is weekly aggregated
        daily_seasonality=False,
        seasonality_mode='multiplicative' 
    )
    
    # Custom monthly seasonality (30.5 days approx)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # US Holidays for retail event shocks
    model.add_country_holidays(country_name='US')
    
    # Exogenous Promo Intensity
    model.add_regressor('Promo')
    
    logger.info("Fitting Prophet model...")
    model.fit(m_df)
    
    # Prepare future dataframe
    future = test_df.rename(columns={'Date': 'ds'}).copy()
    
    logger.info("Generating Prophet forecast...")
    forecast = model.predict(future)
    
    return forecast['yhat'].values
