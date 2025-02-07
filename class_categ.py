import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def clean_and_prepare_data(df):
    """Clean and prepare the initial data"""
    df['incident_datetime'] = pd.to_datetime(df['incident_start_time_cst'], errors='coerce')
    df = df.dropna(subset=['incident_datetime']).copy()
    df['year_tmp'] = df['incident_datetime'].dt.year
    df = df[(df['year_tmp'] >= 2020) & (df['year_tmp'] <= 2025)]
    return df

def aggregate_monthly_binary(df):
    """Create monthly binary aggregation"""
    df['year'] = df['incident_datetime'].dt.year
    df['month'] = df['incident_datetime'].dt.month

    grouped = (
        df.groupby(['causer_acronym','year','month'])
          .size()
          .reset_index(name='ticket_count')
    )
    grouped['has_outage'] = (grouped['ticket_count'] >= 1).astype(int)

    # Create date column
    grouped['date'] = pd.to_datetime(
        grouped['year'].astype(str) + '-' +
        grouped['month'].astype(str) + '-01'
    )

    return grouped

def train_and_forecast_exponential_smoothing(merged, causer, steps=12):
    """
    Fit Holt-Winters model and forecast future values

    Parameters:
    merged: DataFrame with columns [causer_acronym, date, has_outage, year]
    causer: str, specific causer to analyze
    steps: int, number of months to forecast
    """
    cdf = merged[merged['causer_acronym'] == causer].copy()
    cdf = cdf[(cdf['year'] >= 2020) & (cdf['year'] <= 2024)].copy()

    # Set date as index
    cdf.set_index('date', inplace=True)
    cdf = cdf.asfreq('MS')  # monthly start freq
    cdf['has_outage'].fillna(0, inplace=True)

    y = cdf['has_outage']

    # Fit model and forecast
    model = ExponentialSmoothing(y, trend=None, seasonal='add', seasonal_periods=12)
    res = model.fit()
    fcast_vals = res.forecast(steps=steps)

    # Calculate confidence intervals
    residuals = res.fittedvalues - y
    std_err = np.std(residuals)
    alpha = 1.96  # ~95% CI
    lower = fcast_vals - alpha * std_err
    upper = fcast_vals + alpha * std_err

    df_fcast = pd.DataFrame({
        'forecast': fcast_vals,
        'lower_ci': lower,
        'upper_ci': upper
    })

    return df_fcast

# Example usage:
if __name__ == "__main__":
    # Assuming you have your data in a DataFrame called 'return_df'
    # df = return_df  # or df = pd.read_csv("your_data.csv")

    # Clean and prepare data
    # df = clean_and_prepare_data(df)

    # Create monthly aggregation
    # merged = aggregate_monthly_binary(df)

    # Run forecasting for a specific causer
    # example_causer = "Digital IDP"
    # forecast = train_and_forecast_exponential_smoothing(merged, example_causer, steps=12)
    # print("\nForecast for 2025:")
    # print(forecast)
