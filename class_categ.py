import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def clean_and_prepare_data(df):
    """Clean and prepare the initial data"""
    df['incident_datetime'] = pd.to_datetime(df['incident_start_time_cst'], errors='coerce')
    df = df.dropna(subset=['incident_datetime']).copy()
    df['year_tmp'] = df['incident_datetime'].dt.year
    df = df[(df['year_tmp'] >= 2020) & (df['year_tmp'] <= 2025)]

    # Drop invalid time components
    def invalid_time_components(dt):
        return (dt.minute > 59) or (dt.hour > 23)
    invalid_mask = df['incident_datetime'].apply(invalid_time_components)
    df = df[~invalid_mask].copy()

    return df

def aggregate_monthly_binary(df):
    """Create monthly binary aggregation with placeholders"""
    df['year'] = df['incident_datetime'].dt.year
    df['month'] = df['incident_datetime'].dt.month

    # Count tickets and create binary indicator
    grouped = (
        df.groupby(['causer_acronym','year','month'])
          .size()
          .reset_index(name='ticket_count')
    )
    grouped['has_outage'] = (grouped['ticket_count'] >= 1).astype(int)

    # Create all possible combinations for 2020-2025
    all_causers = df['causer_acronym'].unique()
    years = range(2020, 2026)
    months = range(1, 13)
    combos = [(c, y, m) for c in all_causers for y in years for m in months]
    combos_df = pd.DataFrame(combos, columns=['causer_acronym','year','month'])

    # Merge with actual data
    merged = pd.merge(
        combos_df,
        grouped[['causer_acronym','year','month','has_outage']],
        on=['causer_acronym','year','month'],
        how='left'
    )
    merged['has_outage'] = merged['has_outage'].fillna(0).astype(int)

    # Sort for consistency
    merged.sort_values(['causer_acronym','year','month'], inplace=True)
    return merged

def build_basic_features(merged):
    """Add engineered features"""
    # Cyclical month encoding
    merged['month_sin'] = np.sin(2*np.pi * merged['month']/12)
    merged['month_cos'] = np.cos(2*np.pi * merged['month']/12)

    # Lag feature
    merged['lag1_outage'] = merged.groupby('causer_acronym')['has_outage'].shift(1).fillna(0)

    # Create proper datetime
    merged['date'] = pd.to_datetime(
        merged['year'].astype(str) + '-' +
        merged['month'].astype(str) + '-01'
    )

    return merged

def train_and_forecast_exponential_smoothing(merged, causer, steps=12):
    """
    Fit Holt-Winters model and forecast future values

    Parameters:
    merged: DataFrame with columns [causer_acronym, date, has_outage, year]
    causer: str, specific causer to analyze
    steps: int, number of months to forecast
    """
    # Filter data for specific causer and time range
    cdf = merged[merged['causer_acronym'] == causer].copy()
    cdf = cdf[(cdf['year'] >= 2020) & (cdf['year'] <= 2024)].copy()

    # Set date as index and ensure monthly frequency
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

    # 1. Clean and prepare data
    # df = clean_and_prepare_data(df)

    # 2. Create monthly aggregation with placeholders
    # merged = aggregate_monthly_binary(df)

    # 3. Add engineered features
    # merged = build_basic_features(merged)

    # 4. Run forecasting for a specific causer
    # example_causer = "Digital IDP"
    # forecast = train_and_forecast_exponential_smoothing(merged, example_causer, steps=12)
    # print("\nForecast for 2025:")
    # print(forecast)
