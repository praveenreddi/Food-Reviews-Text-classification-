import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import norm

def train_and_forecast_exponential_smoothing(merged, causer, steps=12):
    """
    Modified version with improved confidence interval calculation
    """
    # Filter and prepare data
    cdf = merged[merged['causer_acronym'] == causer].copy()
    cdf = cdf[(cdf['year'] >= 2020) & (cdf['year'] <= 2024)].copy()

    cdf.set_index('date', inplace=True)
    cdf = cdf.asfreq('MS')
    cdf['has_outage'].fillna(0, inplace=True)

    y = cdf['has_outage']

    # Fit model
    model = ExponentialSmoothing(y,
                                trend=None,
                                seasonal='add',
                                seasonal_periods=12)
    res = model.fit()

    # Generate forecast
    fcast_vals = res.forecast(steps=steps)

    # Calculate prediction intervals using a different approach
    residuals = res.fittedvalues - y
    sigma = np.sqrt(np.sum(residuals**2) / (len(residuals) - 1))

    # Calculate confidence intervals using prediction standard errors
    z_score = norm.ppf(0.975)  # 95% confidence level
    pred_std = np.sqrt(sigma**2 * (1 + 1/len(y)))

    lower = fcast_vals - z_score * pred_std
    upper = fcast_vals + z_score * pred_std

    # Bound the forecasts and CIs between 0 and 1 with a more nuanced approach
    fcast_vals = np.clip(fcast_vals, 0.001, 0.999)
    lower = np.clip(lower, 0.001, 0.999)
    upper = np.clip(upper, 0.001, 0.999)

    # Ensure lower <= forecast <= upper
    lower = np.minimum(lower, fcast_vals)
    upper = np.maximum(upper, fcast_vals)

    # Create forecast DataFrame with date index
    forecast_dates = pd.date_range(start=y.index[-1] + pd.DateOffset(months=1),
                                 periods=steps,
                                 freq='MS')

    df_fcast = pd.DataFrame({
        'forecast': fcast_vals,
        'lower_ci': lower,
        'upper_ci': upper
    }, index=forecast_dates)

    return df_fcast, y

def plot_forecast_with_history(forecast_df, history, causer_name):
    """
    Plot the forecast with history and confidence intervals
    """
    plt.figure(figsize=(12, 6))

    # Plot historical values
    plt.plot(history.index, history.values,
            label='Historical', color='blue', marker='o', markersize=4)

    # Plot forecast
    plt.plot(forecast_df.index, forecast_df['forecast'],
            label='Forecast', color='red', linestyle='--')

    # Plot confidence intervals
    plt.fill_between(forecast_df.index,
                    forecast_df['lower_ci'],
                    forecast_df['upper_ci'],
                    color='red', alpha=0.2,
                    label='95% Confidence Interval')

    plt.title(f'Outage Forecast for {causer_name}')
    plt.xlabel('Date')
    plt.ylabel('Probability of Outage')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Set y-axis limits with some padding
    plt.ylim(-0.05, 1.05)

    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()

    return plt

def print_forecast_summary(forecast_df):
    """
    Print a nicely formatted forecast summary
    """
    print("\nMonthly Forecast Summary:")
    print("-" * 70)
    print(f"{'Month':<12} {'Forecast':>10} {'Lower CI':>12} {'Upper CI':>12} {'CI Width':>12}")
    print("-" * 70)

    for idx, row in forecast_df.iterrows():
        ci_width = row['upper_ci'] - row['lower_ci']
        print(f"{idx.strftime('%Y-%m'):<12} "
              f"{row['forecast']:>10.3f} "
              f"{row['lower_ci']:>12.3f} "
              f"{row['upper_ci']:>12.3f} "
              f"{ci_width:>12.3f}")

# Example usage:
if __name__ == "__main__":
    # Assuming you have your data prepared
    # merged = ... [your data preparation steps]

    example_causer = "Digital IDP"
    # forecast_df, history = train_and_forecast_exponential_smoothing(merged, example_causer)

    # Create visualization
    # fig = plot_forecast_with_history(forecast_df, history, example_causer)
    # plt.show()

    # Print detailed forecast summary
    # print_forecast_summary(forecast_df)
