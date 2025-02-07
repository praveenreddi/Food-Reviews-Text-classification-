import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import norm
from datetime import datetime
import os

def clean_and_prepare_data(df):
    """Clean and prepare the initial data"""
    print("Cleaning and preparing data...")

    # Convert to datetime
    df['incident_datetime'] = pd.to_datetime(df['incident_start_time_cst'], errors='coerce')

    # Remove rows with null datetime
    df = df.dropna(subset=['incident_datetime']).copy()

    # Extract year for filtering
    df['year'] = df['incident_datetime'].dt.year
    df = df[(df['year'] >= 2020) & (df['year'] <= 2025)]

    # Remove invalid time entries
    def invalid_time_components(dt):
        return (dt.minute > 59) or (dt.hour > 23)
    invalid_mask = df['incident_datetime'].apply(invalid_time_components)
    df = df[~invalid_mask].copy()

    return df

def aggregate_monthly_binary(df):
    """Create monthly binary aggregation with placeholders"""
    print("Aggregating monthly data...")

    # Extract month
    df['month'] = df['incident_datetime'].dt.month

    # Create binary indicator for outages
    grouped = (
        df.groupby(['causer_acronym', 'year', 'month'])
        .size()
        .reset_index(name='ticket_count')
    )
    grouped['has_outage'] = (grouped['ticket_count'] >= 1).astype(int)

    # Create all possible combinations
    all_causers = df['causer_acronym'].unique()
    years = range(2020, 2026)
    months = range(1, 13)
    combos = [(c, y, m) for c in all_causers for y in years for m in months]
    combos_df = pd.DataFrame(combos, columns=['causer_acronym', 'year', 'month'])

    # Merge with actual data
    merged = pd.merge(combos_df, grouped,
                     on=['causer_acronym', 'year', 'month'],
                     how='left')
    merged['has_outage'] = merged['has_outage'].fillna(0)

    return merged

def build_basic_features(merged):
    """Add engineered features"""
    print("Building features...")

    # Cyclical month encoding
    merged['month_sin'] = np.sin(2*np.pi * merged['month']/12)
    merged['month_cos'] = np.cos(2*np.pi * merged['month']/12)

    # Create lag feature
    merged['lag1_outage'] = merged.groupby('causer_acronym')['has_outage'].shift(1).fillna(0)

    # Create proper datetime
    merged['date'] = pd.to_datetime(
        merged['year'].astype(str) + '-' +
        merged['month'].astype(str) + '-01'
    )

    return merged

def train_and_forecast_exponential_smoothing(merged, causer, steps=12):
    """Train model and generate forecasts"""
    try:
        cdf = merged[merged['causer_acronym'] == causer].copy()
        cdf = cdf[(cdf['year'] >= 2020) & (cdf['year'] <= 2024)].copy()

        cdf.set_index('date', inplace=True)
        cdf = cdf.asfreq('MS')
        cdf['has_outage'].fillna(0, inplace=True)

        y = cdf['has_outage']

        model = ExponentialSmoothing(y, trend=None, seasonal='add', seasonal_periods=12)
        res = model.fit()
        fcast_vals = res.forecast(steps=steps)

        residuals = res.fittedvalues - y
        sigma = np.sqrt(np.sum(residuals**2) / (len(residuals) - 1))
        z_score = norm.ppf(0.975)
        pred_std = np.sqrt(sigma**2 * (1 + 1/len(y)))

        lower = fcast_vals - z_score * pred_std
        upper = fcast_vals + z_score * pred_std

        fcast_vals = np.clip(fcast_vals, 0.001, 0.999)
        lower = np.clip(lower, 0.001, 0.999)
        upper = np.clip(upper, 0.001, 0.999)

        lower = np.minimum(lower, fcast_vals)
        upper = np.maximum(upper, fcast_vals)

        forecast_dates = pd.date_range(start=y.index[-1] + pd.DateOffset(months=1),
                                     periods=steps,
                                     freq='MS')

        df_fcast = pd.DataFrame({
            'forecast': fcast_vals,
            'lower_ci': lower,
            'upper_ci': upper
        }, index=forecast_dates)

        return df_fcast, y, True
    except Exception as e:
        print(f"Error processing {causer}: {str(e)}")
        return None, None, False

def process_all_applications(merged):
    """Process all applications and compile results"""
    applications = merged['causer_acronym'].unique()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"forecast_results_{timestamp}"
    plots_dir = f"{output_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    all_results = []

    for app in applications:
        print(f"\nProcessing {app}...")
        forecast_df, history, success = train_and_forecast_exponential_smoothing(merged, app)

        if success:
            app_results = forecast_df.copy()
            app_results['application'] = app
            app_results['date'] = app_results.index
            all_results.append(app_results)

            fig = plot_forecast_with_history(forecast_df, history, app)
            plt.savefig(f"{plots_dir}/{app.replace('/', '_')}_forecast.png")
            plt.close()

            print(f"Successfully processed {app}")
        else:
            print(f"Failed to process {app}")

    if all_results:
        final_results = pd.concat(all_results, axis=0)

        csv_file = f"{output_dir}/forecast_results.csv"
        final_results.to_csv(csv_file, index=False)

        summary_stats = create_summary_statistics(final_results)
        summary_file = f"{output_dir}/forecast_summary.csv"
        summary_stats.to_csv(summary_file, index=True)

        return final_results, csv_file
    else:
        print("No results to save!")
        return None, None

def create_summary_statistics(final_results):
    """Create summary statistics"""
    summary_stats = final_results.groupby('application').agg({
        'forecast': ['mean', 'min', 'max'],
        'lower_ci': 'mean',
        'upper_ci': 'mean'
    }).round(3)

    summary_stats['ci_width'] = (summary_stats[('upper_ci', 'mean')] -
                                summary_stats[('lower_ci', 'mean')]).round(3)

    return summary_stats

def plot_forecast_with_history(forecast_df, history, causer_name):
    """Create forecast plots"""
    plt.figure(figsize=(12, 6))

    plt.plot(history.index, history.values,
            label='Historical', color='blue', marker='o', markersize=4)

    plt.plot(forecast_df.index, forecast_df['forecast'],
            label='Forecast', color='red', linestyle='--')

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
    plt.ylim(-0.05, 1.05)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return plt

# Main execution
if __name__ == "__main__":
    # Read the input data
    try:
        print("Reading input data...")
        # Replace 'your_input_file.csv' with your actual file name
        input_file = 'your_input_file.csv'
        df = pd.read_csv(input_file)

        # Data preparation pipeline
        cleaned_df = clean_and_prepare_data(df)
        monthly_data = aggregate_monthly_binary(cleaned_df)
        merged = build_basic_features(monthly_data)

        # Process all applications
        print("\nStarting forecast processing...")
        results, csv_file = process_all_applications(merged)

        if results is not None:
            print(f"\nResults successfully saved to: {csv_file}")
            print("\nSample of results:")
            print(results.head())

            # Print summary statistics
            print("\nSummary Statistics:")
            summary_stats = create_summary_statistics(results)
            print(summary_stats)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
