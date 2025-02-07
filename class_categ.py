import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from lifelines import WeibullAFTFitter, LogNormalAFTFitter
from scipy.stats import expon
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(df):
    """Step 1: Load and preprocess the data"""
    # Convert to datetime and fix any incorrect dates
    df['incident_start_time_cst'] = pd.to_datetime(df['incident_start_time_cst'], errors='coerce')

    # Remove rows with invalid dates
    df = df.dropna(subset=['incident_start_time_cst'])

    # Extract time features
    df['year'] = df['incident_start_time_cst'].dt.year
    df['month'] = df['incident_start_time_cst'].dt.month
    df['day'] = df['incident_start_time_cst'].dt.day
    df['hour'] = df['incident_start_time_cst'].dt.hour
    df['dayofweek'] = df['incident_start_time_cst'].dt.dayofweek

    # Remove duplicates within same month/year
    df = df.sort_values('incident_start_time_cst')
    df = df.drop_duplicates(subset=['causer_acronym', 'year', 'month'], keep='first')

    return df

def create_features(data):
    """Step 2: Feature Engineering"""
    features = pd.DataFrame()

    # Basic time features
    features['month'] = data['month']
    features['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
    features['is_business_hours'] = ((data['hour'] >= 9) & (data['hour'] <= 17)).astype(int)

    # Seasonal features
    features['season'] = pd.cut(data['month'],
                              bins=[0,3,6,9,12],
                              labels=['Winter', 'Spring', 'Summer', 'Fall'])

    # Calculate time since last incident
    data = data.sort_values('incident_start_time_cst')
    data['time_since_last'] = data['incident_start_time_cst'].diff().dt.total_seconds() / (24*3600)

    return features, data

def advanced_survival_analysis(data):
    """Step 3: Advanced Survival Analysis using Weibull AFT"""
    # Prepare survival data
    survival_data = pd.DataFrame()
    survival_data['duration'] = data['time_since_last'].fillna(30)  # Fill first value with 30 days
    survival_data['month'] = data['month']
    survival_data['event'] = 1  # All events are observed

    # Fit Weibull AFT model
    aft = WeibullAFTFitter()
    try:
        aft.fit(survival_data, duration_col='duration', event_col='event')
        future_months = pd.DataFrame({'month': range(1, 13)})
        predictions = aft.predict_median(future_months)
        return np.clip(1 / predictions, 0, 1)  # Convert to probability
    except:
        return np.zeros(12) + 0.5  # Return neutral prediction if model fails

def exponential_smoothing_predict(data):
    """Step 4: Exponential Smoothing"""
    monthly_counts = pd.Series(index=pd.date_range(start='2020-01-01',
                                                 end='2024-12-31',
                                                 freq='M'))

    for idx, row in data.iterrows():
        date = pd.Timestamp(year=row['year'], month=row['month'], day=1)
        monthly_counts[date] = 1

    monthly_counts = monthly_counts.fillna(0)

    try:
        model = ExponentialSmoothing(monthly_counts,
                                   seasonal_periods=12,
                                   trend='add',
                                   seasonal='add')
        fitted_model = model.fit()
        predictions = fitted_model.forecast(12)
        return np.clip(predictions, 0, 1)
    except:
        return np.zeros(12) + 0.5

def mathematical_model(data):
    """Step 5: Mathematical Model - Modified Exponential"""
    monthly_rates = []
    for month in range(1, 13):
        month_data = data[data['month'] == month]
        if len(month_data) > 0:
            time_diffs = month_data['time_since_last'].dropna()
            if len(time_diffs) > 0:
                rate = 1 / time_diffs.mean() if time_diffs.mean() > 0 else 0.5
                monthly_rates.append(rate)
            else:
                monthly_rates.append(0.5)
        else:
            monthly_rates.append(0.5)

    return np.clip(monthly_rates, 0, 1)

def analyze_application(df, app_name):
    """Step 6: Main Analysis Function"""
    print(f"\nAnalyzing {app_name}...")

    # Filter data for specific application
    app_data = df[df['causer_acronym'] == app_name].copy()

    # Create features
    features, app_data = create_features(app_data)

    # Generate predictions
    predictions = pd.DataFrame()
    predictions['month'] = range(1, 13)

    # Run all models
    predictions['survival'] = advanced_survival_analysis(app_data)
    predictions['exp_smoothing'] = exponential_smoothing_predict(app_data)
    predictions['mathematical'] = mathematical_model(app_data)

    # Calculate ensemble with weights
    weights = {
        'survival': 0.4,
        'exp_smoothing': 0.3,
        'mathematical': 0.3
    }

    predictions['ensemble'] = sum(predictions[model] * weight
                                for model, weight in weights.items())

    # Calculate confidence intervals
    std_dev = predictions[weights.keys()].std(axis=1)
    predictions['ci_lower'] = np.clip(predictions['ensemble'] - 1.96 * std_dev, 0, 1)
    predictions['ci_upper'] = np.clip(predictions['ensemble'] + 1.96 * std_dev, 0, 1)

    return predictions

def main():
    """Step 7: Main Execution"""
    # Read data
    df = pd.read_csv('your_data.csv')
    df = load_and_preprocess_data(df)

    # Create Excel writer
    with pd.ExcelWriter('outage_predictions_2025.xlsx') as writer:
        # Process each application
        for app in df['causer_acronym'].unique():
            # Generate predictions
            predictions = analyze_application(df, app)

            # Save to Excel sheet
            predictions.to_excel(writer, sheet_name=app[:31], index=False)

            # Create visualization
            plt.figure(figsize=(15, 8))
            for col in ['survival', 'exp_smoothing', 'mathematical']:
                plt.plot(predictions['month'], predictions[col], '--',
                        alpha=0.3, label=col)

            plt.plot(predictions['month'], predictions['ensemble'], 'k-',
                    linewidth=3, label='Ensemble')
            plt.fill_between(predictions['month'],
                           predictions['ci_lower'],
                           predictions['ci_upper'],
                           color='gray', alpha=0.2,
                           label='95% CI')

            plt.title(f'Outage Predictions 2025 - {app}')
            plt.xlabel('Month')
            plt.ylabel('Probability')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'prediction_plot_{app}.png')
            plt.close()

            print(f"Completed analysis for {app}")

if __name__ == "__main__":
    main()
