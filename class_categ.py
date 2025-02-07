import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import weibull_min, gamma, poisson
import warnings
warnings.filterwarnings('ignore')

# Step 1: Data Loading and Preprocessing
def load_and_preprocess_data(df):
    """
    Load and preprocess the data with feature engineering
    """
    # Convert to datetime
    df['incident_start_time_cst'] = pd.to_datetime(df['incident_start_time_cst'])

    # Extract basic time features
    df['year'] = df['incident_start_time_cst'].dt.year
    df['month'] = df['incident_start_time_cst'].dt.month
    df['day'] = df['incident_start_time_cst'].dt.day
    df['hour'] = df['incident_start_time_cst'].dt.hour
    df['day_of_week'] = df['incident_start_time_cst'].dt.dayofweek

    # Remove duplicates within same month/year for each application
    df = df.sort_values('incident_start_time_cst')
    df = df.drop_duplicates(subset=['causer_acronym', 'year', 'month'], keep='first')

    return df

# Step 2: Feature Engineering
def create_features(df):
    """
    Create additional features for better prediction
    """
    # Create time-based features
    features = pd.DataFrame()
    features['month'] = df['month']
    features['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    features['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

    # Create seasonal indicators
    features['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    features['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    features['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
    features['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)

    # Calculate historical incident rates
    monthly_rates = df.groupby('month').size() / len(df['year'].unique())
    features['historical_rate'] = features['month'].map(monthly_rates)

    return features

# Step 3: Model Definitions
def xgboost_predict(X_train, y_train, X_pred):
    """XGBoost prediction"""
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)
    return np.clip(model.predict(X_pred), 0, 1)

def random_forest_predict(X_train, y_train, X_pred):
    """Random Forest prediction"""
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return np.clip(rf.predict(X_pred), 0, 1)

def survival_analysis_predict(data):
    """Weibull Survival Analysis"""
    intervals = np.diff(data['incident_start_time_cst'].values)
    intervals = np.array([x.total_seconds() / (24*3600) for x in intervals])
    params = weibull_min.fit(intervals)
    times = np.arange(1, 13)
    return np.clip(1 - weibull_min.sf(times, *params), 0, 1)

def poisson_process_predict(data):
    """Enhanced Poisson Process prediction"""
    monthly_rates = data.groupby('month').size().values
    rate = monthly_rates.mean()
    times = np.arange(1, 13)
    return np.clip(1 - np.exp(-rate * times/12), 0, 1)

def sarima_predict(data):
    """SARIMA prediction with seasonal components"""
    monthly_counts = pd.Series(index=pd.date_range(start='2020-01-01', end='2024-12-31', freq='M'))
    for idx, row in data.iterrows():
        date = pd.Timestamp(year=row['year'], month=row['month'], day=1)
        monthly_counts[date] = 1
    monthly_counts = monthly_counts.fillna(0)

    model = SARIMAX(monthly_counts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=False)
    return np.clip(results.forecast(steps=12), 0, 1)

# Step 4: Main Analysis Function
def analyze_application(df, app_name):
    """Analyze single application data"""
    print(f"\nAnalyzing {app_name}...")

    # Filter data for specific application
    app_data = df[df['causer_acronym'] == app_name].copy()

    # Create features
    features = create_features(app_data)

    # Prepare training data
    X = features.drop(['historical_rate'], axis=1)
    y = features['historical_rate']

    # Prepare prediction features
    X_pred = pd.DataFrame()
    for month in range(1, 13):
        month_features = pd.DataFrame({
            'month': [month],
            'is_weekend': [0],  # Average prediction
            'is_night': [0],    # Average prediction
            'is_spring': [1 if month in [3,4,5] else 0],
            'is_summer': [1 if month in [6,7,8] else 0],
            'is_fall': [1 if month in [9,10,11] else 0],
            'is_winter': [1 if month in [12,1,2] else 0]
        })
        X_pred = pd.concat([X_pred, month_features])

    # Generate predictions from all models
    predictions = pd.DataFrame()
    predictions['month'] = range(1, 13)

    # Run all models
    predictions['xgboost'] = xgboost_predict(X, y, X_pred)
    predictions['random_forest'] = random_forest_predict(X, y, X_pred)
    predictions['survival'] = survival_analysis_predict(app_data)
    predictions['poisson'] = poisson_process_predict(app_data)
    predictions['sarima'] = sarima_predict(app_data)

    # Calculate ensemble prediction with weights
    weights = {
        'xgboost': 0.25,
        'random_forest': 0.20,
        'survival': 0.20,
        'poisson': 0.15,
        'sarima': 0.20
    }

    predictions['ensemble'] = sum(predictions[model] * weight
                                for model, weight in weights.items())

    # Calculate confidence intervals
    std_dev = predictions[weights.keys()].std(axis=1)
    predictions['ci_lower'] = np.clip(predictions['ensemble'] - 1.96 * std_dev, 0, 1)
    predictions['ci_upper'] = np.clip(predictions['ensemble'] + 1.96 * std_dev, 0, 1)

    return predictions

# Step 5: Visualization Function
def visualize_predictions(predictions, app_name):
    """Create detailed visualization"""
    plt.figure(figsize=(15, 8))

    # Plot individual model predictions
    for col in ['xgboost', 'random_forest', 'survival', 'poisson', 'sarima']:
        plt.plot(predictions['month'], predictions[col], '--', alpha=0.3, label=col)

    # Plot ensemble prediction
    plt.plot(predictions['month'], predictions['ensemble'], 'k-',
            linewidth=3, label='Ensemble')

    # Plot confidence interval
    plt.fill_between(predictions['month'], predictions['ci_lower'],
                    predictions['ci_upper'], color='gray', alpha=0.2,
                    label='95% CI')

    plt.title(f'Outage Predictions 2025 - {app_name}')
    plt.xlabel('Month')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'predictions_{app_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Step 6: Main Execution
def main():
    # Read data
    df = pd.read_csv('your_data.csv')
    df = load_and_preprocess_data(df)

    # Process each application
    all_results = {}
    for app in df['causer_acronym'].unique():
        # Generate predictions
        predictions = analyze_application(df, app)

        # Save predictions to CSV
        predictions.to_csv(f'predictions_{app}.csv', index=False)

        # Create visualization
        visualize_predictions(predictions, app)

        all_results[app] = predictions
        print(f"Completed analysis for {app}")

    return all_results

if __name__ == "__main__":
    results = main()
