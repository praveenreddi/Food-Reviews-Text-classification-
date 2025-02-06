import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

### 1. Data Preparation and Automatic Risk Calculation
def calculate_month_risks(df):
    """
    Automatically calculate risk scores for each month based on historical patterns
    """
    # Convert to datetime if not already
    df['incident_start_time_cst'] = pd.to_datetime(df['incident_start_time_cst'])

    # Extract month
    df['month'] = df['incident_start_time_cst'].dt.month

    # Calculate monthly frequencies
    monthly_counts = df['month'].value_counts()

    # Calculate risk scores (normalized between 0 and 1)
    max_count = monthly_counts.max()
    month_risks = (monthly_counts / max_count).to_dict()

    # Fill missing months with minimum risk
    min_risk = min(month_risks.values()) if month_risks else 0
    for month in range(1, 13):
        if month not in month_risks:
            month_risks[month] = min_risk

    print("\nAutomatically Calculated Month Risks:")
    for month in sorted(month_risks.keys()):
        print(f"Month {month}: {month_risks[month]:.3f}")

    return month_risks

def prepare_data(df):
    """
    Prepare and engineer features from raw data
    """
    # Convert to datetime
    df['incident_start_time_cst'] = pd.to_datetime(df['incident_start_time_cst'])

    # Extract basic time features
    df['year'] = df['incident_start_time_cst'].dt.year
    df['month'] = df['incident_start_time_cst'].dt.month
    df['day'] = df['incident_start_time_cst'].dt.day
    df['hour'] = df['incident_start_time_cst'].dt.hour
    df['day_of_week'] = df['incident_start_time_cst'].dt.dayofweek

    # Calculate month risks
    month_risks = calculate_month_risks(df)
    df['month_risk'] = df['month'].map(month_risks)

    # Create rolling features
    df = df.sort_values('incident_start_time_cst')
    df['days_since_last_outage'] = df['incident_start_time_cst'].diff().dt.days

    # Calculate rolling outage counts
    df['outages_last_90days'] = df['incident_start_time_cst'].rolling('90D').count()
    df['outages_last_180days'] = df['incident_start_time_cst'].rolling('180D').count()

    # Business hours and seasonal features
    df['is_business_hours'] = ((df['hour'] >= 7) & (df['hour'] <= 15)).astype(int)
    df['is_quarter_end'] = df['month'].isin([3, 6, 9, 12]).astype(int)

    return df

### 2. Visualization of Patterns
def visualize_patterns(df):
    """
    Create visualizations of outage patterns
    """
    plt.figure(figsize=(15, 10))

    # Monthly distribution
    plt.subplot(2, 2, 1)
    monthly_counts = df['month'].value_counts().sort_index()
    sns.barplot(x=monthly_counts.index, y=monthly_counts.values)
    plt.title('Outages by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Outages')

    # Yearly trend
    plt.subplot(2, 2, 2)
    yearly_counts = df['year'].value_counts().sort_index()
    sns.lineplot(x=yearly_counts.index, y=yearly_counts.values)
    plt.title('Yearly Trend of Outages')
    plt.xlabel('Year')
    plt.ylabel('Number of Outages')

    # Hour of day distribution
    plt.subplot(2, 2, 3)
    sns.histplot(df['hour'], bins=24)
    plt.title('Outages by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Count')

    # Month risk scores
    plt.subplot(2, 2, 4)
    sns.scatterplot(x=df['month'], y=df['month_risk'])
    plt.title('Calculated Month Risk Scores')
    plt.xlabel('Month')
    plt.ylabel('Risk Score')

    plt.tight_layout()
    plt.show()

### 3. Model Training
def train_model(df):
    """
    Train XGBoost model with automatically engineered features
    """
    # Prepare features
    feature_cols = ['month', 'day', 'hour', 'day_of_week', 'month_risk',
                   'is_business_hours', 'is_quarter_end',
                   'outages_last_90days', 'outages_last_180days']

    X = df[feature_cols]
    y = np.ones(len(df))  # Since all rows represent outages

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Train model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    return model, feature_cols

### 4. Predict 2025 Outages
def predict_2025(model, feature_cols, month_risks):
    """
    Generate predictions for each month in 2025
    """
    # Create features for 2025
    future_months = pd.DataFrame({
        'month': range(1, 13),
        'day': 15,  # Middle of month
        'hour': 12,  # Middle of day
        'day_of_week': 0,
        'is_business_hours': 1,
        'is_quarter_end': [1 if m in [3,6,9,12] else 0 for m in range(1,13)],
        'outages_last_90days': 0,  # Will need to be updated based on predictions
        'outages_last_180days': 0   # Will need to be updated based on predictions
    })

    # Add month risks
    future_months['month_risk'] = future_months['month'].map(month_risks)

    # Generate predictions
    probabilities = model.predict_proba(future_months[feature_cols])

    # Create results DataFrame
    predictions = pd.DataFrame({
        'Month': future_months['month'],
        'Risk_Score': future_months['month_risk'],
        'Outage_Probability': probabilities[:, 1]
    })

    return predictions

### 5. Main Execution
def main(data):
    # Prepare data
    processed_df = prepare_data(data)

    # Visualize patterns
    visualize_patterns(processed_df)

    # Train model
    model, feature_cols = train_model(processed_df)

    # Get month risks
    month_risks = calculate_month_risks(processed_df)

    # Predict 2025
    predictions = predict_2025(model, feature_cols, month_risks)

    # Display results
    print("\nPredicted Outage Probabilities for 2025:")
    print(predictions.sort_values('Outage_Probability', ascending=False))

    # Visualize predictions
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Month', y='Outage_Probability', data=predictions)
    plt.title('Predicted Outage Probabilities by Month (2025)')
    plt.xlabel('Month')
    plt.ylabel('Probability of Outage')
    plt.show()

    return predictions

# Execute the analysis
# Assuming your data is in a DataFrame called 'data'
data = pd.DataFrame({
    'incident_start_time_cst': your_datetime_column,
    # ... other columns
})

predictions = main(data)
