"""
FILE: all_approaches_single_script.py
Dependencies (install if needed):
  pip install pandas numpy xgboost scikit-learn statsmodels lifelines
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.utils import resample
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from lifelines import CoxPHFitter

# ---------------------------------------------------------
# 1) DATA PREP & CLEANING
# ---------------------------------------------------------

def clean_and_prepare_data(df):
    """
    1. Parse 'incident_start_time_cst' to datetime.
    2. Drop obviously invalid rows or fix them if possible:
       - If year > 2025 or < 2020, drop or treat as invalid.
       - If hours, minutes, seconds are out of range, fix or drop.
    3. Restrict final set to 2020–2024 for training, but also
       create placeholders for 2025 so we can later forecast.
    4. Return the cleaned DataFrame with a new column 'incident_datetime'.
    """

    # Try to coerce to datetime
    df['incident_datetime'] = pd.to_datetime(df['incident_start_time_cst'], errors='coerce')

    # Drop rows that could not be parsed
    df = df.dropna(subset=['incident_datetime']).copy()

    # Example: Drop or correct obviously invalid years or times
    # Here, let's just drop rows with year outside 2020–2025
    df['year_tmp'] = df['incident_datetime'].dt.year
    df = df[(df['year_tmp'] >= 2020) & (df['year_tmp'] <= 2025)]

    # Similarly, if you detect times with invalid minutes (e.g. 70, 76, 79),
    # you can drop them or approximate. Below is a simple approach:
    # We'll identify them and drop those rows:
    def invalid_time_components(dt):
        return (dt.minute > 59) or (dt.hour > 23)

    # Mask for invalid
    invalid_mask = df['incident_datetime'].apply(invalid_time_components)
    df = df[~invalid_mask].copy()

    # Return cleaned
    return df


def aggregate_monthly_binary(df):
    """
    Group by (causer_acronym, year, month) => has_outage = 1 if >=1 outages in that month.
    Also ensure placeholders for all months 2020–2025 (since we want to forecast 2025).
    """

    # Extract year, month from the cleaned datetime
    df['year'] = df['incident_datetime'].dt.year
    df['month'] = df['incident_datetime'].dt.month

    # Count tickets
    grouped = (
        df.groupby(['causer_acronym','year','month'])
          .size()
          .reset_index(name='ticket_count')
    )
    grouped['has_outage'] = (grouped['ticket_count'] >= 1).astype(int)

    # We'll include 2020–2025 for combos
    all_causers = df['causer_acronym'].unique()
    years = range(2020, 2026)   # up to 2025
    months = range(1, 13)
    combos = [(c, y, m) for c in all_causers for y in years for m in months]
    combos_df = pd.DataFrame(combos, columns=['causer_acronym','year','month'])

    merged = pd.merge(
        combos_df,
        grouped[['causer_acronym','year','month','has_outage']],
        on=['causer_acronym','year','month'],
        how='left'
    )
    merged['has_outage'] = merged['has_outage'].fillna(0).astype(int)

    # Sort for consistent ordering
    merged.sort_values(['causer_acronym','year','month'], inplace=True)
    return merged


def build_basic_features(merged):
    """
    Add cyclical month features + lag1_outage.
    Creates a 'date' column (YYYY-MM-01).
    """
    merged['month_sin'] = np.sin(2*np.pi * merged['month']/12)
    merged['month_cos'] = np.cos(2*np.pi * merged['month']/12)

    # Group-lag
    merged['lag1_outage'] = merged.groupby('causer_acronym')['has_outage'].shift(1).fillna(0)

    # Create date
    merged['date'] = pd.to_datetime(
        merged['year'].astype(str) + '-' + merged['month'].astype(str) + '-01'
    )

    return merged


# ---------------------------------------------------------
# 2) CLASSIFICATION APPROACH (XGBoost) + Weighted Recent Years
# ---------------------------------------------------------

def train_xgb_classification(merged):
    """
    Train on 2020–2024 (all rows) with heavier weights for 2023–2024.
    Return the fitted model.
    """
    # Filter training data
    train_df = merged[(merged['year'] >= 2020) & (merged['year'] <= 2024)].copy()

    features = ['month_sin','month_cos','lag1_outage']
    target   = 'has_outage'

    def year_to_weight(y):
        if y == 2023:
            return 2.0
        elif y == 2024:
            return 3.0
        else:
            return 1.0

    sample_w = train_df['year'].apply(year_to_weight).values

    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, use_label_encoder=False)
    model.fit(train_df[features], train_df[target], sample_weight=sample_w)

    return model


def predict_2025_xgb(xgb_model, merged, n_bootstraps=50):
    """
    Predict monthly outage probability for 2025 rows,
    plus a simple bootstrap-based confidence interval for the *average* predicted probability.
    Returns a DataFrame with columns:
      causer_acronym, year=2025, month, predicted_prob, lower_ci, upper_ci (global for average).
    """

    # Subset 2025
    df_2025 = merged[merged['year'] == 2025].copy()
    if df_2025.empty:
        raise ValueError("No 2025 data found. Ensure you included 2025 in your combos.")

    features = ['month_sin','month_cos','lag1_outage']

    # Predict
    df_2025['predicted_prob'] = xgb_model.predict_proba(df_2025[features])[:,1]

    # Bootstrap approach for overall average probability
    boot_means = []
    for i in range(n_bootstraps):
        df_samp = resample(df_2025, random_state=42+i)
        probs_samp = xgb_model.predict_proba(df_samp[features])[:,1]
        boot_means.append(np.mean(probs_samp))

    lower_ci = np.percentile(boot_means, 2.5)
    upper_ci = np.percentile(boot_means, 97.5)

    df_2025['avg_prob_boot_mean'] = np.mean(df_2025['predicted_prob'])
    df_2025['avg_prob_lower_ci']  = lower_ci
    df_2025['avg_prob_upper_ci']  = upper_ci

    return df_2025


# ---------------------------------------------------------
# 3) EXPONENTIAL SMOOTHING (Holt-Winters)
# ---------------------------------------------------------

def train_and_forecast_exponential_smoothing(merged, causer, steps=12):
    """
    Example: For a single causer_acronym, get the monthly 0/1 series (2020–2024),
    fit a Holt-Winters model, then forecast 'steps' months (which might be 2025).
    Returns a DataFrame with forecast + naive confidence intervals.
    """

    cdf = merged[merged['causer_acronym'] == causer].copy()
    cdf = cdf[(cdf['year'] >= 2020) & (cdf['year'] <= 2024)].copy()

    # Index by date
    cdf.set_index('date', inplace=True)
    cdf = cdf.asfreq('MS')  # monthly start freq
    cdf['has_outage'].fillna(0, inplace=True)

    y = cdf['has_outage']

    # Fit
    model = ExponentialSmoothing(y, trend=None, seasonal='add', seasonal_periods=12)
    res = model.fit()

    # Forecast
    fcast_vals = res.forecast(steps=steps)

    # Confidence intervals (naive)
    # we estimate residual std dev
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


# ---------------------------------------------------------
# 4) SARIMA (ARIMA)
# ---------------------------------------------------------

def train_and_forecast_sarima(merged, causer, steps=12):
    """
    For a single causer_acronym, monthly 0/1 data from 2020–2024,
    fit a SARIMA model (1,1,1)(0,1,1,12) then forecast steps months (2025).
    Returns a DataFrame with forecast + confidence intervals from model.
    """

    cdf = merged[merged['causer_acronym'] == causer].copy()
    cdf = cdf[(cdf['year'] >= 2020) & (cdf['year'] <= 2024)].copy()

    cdf.set_index('date', inplace=True)
    cdf = cdf.asfreq('MS')
    cdf['has_outage'].fillna(0, inplace=True)

    y = cdf['has_outage']

    model = SARIMAX(y, order=(1,1,1), seasonal_order=(0,1,1,12), trend='n',
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    forecast_obj = results.get_forecast(steps=steps)
    mean_forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    df_out = pd.DataFrame({
        'forecast': mean_forecast
    }, index=mean_forecast.index)
    df_out[['lower_ci','upper_ci']] = conf_int

    return df_out


# ---------------------------------------------------------
# 5) SURVIVAL ANALYSIS
# ---------------------------------------------------------

def prepare_survival_data(df, causer):
    """
    For survival analysis, we want time-to-next-outage.
    We'll filter for the chosen causer, sort by incident_datetime,
    then compute difference to the next outage.

    event=1 if the next outage is observed, 0 if censored.
    T = number of days to next outage.
    """
    cdf = df[df['causer_acronym'] == causer].copy()
    cdf = cdf.sort_values('incident_datetime').reset_index(drop=True)

    cdf['next_datetime'] = cdf['incident_datetime'].shift(-1)
    cdf['time_diff'] = (cdf['next_datetime'] - cdf['incident_datetime']).dt.days

    # The last record has no "next outage" => censored
    cdf['event'] = np.where(cdf['time_diff'].notnull(), 1, 0)

    # Drop the last row if time_diff is NaN (censored) or keep it with event=0
    cdf.dropna(subset=['time_diff'], inplace=True)

    # Minimal survival set
    sdata = pd.DataFrame({
        'T': cdf['time_diff'],
        'event': cdf['event']
    })
    # You can add more covariates if you like (month, year, etc.)
    # We'll just do a dummy feature for demonstration.
    sdata['dummy_feature'] = 1.0

    return sdata


def run_cox_survival_analysis(sdata):
    """
    Fit a Cox Proportional Hazards model on the survival data.
    Returns the fitted model plus a summary.
    """
    cph = CoxPHFitter()
    cph.fit(sdata, duration_col='T', event_col='event', show_progress=False)
    return cph


# ---------------------------------------------------------
# MAIN EXAMPLE OF USAGE
# ---------------------------------------------------------

if __name__ == "__main__":

    # --------------------------------------------------------------------
    # A) Suppose you have a dataframe 'return_df' from the uploaded doc.
    #    We'll just call it df here.
    #    For example:
    ######################################################################
    # df = return_df  # <--- you already have it in your environment
    #
    # OR if you read from CSV:
    # df = pd.read_csv("some_outage_data.csv")
    ######################################################################

    # We'll pretend we have it:
    print("=== SAMPLE RUN of All Approaches ===")

    # 1) Clean data
    # df = clean_and_prepare_data(df)  # UNCOMMENT once you have real df

    # 2) Create monthly binary aggregator + placeholder for 2025
    # merged = aggregate_monthly_binary(df)

    # 3) Add features (cyclical month, lag1_outage, date)
    # merged = build_basic_features(merged)

    # ==========================
    # Classification (XGBoost)
    # ==========================
    # xgb_model = train_xgb_classification(merged)
    # predictions_2025 = predict_2025_xgb(xgb_model, merged, n_bootstraps=50)
    # print("XGBoost Predictions for 2025 (first few rows):")
    # print(predictions_2025.head(10))

    # ==========================
    # Exponential Smoothing
    # ==========================
    # example_causer = "Digital IDP"
    # es_forecast = train_and_forecast_exponential_smoothing(merged, example_causer, steps=12)
    # print("\nHolt-Winters forecast for 2025 (first few rows):")
    # print(es_forecast.head())

    # ==========================
    # SARIMA (ARIMA)
    # ==========================
    # sarima_forecast = train_and_forecast_sarima(merged, example_causer, steps=12)
    # print("\nSARIMA forecast for 2025 (first few rows):")
    # print(sarima_forecast.head())

    # ==========================
    # Survival Analysis
    # ==========================
    # sdata = prepare_survival_data(df, example_causer)
    # cph_model = run_cox_survival_analysis(sdata)
    # print("\nCOX PH Model summary:")
    # print(cph_model.summary)
    #
    # # Example: get survival function for the 'average' row
    # print("\nSurvival function at a range of times:")
    # print(cph_model.predict_survival_function(sdata.iloc[[0]]).head(30))

    print("""
All stubs/methods are defined in this single file.
Uncomment and adapt as needed once you have your 'return_df' dataframe
plugged in. Then you can run the classification, Exponential Smoothing,
SARIMA, and survival analysis approaches in one place.
""")
