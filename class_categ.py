"""
FILE: simple_outage_forecast.py
Dependencies to install once (if needed):
  pip install pandas numpy xgboost scikit-learn statsmodels openpyxl
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ------------------------------------------------------------------
# STEP 1: CLEAN THE RAW DATA
# ------------------------------------------------------------------

def clean_data(df):
    """
    1) Convert 'incident_start_time_cst' to datetime, dropping rows that fail.
    2) Remove obviously invalid years (<2020 or >2025) for simplicity.
    3) Remove times if minute>59 or hour>23, to avoid confusion.
    """
    # Convert
    df['incident_datetime'] = pd.to_datetime(df['incident_start_time_cst'], errors='coerce')
    df.dropna(subset=['incident_datetime'], inplace=True)

    # Keep only years 2020..2025 (so we can see up to 2024 for training + 2025 for future months)
    df['year_temp'] = df['incident_datetime'].dt.year
    df = df[(df['year_temp'] >= 2020) & (df['year_temp'] <= 2025)]

    # Drop invalid hour or minute
    def invalid_time(dt):
        return (dt.hour > 23) or (dt.minute > 59)
    mask_invalid = df['incident_datetime'].apply(invalid_time)
    df = df[~mask_invalid].copy()

    df.drop(columns=['year_temp'], inplace=True, errors='ignore')
    return df


# ------------------------------------------------------------------
# STEP 2: AGGREGATE TO MONTHLY => has_outage=1 if >=1 in that month
# ------------------------------------------------------------------

def monthly_aggregation(df):
    """
    Group by (causer_acronym, year, month) => has_outage=1 if any outage that month.
    Also ensures we have rows for every month from 2020..2025 if you want
    to forecast 2025. For 2025 we assume we have no "actual" outages yet.
    """
    df['year'] = df['incident_datetime'].dt.year
    df['month'] = df['incident_datetime'].dt.month

    grouped = (
        df.groupby(['causer_acronym','year','month'])
          .size()
          .reset_index(name='count')
    )
    grouped['has_outage'] = (grouped['count'] >= 1).astype(int)

    all_causers = df['causer_acronym'].unique()
    years = range(2020, 2026)  # up to 2025
    months = range(1, 13)
    combos = [(c,y,m) for c in all_causers for y in years for m in months]
    combos_df = pd.DataFrame(combos, columns=['causer_acronym','year','month'])

    merged = combos_df.merge(
        grouped[['causer_acronym','year','month','has_outage']],
        on=['causer_acronym','year','month'],
        how='left'
    )
    merged['has_outage'] = merged['has_outage'].fillna(0).astype(int)

    return merged


# ------------------------------------------------------------------
# STEP 3: CLASSIFICATION MODELS (XGB & RF) FOR 2020–24 => Predict 2025
# ------------------------------------------------------------------

def build_time_index(df):
    """
    Simpler approach: a single numeric 'time_index' = (year - 2020)*12 + (month-1).
    That is the only feature. No "lag", no cyclical.
    """
    df['time_index'] = (df['year'] - 2020)*12 + (df['month'] - 1)
    return df


def train_xgb(df):
    """
    Train on 2020..2024 => Weighted for 2023=2, 2024=3
    X feature = only time_index
    y = has_outage
    Return fitted model.
    """
    train_df = df[(df['year'] >= 2020) & (df['year'] <= 2024)]
    X_train = train_df[['time_index']]
    y_train = train_df['has_outage']

    def year_to_weight(y):
        if y == 2023:
            return 2.0
        elif y == 2024:
            return 3.0
        else:
            return 1.0

    weights = train_df['year'].apply(year_to_weight).values

    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, use_label_encoder=False)
    model.fit(X_train, y_train, sample_weight=weights)

    return model


def train_rf(df):
    """
    Train RandomForest the same way
    """
    train_df = df[(df['year'] >= 2020) & (df['year'] <= 2024)]
    X_train = train_df[['time_index']]
    y_train = train_df['has_outage']

    def year_to_weight(y):
        if y == 2023:
            return 2.0
        elif y == 2024:
            return 3.0
        else:
            return 1.0

    weights = train_df['year'].apply(year_to_weight).values

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)

    return model


def predict_with_bootstrap(df_2025, model, model_name, n_boot=30):
    """
    For each row in df_2025, get a predicted probability plus a naive bootstrap
    confidence interval. We'll do a simplistic "input-perturbation" approach:
      1) For row i, we do 'n_boot' times:
         - sample a random row from df_2025 (resample),
         - predict prob for that row's time_index,
         - collect in a list => get 2.5% and 97.5%ile => ci
    We then clip to [0,1].
    """
    X_2025 = df_2025[['time_index']].values
    base_probs = model.predict_proba(X_2025)[:,1]

    lower_list = []
    upper_list = []

    # We keep the code short & naive
    all_time_idxs = X_2025.ravel()  # array of the time_index values

    for i in range(len(X_2025)):
        row_probs = []
        for _ in range(n_boot):
            # pick random row from 2025 set
            random_idx = np.random.randint(low=0, high=len(X_2025))
            time_val = all_time_idxs[random_idx]
            # predict prob
            p = model.predict_proba([[time_val]])[0,1]
            row_probs.append(p)

        low_i = np.percentile(row_probs, 2.5)
        up_i = np.percentile(row_probs, 97.5)
        # clip
        low_i = max(0.0, min(1.0, low_i))
        up_i = max(0.0, min(1.0, up_i))

        lower_list.append(low_i)
        upper_list.append(up_i)

    df_2025[f'{model_name}_prob'] = base_probs.clip(0,1)
    df_2025[f'{model_name}_lower'] = lower_list
    df_2025[f'{model_name}_upper'] = upper_list

    return df_2025


# ------------------------------------------------------------------
# STEP 4: EXPONENTIAL SMOOTHING
# ------------------------------------------------------------------

def exponential_smoothing_forecast(df, causer):
    """
    Simpler approach for each causer:
      1) Filter df for that causer, years=2020..2024
      2) Sort by ascending (year, month)
      3) Fit Holt-Winters on the monthly 0/1 series
      4) Forecast 12 steps => 2025
      5) Compute naive +/-1.96*std(residuals) for CI
      6) Clip to [0,1], return result
    """
    # Filter
    cdf = df[(df['causer_acronym']==causer) &
             (df['year']>=2020) & (df['year']<=2024)].copy()

    # Build date
    cdf['date'] = pd.to_datetime(
        cdf['year'].astype(str) + '-' + cdf['month'].astype(str) + '-01')

    cdf.set_index('date', inplace=True)
    cdf.sort_index(inplace=True)
    cdf = cdf.asfreq('MS')  # ensure monthly

    # Fill missing months in case some are missing
    cdf['has_outage'] = cdf['has_outage'].fillna(0)

    y = cdf['has_outage']

    # Fit
    # Because it's 0/1 data, we might do seasonal=None or 'add'
    # but let's keep it simple:
    model = ExponentialSmoothing(y, trend=None, seasonal=None)
    res = model.fit()

    # Forecast 12 months => 2025
    fcast_vals = res.forecast(steps=12)

    # CI via naive approach
    residuals = res.fittedvalues - y
    std_err   = np.std(residuals)
    alpha     = 1.96
    lower     = fcast_vals - alpha*std_err
    upper     = fcast_vals + alpha*std_err

    # Clip to [0,1]
    fcast_clipped = fcast_vals.clip(0,1)
    lower_clipped = lower.clip(0,1)
    upper_clipped = upper.clip(0,1)

    # Build a DataFrame with columns: year=2025, month=1..12, ES_prob, ES_lower, ES_upper
    # Index is monthly from last known date +1 month
    f_df = pd.DataFrame({
        'ES_prob': fcast_clipped,
        'ES_lower': lower_clipped,
        'ES_upper': upper_clipped
    }, index=fcast_vals.index)

    # Convert index => year, month
    f_df['year'] = f_df.index.year
    f_df['month'] = f_df.index.month
    f_df['causer_acronym'] = causer
    f_df.reset_index(drop=True, inplace=True)

    return f_df[['causer_acronym','year','month','ES_prob','ES_lower','ES_upper']]


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main(return_df):
    # 1) Clean
    df_cleaned = clean_data(return_df)

    # 2) Monthly aggregator
    monthly_df = monthly_aggregation(df_cleaned)

    # 3) Add a simple numeric time_index = (year-2020)*12 + (month-1)
    monthly_df = build_time_index(monthly_df)

    # 4) Train XGB & RF
    xgb_model = train_xgb(monthly_df)
    rf_model  = train_rf(monthly_df)

    # 5) Predict for 2025 with each model
    df_2025 = monthly_df[monthly_df['year'] == 2025].copy()
    if df_2025.empty:
        print("No 2025 data in monthly aggregator. Possibly no combos? Check code.")
        return

    # XGB predictions
    df_2025_xgb = predict_with_bootstrap(df_2025.copy(), xgb_model, "XGB", n_boot=30)

    # RF predictions
    df_2025_rf  = predict_with_bootstrap(df_2025.copy(), rf_model,  "RF",  n_boot=30)

    # Merge them
    merged_preds = pd.merge(
        df_2025_xgb,
        df_2025_rf[['causer_acronym','year','month','RF_prob','RF_lower','RF_upper']],
        on=['causer_acronym','year','month'], how='left'
    )

    # 6) For each causer, run Exponential Smoothing and combine
    # Build an empty list to accumulate sub-DataFrames
    es_list = []
    all_causers = monthly_df['causer_acronym'].unique()
    for c in all_causers:
        es_result = exponential_smoothing_forecast(monthly_df, c)
        es_list.append(es_result)

    es_df = pd.concat(es_list, ignore_index=True)

    # Merge ES with classification predictions
    final_2025 = pd.merge(
        merged_preds,
        es_df,
        on=['causer_acronym','year','month'],
        how='left'
    )

    # optional: reorder columns
    final_2025 = final_2025[[
        'causer_acronym','year','month',
        'XGB_prob','XGB_lower','XGB_upper',
        'RF_prob','RF_lower','RF_upper',
        'ES_prob','ES_lower','ES_upper'
    ]]

    # 7) Save final table to CSV
    final_2025.to_csv("predictions_2025.csv", index=False)
    print("Saved final 2025 predictions (XGB, RF, ES) to 'predictions_2025.csv'.")


# If you want to run this standalone, you'd do:
if __name__ == "__main__":
    # Suppose you already have your DataFrame (return_df) loaded:
    # return_df = pd.read_csv("some_outage_data.csv")
    # main(return_df)
    print("Please load your DataFrame into 'return_df' then call main(return_df).")
