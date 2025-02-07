"""
FILE: combined_outage_prediction.py

Install dependencies once if needed:
   pip install pandas numpy xgboost scikit-learn statsmodels lifelines openpyxl
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from lifelines import CoxPHFitter

# --------------------------------------------------------------------
# 1) DATA CLEANING
# --------------------------------------------------------------------
def clean_data(df):
    """
    1) Convert 'incident_start_time_cst' to datetime (drop rows that fail).
    2) Remove invalid years (not in 2020..2025).
    3) Remove if hour>23 or minute>59.
    """
    # Convert to datetime
    df['incident_datetime'] = pd.to_datetime(df['incident_start_time_cst'], errors='coerce')
    df.dropna(subset=['incident_datetime'], inplace=True)

    # Filter years
    df['year_tmp'] = df['incident_datetime'].dt.year
    df = df[(df['year_tmp'] >= 2020) & (df['year_tmp'] <= 2025)]

    # Drop invalid times
    def invalid_time(dt):
        return (dt.hour > 23) or (dt.minute > 59)
    mask_invalid = df['incident_datetime'].apply(invalid_time)
    df = df[~mask_invalid].copy()

    df.drop(columns=['year_tmp'], inplace=True, errors='ignore')
    return df


# --------------------------------------------------------------------
# 2) MONTHLY AGGREGATION => has_outage=1 if >=1 ticker that month
# --------------------------------------------------------------------
def aggregate_monthly(df):
    # Extract year, month
    df['year'] = df['incident_datetime'].dt.year
    df['month'] = df['incident_datetime'].dt.month

    # Count outages
    grouped = (
        df.groupby(['causer_acronym','year','month']).size()
          .reset_index(name='count')
    )
    grouped['has_outage'] = (grouped['count'] >= 1).astype(int)

    # Ensure presence of all months 2020..2025 for each causer
    all_causers = df['causer_acronym'].unique()
    years = range(2020, 2026)
    months = range(1, 13)
    combos = [(c, y, m) for c in all_causers for y in years for m in months]
    combos_df = pd.DataFrame(combos, columns=['causer_acronym','year','month'])

    merged = combos_df.merge(
        grouped[['causer_acronym','year','month','has_outage']],
        on=['causer_acronym','year','month'],
        how='left'
    )
    merged['has_outage'] = merged['has_outage'].fillna(0).astype(int)

    return merged


# --------------------------------------------------------------------
# 3) FEATURE ENGINEERING: month_sin, month_cos
# --------------------------------------------------------------------
def add_sin_cos_features(df):
    """
    We treat 'month' as cyclical (1..12).
    month_sin = sin(2π * month/12)
    month_cos = cos(2π * month/12)
    """
    df['month_sin'] = np.sin(2.0 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2.0 * np.pi * df['month'] / 12.0)
    return df


# --------------------------------------------------------------------
# 4) XGBOOST & RANDOM FOREST (CLASSIFICATION)
# --------------------------------------------------------------------
def train_xgb(df):
    """
    Train XGBoost on 2020..2024. Weighted heavier in 2023..24 if desired.
    Features: month_sin, month_cos
    """
    train_data = df[(df['year'] >= 2020) & (df['year'] <= 2024)].copy()

    def year_to_weight(y):
        if y == 2023:
            return 2.0
        elif y == 2024:
            return 3.0
        else:
            return 1.0

    w = train_data['year'].apply(year_to_weight).values

    X_train = train_data[['month_sin','month_cos']]
    y_train = train_data['has_outage']

    model = XGBClassifier(n_estimators=100, max_depth=3, random_state=42, use_label_encoder=False)
    model.fit(X_train, y_train, sample_weight=w)
    return model


def train_rf(df):
    """
    RandomForest with the same approach.
    """
    train_data = df[(df['year'] >= 2020) & (df['year'] <= 2024)].copy()

    def year_to_weight(y):
        if y == 2023:
            return 2.0
        elif y == 2024:
            return 3.0
        else:
            return 1.0

    w = train_data['year'].apply(year_to_weight).values

    X_train = train_data[['month_sin','month_cos']]
    y_train = train_data['has_outage']

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train, sample_weight=w)
    return rf_model


def predict_2025_with_ci(df, model, model_name, n_boot=30):
    """
    For each row in 2025, predict probability plus a naive row-level bootstrap CI.
    We'll do a simple approach:
      - The base probability for row i is model.predict_proba([feature_i])
      - For the CI, we resample from the entire 2025 set,
        grab random features, get predictions -> approximate [2.5%, 97.5%].
    Clip to [0,1].
    """
    df_2025 = df[df['year'] == 2025].copy()
    if df_2025.empty:
        return pd.DataFrame()  # no 2025 data

    X_2025 = df_2025[['month_sin','month_cos']].values
    base_probs = model.predict_proba(X_2025)[:,1]

    lower_list = []
    upper_list = []

    # We'll store the features in an array for naive re-sampling
    all_feats_2025 = df_2025[['month_sin','month_cos']].values

    for i in range(len(df_2025)):
        row_probs = []
        for bs_i in range(n_boot):
            # pick random row from 2025
            idx_rand = np.random.randint(0, len(all_feats_2025))
            rand_feat = all_feats_2025[idx_rand].reshape(1, -1)
            p = model.predict_proba(rand_feat)[0,1]
            row_probs.append(p)
        low_i = np.percentile(row_probs, 2.5)
        up_i  = np.percentile(row_probs, 97.5)
        lower_list.append(np.clip(low_i, 0, 1))
        upper_list.append(np.clip(up_i, 0, 1))

    df_2025[f'{model_name}_prob']  = np.clip(base_probs, 0, 1)
    df_2025[f'{model_name}_lower'] = lower_list
    df_2025[f'{model_name}_upper'] = upper_list

    return df_2025


# --------------------------------------------------------------------
# 5) EXPONENTIAL SMOOTHING
# --------------------------------------------------------------------
def exponential_smoothing_forecast(df, causer):
    """
    For a single causer:
      - 2020..24 => fit on monthly has_outage
      - forecast 12 months => 2025
      - naive +/-1.96*std residual => clip [0,1]
    """
    cdf = df[(df['causer_acronym']==causer) & (df['year']>=2020) & (df['year']<=2024)].copy()
    if cdf.empty:
        return pd.DataFrame()

    # Build a date index
    cdf['date'] = pd.to_datetime(cdf['year'].astype(str) + '-' + cdf['month'].astype(str) + '-01')
    cdf.set_index('date', inplace=True)
    cdf = cdf.asfreq('MS').sort_index()
    cdf['has_outage'] = cdf['has_outage'].fillna(0)

    # Fit
    model = ExponentialSmoothing(cdf['has_outage'], trend=None, seasonal=None)
    fit_res = model.fit()

    # Forecast 12 steps (months)
    fcast = fit_res.forecast(steps=12)

    # Confidence intervals
    residuals = fit_res.fittedvalues - cdf['has_outage']
    std_resid = np.std(residuals)
    alpha = 1.96
    lower = fcast - alpha*std_resid
    upper = fcast + alpha*std_resid

    # Clip to [0,1]
    fcast = fcast.clip(0,1)
    lower = lower.clip(0,1)
    upper = upper.clip(0,1)

    df_out = pd.DataFrame({
        'ES_prob': fcast,
        'ES_lower': lower,
        'ES_upper': upper
    }, index=fcast.index)

    df_out['year'] = df_out.index.year
    df_out['month'] = df_out.index.month
    df_out['causer_acronym'] = causer
    df_out.reset_index(drop=True, inplace=True)

    return df_out[['causer_acronym','year','month','ES_prob','ES_lower','ES_upper']]


# --------------------------------------------------------------------
# 6) SURVIVAL ANALYSIS
# --------------------------------------------------------------------
def build_survival_data(raw_df, causer):
    """
    Build time-to-next-outage for the given causer.
    T = days until next outage
    event=1 if observed, else 0
    """
    cdf = raw_df[raw_df['causer_acronym'] == causer].copy()
    if cdf.empty:
        return pd.DataFrame()
    cdf.sort_values('incident_datetime', inplace=True)

    cdf['next_dt'] = cdf['incident_datetime'].shift(-1)
    cdf['time_diff'] = (cdf['next_dt'] - cdf['incident_datetime']).dt.days
    cdf['event'] = np.where(cdf['time_diff'].notnull(), 1, 0)
    cdf.dropna(subset=['time_diff'], inplace=True)

    sdata = pd.DataFrame({
        'T': cdf['time_diff'],
        'event': cdf['event']
    })
    # Example dummy feature
    sdata['dummy_feat'] = 1.0
    return sdata


def monthly_prob_survival(cph, months=12):
    """
    For demonstration: Probability of at least one outage in each of the next 'months'.
    We assume each month ~ 30 days.
    Probability(event in month) ~ S(t0)-S(t1).
    """
    # We'll create a generic row with 'dummy_feat'=1
    test_row = pd.DataFrame({'dummy_feat':[1.0]})
    times = np.arange(0, months*30+1)  # 0..(months*30)

    sf = cph.predict_survival_function(test_row, times=times)  # shape: (1, len(times))
    sf_vals = sf.iloc[0].values  # 1D array

    probs = []
    for m in range(1, months+1):
        t0 = (m-1)*30
        t1 = m*30
        pm = sf_vals[t0] - sf_vals[t1]
        pm = np.clip(pm, 0, 1)
        probs.append(pm)
    return probs


def run_survival_analysis(raw_df, monthly_df, causer):
    """
    1) Build survival data for 'causer'
    2) Fit CoxPH
    3) Convert survival function => 12 monthly probabilities in 2025
    Return a DataFrame with columns [causer_acronym, year=2025, month, Survival_prob]
    """
    sdata = build_survival_data(raw_df, causer)
    if sdata.empty:
        return pd.DataFrame()

    cph = CoxPHFitter()
    cph.fit(sdata, duration_col='T', event_col='event', show_progress=False)

    # 12 months => 1..12
    probs = monthly_prob_survival(cph, months=12)

    out = pd.DataFrame({
        'causer_acronym': causer,
        'year': 2025,
        'month': range(1,13),
        'Survival_prob': probs
    })
    return out


# --------------------------------------------------------------------
# 7) MAIN: Everything + Save
# --------------------------------------------------------------------
def main(return_df):
    # A) Clean raw
    df_cleaned = clean_data(return_df)

    # B) Monthly aggregator
    monthly_df = aggregate_monthly(df_cleaned)

    # C) Add sin/cos for month
    monthly_df = add_sin_cos_features(monthly_df)

    # D) Classification (XGB, RF)
    xgb_model = train_xgb(monthly_df)
    rf_model  = train_rf(monthly_df)

    # E) Predict for 2025
    xgb_2025 = predict_2025_with_ci(monthly_df, xgb_model, "XGB", n_boot=30)
    rf_2025  = predict_2025_with_ci(monthly_df, rf_model,  "RF",  n_boot=30)

    # Merge classification results
    preds_2025 = pd.merge(
        xgb_2025,
        rf_2025[['causer_acronym','year','month','RF_prob','RF_lower','RF_upper']],
        on=['causer_acronym','year','month'],
        how='left'
    )

    # F) Exponential Smoothing => forecast for each causer
    #    We'll loop through all causers, combine results
    es_list = []
    for c in monthly_df['causer_acronym'].unique():
        es_part = exponential_smoothing_forecast(monthly_df, c)
        es_list.append(es_part)
    es_df = pd.concat(es_list, ignore_index=True)

    # Filter only year=2025
    es_2025 = es_df[es_df['year'] == 2025].copy()

    # Merge ES results
    final_2025 = pd.merge(
        preds_2025,
        es_2025[['causer_acronym','year','month','ES_prob','ES_lower','ES_upper']],
        on=['causer_acronym','year','month'],
        how='left'
    )

    # G) Survival Analysis => produce monthly probabilities for 2025
    surv_list = []
    for c in monthly_df['causer_acronym'].unique():
        s_df = run_survival_analysis(df_cleaned, monthly_df, c)
        if not s_df.empty:
            surv_list.append(s_df)
    if surv_list:
        surv_final = pd.concat(surv_list, ignore_index=True)
        # Merge
        final_2025 = pd.merge(
            final_2025,
            surv_final[['causer_acronym','year','month','Survival_prob']],
            on=['causer_acronym','year','month'],
            how='left'
        )
    else:
        final_2025['Survival_prob'] = np.nan

    # Reorder columns
    final_2025 = final_2025[[
        'causer_acronym','year','month',
        'XGB_prob','XGB_lower','XGB_upper',
        'RF_prob','RF_lower','RF_upper',
        'ES_prob','ES_lower','ES_upper',
        'Survival_prob'
    ]]

    # Save
    final_2025.to_csv("final_predictions_2025.csv", index=False)
    print("--- SAVED final_predictions_2025.csv ---")


# --------------------------------------------------------------------
# Usage
# --------------------------------------------------------------------
if __name__ == "__main__":
    """
    Example usage:
       df_raw = pd.read_csv("some_tickets_data.csv")
       main(df_raw)
    """
    print("This script defines 'main(return_df)'. Load your data as 'return_df' then call main(return_df).")
