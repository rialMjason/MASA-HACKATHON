"""
Python Script: Best Predictive Indicators for GHG Emissions per Capita - ALL 11 SEA COUNTRIES
Using all possible combinations with model selection criteria (CV, AIC, AICc, BIC)
"""

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

SEA_COUNTRIES = ['BRN', 'KHM', 'IDN', 'LAO', 'MYS', 'MMR', 'PHL', 'SGP', 'THA', 'TLS', 'VNM']
COUNTRY_NAMES = {
    'BRN': 'Brunei',
    'KHM': 'Cambodia',
    'IDN': 'Indonesia',
    'LAO': 'Laos',
    'MYS': 'Malaysia',
    'MMR': 'Myanmar',
    'PHL': 'Philippines',
    'SGP': 'Singapore',
    'THA': 'Thailand',
    'TLS': 'Timor-Leste',
    'VNM': 'Vietnam'
}

# ============================================================================
# Step 1: Load and prepare data (once for all countries)
# ============================================================================

print("\n" + "="*80)
print("LOADING AND PREPARING DATA FOR ALL 11 SEA COUNTRIES")
print("="*80)

# Load the World Bank WDI data
df = pd.read_csv('WB_WDI_WIDEF.csv')
print(f"✓ Loaded dataset with shape: {df.shape}")

# Identify year columns (numeric column names)
year_cols = [col for col in df.columns if str(col).isdigit()]
year_cols_sorted = sorted(year_cols)

if not year_cols_sorted:
    print("ERROR: No year columns found!")
    exit(1)

print(f"✓ Found {len(year_cols_sorted)} year columns: {year_cols_sorted[0]} to {year_cols_sorted[-1]}")

# Identify the indicator label column
indicator_col = 'INDICATOR_LABEL'
country_col = 'REF_AREA'

if indicator_col not in df.columns:
    print(f"ERROR: Column '{indicator_col}' not found.")
    exit(1)

# Target variable
target_name = "Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)"

# ============================================================================
# Function to calculate metrics
# ============================================================================

def calculate_metrics(X_subset, y, feature_indices):
    """Calculate CV, AIC, AICc, BIC for a linear regression model"""
    
    X_design = np.column_stack([np.ones(len(y)), X_subset])
    
    try:
        beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
        
        y_pred = X_design @ beta
        residuals = y - y_pred
        rss = np.sum(residuals**2)
        
        n = len(y)
        p = X_subset.shape[1]
        
        aic = 2*p + n*np.log(rss/n)
        aicc = aic + (2*p*(p+1))/(n-p-1) if (n-p-1) > 0 else np.inf
        bic = p*np.log(n) + n*np.log(rss/n)
        
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (rss/ss_tot) if ss_tot > 0 else 0
        
        k = min(5, n // 3)
        kf = KFold(n_splits=k, shuffle=True, random_state=123)
        cv_errors = []
        
        for train_idx, test_idx in kf.split(X_subset):
            X_train, X_test = X_subset[train_idx], X_subset[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            X_train_design = np.column_stack([np.ones(len(y_train)), X_train])
            beta_train = np.linalg.lstsq(X_train_design, y_train, rcond=None)[0]
            
            X_test_design = np.column_stack([np.ones(len(y_test)), X_test])
            y_pred_test = X_test_design @ beta_train
            mse = np.mean((y_test - y_pred_test)**2)
            cv_errors.append(mse)
        
        cv_rmse = np.sqrt(np.mean(cv_errors)) if cv_errors else np.inf
        
        return {
            'cv_rmse': cv_rmse,
            'aic': aic,
            'aicc': aicc,
            'bic': bic,
            'rss': rss,
            'r_squared': r_squared,
            'n_predictors': p
        }
    
    except (np.linalg.LinAlgError, ValueError):
        return None

# ============================================================================
# Process each country
# ============================================================================

all_country_results = {}

for country_code in SEA_COUNTRIES:
    
    print("\n" + "="*80)
    print(f"ANALYZING {COUNTRY_NAMES[country_code].upper()} ({country_code})")
    print("="*80)
    
    # Filter for this country
    country_data = df[df[country_col] == country_code].copy()
    
    if country_data.empty:
        print(f"✗ No data for {country_code}")
        continue
    
    print(f"✓ Country data shape: {country_data.shape}")
    
    # Create indicators dictionary
    indicators_dict = {}
    for idx, row in country_data.iterrows():
        indicator_name = row[indicator_col]
        values = pd.to_numeric(row[year_cols_sorted], errors='coerce')
        if indicator_name not in indicators_dict:
            indicators_dict[indicator_name] = values.values
    
    # Check if target exists
    if target_name not in indicators_dict:
        print(f"✗ Target variable not found for {country_code}")
        continue
    
    target = indicators_dict[target_name]
    target_non_null = np.sum(~np.isnan(target))
    
    if target_non_null < 3:
        print(f"✗ Insufficient target data ({target_non_null} points) for {country_code}")
        continue
    
    print(f"✓ Target variable found: {target_non_null} data points")
    
    # Create feature matrix excluding GHG-related targets
    exclude_indicators = {
        target_name,
        "Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)",
        "Total greenhouse gas emissions excluding LULUCF (Mt CO2e)",
        "Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)",
        "Total greenhouse gas emissions excluding LULUCF (% change from 1990)",
    }
    
    feature_indicators = [ind for ind in indicators_dict.keys() if ind not in exclude_indicators]
    
    # Filter by minimum data points
    min_data_points = 5
    feature_indicators_filtered = []
    for ind in feature_indicators:
        non_null_count = np.sum(~np.isnan(indicators_dict[ind]))
        if non_null_count >= min_data_points:
            feature_indicators_filtered.append(ind)
    
    feature_indicators = feature_indicators_filtered
    print(f"✓ Available features: {len(feature_indicators)}")
    
    if len(feature_indicators) < 2:
        print(f"✗ Insufficient features for {country_code}")
        continue
    
    # Create data matrix
    data_list = []
    target_list = []
    min_features_required = min(10, max(3, len(feature_indicators) // 3))
    
    for i in range(len(target)):
        if not np.isnan(target[i]):
            features_valid = []
            for ind in feature_indicators:
                val = indicators_dict[ind][i] if i < len(indicators_dict[ind]) else np.nan
                features_valid.append(val)
            
            if np.sum(~np.isnan(features_valid)) >= min_features_required:
                data_list.append(features_valid)
                target_list.append(target[i])
    
    if not data_list:
        print(f"✗ Not enough complete data for {country_code}")
        continue
    
    X = np.array(data_list)
    y = np.array(target_list)
    
    print(f"✓ Dataset prepared: {X.shape[0]} observations, {X.shape[1]} features")
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    # Limit features for computational efficiency
    max_features_for_analysis = 15
    if X.shape[1] > max_features_for_analysis:
        variances = np.var(X, axis=0)
        top_indices = np.argsort(-variances)[:max_features_for_analysis]
        X = X[:, top_indices]
        feature_indicators = [feature_indicators[i] for i in sorted(top_indices)]
        print(f"✓ Reduced to {X.shape[1]} features with highest variance")
    
    # ========================================================================
    # All possible subsets regression for this country
    # ========================================================================
    
    print(f"\nTesting models for {COUNTRY_NAMES[country_code]}...", flush=True)
    
    max_features = min(12, X.shape[1])
    results = []
    
    for p in range(1, max_features + 1):
        all_combos = list(combinations(range(X.shape[1]), p))
        
        max_combos_per_size = 150
        if len(all_combos) > max_combos_per_size:
            np.random.seed(123)
            combo_indices = np.random.choice(len(all_combos), max_combos_per_size, replace=False)
            combos = [all_combos[i] for i in combo_indices]
        else:
            combos = all_combos
        
        for combo in combos:
            X_subset = X[:, list(combo)]
            metrics = calculate_metrics(X_subset, y, combo)
            
            if metrics is not None:
                metrics['feature_indices'] = combo
                metrics['features'] = [feature_indicators[i] for i in combo]
                results.append(metrics)
    
    if not results:
        print(f"✗ No valid models for {country_code}")
        continue
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('bic').reset_index(drop=True)
    
    # Store results for this country
    all_country_results[country_code] = {
        'country_name': COUNTRY_NAMES[country_code],
        'results_df': results_df,
        'best_model': results_df.iloc[0],
        'best_cv': results_df.loc[results_df['cv_rmse'].idxmin()],
        'best_aic': results_df.loc[results_df['aic'].idxmin()],
        'best_bic': results_df.loc[results_df['bic'].idxmin()]
    }
    
    # Display summary for this country
    best = results_df.iloc[0]
    print(f"\n✓ Analysis complete for {COUNTRY_NAMES[country_code]}")
    print(f"   Best Model (BIC): {int(best['n_predictors'])} predictors")
    print(f"   CV-RMSE: {best['cv_rmse']:.4f} | BIC: {best['bic']:.2f} | R²: {best['r_squared']:.4f}")

# ============================================================================
# Compile Summary Report
# ============================================================================

print("\n" + "="*80)
print("SUMMARY FOR ALL 11 SEA COUNTRIES")
print("="*80)

summary_records = []

for country_code in SEA_COUNTRIES:
    if country_code not in all_country_results:
        print(f"✗ {COUNTRY_NAMES[country_code]}: Analysis failed or insufficient data")
        continue
    
    result = all_country_results[country_code]
    best = result['best_model']
    
    summary_records.append({
        'Country_Code': country_code,
        'Country_Name': result['country_name'],
        'Num_Predictors': int(best['n_predictors']),
        'CV_RMSE': best['cv_rmse'],
        'AIC': best['aic'],
        'BIC': best['bic'],
        'R_Squared': best['r_squared'],
        'Predictors': ' | '.join(best['features'])
    })

summary_df = pd.DataFrame(summary_records)
summary_df = summary_df.sort_values('BIC')

print("\nBEST MODELS BY BIC (RECOMMENDED):\n")
print(summary_df[['Country_Name', 'Num_Predictors', 'CV_RMSE', 'BIC', 'R_Squared']].to_string(index=False))

# Save comprehensive results
summary_df.to_csv('model_outputs/sea_countries_best_indicators.csv', index=False)
print("\n✓ Saved: model_outputs/sea_countries_best_indicators.csv")

# ============================================================================
# Detailed Summary File
# ============================================================================

with open('model_outputs/sea_countries_summary.txt', 'w') as f:
    f.write("="*100 + "\n")
    f.write("BEST PREDICTIVE INDICATORS FOR GHG EMISSIONS PER CAPITA - ALL 11 SEA COUNTRIES\n")
    f.write("="*100 + "\n\n")
    
    for idx, row in summary_df.iterrows():
        country_code = row['Country_Code']
        result = all_country_results[country_code]
        best = result['best_model']
        
        f.write(f"\n{idx+1}. {row['Country_Name'].upper()} ({country_code})\n")
        f.write("-" * 100 + "\n")
        f.write(f"Number of Predictors: {int(best['n_predictors'])}\n\n")
        f.write("Selected Indicators:\n")
        for i, feat in enumerate(best['features'], 1):
            f.write(f"  {i}. {feat}\n")
        
        f.write(f"\nModel Performance:\n")
        f.write(f"  Cross-Validation RMSE: {best['cv_rmse']:.6f}\n")
        f.write(f"  AIC:                   {best['aic']:.2f}\n")
        f.write(f"  AICc:                  {best['aicc']:.2f}\n")
        f.write(f"  BIC:                   {best['bic']:.2f}\n")
        f.write(f"  R-squared:             {best['r_squared']:.6f}\n")

print("✓ Saved: model_outputs/sea_countries_summary.txt")

# ============================================================================
# Individual Country Reports
# ============================================================================

for country_code in SEA_COUNTRIES:
    if country_code not in all_country_results:
        continue
    
    result = all_country_results[country_code]
    country_name = result['country_name']
    results_df = result['results_df']
    
    filename = f"model_outputs/best_indicators_{country_code}_{country_name.lower().replace(' ', '_')}.csv"
    results_df.to_csv(filename, index=False)

print("\n✓ Saved individual country reports in model_outputs/")

print("\n" + "="*80)
print("MULTI-COUNTRY ANALYSIS COMPLETE")
print("="*80)
print(f"\nSuccessfully analyzed {len(all_country_results)} countries")
print("\nOutput files:")
print("  - model_outputs/sea_countries_best_indicators.csv")
print("  - model_outputs/sea_countries_summary.txt")
print("  - model_outputs/best_indicators_[CODE]_[COUNTRY].csv (individual reports)")
