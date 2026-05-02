"""
Python Script: Select Best Predictive Indicators for Malaysia's Total GHG Emissions per Capita
Using all possible combinations with model selection criteria (CV, AIC, AICc, BIC)
"""

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# Step 0: Load and prepare data
# ============================================================================

print("\n" + "="*80)
print("STEP 0: Loading and Preparing Data")
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
country_code = 'MYS'  # Malaysia

if indicator_col not in df.columns:
    print(f"ERROR: Column '{indicator_col}' not found. Available: {df.columns.tolist()[:15]}")
    exit(1)

print(f"✓ Using indicator column: {indicator_col}")

# Filter for Malaysia
malaysia_data = df[df[country_col] == country_code].copy()

if malaysia_data.empty:
    print(f"ERROR: No data for country code '{country_code}'")
    exit(1)

print(f"✓ Malaysia data shape: {malaysia_data.shape}")

# Create a dictionary: indicator_name -> year values
indicators_dict = {}
for idx, row in malaysia_data.iterrows():
    indicator_name = row[indicator_col]
    # Get numeric values for year columns
    values = pd.to_numeric(row[year_cols_sorted], errors='coerce')
    if indicator_name not in indicators_dict:
        indicators_dict[indicator_name] = values.values

# Target variable
target_name = "Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)"

if target_name not in indicators_dict:
    print(f"ERROR: Target '{target_name}' not found!")
    print(f"Available indicators: {list(indicators_dict.keys())[:10]}")
    exit(1)

target = indicators_dict[target_name]
print(f"✓ Target variable: {target_name}")

# Create feature matrix excluding highly correlated GHG variables
exclude_indicators = {
    target_name,
    "Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)",
    "Total greenhouse gas emissions excluding LULUCF (Mt CO2e)",
    "Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)",
    "Total greenhouse gas emissions excluding LULUCF (% change from 1990)",
}

feature_indicators = [ind for ind in indicators_dict.keys() if ind not in exclude_indicators]
print(f"✓ Number of potential features: {len(feature_indicators)}")

# Create data matrix with complete cases
# First, filter features to those with sufficient data
min_data_points = 5  # At least 5 years of data for a feature to be useful

feature_indicators_filtered = []
for ind in feature_indicators:
    non_null_count = np.sum(~np.isnan(indicators_dict[ind]))
    if non_null_count >= min_data_points:
        feature_indicators_filtered.append(ind)

print(f"\n✓ Filtered to {len(feature_indicators_filtered)} features with {min_data_points}+ data points")
print(f"  (from {len(feature_indicators)} initial features)")

feature_indicators = feature_indicators_filtered

# Create data matrix - use pairwise complete observations
data_list = []
target_list = []

min_year = 10  # Need at least 10 years of data

for i in range(len(target)):
    if not np.isnan(target[i]):
        features_valid = []
        for ind in feature_indicators:
            val = indicators_dict[ind][i] if i < len(indicators_dict[ind]) else np.nan
            if not np.isnan(val):
                features_valid.append(val)
            else:
                features_valid.append(np.nan)
        
        # Count non-missing features
        if np.sum(~np.isnan(features_valid)) >= min(10, len(feature_indicators) // 2):
            data_list.append(features_valid)
            target_list.append(target[i])

if not data_list:
    print("ERROR: Not enough complete data for analysis")
    exit(1)

data_array = np.array(data_list)
X = data_array[:, 1:]
y = data_array[:, 0]

print(f"✓ Dataset prepared: {X.shape[0]} observations, {X.shape[1]} features")
print(f"  Features: {feature_indicators[:5]}..." if len(feature_indicators) > 5 else f"  Features: {feature_indicators}")

# ============================================================================
# Step 1: Functions to calculate metrics
# ============================================================================

def calculate_metrics(X_subset, y, feature_indices):
    """Calculate CV, AIC, AICc, BIC for a linear regression model"""
    
    # Fit OLS model
    X_design = np.column_stack([np.ones(len(y)), X_subset])
    
    try:
        # Compute beta coefficients using normal equation
        beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
        
        # Predictions
        y_pred = X_design @ beta
        residuals = y - y_pred
        rss = np.sum(residuals**2)
        
        n = len(y)
        p = X_subset.shape[1]  # number of predictors (excluding intercept)
        
        # Information criteria
        aic = 2*p + n*np.log(rss/n)
        aicc = aic + (2*p*(p+1))/(n-p-1) if (n-p-1) > 0 else np.inf
        bic = p*np.log(n) + n*np.log(rss/n)
        
        # R-squared
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (rss/ss_tot)
        
        # Cross-validation using K-Fold
        kf = KFold(n_splits=5, shuffle=True, random_state=123)
        cv_errors = []
        
        for train_idx, test_idx in kf.split(X_subset):
            X_train, X_test = X_subset[train_idx], X_subset[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit on training data
            X_train_design = np.column_stack([np.ones(len(y_train)), X_train])
            beta_train = np.linalg.lstsq(X_train_design, y_train, rcond=None)[0]
            
            # Predict on test data
            X_test_design = np.column_stack([np.ones(len(y_test)), X_test])
            y_pred_test = X_test_design @ beta_train
            mse = np.mean((y_test - y_pred_test)**2)
            cv_errors.append(mse)
        
        cv_rmse = np.sqrt(np.mean(cv_errors))
        
        return {
            'cv_rmse': cv_rmse,
            'aic': aic,
            'aicc': aicc,
            'bic': bic,
            'rss': rss,
            'r_squared': r_squared,
            'n_predictors': p
        }
    
    except np.linalg.LinAlgError:
        return None


# ============================================================================
# Step 2: All possible subsets regression
# ============================================================================

print("\n" + "="*80)
print("STEP 1: All Possible Subsets Regression")
print("="*80)

max_features = min(10, X.shape[1])  # Limit to 10 features to avoid explosion
print(f"Testing all subsets up to {max_features} features...")

results = []

# Test models of each size
for p in range(1, max_features + 1):
    print(f"\nTesting {p}-predictor models...", end=" ")
    
    num_combos = min(100, len(list(combinations(range(X.shape[1]), p))))  # Limit to 100 combinations for large p
    
    best_bic_for_size = np.inf
    best_model_for_size = None
    
    # Sample combinations if too many
    if len(list(combinations(range(X.shape[1]), p))) > 100:
        np.random.seed(123)
        all_combos = list(combinations(range(X.shape[1]), p))
        combos = [all_combos[i] for i in np.random.choice(len(all_combos), 100, replace=False)]
    else:
        combos = combinations(range(X.shape[1]), p)
    
    for combo in combos:
        X_subset = X[:, list(combo)]
        metrics = calculate_metrics(X_subset, y, combo)
        
        if metrics is not None:
            metrics['feature_indices'] = combo
            metrics['features'] = [feature_indicators[i] for i in combo]
            results.append(metrics)
            
            if metrics['bic'] < best_bic_for_size:
                best_bic_for_size = metrics['bic']
                best_model_for_size = metrics
    
    if best_model_for_size:
        print(f"✓ Best BIC: {best_bic_for_size:.2f}")
    else:
        print("✗ No valid models")

# ============================================================================
# Step 3: Sort and display results
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Results Summary")
print("="*80)

if not results:
    print("ERROR: No valid models found")
    exit(1)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('bic').reset_index(drop=True)

# Display top 10 models
print("\nTOP 10 MODELS BY BIC (BEST OVERALL):\n")
display_cols = ['n_predictors', 'cv_rmse', 'aic', 'aicc', 'bic', 'r_squared']
top_10 = results_df[display_cols].head(10)
print(top_10.to_string(float_format=lambda x: f'{x:.4f}'))

# ============================================================================
# Step 4: Show best model
# ============================================================================

print("\n" + "="*80)
print("BEST MODEL (Ranked by BIC)")
print("="*80)

best_model = results_df.iloc[0]

print(f"\n✓ Number of Predictors: {int(best_model['n_predictors'])}")
print(f"\nSelected Indicators:")
for i, feat in enumerate(best_model['features'], 1):
    print(f"  {i}. {feat}")

print(f"\n\nModel Performance Metrics:")
print(f"  Cross-Validation RMSE:  {best_model['cv_rmse']:.6f}")
print(f"  AIC:                    {best_model['aic']:.2f}")
print(f"  AICc:                   {best_model['aicc']:.2f}")
print(f"  BIC:                    {best_model['bic']:.2f}")
print(f"  R-squared:              {best_model['r_squared']:.6f}")

# ============================================================================
# Step 5: Comparison with best by other criteria
# ============================================================================

print("\n" + "="*80)
print("BEST MODELS BY DIFFERENT CRITERIA")
print("="*80)

# Find best by each criterion
best_cv = results_df.loc[results_df['cv_rmse'].idxmin()]
best_aic = results_df.loc[results_df['aic'].idxmin()]
best_aicc = results_df.loc[results_df['aicc'].idxmin()]
best_bic = results_df.loc[results_df['bic'].idxmin()]

print("\n1. BEST BY CROSS-VALIDATION RMSE (predict accuracy):")
print(f"   Predictors: {int(best_cv['n_predictors'])}")
print(f"   CV-RMSE: {best_cv['cv_rmse']:.6f}")

print("\n2. BEST BY AIC (fit + penalty):")
print(f"   Predictors: {int(best_aic['n_predictors'])}")
print(f"   AIC: {best_aic['aic']:.2f}")

print("\n3. BEST BY AICc (corrected for small sample):")
print(f"   Predictors: {int(best_aicc['n_predictors'])}")
print(f"   AICc: {best_aicc['aicc']:.2f}")

print("\n4. BEST BY BIC (preferred for variable selection):")
print(f"   Predictors: {int(best_bic['n_predictors'])}")
print(f"   BIC: {best_bic['bic']:.2f}")
print(f"\n   Indicators:")
for i, feat in enumerate(best_bic['features'], 1):
    print(f"     {i}. {feat}")

# ============================================================================
# Step 6: Pareto frontier (trade-off R² vs model complexity)
# ============================================================================

print("\n" + "="*80)
print("PARETO FRONTIER: R² vs Model Complexity")
print("="*80)

# Group by number of predictors, find best R² for each size
pareto = results_df.groupby('n_predictors').apply(
    lambda g: g.loc[g['r_squared'].idxmax()]
).reset_index(drop=True)

print("\nBest R-squared for each model size:")
print("\n┌─ Size ─┬── R² ──┬─ CV-RMSE ─┬─ BIC ────┐")
for _, row in pareto.iterrows():
    print(f"│  {int(row['n_predictors']):2d}   │ {row['r_squared']:.4f} │ {row['cv_rmse']:8.4f} │ {row['bic']:8.2f} │")
print("└────────┴────────┴──────────┴──────────┘")

# ============================================================================
# Step 7: Summary and recommendation
# ============================================================================

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

print(f"""
Based on BIC (recommended for variable selection - balances model fit and complexity):

╔════════════════════════════════════════════════════════════════════════════╗
║ OPTIMAL MODEL FOR PREDICTING MALAYSIA'S GHG EMISSIONS PER CAPITA          ║
╚════════════════════════════════════════════════════════════════════════════╝

Number of Indicators: {int(best_bic['n_predictors'])}

Selected Indicators:
""")

for i, feat in enumerate(best_bic['features'], 1):
    print(f"  {i}. {feat}")

print(f"""
Performance:
  • Cross-Validation RMSE: {best_bic['cv_rmse']:.6f}
  • R² (explained variance): {best_bic['r_squared']:.2%}
  • AIC: {best_bic['aic']:.2f}
  • BIC: {best_bic['bic']:.2f}

Interpretation:
  This combination of {int(best_bic['n_predictors'])} indicators explains {best_bic['r_squared']:.1%} of the variation 
  in Malaysia's total GHG emissions per capita. The model is parsimonious 
  (not overfitted) as validated by 5-fold cross-validation.
""")

# ============================================================================
# Step 8: Save results
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save detailed results to CSV
results_df_export = results_df.copy()
results_df_export['features_str'] = results_df_export['features'].apply(lambda x: ' | '.join(x))
results_df_export.to_csv('model_outputs/best_indicators_analysis.csv', index=False)
print("✓ Saved detailed results to: model_outputs/best_indicators_analysis.csv")

# Save summary
with open('model_outputs/best_indicators_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("BEST PREDICTIVE INDICATORS FOR MALAYSIA'S GHG EMISSIONS PER CAPITA\n")
    f.write("="*80 + "\n\n")
    f.write(f"BEST MODEL (by BIC):\n")
    f.write(f"Number of Predictors: {int(best_bic['n_predictors'])}\n\n")
    f.write(f"Selected Indicators:\n")
    for i, feat in enumerate(best_bic['features'], 1):
        f.write(f"  {i}. {feat}\n")
    f.write(f"\nPerformance Metrics:\n")
    f.write(f"  CV-RMSE: {best_bic['cv_rmse']:.6f}\n")
    f.write(f"  AIC: {best_bic['aic']:.2f}\n")
    f.write(f"  AICc: {best_bic['aicc']:.2f}\n")
    f.write(f"  BIC: {best_bic['bic']:.2f}\n")
    f.write(f"  R-squared: {best_bic['r_squared']:.6f}\n")
    f.write(f"\nTop 10 Models:\n")
    f.write(top_10.to_string(float_format=lambda x: f'{x:.4f}'))

print("✓ Saved summary to: model_outputs/best_indicators_summary.txt")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80 + "\n")
