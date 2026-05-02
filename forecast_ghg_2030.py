"""
GHG Emissions Per Capita Forecasting to 2030 - ALL 11 SEA COUNTRIES
Includes: Preprocessing, Validation, and Model Details
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
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

FORECAST_YEAR = 2030

# ============================================================================
# STEP 1: LOAD DATA AND BEST INDICATORS
# ============================================================================

print("\n" + "="*100)
print("STEP 1: LOADING DATA AND BEST INDICATORS")
print("="*100)

# Load main dataset
df = pd.read_csv('WB_WDI_WIDEF.csv')
print(f"✓ Loaded main dataset: {df.shape}")

# Load best indicators CSV
best_indicators_df = pd.read_csv('model_outputs/sea_countries_best_indicators.csv')
print(f"✓ Loaded best indicators for {len(best_indicators_df)} countries")

# Identify year columns
year_cols = sorted([col for col in df.columns if str(col).isdigit()])
year_cols_int = [int(col) for col in year_cols]
max_year = max(year_cols_int)

print(f"✓ Year range in data: {min(year_cols_int)} - {max_year}")

# ============================================================================
# STEP 2: PREPROCESSING AND MODEL FITTING FOR EACH COUNTRY
# ============================================================================

print("\n" + "="*100)
print("STEP 2: PREPROCESSING, VALIDATION, AND FORECASTING FOR EACH COUNTRY")
print("="*100)

all_forecasts = []
detailed_reports = []

for country_code in SEA_COUNTRIES:
    
    print(f"\n{'='*100}")
    print(f"COUNTRY: {COUNTRY_NAMES[country_code].upper()} ({country_code})")
    print(f"{'='*100}")
    
    # Get best indicators for this country
    country_row = best_indicators_df[best_indicators_df['Country_Code'] == country_code]
    
    if country_row.empty:
        print(f"✗ No indicators found for {country_code}")
        continue
    
    country_row = country_row.iloc[0]
    predictors_str = country_row['Predictors']
    predictors_list = [p.strip() for p in predictors_str.split('|')]
    num_predictors = len(predictors_list)
    
    print(f"\n[MODEL SELECTION] Selected {num_predictors} predictors (BIC-optimized):")
    for i, pred in enumerate(predictors_list, 1):
        print(f"  {i}. {pred}")
    
    # Filter data for this country
    country_data = df[df['REF_AREA'] == country_code].copy()
    
    if country_data.empty:
        print(f"✗ No country data found")
        continue
    
    # ========================================================================
    # PREPROCESSING STEP 1: Extract Target and Features
    # ========================================================================
    
    print(f"\n[PREPROCESSING] Step 1: Extracting Target and Features")
    
    target_name = "Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)"
    
    # Check if target exists
    target_rows = country_data[country_data['INDICATOR_LABEL'] == target_name]
    if target_rows.empty:
        print(f"✗ Target variable not found")
        continue
    
    target_values = pd.to_numeric(target_rows.iloc[0][year_cols], errors='coerce')
    print(f"  ✓ Target variable: {target_name}")
    print(f"    - Data points with values: {target_values.notna().sum()} / {len(target_values)}")
    
    # Extract features
    features_data = []
    feature_names_found = []
    
    for pred_name in predictors_list:
        pred_rows = country_data[country_data['INDICATOR_LABEL'] == pred_name]
        if not pred_rows.empty:
            pred_values = pd.to_numeric(pred_rows.iloc[0][year_cols], errors='coerce')
            features_data.append(pred_values.values)
            feature_names_found.append(pred_name)
    
    if len(feature_names_found) < len(predictors_list):
        print(f"⚠ Warning: Only {len(feature_names_found)}/{len(predictors_list)} features found in dataset")
    
    # ========================================================================
    # PREPROCESSING STEP 2: Align Data and Handle Missing Values
    # ========================================================================
    
    print(f"\n[PREPROCESSING] Step 2: Data Alignment and Missing Value Handling")
    
    X_raw = np.array(features_data).T
    y_raw = target_values.values
    
    print(f"  ✓ Data shape before imputation: X={X_raw.shape}, y={y_raw.shape}")
    
    # Create matrix aligned by years
    valid_idx = ~np.isnan(y_raw)
    X_valid = X_raw[valid_idx]
    y_valid = y_raw[valid_idx]
    valid_years = np.array(year_cols_int)[valid_idx]
    
    print(f"  ✓ Valid target samples: {len(y_valid)}")
    print(f"    - Year range: {valid_years[0]} - {valid_years[-1]}")
    print(f"    - Missing features per year: {np.sum(np.isnan(X_valid), axis=1)}")
    
    # Impute missing features using mean strategy
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_valid)
    
    print(f"  ✓ Imputed missing feature values using mean strategy")
    print(f"    - Data shape after imputation: {X_imputed.shape}")
    
    # ========================================================================
    # PREPROCESSING STEP 3: Feature Scaling
    # ========================================================================
    
    print(f"\n[PREPROCESSING] Step 3: Feature Scaling")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    print(f"  ✓ Scaled features using StandardScaler")
    print(f"    - Feature means (should be ~0): {np.mean(X_scaled, axis=0).round(6)}")
    print(f"    - Feature stds (should be ~1): {np.std(X_scaled, axis=0).round(6)}")
    
    # ========================================================================
    # VALIDATION STEP 1: Cross-Validation
    # ========================================================================
    
    print(f"\n[VALIDATION] Step 1: K-Fold Cross-Validation (k=5)")
    
    model = LinearRegression()
    k = min(5, len(y_valid) // 3)
    kf = KFold(n_splits=k, shuffle=True, random_state=123)
    
    cv_scores = []
    cv_r2_scores = []
    fold_info = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_valid[train_idx], y_valid[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = np.mean((y_test - y_pred)**2)
        rmse = np.sqrt(mse)
        r2 = model.score(X_test, y_test)
        
        cv_scores.append(rmse)
        cv_r2_scores.append(r2)
        fold_info.append({
            'fold': fold_idx,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'rmse': rmse,
            'r2': r2
        })
    
    cv_rmse_mean = np.mean(cv_scores)
    cv_rmse_std = np.std(cv_scores)
    cv_r2_mean = np.mean(cv_r2_scores)
    
    print(f"  Fold Results:")
    for info in fold_info:
        print(f"    Fold {info['fold']}: train={info['train_size']}, test={info['test_size']}, " +
              f"RMSE={info['rmse']:.4f}, R²={info['r2']:.4f}")
    
    print(f"  ✓ Cross-Validation Summary:")
    print(f"    - Mean CV-RMSE: {cv_rmse_mean:.6f} (±{cv_rmse_std:.6f})")
    print(f"    - Mean CV-R²: {cv_r2_mean:.6f}")
    
    # ========================================================================
    # VALIDATION STEP 2: Full Model Training
    # ========================================================================
    
    print(f"\n[VALIDATION] Step 2: Full Model Training")
    
    model_full = LinearRegression()
    model_full.fit(X_scaled, y_valid)
    
    y_pred_full = model_full.predict(X_scaled)
    residuals = y_valid - y_pred_full
    mse_full = np.mean(residuals**2)
    rmse_full = np.sqrt(mse_full)
    r2_full = model_full.score(X_scaled, y_valid)
    
    print(f"  ✓ Full Model Performance (on all training data):")
    print(f"    - RMSE: {rmse_full:.6f}")
    print(f"    - R²: {r2_full:.6f}")
    print(f"    - Mean Residual: {np.mean(residuals):.6f}")
    print(f"    - Residual Std: {np.std(residuals):.6f}")
    
    # ========================================================================
    # MODEL COEFFICIENTS
    # ========================================================================
    
    print(f"\n[MODEL] Fitted Coefficients")
    
    print(f"  Intercept: {model_full.intercept_:.6f}")
    print(f"  Coefficients (on scaled features):")
    for feat_name, coef in zip(feature_names_found, model_full.coef_):
        print(f"    - {feat_name}: {coef:.6f}")
    
    # ========================================================================
    # FORECASTING STEP: Predict to 2030
    # ========================================================================
    
    print(f"\n[FORECASTING] Predicting to {FORECAST_YEAR}")
    
    # Get the most recent year's feature values
    latest_year_idx = len(valid_years) - 1
    latest_year = valid_years[latest_year_idx]
    latest_features_scaled = X_scaled[latest_year_idx].reshape(1, -1)
    
    print(f"  ✓ Latest year in data: {latest_year}")
    print(f"    - Target value: {y_valid[latest_year_idx]:.4f} t CO2e/capita")
    print(f"    - Latest feature values (original scale):")
    for feat_name, feat_val in zip(feature_names_found, X_imputed[latest_year_idx]):
        print(f"      • {feat_name}: {feat_val:.2e}")
    
    # For forecasting, we'll use a simple approach:
    # Assume feature growth rate based on recent years (last 5 years average)
    
    years_for_projection = min(5, len(valid_years) - 1)
    
    if years_for_projection > 1:
        X_recent = X_imputed[-years_for_projection:]
        growth_rates = []
        
        for feature_idx in range(X_recent.shape[1]):
            if X_recent[0, feature_idx] != 0:
                feature_cagr = (X_recent[-1, feature_idx] / X_recent[0, feature_idx]) ** (
                    1 / (years_for_projection - 1)
                ) - 1
                growth_rates.append(feature_cagr)
            else:
                growth_rates.append(0)
        
        print(f"  ✓ Estimated feature growth rates (from last {years_for_projection} years):")
        for feat_name, growth_rate in zip(feature_names_found, growth_rates):
            print(f"    - {feat_name}: {growth_rate*100:.2f}% annually")
        
        # Project features to 2030
        years_to_project = FORECAST_YEAR - latest_year
        X_2030 = X_imputed[-1].copy()
        
        for feature_idx, growth_rate in enumerate(growth_rates):
            X_2030[feature_idx] = X_imputed[-1, feature_idx] * ((1 + growth_rate) ** years_to_project)
        
        # Scale the projected features
        X_2030_scaled = scaler.transform(X_2030.reshape(1, -1))
        
        # Make prediction
        y_2030_pred = model_full.predict(X_2030_scaled)[0]
        
        print(f"\n  ✓ Projected features to {FORECAST_YEAR}:")
        for feat_name, feat_val_2030, feat_val_now in zip(
            feature_names_found, X_2030, X_imputed[-1]
        ):
            change_pct = (feat_val_2030 / feat_val_now - 1) * 100 if feat_val_now != 0 else 0
            print(f"    - {feat_name}:")
            print(f"      Now ({latest_year}): {feat_val_now:.2e}")
            print(f"      2030: {feat_val_2030:.2e} ({change_pct:+.1f}%)")
        
        forecast_change = y_2030_pred - y_valid[-1]
        forecast_pct = (forecast_change / y_valid[-1]) * 100 if y_valid[-1] != 0 else 0
        
        print(f"\n  ✓ GHG EMISSIONS FORECAST:")
        print(f"    - {latest_year}: {y_valid[-1]:.4f} t CO2e/capita")
        print(f"    - {FORECAST_YEAR}: {y_2030_pred:.4f} t CO2e/capita")
        print(f"    - Change: {forecast_change:+.4f} ({forecast_pct:+.1f}%)")
    
    else:
        # Not enough historical data for growth rate estimation
        print(f"  ⚠ Insufficient historical data for growth rate projection")
        y_2030_pred = y_valid[-1]  # Use last known value
    
    # Store forecast
    all_forecasts.append({
        'Country_Code': country_code,
        'Country_Name': COUNTRY_NAMES[country_code],
        'Num_Predictors': num_predictors,
        'Latest_Year': latest_year,
        'Latest_GHG_Per_Capita': y_valid[-1],
        'GHG_2030_Forecast': y_2030_pred,
        'Change_2030': y_2030_pred - y_valid[-1],
        'Pct_Change_2030': ((y_2030_pred - y_valid[-1]) / y_valid[-1] * 100) if y_valid[-1] != 0 else 0,
        'CV_RMSE': cv_rmse_mean,
        'CV_R2': cv_r2_mean,
        'Full_R2': r2_full,
        'Model_RMSE': rmse_full
    })
    
    # Store detailed report
    detailed_reports.append({
        'code': country_code,
        'name': COUNTRY_NAMES[country_code],
        'predictors': feature_names_found,
        'coefficients': model_full.coef_,
        'intercept': model_full.intercept_,
        'cv_rmse': cv_rmse_mean,
        'r2': r2_full,
        'latest_year': latest_year,
        'latest_ghg': y_valid[-1],
        'forecast_2030': y_2030_pred
    })

# ============================================================================
# STEP 3: COMPILATION AND EXPORT
# ============================================================================

print("\n" + "="*100)
print("STEP 3: COMPILING RESULTS AND EXPORTING")
print("="*100)

if all_forecasts:
    forecasts_df = pd.DataFrame(all_forecasts)
    forecasts_df = forecasts_df.sort_values('GHG_2030_Forecast', ascending=False)
    
    print("\nFORECAST SUMMARY - GHG EMISSIONS PER CAPITA TO 2030:\n")
    display_cols = ['Country_Name', 'Latest_Year', 'Latest_GHG_Per_Capita', 
                    'GHG_2030_Forecast', 'Change_2030', 'Pct_Change_2030']
    print(forecasts_df[display_cols].to_string(index=False))
    
    # Save forecast
    forecasts_df.to_csv('model_outputs/ghg_forecast_2030.csv', index=False)
    print("\n✓ Saved: model_outputs/ghg_forecast_2030.csv")

# ============================================================================
# STEP 4: DETAILED REPORT FILE
# ============================================================================

print("\nGenerating detailed report...")

with open('model_outputs/ghg_forecast_2030_detailed.txt', 'w') as f:
    f.write("="*100 + "\n")
    f.write("GHG EMISSIONS PER CAPITA FORECAST TO 2030 - DETAILED REPORT\n")
    f.write("ALL 11 SEA COUNTRIES - WITH PREPROCESSING, VALIDATION, AND MODEL DETAILS\n")
    f.write("="*100 + "\n\n")
    
    for forecast in all_forecasts:
        code = forecast['Country_Code']
        report = next((r for r in detailed_reports if r['code'] == code), None)
        
        if not report:
            continue
        
        f.write(f"\n{'='*100}\n")
        f.write(f"{forecast['Country_Name'].upper()} ({code})\n")
        f.write(f"{'='*100}\n\n")
        
        f.write(f"[MODEL SELECTION]\n")
        f.write(f"Number of Predictors: {forecast['Num_Predictors']}\n")
        f.write(f"Selected Indicators:\n")
        for i, pred in enumerate(report['predictors'], 1):
            f.write(f"  {i}. {pred}\n")
        
        f.write(f"\n[PREPROCESSING]\n")
        f.write(f"• Data alignment and imputation: Mean strategy for missing features\n")
        f.write(f"• Feature scaling: StandardScaler (mean=0, std=1)\n")
        f.write(f"• Training samples: {len(all_forecasts)} time periods\n")
        
        f.write(f"\n[VALIDATION METRICS]\n")
        f.write(f"• Cross-Validation RMSE (5-fold): {forecast['CV_RMSE']:.6f}\n")
        f.write(f"• Cross-Validation R²: {forecast['CV_R2']:.6f}\n")
        f.write(f"• Full Model RMSE: {forecast['Model_RMSE']:.6f}\n")
        f.write(f"• Full Model R²: {forecast['Full_R2']:.6f}\n")
        
        f.write(f"\n[MODEL COEFFICIENTS]\n")
        f.write(f"Intercept (scaled features): {report['intercept']:.6f}\n")
        f.write(f"Regression Coefficients:\n")
        for pred, coef in zip(report['predictors'], report['coefficients']):
            f.write(f"  • {pred}: {coef:.6f}\n")
        
        f.write(f"\n[GHG EMISSIONS FORECAST]\n")
        f.write(f"Latest Year: {forecast['Latest_Year']}\n")
        f.write(f"  GHG Emissions: {forecast['Latest_GHG_Per_Capita']:.4f} t CO2e/capita\n\n")
        f.write(f"Forecast Year: 2030\n")
        f.write(f"  GHG Emissions: {forecast['GHG_2030_Forecast']:.4f} t CO2e/capita\n\n")
        f.write(f"Change 2030 vs {forecast['Latest_Year']}:\n")
        f.write(f"  Absolute: {forecast['Change_2030']:+.4f} t CO2e/capita\n")
        f.write(f"  Percentage: {forecast['Pct_Change_2030']:+.1f}%\n")

print("✓ Saved: model_outputs/ghg_forecast_2030_detailed.txt")

print("\n" + "="*100)
print("FORECASTING COMPLETE")
print("="*100)
print("\nOutput files:")
print("  - model_outputs/ghg_forecast_2030.csv")
print("  - model_outputs/ghg_forecast_2030_detailed.txt")
