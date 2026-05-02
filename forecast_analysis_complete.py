"""
Extended GHG Forecast Analysis: Year-by-Year Projections + Residual Analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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

FORECAST_YEARS = list(range(2025, 2031))  # 2025-2030

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*100)
print("LOADING DATA FOR YEAR-BY-YEAR FORECAST & RESIDUAL ANALYSIS")
print("="*100)

df = pd.read_csv('WB_WDI_WIDEF.csv')
best_indicators_df = pd.read_csv('model_outputs/sea_countries_best_indicators.csv')

year_cols = sorted([col for col in df.columns if str(col).isdigit()])
year_cols_int = [int(col) for col in year_cols]

all_yearly_forecasts = []
all_residuals = []
residuals_summary = []

# ============================================================================
# PROCESS EACH COUNTRY
# ============================================================================

for country_code in SEA_COUNTRIES:
    
    print(f"\n{'='*100}")
    print(f"PROCESSING: {COUNTRY_NAMES[country_code].upper()} ({country_code})")
    print(f"{'='*100}")
    
    # Get best indicators
    country_row = best_indicators_df[best_indicators_df['Country_Code'] == country_code]
    
    if country_row.empty:
        print(f"✗ No indicators found")
        continue
    
    country_row = country_row.iloc[0]
    predictors_str = country_row['Predictors']
    predictors_list = [p.strip() for p in predictors_str.split('|')]
    
    # Filter country data
    country_data = df[df['REF_AREA'] == country_code].copy()
    
    if country_data.empty:
        print(f"✗ No country data found")
        continue
    
    target_name = "Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)"
    
    # Extract target and features
    target_rows = country_data[country_data['INDICATOR_LABEL'] == target_name]
    if target_rows.empty:
        print(f"✗ Target variable not found")
        continue
    
    target_values = pd.to_numeric(target_rows.iloc[0][year_cols], errors='coerce')
    
    # Extract features
    features_data = []
    for pred_name in predictors_list:
        pred_rows = country_data[country_data['INDICATOR_LABEL'] == pred_name]
        if not pred_rows.empty:
            pred_values = pd.to_numeric(pred_rows.iloc[0][year_cols], errors='coerce')
            features_data.append(pred_values.values)
    
    # Align and prepare data
    X_raw = np.array(features_data).T
    y_raw = target_values.values
    
    valid_idx = ~np.isnan(y_raw)
    X_valid = X_raw[valid_idx]
    y_valid = y_raw[valid_idx]
    valid_years = np.array(year_cols_int)[valid_idx]
    
    # Preprocess: Impute and Scale
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_valid)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Train model
    model = LinearRegression()
    model.fit(X_scaled, y_valid)
    
    # Get predictions on training data for residuals
    y_pred_train = model.predict(X_scaled)
    residuals_train = y_valid - y_pred_train
    
    print(f"\n[TRAINING DATA RESIDUALS]")
    print(f"Mean Residual: {np.mean(residuals_train):.6f}")
    print(f"Std Residual: {np.std(residuals_train):.6f}")
    print(f"Min Residual: {np.min(residuals_train):.6f}")
    print(f"Max Residual: {np.max(residuals_train):.6f}")
    
    # Store residuals
    for year, actual, predicted, residual in zip(valid_years, y_valid, y_pred_train, residuals_train):
        all_residuals.append({
            'Country_Code': country_code,
            'Country_Name': COUNTRY_NAMES[country_code],
            'Year': year,
            'Actual_GHG': actual,
            'Predicted_GHG': predicted,
            'Residual': residual,
            'Abs_Residual': abs(residual),
            'Pct_Error': (residual / actual * 100) if actual != 0 else 0
        })
    
    # Calculate summary statistics
    rmse_train = np.sqrt(np.mean(residuals_train**2))
    mae_train = np.mean(np.abs(residuals_train))
    
    residuals_summary.append({
        'Country_Code': country_code,
        'Country_Name': COUNTRY_NAMES[country_code],
        'Sample_Size': len(y_valid),
        'RMSE': rmse_train,
        'MAE': mae_train,
        'Mean_Residual': np.mean(residuals_train),
        'Std_Residual': np.std(residuals_train),
        'Min_Residual': np.min(residuals_train),
        'Max_Residual': np.max(residuals_train)
    })
    
    # ========================================================================
    # YEAR-BY-YEAR FORECASTING (2025-2030)
    # ========================================================================
    
    print(f"\n[YEAR-BY-YEAR FORECAST]")
    
    latest_year = valid_years[-1]
    latest_features = X_imputed[-1]
    
    # Calculate growth rates from last 5 years
    years_for_projection = min(5, len(valid_years) - 1)
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
    
    # Generate forecasts for each year 2025-2030
    X_current = latest_features.copy()
    
    for forecast_year in FORECAST_YEARS:
        years_ahead = forecast_year - latest_year
        
        # Project features
        X_projected = X_current.copy()
        for feature_idx, growth_rate in enumerate(growth_rates):
            X_projected[feature_idx] = X_current[feature_idx] * ((1 + growth_rate) ** years_ahead)
        
        # Scale and predict
        X_projected_scaled = scaler.transform(X_projected.reshape(1, -1))
        ghg_forecast = model.predict(X_projected_scaled)[0]
        
        all_yearly_forecasts.append({
            'Country_Code': country_code,
            'Country_Name': COUNTRY_NAMES[country_code],
            'Forecast_Year': forecast_year,
            'GHG_Per_Capita': ghg_forecast
        })
        
        print(f"  {forecast_year}: {ghg_forecast:.4f} t CO2e/capita")

# ============================================================================
# COMPILE AND EXPORT RESULTS
# ============================================================================

print("\n" + "="*100)
print("COMPILING RESULTS")
print("="*100)

# Convert to DataFrames
yearly_df = pd.DataFrame(all_yearly_forecasts)
residuals_df = pd.DataFrame(all_residuals)
residuals_summary_df = pd.DataFrame(residuals_summary)

# Save CSV files
yearly_df.to_csv('model_outputs/ghg_yearly_forecast_2025_2030.csv', index=False)
residuals_df.to_csv('model_outputs/ghg_model_residuals_detailed.csv', index=False)
residuals_summary_df.to_csv('model_outputs/ghg_residuals_summary.csv', index=False)

print("\n✓ Saved CSV files:")
print("  - model_outputs/ghg_yearly_forecast_2025_2030.csv")
print("  - model_outputs/ghg_model_residuals_detailed.csv")
print("  - model_outputs/ghg_residuals_summary.csv")

# ============================================================================
# GENERATE COMPREHENSIVE TEXT REPORT
# ============================================================================

report_text = ""

report_text += "="*100 + "\n"
report_text += "GHG EMISSIONS FORECAST ANALYSIS: YEAR-BY-YEAR PROJECTIONS & RESIDUAL ANALYSIS\n"
report_text += "ALL 11 SEA COUNTRIES\n"
report_text += "="*100 + "\n\n"

# ========================================================================
# SECTION 1: YEAR-BY-YEAR FORECASTS
# ========================================================================

report_text += "="*100 + "\n"
report_text += "SECTION 1: YEAR-BY-YEAR GHG EMISSIONS FORECASTS (2025-2030)\n"
report_text += "="*100 + "\n\n"

for country_code in SEA_COUNTRIES:
    country_forecasts = yearly_df[yearly_df['Country_Code'] == country_code]
    
    if country_forecasts.empty:
        continue
    
    report_text += f"\n{COUNTRY_NAMES[country_code].upper()} ({country_code})\n"
    report_text += "-" * 100 + "\n"
    report_text += f"{'Year':<10} {'GHG (t CO2e/capita)':<20} {'YoY Change':<20} {'YoY % Change':<20}\n"
    report_text += "-" * 100 + "\n"
    
    prev_ghg = None
    for _, row in country_forecasts.iterrows():
        year = int(row['Forecast_Year'])
        ghg = row['GHG_Per_Capita']
        
        if prev_ghg is not None:
            yoy_change = ghg - prev_ghg
            yoy_pct = (yoy_change / prev_ghg * 100) if prev_ghg != 0 else 0
            report_text += f"{year:<10} {ghg:<20.4f} {yoy_change:<20.4f} {yoy_pct:<20.2f}%\n"
        else:
            report_text += f"{year:<10} {ghg:<20.4f} {'N/A':<20} {'N/A':<20}\n"
        
        prev_ghg = ghg
    
    report_text += "\n"

# ========================================================================
# SECTION 2: RESIDUAL ANALYSIS
# ========================================================================

report_text += "\n" + "="*100 + "\n"
report_text += "SECTION 2: MODEL RESIDUAL ANALYSIS (HISTORICAL DATA)\n"
report_text += "="*100 + "\n\n"

report_text += "RESIDUAL SUMMARY BY COUNTRY:\n"
report_text += "-" * 100 + "\n"
report_text += f"{'Country':<20} {'N':<8} {'RMSE':<12} {'MAE':<12} {'Mean Res':<12} {'Std Res':<12} {'Min':<12} {'Max':<12}\n"
report_text += "-" * 100 + "\n"

for _, row in residuals_summary_df.iterrows():
    report_text += (f"{row['Country_Name']:<20} {int(row['Sample_Size']):<8} "
                    f"{row['RMSE']:<12.6f} {row['MAE']:<12.6f} "
                    f"{row['Mean_Residual']:<12.6f} {row['Std_Residual']:<12.6f} "
                    f"{row['Min_Residual']:<12.6f} {row['Max_Residual']:<12.6f}\n")

report_text += "\n"

# ========================================================================
# SECTION 3: DETAILED RESIDUALS BY COUNTRY & YEAR
# ========================================================================

report_text += "\n" + "="*100 + "\n"
report_text += "SECTION 3: DETAILED RESIDUALS BY COUNTRY & YEAR\n"
report_text += "="*100 + "\n\n"

for country_code in SEA_COUNTRIES:
    country_residuals = residuals_df[residuals_df['Country_Code'] == country_code]
    
    if country_residuals.empty:
        continue
    
    report_text += f"\n{COUNTRY_NAMES[country_code].upper()} ({country_code})\n"
    report_text += "-" * 100 + "\n"
    report_text += f"{'Year':<8} {'Actual GHG':<18} {'Predicted GHG':<18} {'Residual':<18} {'Abs Error':<18} {'% Error':<15}\n"
    report_text += "-" * 100 + "\n"
    
    for _, row in country_residuals.iterrows():
        report_text += (f"{int(row['Year']):<8} {row['Actual_GHG']:<18.6f} "
                        f"{row['Predicted_GHG']:<18.6f} {row['Residual']:<18.6f} "
                        f"{row['Abs_Residual']:<18.6f} {row['Pct_Error']:<15.2f}%\n")
    
    report_text += "\n"

# ========================================================================
# SECTION 4: RESIDUAL STATISTICS
# ========================================================================

report_text += "\n" + "="*100 + "\n"
report_text += "SECTION 4: RESIDUAL STATISTICS & DIAGNOSTICS\n"
report_text += "="*100 + "\n\n"

for country_code in SEA_COUNTRIES:
    country_residuals = residuals_df[residuals_df['Country_Code'] == country_code]
    
    if country_residuals.empty:
        continue
    
    residuals = country_residuals['Residual'].values
    
    report_text += f"\n{COUNTRY_NAMES[country_code].upper()} ({country_code})\n"
    report_text += "-" * 100 + "\n"
    
    report_text += f"Mean Residual:           {np.mean(residuals):>12.6f} (Should be ~0 for unbiased model)\n"
    report_text += f"Standard Deviation:      {np.std(residuals):>12.6f} (Model uncertainty)\n"
    report_text += f"Minimum Residual:        {np.min(residuals):>12.6f}\n"
    report_text += f"Maximum Residual:        {np.max(residuals):>12.6f}\n"
    report_text += f"Mean Absolute Error:     {np.mean(np.abs(residuals)):>12.6f}\n"
    report_text += f"Root Mean Squared Error: {np.sqrt(np.mean(residuals**2)):>12.6f}\n"
    
    # Percentiles
    report_text += f"\nPercentiles:\n"
    for percentile in [25, 50, 75, 90, 95]:
        value = np.percentile(residuals, percentile)
        report_text += f"  {percentile}th percentile: {value:>12.6f}\n"
    
    report_text += "\n"

# ========================================================================
# SECTION 5: FORECAST SUMMARY TABLE
# ========================================================================

report_text += "\n" + "="*100 + "\n"
report_text += "SECTION 5: COMPREHENSIVE FORECAST SUMMARY (2025-2030)\n"
report_text += "="*100 + "\n\n"

pivot_table = yearly_df.pivot_table(
    values='GHG_Per_Capita',
    index='Country_Name',
    columns='Forecast_Year',
    aggfunc='first'
)

report_text += f"{'Country':<20}"
for year in sorted(pivot_table.columns):
    report_text += f"{year:<15}"
report_text += "\n"
report_text += "-" * 100 + "\n"

for idx, row in pivot_table.iterrows():
    report_text += f"{idx:<20}"
    for val in row:
        if pd.notna(val):
            report_text += f"{val:<15.4f}"
        else:
            report_text += f"{'N/A':<15}"
    report_text += "\n"

report_text += "\n"

# ========================================================================
# SECTION 6: FORECAST INTERPRETATION GUIDE
# ========================================================================

report_text += "\n" + "="*100 + "\n"
report_text += "SECTION 6: FORECAST INTERPRETATION GUIDE\n"
report_text += "="*100 + "\n\n"

report_text += """HOW TO INTERPRET THE RESULTS:

1. YEARLY FORECASTS (2025-2030):
   - Values represent predicted GHG emissions per capita in t CO2e
   - Based on linear regression fitted to historical economic indicators
   - Assumes continuation of past 5-year growth rates for all predictors

2. RESIDUALS:
   - Residual = Actual - Predicted value from model on historical data
   - Positive residual: Model underpredicted (actual > predicted)
   - Negative residual: Model overpredicted (actual < predicted)
   
3. KEY RESIDUAL METRICS:
   - RMSE (Root Mean Squared Error): Average magnitude of prediction error
   - MAE (Mean Absolute Error): Average absolute deviation
   - Mean Residual: Should be ~0 for unbiased model
   - Std Deviation: Spread of errors around zero

4. MODEL QUALITY ASSESSMENT:
   - Large RMSE → Large forecast uncertainty → Use with caution
   - Small mean residual → Unbiased model (not systematically over/under-predicting)
   - High std deviation → Inconsistent model performance across years

5. CONFIDENCE INTERVALS:
   For each forecast: GHG ± 1.96 × RMSE ≈ 95% confidence interval
   
   Example: If RMSE=0.1, forecast=5.0
   95% CI = 5.0 ± 0.196 = [4.80, 5.20]

"""

report_text += "\n" + "="*100 + "\n"
report_text += "END OF REPORT\n"
report_text += "="*100 + "\n"

# Save text report
with open('model_outputs/ghg_forecast_analysis_complete.txt', 'w') as f:
    f.write(report_text)

print("✓ Saved: model_outputs/ghg_forecast_analysis_complete.txt")

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
print("\nOutput files generated:")
print("  1. ghg_yearly_forecast_2025_2030.csv - Year-by-year forecasts")
print("  2. ghg_model_residuals_detailed.csv - Historical residuals by year")
print("  3. ghg_residuals_summary.csv - Summary statistics by country")
print("  4. ghg_forecast_analysis_complete.txt - Comprehensive text report")
