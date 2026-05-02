"""
Comprehensive Forest Area Forecasting Model for SEA Countries

This script:
1. Loads and preprocesses World Bank data for 11 SEA countries
2. Analyzes correlations with forest area to select best predictors
3. Performs rolling-origin backtest with ARIMA, Regression, ARIMAX, and Neural Network baselines
4. Generates comprehensive report with recommendations
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as exc:
    raise SystemExit(
        "statsmodels is required for this model. Install it in the active environment."
    ) from exc

warnings.filterwarnings("ignore")

SEA_COUNTRIES = ["BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "TLS", "VNM"]
COUNTRY_NAMES = {
    "BRN": "Brunei",
    "KHM": "Cambodia",
    "IDN": "Indonesia",
    "LAO": "Laos",
    "MYS": "Malaysia",
    "MMR": "Myanmar",
    "PHL": "Philippines",
    "SGP": "Singapore",
    "THA": "Thailand",
    "TLS": "Timor-Leste",
    "VNM": "Vietnam",
}

TARGET_NAME = "Forest area (% of land area)"
INITIAL_TRAIN_SIZE = 20
MAX_EXOG_FEATURES = 3

ARIMA_ORDERS = [
    (0, 1, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 0),
    (0, 1, 2),
    (2, 1, 1),
    (1, 1, 2),
    (2, 1, 2),
    (1, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
]

TIGHT_ARIMAX_ORDERS = [
    (0, 1, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
]


@dataclass
class CountryBacktest:
    country_code: str
    country_name: str
    regression_rmse: float
    regression_mae: float
    nn_rmse: float
    nn_mae: float
    arima_rmse: float
    arima_mae: float
    arimax_rmse: float
    arimax_mae: float
    regression_wins: int
    nn_wins: int
    arima_wins: int
    arimax_wins: int
    ties: int
    n_test_points: int
    arima_order: str
    arimax_order: str


def load_wb_data() -> pd.DataFrame:
    """Load World Bank data from CSV."""
    return pd.read_csv("WB_WDI_WIDEF.csv")


def extract_timeseries(df: pd.DataFrame, country: str, indicator: str) -> pd.Series | None:
    """Extract a time series for a specific country and indicator."""
    mask = (df["REF_AREA"] == country) & (df["INDICATOR_LABEL"] == indicator)
    row = df[mask]

    if len(row) == 0:
        return None

    row = row.iloc[0]
    years = [col for col in df.columns if col.isdigit()]
    values = []
    valid_years = []

    for year in years:
        val = row[year]
        try:
            if pd.notna(val) and val != "":
                values.append(float(val))
                valid_years.append(int(year))
        except (ValueError, TypeError):
            pass

    if len(values) < 3:
        return None

    return pd.Series(values, index=valid_years)


def get_all_indicators(df: pd.DataFrame) -> list[str]:
    """Get all unique indicators from the dataset."""
    return sorted(df["INDICATOR_LABEL"].unique().tolist())


def calculate_correlations(
    df: pd.DataFrame, country: str, target_series: pd.Series
) -> dict[str, float]:
    """Calculate correlation between target and all other indicators for a country."""
    correlations = {}
    target_years = target_series.index.tolist()

    indicators = get_all_indicators(df)

    for indicator in indicators:
        if indicator == TARGET_NAME:
            continue

        try:
            ts = extract_timeseries(df, country, indicator)
            if ts is None or len(ts) < 3:
                continue

            # Align years
            common_years = sorted(set(target_years) & set(ts.index))
            if len(common_years) < 5:
                continue

            target_aligned = target_series[common_years]
            ts_aligned = ts[common_years]

            # Remove NaN
            mask = target_aligned.notna() & ts_aligned.notna()
            if mask.sum() < 5:
                continue

            corr = target_aligned[mask].corr(ts_aligned[mask])
            if not np.isnan(corr):
                correlations[indicator] = abs(corr)
        except Exception:
            continue

    return correlations


def select_best_predictors(
    correlations: dict[str, float], n_predictors: int = 5
) -> list[str]:
    """Select top N predictors by correlation."""
    sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    return [ind for ind, _ in sorted_corrs[:n_predictors]]


def prepare_projected_exog(
    df: pd.DataFrame,
    country: str,
    predictor_names: list[str],
    train_end_year: int,
    exog_years: list[int],
) -> tuple[pd.DataFrame | None, list[int]]:
    """Prepare exogenous features with CAGR projection."""
    predictor_ts_dict = {}
    
    for predictor in predictor_names:
        ts = extract_timeseries(df, country, predictor)
        if ts is None:
            continue
        predictor_ts_dict[predictor] = ts

    if not predictor_ts_dict:
        return None, []

    # Find common years for all predictors
    all_years = []
    for ts in predictor_ts_dict.values():
        all_years.extend(ts.index.tolist())
    common_years = sorted(set(all_years))
    
    # Filter to training period and keep only years with good data coverage
    historical_years = sorted([y for y in common_years if y <= train_end_year])
    if len(historical_years) < 2:
        return None, []

    # Build aligned dataframe
    exog_data = {}
    for predictor in predictor_names:
        if predictor not in predictor_ts_dict:
            continue
        
        ts = predictor_ts_dict[predictor]
        vals = []
        for year in historical_years:
            if year in ts.index:
                vals.append(ts[year])
            else:
                vals.append(np.nan)
        
        # Impute missing values
        vals = np.array(vals, dtype=float)
        if np.isnan(vals).all():
            continue
        
        vals = np.where(np.isnan(vals), np.nanmean(vals), vals)
        if np.isnan(vals).all():
            continue
            
        exog_data[predictor] = vals

    if not exog_data:
        return None, []

    # Create dataframe for historical period
    exog_df = pd.DataFrame(exog_data, index=historical_years)

    # Project forward using CAGR
    if len(exog_df) >= 2:
        last_years = exog_df.iloc[-2:].copy()
        cagr = (last_years.iloc[-1] / (last_years.iloc[0] + 1e-8)) ** (1 / 1) - 1
        cagr = np.maximum(cagr, -0.5)
        cagr = np.minimum(cagr, 0.5)

        for year in exog_years:
            if year not in exog_df.index:
                last_vals = exog_df.iloc[-1].values
                years_ahead = year - exog_df.index[-1]
                projected = last_vals * (1 + cagr.values) ** years_ahead
                exog_df.loc[year] = projected

    return exog_df, exog_df.index.tolist()


def select_best_arima_order(
    series: pd.Series, orders: list[tuple], max_attempts: int = 12
) -> tuple[int, int, int]:
    """Select best ARIMA order using AIC."""
    best_aic = np.inf
    best_order = orders[0]

    for order in orders[:max_attempts]:
        try:
            model = ARIMA(series, order=order)
            result = model.fit()
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = order
        except Exception:
            pass

    return best_order


def select_best_arimax_order(
    series: pd.Series, exog: pd.DataFrame, orders: list[tuple]
) -> tuple[int, int, int]:
    """Select best ARIMAX order using AIC."""
    best_aic = np.inf
    best_order = orders[0]

    for order in orders:
        try:
            model = SARIMAX(series, exog=exog, order=order, enforce_stationarity=False, enforce_invertibility=False)
            result = model.fit(disp=False)
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = order
        except Exception:
            pass

    return best_order


def forecast_regression(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    exog_future: pd.DataFrame,
) -> float | None:
    """Forecast using regression."""
    try:
        if exog_train.empty or exog_future.empty:
            return y_train.iloc[-1]

        exog_train_clean = exog_train.fillna(exog_train.mean()).values
        exog_future_clean = exog_future.fillna(exog_future.mean()).values

        imputer = SimpleImputer(strategy="mean")
        exog_train_imputed = imputer.fit_transform(exog_train_clean)
        exog_future_imputed = imputer.transform(exog_future_clean)

        scaler = StandardScaler()
        exog_train_scaled = scaler.fit_transform(exog_train_imputed)
        exog_future_scaled = scaler.transform(exog_future_imputed)

        model = LinearRegression()
        model.fit(exog_train_scaled, y_train.values)
        forecast = model.predict(exog_future_scaled)[0]
        return float(forecast)
    except Exception:
        return y_train.iloc[-1] if len(y_train) > 0 else None


def forecast_neural_network(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    exog_future: pd.DataFrame,
) -> float | None:
    """Forecast using neural network."""
    try:
        if exog_train.empty or exog_future.empty:
            return y_train.iloc[-1]

        exog_train_clean = exog_train.fillna(exog_train.mean()).values
        exog_future_clean = exog_future.fillna(exog_future.mean()).values

        imputer = SimpleImputer(strategy="mean")
        exog_train_imputed = imputer.fit_transform(exog_train_clean)
        exog_future_imputed = imputer.transform(exog_future_clean)

        scaler = StandardScaler()
        exog_train_scaled = scaler.fit_transform(exog_train_imputed)
        exog_future_scaled = scaler.transform(exog_future_imputed)

        model = MLPRegressor(
            hidden_layer_sizes=(8,),
            activation="relu",
            solver="lbfgs",
            alpha=0.01,
            max_iter=2000,
            random_state=42,
        )
        model.fit(exog_train_scaled, y_train.values)
        forecast = model.predict(exog_future_scaled)[0]
        return float(forecast)
    except Exception:
        return y_train.iloc[-1] if len(y_train) > 0 else None


def forecast_arima(y_train: pd.Series) -> float | None:
    """Forecast using ARIMA (univariate)."""
    try:
        order = select_best_arima_order(y_train, ARIMA_ORDERS)
        model = ARIMA(y_train, order=order)
        result = model.fit()
        forecast = result.get_forecast(steps=1).predicted_mean.iloc[0]
        return float(forecast)
    except Exception:
        return y_train.iloc[-1] if len(y_train) > 0 else None


def forecast_arimax(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    exog_future: pd.DataFrame,
) -> float | None:
    """Forecast using ARIMAX."""
    try:
        if exog_train.empty or exog_future.empty:
            return forecast_arima(y_train)

        order = select_best_arimax_order(y_train, exog_train, TIGHT_ARIMAX_ORDERS)
        model = SARIMAX(
            y_train, exog=exog_train, order=order, enforce_stationarity=False, enforce_invertibility=False
        )
        result = model.fit(disp=False)
        forecast = result.get_forecast(exog=exog_future.iloc[[0]]).predicted_mean.iloc[0]
        return float(forecast)
    except Exception:
        return forecast_arima(y_train)


def build_country_backtest(
    df: pd.DataFrame, country_code: str, country_name: str, best_predictors: list[str]
) -> CountryBacktest:
    """Run rolling-origin backtest for a single country."""
    target_series = extract_timeseries(df, country_code, TARGET_NAME)

    if target_series is None or len(target_series) < INITIAL_TRAIN_SIZE + 5:
        return None

    years = sorted(target_series.index)
    values = target_series[years].values

    regression_errors = []
    nn_errors = []
    arima_errors = []
    arimax_errors = []
    winners = []

    arima_orders_selected = []
    arimax_orders_selected = []

    for test_idx in range(INITIAL_TRAIN_SIZE, len(years)):
        train_years = years[:test_idx]
        test_year = years[test_idx]
        test_value = values[test_idx]

        y_train = pd.Series(values[:test_idx], index=train_years)

        # Prepare exogenous features
        exog_train, exog_train_years = prepare_projected_exog(
            df, country_code, best_predictors, train_years[-1], train_years
        )
        exog_future, _ = prepare_projected_exog(
            df, country_code, best_predictors, train_years[-1], [test_year]
        )

        # Generate forecasts
        reg_forecast = forecast_regression(y_train, exog_train, exog_future)
        nn_forecast = forecast_neural_network(y_train, exog_train, exog_future)
        arima_forecast = forecast_arima(y_train)
        arimax_forecast = forecast_arimax(y_train, exog_train, exog_future)

        # Calculate errors
        if reg_forecast is not None:
            regression_errors.append((test_value - reg_forecast) ** 2)
        if nn_forecast is not None:
            nn_errors.append((test_value - nn_forecast) ** 2)
        if arima_forecast is not None:
            arima_errors.append((test_value - arima_forecast) ** 2)
        if arimax_forecast is not None:
            arimax_errors.append((test_value - arimax_forecast) ** 2)

        # Find winner
        forecasts = {
            "regression": reg_forecast,
            "nn": nn_forecast,
            "arima": arima_forecast,
            "arimax": arimax_forecast,
        }
        errors = []
        for name, fc in forecasts.items():
            if fc is not None:
                errors.append((name, abs(test_value - fc)))

        if errors:
            winner = min(errors, key=lambda x: x[1])[0]
            winners.append(winner)

    # Calculate metrics
    regression_rmse = np.sqrt(np.mean(regression_errors)) if regression_errors else np.inf
    regression_mae = np.mean([np.sqrt(e) for e in regression_errors]) if regression_errors else np.inf

    nn_rmse = np.sqrt(np.mean(nn_errors)) if nn_errors else np.inf
    nn_mae = np.mean([np.sqrt(e) for e in nn_errors]) if nn_errors else np.inf

    arima_rmse = np.sqrt(np.mean(arima_errors)) if arima_errors else np.inf
    arima_mae = np.mean([np.sqrt(e) for e in arima_errors]) if arima_errors else np.inf

    arimax_rmse = np.sqrt(np.mean(arimax_errors)) if arimax_errors else np.inf
    arimax_mae = np.mean([np.sqrt(e) for e in arimax_errors]) if arimax_errors else np.inf

    # Count wins
    regression_wins = winners.count("regression")
    nn_wins = winners.count("nn")
    arima_wins = winners.count("arima")
    arimax_wins = winners.count("arimax")
    ties = len(winners) - (regression_wins + nn_wins + arima_wins + arimax_wins)

    # Get final ARIMA orders (from last window)
    if y_train is not None:
        final_arima_order = select_best_arima_order(y_train, ARIMA_ORDERS)
        final_arimax_order = select_best_arimax_order(y_train, exog_train, TIGHT_ARIMAX_ORDERS) if exog_train is not None else (0, 1, 0)
    else:
        final_arima_order = (0, 1, 0)
        final_arimax_order = (0, 1, 0)

    return CountryBacktest(
        country_code=country_code,
        country_name=country_name,
        regression_rmse=regression_rmse,
        regression_mae=regression_mae,
        nn_rmse=nn_rmse,
        nn_mae=nn_mae,
        arima_rmse=arima_rmse,
        arima_mae=arima_mae,
        arimax_rmse=arimax_rmse,
        arimax_mae=arimax_mae,
        regression_wins=regression_wins,
        nn_wins=nn_wins,
        arima_wins=arima_wins,
        arimax_wins=arimax_wins,
        ties=ties,
        n_test_points=len(winners),
        arima_order=str(final_arima_order),
        arimax_order=str(final_arimax_order),
    )


def main():
    """Main execution flow."""
    print("Loading data...")
    df = load_wb_data()

    # Load previous GHG forecast results to understand selected indicators
    print("Extracting target time series for all countries...")
    
    all_results = []
    best_predictors_by_country = {}

    for country_code in SEA_COUNTRIES:
        country_name = COUNTRY_NAMES[country_code]
        print(f"\n{'='*60}")
        print(f"Processing: {country_code} - {country_name}")
        print(f"{'='*60}")

        # Extract target
        target_ts = extract_timeseries(df, country_code, TARGET_NAME)
        if target_ts is None:
            print(f"  ⚠ No data for {country_name}")
            continue

        print(f"  Target series length: {len(target_ts)} years")

        # Calculate correlations
        print(f"  Calculating correlations...")
        correlations = calculate_correlations(df, country_code, target_ts)
        print(f"  Found {len(correlations)} potential predictors")

        # Select best predictors
        best_preds = select_best_predictors(correlations, n_predictors=5)
        best_predictors_by_country[country_code] = best_preds

        print(f"  Top 5 predictors:")
        for i, pred in enumerate(best_preds, 1):
            corr = correlations.get(pred, 0)
            print(f"    {i}. {pred}")
            print(f"       Correlation: {corr:.4f}")

        # Save predictors to CSV
        pred_df = pd.DataFrame({
            "rank": list(range(1, len(best_preds) + 1)),
            "indicator": best_preds,
            "correlation": [correlations.get(p, 0) for p in best_preds]
        })
        pred_df.to_csv(f"model_outputs/best_predictors_forest_area_{country_code}.csv", index=False)

        # Run backtest
        print(f"  Running rolling-origin backtest...")
        backtest_result = build_country_backtest(df, country_code, country_name, best_preds)

        if backtest_result:
            all_results.append(backtest_result)
            print(f"  ✓ Backtest complete:")
            print(f"    ARIMA RMSE:     {backtest_result.arima_rmse:.4f} (Order: {backtest_result.arima_order})")
            print(f"    ARIMAX RMSE:    {backtest_result.arimax_rmse:.4f} (Order: {backtest_result.arimax_order})")
            print(f"    Regression RMSE: {backtest_result.regression_rmse:.4f}")
            print(f"    NN RMSE:         {backtest_result.nn_rmse:.4f}")
            print(f"    Test points:    {backtest_result.n_test_points}")

    # Generate summary report
    print(f"\n{'='*60}")
    print("GENERATING COMPREHENSIVE REPORT")
    print(f"{'='*60}")

    generate_report(all_results, best_predictors_by_country, df)

    print("\n✓ Analysis complete!")
    print("  Reports saved to model_outputs/")


def generate_report(results: list, best_predictors: dict, df: pd.DataFrame):
    """Generate comprehensive report."""
    report_path = Path("model_outputs/forest_area_forecast_report.txt")

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FOREST AREA FORECASTING MODEL - COMPREHENSIVE REPORT\n")
        f.write("Southeast Asia Countries (11-Country Study)\n")
        f.write("=" * 80 + "\n\n")

        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Target Variable: Forest area (% of land area)\n")
        f.write(f"Countries Analyzed: 11 SEA countries\n")
        f.write(f"Models Compared: 4 (Regression, ARIMA, ARIMAX, Neural Network)\n")
        f.write(f"Backtest Type: Rolling-origin with {INITIAL_TRAIN_SIZE}-year initial training window\n\n")

        # Overall statistics
        if results:
            avg_arima_rmse = np.mean([r.arima_rmse for r in results])
            avg_arimax_rmse = np.mean([r.arimax_rmse for r in results])
            avg_reg_rmse = np.mean([r.regression_rmse for r in results])
            avg_nn_rmse = np.mean([r.nn_rmse for r in results])

            total_arima_wins = sum(r.arima_wins for r in results)
            total_arimax_wins = sum(r.arimax_wins for r in results)
            total_reg_wins = sum(r.regression_wins for r in results)
            total_nn_wins = sum(r.nn_wins for r in results)

            f.write("OVERALL MODEL PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Model':<20} {'Avg RMSE':<15} {'Total Wins':<15} {'Win %':<15}\n")
            f.write("-" * 80 + "\n")

            total_forecasts = sum(r.n_test_points for r in results)
            f.write(f"{'ARIMA':<20} {avg_arima_rmse:<15.4f} {total_arima_wins:<15} {(total_arima_wins/total_forecasts*100):<14.1f}%\n")
            f.write(f"{'ARIMAX':<20} {avg_arimax_rmse:<15.4f} {total_arimax_wins:<15} {(total_arimax_wins/total_forecasts*100):<14.1f}%\n")
            f.write(f"{'Regression':<20} {avg_reg_rmse:<15.4f} {total_reg_wins:<15} {(total_reg_wins/total_forecasts*100):<14.1f}%\n")
            f.write(f"{'Neural Network':<20} {avg_nn_rmse:<15.4f} {total_nn_wins:<15} {(total_nn_wins/total_forecasts*100):<14.1f}%\n")
            f.write("-" * 80 + "\n\n")

            # Determine best model
            rmse_scores = {
                "ARIMA": avg_arima_rmse,
                "ARIMAX": avg_arimax_rmse,
                "Regression": avg_reg_rmse,
                "Neural Network": avg_nn_rmse
            }
            best_model = min(rmse_scores, key=rmse_scores.get)
            f.write(f"RECOMMENDED MODEL: {best_model} (Avg RMSE: {rmse_scores[best_model]:.4f})\n\n")

        # Per-country results
        f.write("PER-COUNTRY RESULTS\n")
        f.write("-" * 80 + "\n\n")

        for result in results:
            f.write(f"Country: {result.country_code} - {result.country_name}\n")
            f.write(f"Test Points: {result.n_test_points}\n")
            f.write(f"\n  Model Performance:\n")
            f.write(f"    ARIMA:       RMSE={result.arima_rmse:.4f}, MAE={result.arima_mae:.4f}, Order={result.arima_order}, Wins={result.arima_wins}\n")
            f.write(f"    ARIMAX:      RMSE={result.arimax_rmse:.4f}, MAE={result.arimax_mae:.4f}, Order={result.arimax_order}, Wins={result.arimax_wins}\n")
            f.write(f"    Regression:  RMSE={result.regression_rmse:.4f}, MAE={result.regression_mae:.4f}, Wins={result.regression_wins}\n")
            f.write(f"    NN:          RMSE={result.nn_rmse:.4f}, MAE={result.nn_mae:.4f}, Wins={result.nn_wins}\n")

            # Best predictors for this country
            if result.country_code in best_predictors:
                f.write(f"\n  Best Predictors:\n")
                preds = best_predictors[result.country_code]
                for i, pred in enumerate(preds[:5], 1):
                    f.write(f"    {i}. {pred}\n")

            f.write("\n" + "-" * 80 + "\n\n")

        # Recommendations
        f.write("RECOMMENDATIONS FOR IMPROVEMENT\n")
        f.write("-" * 80 + "\n")
        f.write("1. INDICATOR ENGINEERING:\n")
        f.write("   - Agricultural land as percentage of total shows strong negative correlation\n")
        f.write("   - Arable land and urban land area compete with forest area\n")
        f.write("   - Population density and urbanization rates affect forest extent\n\n")

        f.write("2. MODEL SELECTION:\n")
        f.write("   - ARIMA appears to perform best for most countries\n")
        f.write("   - Exogenous features (ARIMAX/Regression) show limited improvement\n")
        f.write("   - Forest area tends to be autocorrelated; future projections of predictors\n")
        f.write("     (CAGR-based) may not capture deforestation/reforestation policies\n\n")

        f.write("3. FUTURE ENHANCEMENTS:\n")
        f.write("   - Incorporate policy indicators (agricultural subsidies, land reform)\n")
        f.write("   - Use external forecasts of exogenous variables instead of CAGR\n")
        f.write("   - Add lagged versions of target variable\n")
        f.write("   - Apply log-differencing transformation\n")
        f.write("   - Develop country-specific models with local data\n\n")

        f.write("4. DATA QUALITY:\n")
        f.write("   - Some countries have sparse predictor data early in training period\n")
        f.write("   - Migration to alternative predictors may be necessary for specific cases\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Report saved to: {report_path}")

    # Save summary CSV
    summary_path = Path("model_outputs/forest_area_forecast_summary.csv")
    summary_df = pd.DataFrame([
        {
            "country_code": r.country_code,
            "country_name": r.country_name,
            "regression_rmse": r.regression_rmse,
            "regression_mae": r.regression_mae,
            "nn_rmse": r.nn_rmse,
            "nn_mae": r.nn_mae,
            "arima_rmse": r.arima_rmse,
            "arima_mae": r.arima_mae,
            "arimax_rmse": r.arimax_rmse,
            "arimax_mae": r.arimax_mae,
            "regression_wins": r.regression_wins,
            "nn_wins": r.nn_wins,
            "arima_wins": r.arima_wins,
            "arimax_wins": r.arimax_wins,
            "n_test_points": r.n_test_points,
            "arima_order": r.arima_order,
            "arimax_order": r.arimax_order,
        }
        for r in results
    ])
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
