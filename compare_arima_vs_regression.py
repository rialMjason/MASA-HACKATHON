"""
Rolling backtest comparing the current regression-based forecast against ARIMA.

The regression baseline matches the existing pipeline logic:
- BIC-selected predictors from model_outputs/sea_countries_best_indicators.csv
- mean imputation
- standard scaling
- linear regression
- one-step-ahead feature projection using recent CAGR

The ARIMA baseline is univariate and uses the target series only.
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
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "statsmodels is required for this comparison. Install it in the active "
        "environment, then rerun this script."
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

TARGET_NAME = "Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)"
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
    arimax_dummy_rmse: float
    arimax_dummy_mae: float
    arimax_dummy_wins: int
    arimax_dummy_order: str


def safe_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def safe_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def select_best_arima_order(series: np.ndarray) -> tuple[int, int, int]:
    best_order = ARIMA_ORDERS[0]
    best_aic = np.inf

    for order in ARIMA_ORDERS:
        try:
            model = ARIMA(series, order=order)
            result = model.fit()
            if np.isfinite(result.aic) and result.aic < best_aic:
                best_aic = float(result.aic)
                best_order = order
        except Exception:
            continue

    return best_order


def select_top_exog_columns(x_imputed: np.ndarray, train_y: np.ndarray, max_features: int = MAX_EXOG_FEATURES) -> np.ndarray:
    if x_imputed.shape[1] == 0:
        return np.array([], dtype=int)

    correlations = []
    for column_idx in range(x_imputed.shape[1]):
        feature = x_imputed[:, column_idx]
        if np.allclose(feature, feature[0]):
            continue
        corr_matrix = np.corrcoef(feature, train_y)
        corr_value = corr_matrix[0, 1]
        if np.isfinite(corr_value):
            correlations.append((column_idx, abs(corr_value)))

    if not correlations:
        return np.array([], dtype=int)

    correlations.sort(key=lambda item: item[1], reverse=True)
    selected = [column_idx for column_idx, _ in correlations[:max_features]]
    return np.array(selected, dtype=int)


def prepare_projected_exog(train_x: np.ndarray, train_y: np.ndarray, forecast_steps: int = 1) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    usable_columns = ~np.all(np.isnan(train_x), axis=0)
    if not np.any(usable_columns):
        return None, None

    train_x = train_x[:, usable_columns]

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    x_imputed = imputer.fit_transform(train_x)
    if x_imputed.shape[1] == 0 or np.isnan(x_imputed).any():
        return None, None

    selected_columns = select_top_exog_columns(x_imputed, train_y)
    if selected_columns.size == 0:
        return None, None

    x_imputed = x_imputed[:, selected_columns]

    recent_window = min(5, len(x_imputed))
    if recent_window < 2:
        projected_features = x_imputed[-1].copy()
    else:
        recent_features = x_imputed[-recent_window:]
        projected_features = recent_features[-1].copy()

        for feature_idx in range(recent_features.shape[1]):
            start_value = recent_features[0, feature_idx]
            end_value = recent_features[-1, feature_idx]
            if start_value > 0 and np.isfinite(start_value) and np.isfinite(end_value):
                growth_rate = (end_value / start_value) ** (1 / (recent_window - 1)) - 1
                projected_features[feature_idx] = end_value * ((1 + growth_rate) ** forecast_steps)

    x_scaled = scaler.fit_transform(x_imputed)
    projected_scaled = scaler.transform(projected_features.reshape(1, -1))
    return x_scaled, projected_scaled


def forecast_regression(train_y: np.ndarray, train_x: np.ndarray, forecast_steps: int = 1) -> float:
    prepared = prepare_projected_exog(train_x, train_y, forecast_steps)
    if prepared[0] is None:
        return float(train_y[-1])

    x_scaled, projected_scaled = prepared
    if x_scaled.shape[1] == 0 or projected_scaled.shape[1] == 0 or not np.isfinite(x_scaled).all() or not np.isfinite(projected_scaled).all():
        return float(train_y[-1])

    model = LinearRegression()
    model.fit(x_scaled, train_y)
    return float(model.predict(projected_scaled)[0])


def forecast_neural_network(train_y: np.ndarray, train_x: np.ndarray, forecast_steps: int = 1) -> float:
    prepared = prepare_projected_exog(train_x, train_y, forecast_steps)
    if prepared[0] is None:
        return float(train_y[-1])

    x_scaled, projected_scaled = prepared
    if x_scaled.shape[1] == 0 or projected_scaled.shape[1] == 0 or not np.isfinite(x_scaled).all() or not np.isfinite(projected_scaled).all():
        return float(train_y[-1])

    model = MLPRegressor(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="lbfgs",
        alpha=0.01,
        max_iter=2000,
        random_state=123,
    )
    model.fit(x_scaled, train_y)
    return float(model.predict(projected_scaled)[0])


def select_best_arimax_order(train_y: np.ndarray, train_x: np.ndarray) -> tuple[int, int, int]:
    prepared = prepare_projected_exog(train_x, train_y)
    if prepared[0] is None:
        return ARIMA_ORDERS[0]

    x_scaled, _ = prepared
    if x_scaled.shape[1] == 0 or not np.isfinite(x_scaled).all():
        return ARIMA_ORDERS[0]
    best_order = ARIMA_ORDERS[0]
    best_aic = np.inf

    for order in TIGHT_ARIMAX_ORDERS:
        try:
            model = SARIMAX(
                train_y,
                exog=x_scaled,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False)
            if np.isfinite(result.aic) and result.aic < best_aic:
                best_aic = float(result.aic)
                best_order = order
        except Exception:
            continue

    return best_order


def build_policy_dummies(years: np.ndarray, country_code: str) -> tuple[list[str], np.ndarray]:
    """Create binary dummy series aligned to `years` for policy events/penalties."""
    years_list = years.tolist()
    dummy_names: list[str] = []
    dummy_arrays: list[np.ndarray] = []

    def make_dummy(active_years: list[int]):
        return np.array([1.0 if y in active_years else 0.0 for y in years_list], dtype=float)

    # Global Shell ruling 2021
    dummy_names.append("Global_Shell_2021")
    dummy_arrays.append(make_dummy([2021]))

    # Net-zero announcement (per-country proxy)
    net_zero_announcements = {
        "BRN": 2020,
        "KHM": 2020,
        "IDN": 2021,
        "LAO": 2021,
        "MYS": 2021,
        "MMR": 2020,
        "PHL": 2021,
        "SGP": 2020,
        "THA": 2020,
        "TLS": 2021,
        "VNM": 2020,
    }
    ann = net_zero_announcements.get(country_code)
    if ann:
        dummy_names.append(f"NetZero_{ann}")
        dummy_arrays.append(make_dummy([y for y in years_list if y >= ann]))

    # Singapore carbon tax
    if country_code == "SGP":
        dummy_names.append("SGP_CarbonTax_2019")
        dummy_arrays.append(make_dummy([y for y in years_list if y >= 2019]))

    # Indonesia ETS and PT Kallista penalty
    if country_code == "IDN":
        dummy_names.append("IDN_ETS_2023")
        dummy_arrays.append(make_dummy([y for y in years_list if y >= 2023]))
        dummy_names.append("PT_Kallista_2014")
        dummy_arrays.append(make_dummy([2014]))

    if not dummy_arrays:
        return [], np.empty((len(years_list), 0))

    exog = np.vstack(dummy_arrays).T
    return dummy_names, exog


def forecast_arimax_with_exog(train_y: np.ndarray, train_exog: np.ndarray, order_candidates: list[tuple[int, int, int]] = TIGHT_ARIMAX_ORDERS) -> float:
    """Fit SARIMAX on train_y with train_exog and forecast using last exog row as projection."""
    try:
        # select best order from tight candidates
        best_order = order_candidates[0]
        best_aic = np.inf
        for order in order_candidates:
            try:
                model = SARIMAX(train_y, exog=train_exog, order=order, enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                if np.isfinite(res.aic) and res.aic < best_aic:
                    best_aic = float(res.aic)
                    best_order = order
            except Exception:
                continue

        model = SARIMAX(train_y, exog=train_exog, order=best_order, enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False)
        # project exog as last observed row
        if train_exog is None or train_exog.size == 0:
            exog_fore = None
        else:
            exog_fore = train_exog[-1].reshape(1, -1)
        forecast = result.forecast(steps=1, exog=exog_fore)
        return float(np.asarray(forecast)[0])
    except Exception:
        return float(train_y[-1])


def forecast_arima(train_y: np.ndarray, order: tuple[int, int, int]) -> float:
    model = ARIMA(train_y, order=order)
    result = model.fit()
    forecast = result.forecast(steps=1)
    return float(np.asarray(forecast)[0])


def forecast_arimax(train_y: np.ndarray, train_x: np.ndarray, order: tuple[int, int, int]) -> float:
    prepared = prepare_projected_exog(train_x, train_y)
    if prepared[0] is None:
        return float(train_y[-1])

    x_scaled, projected_scaled = prepared
    if x_scaled.shape[1] == 0 or projected_scaled.shape[1] == 0 or not np.isfinite(x_scaled).all() or not np.isfinite(projected_scaled).all():
        return float(train_y[-1])
    model = SARIMAX(
        train_y,
        exog=x_scaled,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    forecast = result.forecast(steps=1, exog=projected_scaled)
    return float(np.asarray(forecast)[0])


def build_country_backtest(country_code: str, df: pd.DataFrame, best_indicators_df: pd.DataFrame) -> tuple[CountryBacktest, pd.DataFrame]:
    country_name = COUNTRY_NAMES[country_code]
    country_row = best_indicators_df[best_indicators_df["Country_Code"] == country_code]
    if country_row.empty:
        raise ValueError(f"No predictor row found for {country_code}")

    predictors = [value.strip() for value in country_row.iloc[0]["Predictors"].split("|") if value.strip()]
    country_data = df[df["REF_AREA"] == country_code].copy()
    if country_data.empty:
        raise ValueError(f"No country data found for {country_code}")

    year_cols = sorted([col for col in df.columns if str(col).isdigit()], key=int)
    years = np.array([int(col) for col in year_cols])

    target_rows = country_data[country_data["INDICATOR_LABEL"] == TARGET_NAME]
    if target_rows.empty:
        raise ValueError(f"Target series missing for {country_code}")

    y_raw = pd.to_numeric(target_rows.iloc[0][year_cols], errors="coerce").to_numpy(dtype=float)

    feature_series = []
    for predictor in predictors:
        predictor_rows = country_data[country_data["INDICATOR_LABEL"] == predictor]
        if predictor_rows.empty:
            continue
        feature_series.append(pd.to_numeric(predictor_rows.iloc[0][year_cols], errors="coerce").to_numpy(dtype=float))

    if not feature_series:
        raise ValueError(f"No predictors found for {country_code}")

    x_raw = np.vstack(feature_series).T
    # Build policy/penalty dummies aligned to the year columns and include separately
    dummy_names, dummy_exog = build_policy_dummies(years, country_code)
    valid_mask = ~np.isnan(y_raw)
    x_valid = x_raw[valid_mask]
    if dummy_exog.size == 0:
        dummy_valid = np.empty((len(x_valid), 0))
    else:
        dummy_valid = dummy_exog[valid_mask]
    y_valid = y_raw[valid_mask]
    years_valid = years[valid_mask]

    if len(y_valid) <= INITIAL_TRAIN_SIZE + 2:
        raise ValueError(f"Not enough valid years for {country_code}")

    first_train_y = y_valid[:INITIAL_TRAIN_SIZE]
    first_train_x = x_valid[:INITIAL_TRAIN_SIZE]
    arima_order = select_best_arima_order(first_train_y)
    arimax_order = select_best_arimax_order(first_train_y, first_train_x)

    records = []

    for test_idx in range(INITIAL_TRAIN_SIZE, len(y_valid)):
        train_y = y_valid[:test_idx]
        train_x = x_valid[:test_idx]
        actual = float(y_valid[test_idx])
        forecast_year = int(years_valid[test_idx])

        regression_prediction = forecast_regression(train_y, train_x)
        nn_prediction = forecast_neural_network(train_y, train_x)
        try:
            arima_prediction = forecast_arima(train_y, arima_order)
        except Exception:
            arima_prediction = float(train_y[-1])

        try:
            arimax_prediction = forecast_arimax(train_y, train_x, arimax_order)
        except Exception:
            arimax_prediction = arima_prediction

        # ARIMAX using only policy/penalty dummies as exogenous
        try:
            train_dummy = dummy_valid[:test_idx] if dummy_valid.size else np.empty((0, 0))
            arimax_dummy_prediction = forecast_arimax_with_exog(train_y, train_dummy)
        except Exception:
            arimax_dummy_prediction = arima_prediction

        regression_error = abs(actual - regression_prediction)
        nn_error = abs(actual - nn_prediction)
        arima_error = abs(actual - arima_prediction)
        arimax_error = abs(actual - arimax_prediction)
        arimax_dummy_error = abs(actual - arimax_dummy_prediction)
        # Determine winner (allow ARIMAX_DUMMY as an option)
        errors = {
            "Regression": regression_error,
            "NeuralNetwork": nn_error,
            "ARIMA": arima_error,
            "ARIMAX": arimax_error,
            "ARIMAX_DUMMY": arimax_dummy_error,
        }
        min_err = min(errors.values())
        winners = [k for k, v in errors.items() if abs(v - min_err) < 1e-12]
        if len(winners) == 1:
            winner = winners[0]
        else:
            winner = "Tie"

        records.append(
            {
                "Country_Code": country_code,
                "Country_Name": country_name,
                "Year": forecast_year,
                "Actual_GHG": actual,
                "Regression_Prediction": regression_prediction,
                "NN_Prediction": nn_prediction,
                "ARIMA_Prediction": arima_prediction,
                "ARIMAX_Prediction": arimax_prediction,
                "ARIMAX_DUMMY_Prediction": arimax_dummy_prediction,
                "Regression_Abs_Error": regression_error,
                "NN_Abs_Error": nn_error,
                "ARIMA_Abs_Error": arima_error,
                "ARIMAX_Abs_Error": arimax_error,
                "ARIMAX_DUMMY_Abs_Error": arimax_dummy_error,
                "Winner": winner,
            }
        )

    detail_df = pd.DataFrame(records)
    regression_rmse = safe_rmse(detail_df["Actual_GHG"].to_numpy(), detail_df["Regression_Prediction"].to_numpy())
    regression_mae = safe_mae(detail_df["Actual_GHG"].to_numpy(), detail_df["Regression_Prediction"].to_numpy())
    nn_rmse = safe_rmse(detail_df["Actual_GHG"].to_numpy(), detail_df["NN_Prediction"].to_numpy())
    nn_mae = safe_mae(detail_df["Actual_GHG"].to_numpy(), detail_df["NN_Prediction"].to_numpy())
    arima_rmse = safe_rmse(detail_df["Actual_GHG"].to_numpy(), detail_df["ARIMA_Prediction"].to_numpy())
    arima_mae = safe_mae(detail_df["Actual_GHG"].to_numpy(), detail_df["ARIMA_Prediction"].to_numpy())
    arimax_rmse = safe_rmse(detail_df["Actual_GHG"].to_numpy(), detail_df["ARIMAX_Prediction"].to_numpy())
    arimax_mae = safe_mae(detail_df["Actual_GHG"].to_numpy(), detail_df["ARIMAX_Prediction"].to_numpy())
    arimax_dummy_rmse = safe_rmse(detail_df["Actual_GHG"].to_numpy(), detail_df["ARIMAX_DUMMY_Prediction"].to_numpy())
    arimax_dummy_mae = safe_mae(detail_df["Actual_GHG"].to_numpy(), detail_df["ARIMAX_DUMMY_Prediction"].to_numpy())

    summary = CountryBacktest(
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
        regression_wins=int((detail_df["Winner"] == "Regression").sum()),
        nn_wins=int((detail_df["Winner"] == "NeuralNetwork").sum()),
        arima_wins=int((detail_df["Winner"] == "ARIMA").sum()),
        arimax_wins=int((detail_df["Winner"] == "ARIMAX").sum()),
        ties=int((detail_df["Winner"] == "Tie").sum()),
        n_test_points=len(detail_df),
        arima_order=str(arima_order),
        arimax_order=str(arimax_order),
        arimax_dummy_rmse=arimax_dummy_rmse,
        arimax_dummy_mae=arimax_dummy_mae,
        arimax_dummy_wins=int((detail_df["Winner"] == "ARIMAX_DUMMY").sum()) if "ARIMAX_DUMMY" in detail_df["Winner"].unique() else 0,
        arimax_dummy_order="",
    )

    return summary, detail_df


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    output_dir = repo_root / "model_outputs"

    df = pd.read_csv(repo_root / "WB_WDI_WIDEF.csv")
    best_indicators_df = pd.read_csv(output_dir / "sea_countries_best_indicators.csv")

    summary_rows = []
    detail_frames = []

    for country_code in SEA_COUNTRIES:
        try:
            summary, detail_df = build_country_backtest(country_code, df, best_indicators_df)
            summary_rows.append(summary.__dict__)
            detail_frames.append(detail_df)
            print(
                f"{summary.country_name}: regression RMSE={summary.regression_rmse:.4f}, "
                f"ARIMA RMSE={summary.arima_rmse:.4f}, ARIMA order={summary.arima_order}"
            )
        except Exception as exc:
            print(f"{country_code}: skipped ({exc})")

    if not summary_rows:
        raise SystemExit("No country backtests completed.")

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.concat(detail_frames, ignore_index=True)

    overall_regression_rmse = safe_rmse(detail_df["Actual_GHG"].to_numpy(), detail_df["Regression_Prediction"].to_numpy())
    overall_nn_rmse = safe_rmse(detail_df["Actual_GHG"].to_numpy(), detail_df["NN_Prediction"].to_numpy())
    overall_arima_rmse = safe_rmse(detail_df["Actual_GHG"].to_numpy(), detail_df["ARIMA_Prediction"].to_numpy())
    overall_arimax_rmse = safe_rmse(detail_df["Actual_GHG"].to_numpy(), detail_df["ARIMAX_Prediction"].to_numpy())
    overall_arimax_dummy_rmse = safe_rmse(detail_df["Actual_GHG"].to_numpy(), detail_df["ARIMAX_DUMMY_Prediction"].to_numpy())
    overall_regression_mae = safe_mae(detail_df["Actual_GHG"].to_numpy(), detail_df["Regression_Prediction"].to_numpy())
    overall_nn_mae = safe_mae(detail_df["Actual_GHG"].to_numpy(), detail_df["NN_Prediction"].to_numpy())
    overall_arima_mae = safe_mae(detail_df["Actual_GHG"].to_numpy(), detail_df["ARIMA_Prediction"].to_numpy())
    overall_arimax_mae = safe_mae(detail_df["Actual_GHG"].to_numpy(), detail_df["ARIMAX_Prediction"].to_numpy())
    overall_arimax_dummy_mae = safe_mae(detail_df["Actual_GHG"].to_numpy(), detail_df["ARIMAX_DUMMY_Prediction"].to_numpy())

    regression_wins = int((detail_df["Winner"] == "Regression").sum())
    nn_wins = int((detail_df["Winner"] == "NeuralNetwork").sum())
    arima_wins = int((detail_df["Winner"] == "ARIMA").sum())
    arimax_wins = int((detail_df["Winner"] == "ARIMAX").sum())
    arimax_dummy_wins = int((detail_df["Winner"] == "ARIMAX_DUMMY").sum())
    ties = int((detail_df["Winner"] == "Tie").sum())

    summary_path = output_dir / "forecast_model_comparison_summary.csv"
    detail_path = output_dir / "forecast_model_comparison_detailed.csv"
    report_path = output_dir / "forecast_model_comparison_report.txt"

    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("REGRESSION VS ARIMA ROLLING BACKTEST")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append(f"Initial training window: {INITIAL_TRAIN_SIZE} years")
    report_lines.append("Regression baseline: BIC-selected predictors + imputation + scaling + linear regression")
    report_lines.append("Neural network baseline: MLPRegressor on the same projected exogenous features")
    report_lines.append("ARIMA baseline: univariate ARIMA selected by initial-window AIC")
    report_lines.append("ARIMAX baseline: SARIMAX with projected exogenous predictors selected by initial-window AIC")
    report_lines.append("")
    report_lines.append("OVERALL RESULTS")
    report_lines.append("-" * 100)
    report_lines.append(f"Regression RMSE: {overall_regression_rmse:.6f}")
    report_lines.append(f"Regression MAE:  {overall_regression_mae:.6f}")
    report_lines.append(f"NN RMSE:         {overall_nn_rmse:.6f}")
    report_lines.append(f"NN MAE:          {overall_nn_mae:.6f}")
    report_lines.append(f"ARIMA RMSE:      {overall_arima_rmse:.6f}")
    report_lines.append(f"ARIMA MAE:       {overall_arima_mae:.6f}")
    report_lines.append(f"ARIMAX RMSE:     {overall_arimax_rmse:.6f}")
    report_lines.append(f"ARIMAX MAE:      {overall_arimax_mae:.6f}")
    report_lines.append(f"ARIMAX_DUMMY RMSE:{overall_arimax_dummy_rmse:.6f}")
    report_lines.append(f"ARIMAX_DUMMY MAE: {overall_arimax_dummy_mae:.6f}")
    report_lines.append(f"Regression wins: {regression_wins}")
    report_lines.append(f"NN wins:         {nn_wins}")
    report_lines.append(f"ARIMA wins:      {arima_wins}")
    report_lines.append(f"ARIMAX wins:     {arimax_wins}")
    report_lines.append(f"ARIMAX_DUMMY wins:{arimax_dummy_wins}")
    report_lines.append(f"Ties:            {ties}")
    report_lines.append("")
    report_lines.append("BY COUNTRY")
    report_lines.append("-" * 100)
    for row in summary_df.sort_values("country_name").itertuples(index=False):
        report_lines.append(
            f"{row.country_name:<14} | reg RMSE={row.regression_rmse:.6f} | nn RMSE={row.nn_rmse:.6f} | "
            f"arima RMSE={row.arima_rmse:.6f} | arimax RMSE={row.arimax_rmse:.6f} | reg wins={row.regression_wins:<3} | "
            f"nn wins={row.nn_wins:<3} | arima wins={row.arima_wins:<3} | arimax wins={row.arimax_wins:<3} | "
            f"arima order={row.arima_order} | arimax order={row.arimax_order}"
        )
    report_lines.append("")
    report_lines.append("INTERPRETATION")
    report_lines.append("-" * 100)
    if overall_nn_rmse < min(overall_arimax_rmse, overall_arima_rmse, overall_regression_rmse):
        report_lines.append("The neural network is best on this backtest by RMSE.")
    elif overall_arimax_rmse < min(overall_arima_rmse, overall_regression_rmse):
        report_lines.append("ARIMAX is best on this backtest by RMSE.")
    elif overall_arima_rmse < overall_regression_rmse:
        report_lines.append("ARIMA is better on this backtest by RMSE, but ARIMAX did not improve further.")
    else:
        report_lines.append("The current regression baseline is better on this backtest by RMSE.")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("")
    print(f"Saved summary: {summary_path}")
    print(f"Saved details: {detail_path}")
    print(f"Saved report:  {report_path}")
    print(
        f"Overall RMSE -> regression: {overall_regression_rmse:.6f}, NN: {overall_nn_rmse:.6f}, ARIMA: {overall_arima_rmse:.6f}, ARIMAX: {overall_arimax_rmse:.6f}"
    )


if __name__ == "__main__":
    main()