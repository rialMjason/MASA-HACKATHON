# Final Model Selection

## Model Result
- Forest area uses ARIMAX_DUMMY.
- GHG emissions use ARIMA(1,1,0).
- The final forecast workbook is [model_outputs/final_forecast_2030.xlsx](model_outputs/final_forecast_2030.xlsx).

## Forest Area
Final model: ARIMAX_DUMMY

Reasoning:
- The earlier forest ablation favored the dummy-enhanced model over the plain baseline.
- The policy-aware version is the better fit when we want the forest forecast to reflect interventions instead of only trend.
- This restores the forest implementation to the stronger comparison result.

Key result:
- Improved countries: 8 of 11
- Mean delta RMSE: -0.012138

Reference files:
- model_outputs/forest_area_policy_dummy_ablation.csv
- model_outputs/forest_area_forecast_report.txt

## Residual Diagnostics

### GHG residual summary
The GHG residuals are centered very close to zero across all countries, which is what we want from a well-behaved forecast model. The spread differs by country, so some series are easier to fit than others.

| Country | Sample Size | RMSE | MAE | Mean Residual | Std Residual | Min Residual | Max Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| Brunei | 55 | 2.396121 | 1.902064 | 0.000000 | 2.396121 | -4.142769 | 6.713864 |
| Cambodia | 55 | 0.332163 | 0.253817 | 0.000000 | 0.332163 | -0.814187 | 0.558715 |
| Indonesia | 55 | 0.063277 | 0.047601 | 0.000000 | 0.063277 | -0.144257 | 0.170852 |
| Laos | 55 | 0.499556 | 0.428602 | 0.000000 | 0.499556 | -0.902944 | 0.948735 |
| Malaysia | 55 | 0.791611 | 0.575671 | 0.000000 | 0.791611 | -2.804827 | 1.770562 |
| Myanmar | 55 | 0.089786 | 0.063366 | 0.000000 | 0.089786 | -0.216407 | 0.370068 |
| Philippines | 55 | 0.071132 | 0.057192 | 0.000000 | 0.071132 | -0.132695 | 0.203943 |
| Singapore | 55 | 0.928726 | 0.738100 | 0.000000 | 0.928726 | -2.464795 | 2.229017 |
| Thailand | 55 | 0.115671 | 0.094920 | 0.000000 | 0.115671 | -0.267038 | 0.232633 |
| Timor-Leste | 55 | 0.449285 | 0.337258 | 0.000000 | 0.449285 | -0.731957 | 1.275113 |
| Vietnam | 55 | 0.958946 | 0.925750 | 0.000000 | 0.958946 | -1.162572 | 1.391593 |

Interpretation:
- Mean residuals are approximately zero, so the model is not systematically biased high or low.
- Brunei, Singapore, and Vietnam have larger residual spread, so their GHG series are harder to predict.
- Indonesia, Myanmar, Philippines, and Thailand are much tighter fits.

## Dummy Variables Used

| Dummy | Encoding | Scope | Used In |
|---|---|---|---|
| Global Shell ruling 2021 | 1 in 2021 only | All countries | Forest ARIMAX_DUMMY, GHG ARIMAX_DUMMY benchmark |
| Net-zero announcement | 1 from the announcement year onward | Country-specific proxy | Forest ARIMAX_DUMMY, GHG ARIMAX_DUMMY benchmark |
| Singapore carbon tax | 1 from 2019 onward | Singapore | Forest ARIMAX_DUMMY, GHG ARIMAX_DUMMY benchmark |
| Indonesia ETS | 1 from 2023 onward | Indonesia | Forest ARIMAX_DUMMY, GHG ARIMAX_DUMMY benchmark |
| PT Kallista Alam penalty | 1 in 2014 only | Indonesia | Forest ARIMAX_DUMMY, GHG ARIMAX_DUMMY benchmark |

## GHG Emissions
Final model: ARIMA(1,1,0)

Reasoning:
- The earlier ARIMA selection sometimes collapsed to flat forecasts for several countries.
- Forcing ARIMA(1,1,0) gives a more informative year-to-year GHG trajectory.
- This is the latest model used in the final forecast workbook.

Key result from latest forecast run:
- All 11 countries now show varying GHG forecasts across 2025-2030.
- Forecast file regenerated successfully with 143 total rows.

Reference files:
- model_outputs/final_forecast_2030.xlsx
- final_forecast_2030.py
