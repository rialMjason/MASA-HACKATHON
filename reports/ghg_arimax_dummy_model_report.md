# GHG Forecasting Report: ARIMA

## 1. Model Summary
The final hackathon model for GHG is ARIMA. ARIMAX_DUMMY was evaluated as a policy-aware comparison model using policy and penalty dummy variables as exogenous inputs, and it is compared against the original univariate ARIMA baseline on the same rolling-origin backtest.

## 2. Why Dummies Were Added
Dummy variables let the model react to known interventions and one-off policy shocks without forcing those events to be learned only from the emission history. They are encoded as binary indicators that switch on at the event year and, for announcement-style variables, stay on afterward.

## 3. Dummy Variables Used

| Dummy | Encoding | Scope |
|---|---|---|
| Global Shell ruling 2021 | 1 in 2021 only | All countries |
| Net-zero announcement | 1 from the announcement year onward | Country-specific proxy |
| Singapore carbon tax | 1 from 2019 onward | Singapore |
| Indonesia ETS | 1 from 2023 onward | Indonesia |
| PT Kallista Alam penalty | 1 in 2014 only | Indonesia |

## 4. Evaluation Setup
- Target: Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)
- Countries: 11 Southeast Asian countries
- Validation: rolling-origin backtest
- Comparison: ARIMA vs ARIMAX_DUMMY

## 5. Results
- Mean ARIMA RMSE: 0.442609
- Mean ARIMAX_DUMMY RMSE: 0.440984
- Mean delta RMSE (dummy - original): -0.001625
- Countries improved: 4
- Countries worsened: 7

### Country-level comparison

| Country | ARIMA RMSE | ARIMAX_DUMMY RMSE | Delta RMSE |
|---|---:|---:|---:|
| Brunei | 2.502067 | 2.403337 | -0.098730 |
| Timor-Leste | 0.191240 | 0.180811 | -0.010429 |
| Laos | 0.255023 | 0.251806 | -0.003217 |
| Philippines | 0.067265 | 0.065567 | -0.001698 |
| Thailand | 0.172296 | 0.173045 | 0.000749 |
| Cambodia | 0.098730 | 0.100422 | 0.001693 |
| Malaysia | 0.340268 | 0.351685 | 0.011416 |
| Myanmar | 0.109369 | 0.125921 | 0.016552 |
| Indonesia | 0.143206 | 0.163592 | 0.020385 |
| Singapore | 0.792591 | 0.815057 | 0.022466 |
| Vietnam | 0.196644 | 0.219577 | 0.022933 |

## 6. Interpretation
Even though ARIMAX_DUMMY can edge out ARIMA on mean RMSE in this run, the final submission for GHG is ARIMA to keep the model simpler and more stable across countries.

## 7. Practical Takeaway
Use ARIMAX_DUMMY as a benchmark to measure the impact of policy events, but keep ARIMA as the final GHG forecasting choice for the hackathon submission.