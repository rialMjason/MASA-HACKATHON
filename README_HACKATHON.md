MASA-HACKATHON — Hackathon submission notes

What I changed
- Converted the provided R panel pipeline (EMDAT frequency + WB indicators -> Negative Binomial GLM) into Python and prepended it to `final_forecast_2030_improved.py`.
- Added `main_nb()` to run the NB model independently.

Files added/modified
- Modified: `final_forecast_2030_improved.py` — prepended Python conversion block implementing the panel creation and Negative Binomial GLM fit.
- Added: `README_HACKATHON.md` (this file) — instructions for running and requirements.

How to run
1. Ensure dependencies are installed (recommended in a virtualenv):

```bash
pip install pandas numpy statsmodels
```

If you want to run plotting (requires matplotlib + seaborn):

```bash
pip install matplotlib seaborn
```

2. Place the required data files in the repository root:
- `WB_WDI_WIDEF.csv` (World Bank wide format)
- the EMDAT export file that begins with `public_emdat_custom_request` (CSV or TXT). The script searches for that prefix.

3. Run the Negative Binomial panel model only:

```bash
python -c "from final_forecast_2030_improved import main_nb; main_nb()"
```

4. Produce facet plots (observed + forecasts):

```bash
python scripts/plot_ghg_forest_facets.py
```

Notes: the plotting script will look for `WB_WDI_WIDEF.csv` and, if present,
`model_outputs/final_forecast_2030.xlsx` (created by running the main forecasting
script). If the forecast file exists it will overlay forecast lines and confidence
intervals on the facet plots.

What the block does
- Builds `ghg_long` and `forest_long` from `WB_WDI_WIDEF.csv` by melting year columns.
- Builds a frequency table from EMDAT grouped by `Country` and `Start Year`.
- Maps country names to ISO codes used in the WB file and joins.
- Creates `ghg_lag9` and `forest_lag1` lagged predictors.
- Fits a Negative Binomial GLM with formula: `Frequency ~ ghg_lag9 + forest_lag1 + Year + C(Country)`.

Notes and caveats
- Column names must match: EMDAT must include a `Country` column and a `Start Year` column (exact or similar). If the column naming differs, rename before running.
- The conversion fits a NB GLM using `statsmodels`' `NegativeBinomial` family. This is the closest direct analogue to R's `glm.nb` in a simple script.

If you want, I can also:
- Extract the NB output to CSV under `model_outputs/`.
- Add a small runner script or unit test to validate the transformed panel dataset.

Good luck with the hackathon — tell me if you want packaging (requirements.txt) or further automation.
