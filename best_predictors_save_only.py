"""
Lightweight run to compute best indicator models and save results immediately.
"""
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv('WB_WDI_WIDEF.csv')
# Identify years, indicator, country
year_cols = [c for c in df.columns if str(c).isdigit()]
indicator_col = 'INDICATOR_LABEL'
country_col = 'REF_AREA'
country_code = 'MYS'
malaysia_data = df[df[country_col] == country_code].copy()

# Build indicators dict
indicators = {}
for _, r in malaysia_data.iterrows():
    name = r[indicator_col]
    vals = pd.to_numeric(r[year_cols], errors='coerce').values
    if name not in indicators:
        indicators[name] = vals

target_name = "Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)"
if target_name not in indicators:
    raise SystemExit('Target not found')

target = indicators[target_name]

# Candidate features (exclude direct GHGs)
exclude = {target_name,
           "Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)",
           "Total greenhouse gas emissions excluding LULUCF (Mt CO2e)",
           "Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)",
           "Total greenhouse gas emissions excluding LULUCF (% change from 1990)",}

features = [k for k in indicators.keys() if k not in exclude]
# keep features with >=5 observations
features = [f for f in features if np.sum(~np.isnan(indicators[f])) >= 5]

# assemble rows where target not null and at least some features present
rows_X = []
rows_y = []
for i in range(len(target)):
    if not np.isnan(target[i]):
        vals = [indicators[f][i] if i < len(indicators[f]) else np.nan for f in features]
        if np.sum(~np.isnan(vals)) >= max(3, len(features)//4):
            rows_X.append(vals)
            rows_y.append(target[i])

if len(rows_y) == 0:
    raise SystemExit('No data')

X = np.array(rows_X)
y = np.array(rows_y)
# impute
X = SimpleImputer(strategy='mean').fit_transform(X)
# reduce to top variance features
var_idx = np.argsort(-np.var(X, axis=0))[:15]
X = X[:, var_idx]
features = [features[i] for i in var_idx]

# helper to compute metrics
from math import log
from itertools import combinations

def metrics_for_combo(combo):
    Xs = X[:, list(combo)]
    n = len(y)
    p = Xs.shape[1]
    # fit
    Xd = np.column_stack([np.ones(n), Xs])
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd.dot(coef)
    resid = y - yhat
    rss = np.sum(resid**2)
    aic = 2*p + n*log(rss/n)
    aicc = aic + (2*p*(p+1))/(n-p-1) if (n-p-1)>0 else float('inf')
    bic = p*log(n) + n*log(rss/n)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - rss/ss_tot if ss_tot>0 else 0
    # cv
    k = min(5, max(2, n//3))
    kf = KFold(n_splits=k, shuffle=True, random_state=123)
    mses = []
    for tr, te in kf.split(Xs):
        Xtr = np.column_stack([np.ones(len(tr)), Xs[tr]])
        coef_tr, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
        Xte = np.column_stack([np.ones(len(te)), Xs[te]])
        ypred = Xte.dot(coef_tr)
        mses.append(np.mean((y[te]-ypred)**2))
    cv_rmse = (np.mean(mses)**0.5) if mses else float('inf')
    return dict(n_predictors=p, cv_rmse=cv_rmse, aic=aic, aicc=aicc, bic=bic, r_squared=r2)

# test combinations up to 12 features, limit combos per size
results = []
max_features = min(12, X.shape[1])
for p in range(1, max_features+1):
    combos = list(combinations(range(X.shape[1]), p))
    if len(combos) > 200:
        np.random.seed(123)
        combos = list(np.random.choice(len(combos), 200, replace=False))
        combos = [list(combinations(range(X.shape[1]), p))[i] for i in combos]
    for combo in combos:
        m = metrics_for_combo(combo)
        m['features'] = [features[i] for i in combo]
        results.append(m)

resdf = pd.DataFrame(results)
resdf = resdf.sort_values('bic').reset_index(drop=True)

# Save results immediately
import os
os.makedirs('model_outputs', exist_ok=True)
resdf.to_csv('model_outputs/best_indicators_analysis.csv', index=False)

with open('model_outputs/best_indicators_summary.txt','w') as f:
    best = resdf.iloc[0]
    f.write('BEST MODEL BY BIC\n')
    f.write(f"Predictors: {int(best['n_predictors'])}\n")
    f.write('Indicators:\n')
    for it in best['features']:
        f.write(f" - {it}\n")
    f.write(f"CV-RMSE: {best['cv_rmse']:.6f}\nAIC: {best['aic']:.2f}\nAICc: {best['aicc']:.2f}\nBIC: {best['bic']:.2f}\nR2: {best['r_squared']:.6f}\n")

print('Saved model_outputs/best_indicators_analysis.csv and summary.txt')
