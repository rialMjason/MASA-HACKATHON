#!/usr/bin/env python3
"""
Generate forecast data JSON for the dashboard from model outputs.
"""
from pathlib import Path

import json
import os

import pandas as pd

# Define country code mappings
COUNTRY_MAPPING = {
    'BRN': {'name': 'Brunei', 'code': 'BN'},
    'KHM': {'name': 'Cambodia', 'code': 'KH'},
    'IDN': {'name': 'Indonesia', 'code': 'ID'},
    'LAO': {'name': 'Laos', 'code': 'LA'},
    'MYS': {'name': 'Malaysia', 'code': 'MY'},
    'MMR': {'name': 'Myanmar', 'code': 'MM'},
    'PHL': {'name': 'Philippines', 'code': 'PH'},
    'SGP': {'name': 'Singapore', 'code': 'SG'},
    'THA': {'name': 'Thailand', 'code': 'TH'},
    'TLS': {'name': 'Timor-Leste', 'code': 'TL'},
    'VNM': {'name': 'Vietnam', 'code': 'VN'}
}

COUNTRY_LABELS = {
    'Brunei': 'Brunei Darussalam',
    'Cambodia': 'Cambodia',
    'Indonesia': 'Indonesia',
    'Laos': 'Lao PDR',
    'Malaysia': 'Malaysia',
    'Myanmar': 'Myanmar',
    'Philippines': 'Philippines',
    'Singapore': 'Singapore',
    'Thailand': 'Thailand',
    'Timor-Leste': 'Timor-Leste',
    'Vietnam': 'Vietnam'
}

COUNTRY_NAME_TO_CODE = {value['name']: value['code'] for value in COUNTRY_MAPPING.values()}

GHG_INDICATOR = 'Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)'
FOREST_INDICATOR = 'Forest area (% of land area)'

# Transition risk scores
TRANSITION_RISK_SCORES = {
    'Brunei': 3,
    'Cambodia': 8,
    'Indonesia': 4,
    'Laos': 4,
    'Malaysia': 6,
    'Myanmar': 4,
    'Singapore': 7,
    'Thailand': 6,
    'Timor-Leste': 4,
    'Vietnam': 5,
    'Philippines': 6
}

def convert_to_percent(score, max_score=10):
    """Convert a score to percentage"""
    return (score / max_score) * 100

def extract_historical_series(df, country_label, indicator, year_columns):
    """Extract a historical year/value series from the wide World Bank dataset."""
    mask = (df['REF_AREA_LABEL'] == country_label) & (df['INDICATOR_LABEL'] == indicator)
    if not mask.any():
        return []

    row = df[mask].iloc[0]
    points = []
    for year in year_columns:
        value = row[str(year)]
        if pd.notna(value):
            points.append({'year': int(year), 'value': float(value)})
    return points

def load_historical_data():
    """Load historical GHG and forest series from the wide World Bank dataset."""
    df = pd.read_csv('../WB_WDI_WIDEF.csv', low_memory=False)
    year_columns = [int(col) for col in df.columns if col.isdigit() and 1970 <= int(col) <= 2024]
    year_columns.sort()

    historical = {}
    for country_meta in COUNTRY_MAPPING.values():
        country_name = country_meta['name']
        country_label = COUNTRY_LABELS[country_name]
        code = country_meta['code']
        historical[code] = {
            'ghg': extract_historical_series(df, country_label, GHG_INDICATOR, year_columns),
            'forest': extract_historical_series(df, country_label, FOREST_INDICATOR, year_columns)
        }

    return historical

def load_forecast_workbook():
    """Load 2030 point forecasts and CI bands from the combined workbook."""
    workbook_path = Path(__file__).resolve().parent / '..' / 'model_outputs' / 'final_forecast_2030.xlsx'
    forecast_df = pd.read_excel(workbook_path, sheet_name='Forecasts')

    forecasts = {}
    summary_2030 = {}

    for country_name, group in forecast_df.groupby('Country'):
        code = COUNTRY_NAME_TO_CODE.get(str(country_name))
        if not code:
            continue

        forecasts[code] = {'ghg': [], 'forest': []}
        summary_2030[code] = {}

        for variable in ['GHG', 'Forest']:
            variable_group = group[group['Variable'] == variable].sort_values('Year')
            points = []
            for _, row in variable_group.iterrows():
                year = int(row['Year'])
                forecast_value = float(row['Forecast'])
                lower_ci = float(row['Lower_CI'])
                upper_ci = float(row['Upper_CI'])
                points.append({
                    'year': year,
                    'forecast': forecast_value,
                    'lowerCI': lower_ci,
                    'upperCI': upper_ci,
                })
                if year == 2030:
                    summary_2030[code][variable.lower()] = {
                        'year': year,
                        'forecast': forecast_value,
                        'lowerCI': lower_ci,
                        'upperCI': upper_ci,
                    }

            forecasts[code][variable.lower()] = points

    return forecasts, summary_2030

def combine_series(historical_points, forecast_points):
    """Merge historical and forecast values into a single aligned series."""
    historical_map = {point['year']: point['value'] for point in historical_points}
    forecast_map = {point['year']: point for point in forecast_points}
    years = sorted(set(historical_map.keys()) | set(forecast_map.keys()))

    return {
        'years': years,
        'actual': [historical_map.get(year) for year in years],
        'forecast': [forecast_map.get(year, {}).get('forecast') for year in years],
        'lowerCI': [forecast_map.get(year, {}).get('lowerCI') for year in years],
        'upperCI': [forecast_map.get(year, {}).get('upperCI') for year in years],
    }

def generate_forecast_data():
    """Generate the complete forecast data JSON"""
    
    data = {
        'transitionRiskScores': {},
        'countryCodeMap': COUNTRY_MAPPING,
        'forecast2030': {}
    }
    
    # Add transition risk scores
    for country, score in TRANSITION_RISK_SCORES.items():
        percent = convert_to_percent(score)
        data['transitionRiskScores'][country] = {
            'score': score,
            'percent': percent
        }
    
    # Load historical data and 2030 forecasts
    try:
        historical_data = load_historical_data()
        forecast_data, summary_2030 = load_forecast_workbook()

        data['ghgData'] = {}
        data['forestData'] = {}

        for country_code, series_bundle in historical_data.items():
            country_forecasts = forecast_data.get(country_code, {})
            data['ghgData'][country_code] = combine_series(
                series_bundle.get('ghg', []),
                country_forecasts.get('ghg', [])
            )
            data['forestData'][country_code] = combine_series(
                series_bundle.get('forest', []),
                country_forecasts.get('forest', [])
            )

        data['forecast2030'] = summary_2030
    except Exception as e:
        print(f"Warning: Could not load forecast data: {e}")
    
    return data

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    data = generate_forecast_data()
    
    # Write to JSON file
    with open('forecast-data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print("Generated forecast-data.json")
    print(json.dumps(data, indent=2)[:500] + "...")
