#!/usr/bin/env python3
"""
Generate forecast data JSON for the dashboard from model outputs
"""
import pandas as pd
import json
import os

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

def load_forest_data():
    """Load forest area forecast data"""
    df = pd.read_csv('../model_outputs/forest_area_forecast_detailed.csv')
    
    # Group by country code
    forest_by_country = {}
    for country_code, group in df.groupby('country_code'):
        # Sort by year
        group = group.sort_values('year')
        forest_by_country[country_code] = {
            'years': group['year'].tolist(),
            'actual': group['actual'].tolist(),
            'forecast': group['regression_forecast'].tolist()
        }
    
    return forest_by_country

def load_ghg_data():
    """Load GHG forecast data"""
    ghg_by_country = {}
    
    # Try loading detailed GHG data
    try:
        ghg_df = pd.read_csv('../model_outputs/ghg_arimax_dummy_detailed.csv')
        for country_code, group in ghg_df.groupby('Country_Code'):
            group = group.sort_values('Year')
            ghg_by_country[country_code] = {
                'years': group['Year'].tolist(),
                'actual': group['Actual_GHG'].tolist(),
                'forecast': group['ARIMAX_DUMMY_Prediction'].tolist()
            }
    except Exception as e:
        print(f"Could not load GHG data: {e}")
    
    return ghg_by_country

def generate_forecast_data():
    """Generate the complete forecast data JSON"""
    
    data = {
        'transitionRiskScores': {},
        'countryCodeMap': COUNTRY_MAPPING
    }
    
    # Add transition risk scores
    for country, score in TRANSITION_RISK_SCORES.items():
        percent = convert_to_percent(score)
        data['transitionRiskScores'][country] = {
            'score': score,
            'percent': percent
        }
    
    # Load forest and GHG data
    try:
        forest_data = load_forest_data()
        data['forestData'] = forest_data
    except Exception as e:
        print(f"Warning: Could not load forest data: {e}")
    
    try:
        ghg_data = load_ghg_data()
        data['ghgData'] = ghg_data
    except Exception as e:
        print(f"Warning: Could not load GHG data: {e}")
    
    return data

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    data = generate_forecast_data()
    
    # Write to JSON file
    with open('forecast-data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print("Generated forecast-data.json")
    print(json.dumps(data, indent=2)[:500] + "...")
