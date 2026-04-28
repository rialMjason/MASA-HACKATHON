#!/usr/bin/env python3
"""
Data Converter for Southeast Asia Risk Dashboard
Convert CSV or database data into the format required by the dashboard.
"""

import json
import pandas as pd
from pathlib import Path

# Country codes mapping
COUNTRY_CODES = {
    'Thailand': 'TH',
    'Vietnam': 'VN',
    'Indonesia': 'ID',
    'Philippines': 'PH',
    'Malaysia': 'MY',
    'Singapore': 'SG',
    'Myanmar': 'MM',
    'Cambodia': 'KH',
    'Laos': 'LA',
    'Brunei': 'BN',
    'Timor-Leste': 'TL',
    'East Timor': 'TL'
}

def normalize_value(value, max_scale=10):
    """Convert value to 0-10 scale if needed."""
    try:
        val = float(value)
        if val > max_scale:
            return val / 100 * max_scale  # If appears to be percentage
        return val
    except (ValueError, TypeError):
        return 0

def normalize_percentage(value):
    """Convert value to 0-100 scale."""
    try:
        val = float(value)
        if val > 100:
            return 100
        return val
    except (ValueError, TypeError):
        return 0

def create_country_entry(country_code, physical_risk=0, transition_risk=0, 
                         liability_risk=0, physical_pct=0, transition_pct=0, 
                         liability_pct=0):
    """Create a country data entry."""
    return {
        'code': country_code,
        'physicalRisk': normalize_value(physical_risk),
        'physicalRiskPercent': normalize_percentage(physical_pct),
        'transitionRisk': normalize_value(transition_risk),
        'transitionRiskPercent': normalize_percentage(transition_pct),
        'liabilityRisk': normalize_value(liability_risk),
        'liabilityRiskPercent': normalize_percentage(liability_pct)
    }

def csv_to_dashboard_format(csv_file):
    """
    Convert CSV file to dashboard format.
    
    Expected CSV columns:
    - Country: Country name
    - Physical Risk: Numeric value
    - Physical Risk %: Percentage value
    - Transition Risk: Numeric value
    - Transition Risk %: Percentage value
    - Liability Risk: Numeric value
    - Liability Risk %: Percentage value
    """
    df = pd.read_csv(csv_file)
    
    data = []
    for idx, row in df.iterrows():
        country_name = str(row.get('Country', '')).strip()
        if country_name not in COUNTRY_CODES:
            print(f"Warning: Country '{country_name}' not recognized. Skipping.")
            continue
        
        country_code = COUNTRY_CODES[country_name]
        
        entry = create_country_entry(
            country_code=country_code,
            physical_risk=row.get('Physical Risk', 0),
            transition_risk=row.get('Transition Risk', 0),
            liability_risk=row.get('Liability Risk', 0),
            physical_pct=row.get('Physical Risk %', 0),
            transition_pct=row.get('Transition Risk %', 0),
            liability_pct=row.get('Liability Risk %', 0)
        )
        data.append(entry)
    
    return data

def generate_javascript_code(data):
    """Generate JavaScript code to update the dashboard."""
    return f"updateCountryRiskData({json.dumps(data, indent=2)})"

def main():
    """Main function with examples."""
    print("Southeast Asia Risk Dashboard - Data Converter")
    print("=" * 50)
    
    # Example: Create sample data
    sample_data = [
        create_country_entry('TH', 7.5, 5.2, 3.8, 75, 52, 38),
        create_country_entry('VN', 8.2, 6.1, 4.5, 82, 61, 45),
        create_country_entry('ID', 8.8, 5.9, 4.2, 88, 59, 42),
    ]
    
    print("\nExample JavaScript update code:")
    print(generate_javascript_code(sample_data))
    
    # Example: Save to JSON
    output_file = 'dashboard_data.json'
    with open(output_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    print(f"\n✓ Sample data saved to {output_file}")
    
    # Usage example
    print("\n\nUsage Examples:")
    print("-" * 50)
    print("\n1. From CSV file:")
    print("   >>> data = csv_to_dashboard_format('your_file.csv')")
    print("   >>> code = generate_javascript_code(data)")
    print("   >>> print(code)  # Use in browser console")
    
    print("\n2. Manual entry:")
    print("   >>> entry = create_country_entry('TH', 7.5, 5.2, 3.8, 75, 52, 38)")

if __name__ == '__main__':
    main()
