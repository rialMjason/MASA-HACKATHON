# Southeast Asia Risk Dashboard

An interactive dashboard displaying climate and economic risk metrics for 11 Southeast Asian countries.

## Features

- **Interactive Map**: Displays all 11 Southeast Asian countries with flag markers
- **Click to Zoom**: Click any flag to zoom into that country and view detailed information
- **Risk Metrics**: Shows three types of risk metrics for each country:
  - **Physical Risk**: Climate-related physical hazards (flooding, storms, sea-level rise, etc.)
  - **Transition Risk**: Economic and structural transition risks
  - **Liability Risk**: Legal and financial liability risks
- **States/Provinces**: Lists major states or provinces within each country
- **Country Descriptions**: Provides context about each country's risk profile
- **Responsive Design**: Works on desktop and tablet devices

## Countries Included

1. Thailand 🇹🇭
2. Vietnam 🇻🇳
3. Indonesia 🇮🇩
4. Philippines 🇵🇭
5. Malaysia 🇲🇾
6. Singapore 🇸🇬
7. Myanmar 🇲🇲
8. Cambodia 🇰🇭
9. Laos 🇱🇦
10. Brunei 🇧🇳
11. Timor-Leste 🇹🇱

## How to Use

### Running the Dashboard

1. Open `index.html` in a modern web browser, or
2. Use a local server:
   ```bash
   python -m http.server 8000
   ```
   Then navigate to `http://localhost:8000/dashboard/`

### Viewing Country Details

1. Click on any flag marker on the map
2. The map will zoom into that country
3. The right panel will display:
   - Country name and description
   - Physical, Transition, and Liability risk metrics
   - List of states/provinces
4. Click "Back to Map" to return to the full Southeast Asia view

## Updating Risk Metrics

Risk metrics are currently set to 0 by default. To update them with actual data:

### Option 1: Direct JavaScript Update

In the browser console, use:

```javascript
updateCountryRiskData([
    {
        code: "TH",
        physicalRisk: 7.5,
        transitionRisk: 5.2,
        liabilityRisk: 3.8,
        physicalRiskPercent: 75,
        transitionRiskPercent: 52,
        liabilityRiskPercent: 38
    },
    // Add more countries as needed
]);
```

### Option 2: Modify map-data.js

Update the `countriesData` array in `map-data.js`:

```javascript
{
    name: "Thailand",
    code: "TH",
    flag: "🇹🇭",
    coordinates: [15.870032, 100.992541],
    description: "...",
    states: [...],
    physicalRisk: 7.5,           // Your numeric value
    transitionRisk: 5.2,          // Your numeric value
    liabilityRisk: 3.8,           // Your numeric value
    physicalRiskPercent: 75,       // 0-100 for progress bar
    transitionRiskPercent: 52,     // 0-100 for progress bar
    liabilityRiskPercent: 38       // 0-100 for progress bar
}
```

## Data Structure

Each country object requires:

```javascript
{
    name: "Country Name",
    code: "XX",                    // ISO 3166-1 alpha-2 code
    flag: "🏳️",                    // Country flag emoji
    coordinates: [lat, lng],       // [latitude, longitude]
    description: "...",            // Country description
    states: ["State1", "State2"], // List of states/provinces
    physicalRisk: 0,               // Numeric risk value
    transitionRisk: 0,             // Numeric risk value
    liabilityRisk: 0,              // Numeric risk value
    physicalRiskPercent: 0,        // 0-100 for visual bar
    transitionRiskPercent: 0,      // 0-100 for visual bar
    liabilityRiskPercent: 0        // 0-100 for visual bar
}
```

## File Structure

```
dashboard/
├── index.html          # Main HTML file with styling
├── app.js             # Main JavaScript application logic
├── map-data.js        # Country data and coordinates
└── README.md          # This file
```

## Technologies Used

- **Leaflet.js**: Interactive mapping library
- **OpenStreetMap**: Base map tiles
- **CSS3**: Modern styling with gradients and animations
- **Vanilla JavaScript**: No external dependencies (except Leaflet)

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Future Enhancements

- Add real-time data integration from APIs
- Export reports and charts
- Compare countries side-by-side
- Historical data visualization
- Add more granular regional data
- Implement layers for different risk scenarios
- Add interactive charts and graphs

## Notes

- Flag markers animate with a subtle bounce effect
- Hover over markers to see zoom preview
- Risk bars automatically color-code based on percentage
- All coordinates are approximate country centers for display purposes

## Support

For updates or data corrections, modify the corresponding country object in `map-data.js` and refresh the page.
