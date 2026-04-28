# Deployment & Setup Guide

The Southeast Asia Risk Dashboard is a client-side web application that runs in any modern web browser. No backend server is required.

## Quick Start (No Setup Required)

1. **Open the start page:**
   - Simply open `index-start.html` in your web browser
   - Or navigate to the dashboard folder and open `index.html` directly

2. **Explore the map:**
   - Click on any country flag to zoom in
   - View country details in the right panel
   - Click "Back to Map" to return

## Local Development Server

For better development experience, run a local server:

### Option 1: Python (Recommended)

```bash
# Navigate to the dashboard directory
cd dashboard

# Python 3
python -m http.server 8000

# Or Python 2
python -m SimpleHTTPServer 8000

# Then open http://localhost:8000 in your browser
```

### Option 2: Node.js (http-server)

```bash
# Install globally (if not already installed)
npm install -g http-server

# Run from dashboard directory
cd dashboard
http-server

# Open http://localhost:8080 in your browser
```

### Option 3: Live Server (VS Code Extension)

1. Install "Live Server" extension in VS Code
2. Right-click on `index-start.html` or `index.html`
3. Select "Open with Live Server"
4. Browser will open automatically

### Option 4: Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY dashboard /app

EXPOSE 8000

CMD ["python", "-m", "http.server", "8000"]
```

Build and run:
```bash
docker build -t masa-dashboard .
docker run -p 8000:8000 masa-dashboard
```

## Data Upload Workflow

### Step 1: Prepare Your Data

#### Method A: Using Data Input Tool
1. Open `data-input.html`
2. Enter risk values for each country
3. Click "Update Dashboard"
4. Data is saved automatically

#### Method B: Using CSV
1. Create a CSV file with columns:
   ```
   Country,Physical Risk,Physical Risk %,Transition Risk,Transition Risk %,Liability Risk,Liability Risk %
   Thailand,7.5,75,5.2,52,3.8,38
   Vietnam,8.2,82,6.1,61,4.5,45
   ```

2. Run the converter:
   ```bash
   python dashboard/data_converter.py
   ```

3. Use the generated JavaScript code in browser console

#### Method C: Direct JSON
1. Edit `map-data.js`
2. Update the `countriesData` array with your values
3. Refresh the dashboard

### Step 2: Update Dashboard

#### Via Browser Console:
```javascript
// Paste this in the browser console (F12)
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
    // ... more countries
]);
```

## Deployment Options

### Option 1: GitHub Pages (Free, Static)

```bash
# Push dashboard folder to GitHub
# In your repo settings, enable GitHub Pages
# Point to the dashboard folder
# Access at: https://yourusername.github.io/MASA-HACKATHON/dashboard/
```

### Option 2: Netlify (Free, with CI/CD)

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login to Netlify
netlify login

# Deploy
netlify deploy --dir=dashboard --prod
```

### Option 3: Vercel (Free, with Serverless)

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
cd dashboard
vercel
```

### Option 4: AWS S3 + CloudFront

```bash
# Upload to S3
aws s3 sync dashboard/ s3://your-bucket/dashboard/

# Create CloudFront distribution for CDN
```

### Option 5: Traditional Web Server (Apache, Nginx)

```bash
# Copy dashboard folder to web root
cp -r dashboard/ /var/www/html/risk-dashboard/

# Ensure proper permissions
chmod 755 /var/www/html/risk-dashboard/
```

## File Structure

```
dashboard/
├── index-start.html          # Entry point with quick start guide
├── index.html                # Main dashboard application
├── app.js                    # Application logic and interactivity
├── map-data.js              # Country data and coordinates
├── data-input.html          # Data entry interface
├── data_converter.py        # Python utility to convert CSV data
├── sample-data.json         # Example data in JSON format
├── README.md                # Full documentation
└── deployment-guide.md      # This file
```

## Troubleshooting

### Map not loading
- Check browser console (F12) for errors
- Ensure internet connection (OpenStreetMap tiles need to load)
- Try clearing browser cache

### Data not updating
- Refresh the page after entering data
- Check browser console for JavaScript errors
- Ensure data values are in correct format (numbers, not strings)

### LocalStorage issues
- Clear browser storage: Settings → Clear browsing data
- Try a different browser
- Check if running from `file://` (some browsers restrict localStorage)

### Performance issues
- All on one page is lightweight and fast
- Reduce number of countries if needed
- Use latest browser version

## Customization

### Add/Remove Countries
Edit `map-data.js` - add or remove entries from `countriesData` array

### Change Colors
Edit CSS in `index.html` - look for `background:` and `border-left-color:` properties

### Modify Descriptions
Edit country descriptions in `map-data.js` - update the `description` field

### Change Map Center/Zoom
In `app.js`, find `map.setView()` and modify coordinates and zoom level

## Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
- Mobile browsers (iOS Safari, Chrome Android)

## Environment Variables

None required! This is a fully client-side application.

## Security Considerations

- All data is stored in browser localStorage (not encrypted)
- No data is sent to external servers (except map tiles from OpenStreetMap)
- Use HTTPS for production deployments
- Consider encrypting sensitive data before deployment

## Performance Metrics

- Initial load: < 2 seconds
- Map interaction: 60 FPS
- Data update: Instant
- File sizes:
  - index.html: ~15 KB
  - app.js: ~8 KB
  - map-data.js: ~6 KB
  - Total: ~40 KB (before Leaflet.js CDN)

## Maintenance

### Regular Updates
- Update Leaflet.js version periodically
- Keep browser compatibility in mind
- Monitor OpenStreetMap tile availability

### Backups
- Export data from data-input.html regularly
- Keep a JSON backup of your data
- Version control your modifications

## Support

For issues or questions:
1. Check README.md
2. Review browser console errors
3. Verify data format matches examples
4. Try deployment-guide.md solutions

---

Created for MASA-HACKATHON
Last updated: 2026-04-28
