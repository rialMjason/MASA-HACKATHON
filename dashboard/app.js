// Initialize map
const map = L.map('map').setView([10, 108], 5);

// Add Google Maps tile layer
L.tileLayer('https://{s}.google.com/maps/vt/lyrs=m@221097413,padmc@221205649&hl=en&src=apk&apn=com.google.android.apps.maps&pn=com.google.maps.android&x={x}&y={y}&z={z}&s=Galile', {
    minZoom: 0,
    maxZoom: 21,
    attribution: '© Google Maps',
    subdomains: ['mt0', 'mt1', 'mt2', 'mt3']
}).addTo(map);

// Color scheme for markers
const markerColors = {
    default: '#667eea',
    active: '#764ba2'
};

let countryMarkers = {};
let currentCountry = null;
let countryLayer = null;

// Create custom icon for country flags
function createFlagIcon(country) {
    // Use flagUrl if available, otherwise use emoji
    if (country.flagUrl) {
        // Add special class for Malaysia and Singapore to crop center
        const cropClass = (country.code === 'MY' || country.code === 'SG') ? 'flag-crop-red' : '';
        return L.divIcon({
            className: 'flag-marker',
            html: `<div class="flag-ball">
                        <div class="flag-image-container">
                            <img src="${country.flagUrl}" alt="${country.name} flag" class="flag-image ${cropClass}" onerror="this.parentElement.innerHTML='<div class=\\\"flag-emoji\\\">${country.flagEmoji || '🏴'}</div>'">
                        </div>
                        <div class="ball-shine"></div>
                    </div>`,
            iconSize: [80, 80],
            iconAnchor: [40, 40],
            popupAnchor: [0, -40]
        });
    } else {
        // Fallback to emoji
        return L.divIcon({
            className: 'flag-marker',
            html: `<div class="flag-ball">
                        <div class="flag-image-container">
                            <div class="flag-emoji">${country.flagEmoji}</div>
                        </div>
                        <div class="ball-shine"></div>
                    </div>`,
            iconSize: [80, 80],
            iconAnchor: [40, 40],
            popupAnchor: [0, -40]
        });
    }
}

// Initialize markers for all countries
function initializeMarkers() {
    countriesData.forEach(country => {
        const marker = L.marker(country.coordinates, {
            icon: createFlagIcon(country),
            zIndexOffset: 1000
        }).addTo(map);

        marker.on('click', function() {
            selectCountry(country);
        });

        // Hover effect
        marker.on('mouseover', function() {
            this.setZIndexOffset(1001);
            this.getElement().querySelector('.flag-ball').style.transform = 'scale(1.4)';
            this.getElement().querySelector('.flag-ball').style.boxShadow = '0 16px 32px rgba(102, 126, 234, 0.7), inset -2px -2px 8px rgba(0, 0, 0, 0.15)';
        });

        marker.on('mouseout', function() {
            this.setZIndexOffset(1000);
            this.getElement().querySelector('.flag-ball').style.transform = 'scale(1)';
            this.getElement().querySelector('.flag-ball').style.boxShadow = '0 6px 16px rgba(0, 0, 0, 0.4), inset -2px -2px 8px rgba(0, 0, 0, 0.15)';
        });

        countryMarkers[country.code] = marker;
    });
}

// Select and display country
function selectCountry(country) {
    currentCountry = country;

    // Update info panel
    document.getElementById('countryName').textContent = country.name;
    document.getElementById('placeholderText').style.display = 'none';
    document.getElementById('countryDetails').style.display = 'block';
    document.getElementById('backButton').classList.remove('hidden');

    // Update country description
    document.getElementById('countryDescription').textContent = country.description;

    // Update risk metrics
    updateRiskMetrics(country);

    // Update states list
    updateStatesList(country);

    // Close any existing country layer
    if (countryLayer) {
        map.removeLayer(countryLayer);
    }

    // Fetch and display country borders from GeoJSON
    const countryCode = country.code.toUpperCase();
    fetch(`https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson`)
        .then(res => res.json())
        .then(geojson => {
            // Find and draw this country's borders
            const countryFeature = geojson.features.find(f => 
                f.properties.ISO_A3 === countryCode || 
                f.properties.ISO_A2 === country.code
            );
            
            if (countryFeature && countryFeature.geometry) {
                countryLayer = L.geoJSON(countryFeature, {
                    style: {
                        color: '#FFD700',
                        weight: 3,
                        opacity: 0.9,
                        fill: false,
                        dashArray: '5, 5'
                    }
                }).addTo(map);

                // Fit map bounds to show entire country with padding
                const bounds = countryLayer.getBounds();
                map.fitBounds(bounds, { padding: [50, 50] });
            } else {
                // Fallback to simple zoom if border data not found
                map.setView(country.coordinates, 7);
            }
        })
        .catch(err => {
            console.log('Could not fetch country borders');
            // Fallback to simple zoom
            map.setView(country.coordinates, 7);
        });

    // Bring highlighted country marker to front
    const marker = countryMarkers[country.code];
    if (marker) {
        marker.setZIndexOffset(2000);
    }
}

// Update risk metrics display
function updateRiskMetrics(country) {
    // Physical Risk
    document.getElementById('physicalRisk').textContent = country.physicalRisk.toFixed(1);
    document.getElementById('physicalBar').style.width = country.physicalRiskPercent + '%';

    // Transition Risk
    document.getElementById('transitionRisk').textContent = country.transitionRisk.toFixed(1);
    document.getElementById('transitionBar').style.width = country.transitionRiskPercent + '%';

    // Liability Risk
    document.getElementById('liabilityRisk').textContent = country.liabilityRisk.toFixed(1);
    document.getElementById('liabilityBar').style.width = country.liabilityRiskPercent + '%';

    // Determine bar colors based on risk levels
    updateBarColor('physicalBar', country.physicalRiskPercent);
    updateBarColor('transitionBar', country.transitionRiskPercent);
    updateBarColor('liabilityBar', country.liabilityRiskPercent);
}

// Update bar colors based on risk level
function updateBarColor(barId, percentage) {
    const bar = document.getElementById(barId);
    bar.classList.remove('low', 'medium', 'high');

    if (percentage < 33) {
        bar.classList.add('low');
    } else if (percentage < 66) {
        bar.classList.add('medium');
    } else {
        bar.classList.add('high');
    }
}

// Update states list
function updateStatesList(country) {
    const statesList = document.getElementById('statesList');
    statesList.innerHTML = country.states
        .map(state => `<div class="state-badge">${state}</div>`)
        .join('');
}

// Back button functionality
document.getElementById('backButton').addEventListener('click', function() {
    currentCountry = null;
    document.getElementById('countryName').textContent = 'Southeast Asia Dashboard';
    document.getElementById('placeholderText').style.display = 'block';
    document.getElementById('countryDetails').style.display = 'none';
    document.getElementById('backButton').classList.add('hidden');

    if (countryLayer) {
        map.removeLayer(countryLayer);
    }

    // Reset all marker z-index offsets
    Object.values(countryMarkers).forEach(marker => {
        marker.setZIndexOffset(1000);
    });

    map.setView([10, 108], 5);
});

// Add CSS styles for markers dynamically
const style = document.createElement('style');
style.innerHTML = `
    .flag-marker {
        background: none;
        border: none;
    }

    .flag-ball {
        width: 80px;
        height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border-radius: 50%;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4), inset -2px -2px 8px rgba(0, 0, 0, 0.15);
        background: radial-gradient(circle at 35% 35%, rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.1) 40%, rgba(200, 200, 255, 0.1));
        position: relative;
        animation: ballBounce 0.6s ease-in-out infinite;
        border: 3px solid rgba(255, 255, 255, 0.4);
        overflow: hidden;
    }

    .flag-image-container {
        position: absolute;
        width: 76px;
        height: 76px;
        border-radius: 50%;
        overflow: hidden;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 3;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .flag-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 50%;
    }

    /* For Malaysia and Singapore - show only red parts */
    .flag-image.flag-crop-red {
        object-position: left center;
        width: 150%;
    }

    .flag-emoji {
        font-size: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        height: 100%;
    }

    .ball-shine {
        position: absolute;
        width: 30px;
        height: 30px;
        background: radial-gradient(circle at 40% 40%, rgba(255, 255, 255, 0.8), transparent);
        border-radius: 50%;
        top: 8px;
        left: 8px;
        pointer-events: none;
        z-index: 4;
    }

    .flag-marker:hover .flag-ball {
        box-shadow: 0 16px 32px rgba(102, 126, 234, 0.7), inset -2px -2px 8px rgba(0, 0, 0, 0.15);
    }

    @keyframes ballBounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-8px);
        }
    }
    }

    .leaflet-popup-content-wrapper {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .leaflet-popup-content {
        margin: 0;
        padding: 8px;
    }
`;
document.head.appendChild(style);

// Initialize on page load
window.addEventListener('load', function() {
    initializeMarkers();
});

// Add function to update country data from external source
function updateCountryRiskData(updates) {
    updates.forEach(update => {
        const country = countriesData.find(c => c.code === update.code);
        if (country) {
            country.physicalRisk = update.physicalRisk || 0;
            country.transitionRisk = update.transitionRisk || 0;
            country.liabilityRisk = update.liabilityRisk || 0;
            country.physicalRiskPercent = update.physicalRiskPercent || 0;
            country.transitionRiskPercent = update.transitionRiskPercent || 0;
            country.liabilityRiskPercent = update.liabilityRiskPercent || 0;

            // If this country is currently selected, update display
            if (currentCountry && currentCountry.code === update.code) {
                updateRiskMetrics(country);
            }
        }
    });
}

// Export function for use in console or external scripts
window.updateCountryRiskData = updateCountryRiskData;
