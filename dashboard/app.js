// Initialize map
const map = L.map('map').setView([10, 108], 5);

// Set zoom constraints to prevent excessive zooming
map.setMinZoom(3);
map.setMaxZoom(10);

// Limit map to Southeast Asia region only
// Bounds: [South, West] to [North, East]
const seaBounds = L.latLngBounds(
    L.latLng(-10.6, 90.0),  // Southwest corner (farther west for more SEA coverage)
    L.latLng(29.0, 141.0)   // Northeast corner (Myanmar north, Indonesia east)
);
map.setMaxBounds(seaBounds);
map.on('drag', function() {
    map.panInsideBounds(seaBounds, { animate: false });
});

// Add Google Maps tile layer
L.tileLayer('https://{s}.google.com/maps/vt/lyrs=m@221097413,padmc@221205649&hl=en&src=apk&apn=com.google.android.apps.maps&pn=com.google.maps.android&x={x}&y={y}&z={z}&s=Galile', {
    minZoom: 3,
    maxZoom: 10,
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
let originalCountryCoordinates = {};
let currentFetchRequest = null;

// Create custom icon for country flags
function createFlagIcon(country) {
    // Use flagUrl if available, otherwise use emoji
    if (country.flagUrl) {
        const focusClassMap = {
            MY: 'flag-focus-my',
            SG: 'flag-focus-sg',
            TL: 'flag-focus-tl'
        };
        const focusClass = focusClassMap[country.code] || '';

        return L.divIcon({
            className: 'flag-marker',
            html: `<div class="flag-ball">
                        <div class="flag-image-container">
                            <img src="${country.flagUrl}" alt="${country.name} flag" class="flag-image ${focusClass}" onerror="this.parentElement.innerHTML='<div class=\\\"flag-emoji\\\">${country.flagEmoji || '🏴'}</div>'">
                        </div>
                        <div class="ball-shine"></div>
                    </div>`,
            iconSize: [64, 64],
            iconAnchor: [32, 32],
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
            iconSize: [64, 64],
            iconAnchor: [32, 32],
            popupAnchor: [0, -40]
        });
    }
}

// Initialize markers for all countries
function initializeMarkers() {
    countriesData.forEach(country => {
        // Store original coordinates
        originalCountryCoordinates[country.code] = country.coordinates;

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
            this.getElement().querySelector('.flag-ball').style.transform = 'scale(1.25)';
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
    if (currentCountry && currentCountry.code !== country.code) {
        restoreCountryMarker(currentCountry.code);
    }

    currentCountry = country;

    // Update info panel
    document.getElementById('countryName').textContent = country.name;
    const countryFlagPreview = document.getElementById('countryFlagPreview');
    const countryFlagImage = document.getElementById('countryFlagImage');
    if (countryFlagPreview && countryFlagImage && country.flagUrl) {
        countryFlagImage.src = country.flagUrl;
        countryFlagImage.alt = `${country.name} national flag`;
        countryFlagPreview.classList.add('visible');
    }
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
        countryLayer = null;
    }

    // Create a unique request ID to track this fetch
    const requestId = Date.now() + Math.random();
    currentFetchRequest = requestId;

    // Fetch and display country borders from GeoJSON
    const countryCode = country.code.toUpperCase();
    fetch(`https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson`)
        .then(res => {
            if (!res.ok) {
                throw new Error(`HTTP ${res.status}: Could not fetch country borders`);
            }
            return res.json();
        })
        .then(geojson => {
            // Only process if this is still the current request
            if (currentFetchRequest !== requestId) {
                return;
            }

            // Find and draw this country's borders
            const countryFeature = geojson.features.find(f => 
                f.properties.ISO_A3 === countryCode || 
                f.properties.ISO_A2 === country.code
            );
            
            if (countryFeature && countryFeature.geometry) {
                // Remove any existing layer before adding new one
                if (countryLayer) {
                    map.removeLayer(countryLayer);
                }

                countryLayer = L.geoJSON(countryFeature, {
                    style: {
                        color: '#FFD700',
                        weight: 3,
                        opacity: 0.9,
                        fill: false,
                        dashArray: '5, 5'
                    }
                }).addTo(map);

                // Fit map bounds to show entire country with padding (capped at zoom 9)
                const bounds = countryLayer.getBounds();
                map.fitBounds(bounds, { padding: [50, 50], maxZoom: 9 });
                
                // Move marker to top-right after zoom completes
                map.once('moveend', () => {
                    moveMarkerToTopRight(country);
                });
            } else {
                // Fallback to simple zoom if border data not found
                map.setView(country.coordinates, 7);
            }
        })
        .catch(err => {
            // Only process if this is still the current request
            if (currentFetchRequest !== requestId) {
                return;
            }
            // Silently fallback to simple zoom on error (network issues expected)
            map.setView(country.coordinates, 7);
        });

    // Bring highlighted country marker to front
    const marker = countryMarkers[country.code];
    if (marker) {
        marker.setZIndexOffset(2000);
    }
}

// Move marker to top-right corner of map
function moveMarkerToTopRight(country) {
    const marker = countryMarkers[country.code];
    if (!marker || !currentCountry || currentCountry.code !== country.code) return;

    // Get current map bounds and convert top-right pixel to coordinates
    const topRightPixel = L.point(map.getSize().x - 55, 55); // 55px from top-right corner
    const topRightCoords = map.containerPointToLatLng(topRightPixel);
    
    // Update marker position
    marker.setLatLng(topRightCoords);
}

// Return the active marker to its original location when the map is zoomed again
map.on('zoomstart', function() {
    if (currentCountry) {
        // Cancel any pending fetch requests
        currentFetchRequest = null;
        restoreCountryMarker(currentCountry.code);
    }
});

// Restore a moved marker to its original position
function restoreCountryMarker(code) {
    const marker = countryMarkers[code];
    const originalCoordinates = originalCountryCoordinates[code];

    if (marker && originalCoordinates) {
        marker.setLatLng(originalCoordinates);
        marker.setZIndexOffset(1000);
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
    currentFetchRequest = null;

    document.getElementById('countryName').textContent = 'Southeast Asia Dashboard';
    const countryFlagPreview = document.getElementById('countryFlagPreview');
    const countryFlagImage = document.getElementById('countryFlagImage');
    if (countryFlagPreview) {
        countryFlagPreview.classList.remove('visible');
    }
    if (countryFlagImage) {
        countryFlagImage.removeAttribute('src');
    }
    document.getElementById('placeholderText').style.display = 'block';
    document.getElementById('countryDetails').style.display = 'none';
    document.getElementById('backButton').classList.add('hidden');

    if (countryLayer) {
        map.removeLayer(countryLayer);
        countryLayer = null;
    }

    // Reset all marker positions to original coordinates
    Object.keys(countryMarkers).forEach(code => {
        const marker = countryMarkers[code];
        if (marker && originalCountryCoordinates[code]) {
            marker.setLatLng(originalCountryCoordinates[code]);
            marker.setZIndexOffset(1000);
        }
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
        width: 56px;
        height: 56px;
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
        width: 52px;
        height: 52px;
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
        object-position: center center;
        border-radius: 50%;
        transform-origin: center;
    }

    /* Country-specific focus tuning for small marker balls */
    .flag-image.flag-focus-my {
        object-position: 18% 50%;
        transform: scale(1.24);
    }

    .flag-image.flag-focus-sg {
        object-position: 20% 42%;
        transform: scale(1.24);
    }

    .flag-image.flag-focus-tl {
        object-position: 16% 50%;
        transform: scale(1.22);
    }

    .flag-emoji {
        font-size: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        height: 100%;
    }

    .ball-shine {
        position: absolute;
        width: 20px;
        height: 20px;
        background: radial-gradient(circle at 40% 40%, rgba(255, 255, 255, 0.8), transparent);
        border-radius: 50%;
        top: 6px;
        left: 6px;
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
