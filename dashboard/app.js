// Initialize map
const map = L.map('map').setView([10, 108], 5);

// Add base map layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19,
    minZoom: 3
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
function createFlagIcon(flag) {
    return L.divIcon({
        className: 'flag-marker',
        html: `<div class="flag-bubble" title="Click to zoom">${flag}</div>`,
        iconSize: [50, 50],
        iconAnchor: [25, 25],
        popupAnchor: [0, -25]
    });
}

// Initialize markers for all countries
function initializeMarkers() {
    countriesData.forEach(country => {
        const marker = L.marker(country.coordinates, {
            icon: createFlagIcon(country.flag),
            zIndexOffset: 1000
        }).addTo(map);

        marker.on('click', function() {
            selectCountry(country);
        });

        // Hover effect
        marker.on('mouseover', function() {
            this.setZIndexOffset(1001);
            this.getElement().querySelector('.flag-bubble').style.transform = 'scale(1.2)';
        });

        marker.on('mouseout', function() {
            this.setZIndexOffset(1000);
            this.getElement().querySelector('.flag-bubble').style.transform = 'scale(1)';
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

    // Zoom to country
    map.setView(country.coordinates, 7);

    // Close any existing country layer
    if (countryLayer) {
        map.removeLayer(countryLayer);
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

    map.setView([10, 108], 5);
});

// Add CSS styles for markers dynamically
const style = document.createElement('style');
style.innerHTML = `
    .flag-marker {
        background: none;
        border: none;
    }

    .flag-bubble {
        font-size: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: transform 0.2s ease, filter 0.2s ease;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
        animation: bounce 0.6s ease-in-out infinite;
    }

    .flag-marker:hover .flag-bubble {
        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.4));
    }

    @keyframes bounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-5px);
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
