// Southeast Asia Countries Data
const countriesData = [
    {
        name: "Thailand",
        code: "TH",
        flagUrl: "flags/th.webp",
        coordinates: [15.870032, 100.992541],
        description: "Thailand is a Southeast Asian country known for its tropical climate and diverse ecosystems. The country faces distinct climate-related and economic transition risks due to its dependence on agriculture and tourism.",
        states: ["Bangkok", "Chiang Mai", "Phuket", "Krabi", "Ubon Ratchathani", "Nakhon Ratchasima", "Sukhothai"],
        physicalRisk: 0,
        transitionRisk: 0,
        liabilityRisk: 0,
        physicalRiskPercent: 0,
        transitionRiskPercent: 0,
        liabilityRiskPercent: 0
    },
    {
        name: "Vietnam",
        code: "VN",
        flagUrl: "flags/vn.png",
        coordinates: [14.058804, 108.277199],
        description: "Vietnam is a densely populated Southeast Asian nation with extensive coastlines. The country faces significant physical risks from sea-level rise and typhoons, along with transition risks from rapid industrialization.",
        states: ["Hanoi", "Ho Chi Minh City", "Da Nang", "Hai Phong", "Can Tho", "Hue", "Nha Trang"],
        physicalRisk: 0,
        transitionRisk: 0,
        liabilityRisk: 0,
        physicalRiskPercent: 0,
        transitionRiskPercent: 0,
        liabilityRiskPercent: 0
    },
    {
        name: "Indonesia",
        code: "ID",
        flagUrl: "flags/id.png",
        coordinates: [-0.789275, 113.921327],
        description: "Indonesia is the world's largest archipelago, stretching across multiple islands. It faces significant climate-related physical risks including flooding, earthquakes, and volcanic activity, alongside rapid economic development challenges.",
        states: ["Jakarta", "Surabaya", "Bandung", "Medan", "Semarang", "Makassar", "Bekasi"],
        physicalRisk: 0,
        transitionRisk: 0,
        liabilityRisk: 0,
        physicalRiskPercent: 0,
        transitionRiskPercent: 0,
        liabilityRiskPercent: 0
    },
    {
        name: "Philippines",
        code: "PH",
        flagUrl: "flags/ph.svg",
        coordinates: [12.852926, 121.774017],
        description: "The Philippines is an island nation in Southeast Asia highly vulnerable to typhoons, flooding, and other natural hazards. The country experiences high exposure to physical climate risks and faces transition challenges from economic restructuring.",
        states: ["Manila", "Cebu", "Davao City", "Quezon City", "Caloocan", "Makati City", "Cagayan de Oro"],
        physicalRisk: 0,
        transitionRisk: 0,
        liabilityRisk: 0,
        physicalRiskPercent: 0,
        transitionRiskPercent: 0,
        liabilityRiskPercent: 0
    },
    {
        name: "Malaysia",
        code: "MY",
        flagUrl: "flags/my.png",
        coordinates: [4.210484, 101.975766],
        description: "Malaysia is a Southeast Asian country with a mix of urban development and natural resources. It faces physical risks from flooding and extreme weather, combined with transition risks from economic diversification.",
        states: ["Kuala Lumpur", "George Town", "Johor Bahru", "Shah Alam", "Subang Jaya", "Penang", "Klang"],
        physicalRisk: 0,
        transitionRisk: 0,
        liabilityRisk: 0,
        physicalRiskPercent: 0,
        transitionRiskPercent: 0,
        liabilityRiskPercent: 0
    },
    {
        name: "Singapore",
        code: "SG",
        flagUrl: "flags/sg.png",
        coordinates: [1.3521, 103.8198],
        description: "Singapore is a city-state and global financial hub, highly vulnerable to sea-level rise due to its coastal and low-elevation geography. Despite advanced adaptation capabilities, it faces notable physical risks.",
        states: ["Central Singapore", "East Singapore", "North-East Singapore", "North Singapore", "West Singapore"],
        physicalRisk: 0,
        transitionRisk: 0,
        liabilityRisk: 0,
        physicalRiskPercent: 0,
        transitionRiskPercent: 0,
        liabilityRiskPercent: 0
    },
    {
        name: "Myanmar",
        code: "MM",
        flagUrl: "flags/mm.svg",
        coordinates: [21.913965, 95.956711],
        description: "Myanmar is a Southeast Asian country with diverse geography and weather patterns. It experiences monsoon flooding, cyclones, and droughts, presenting significant physical climate risks alongside development challenges.",
        states: ["Yangon", "Mandalay", "Naypyidaw", "Taunggyi", "Mawlamyine", "Magway", "Bagan"],
        physicalRisk: 0,
        transitionRisk: 0,
        liabilityRisk: 0,
        physicalRiskPercent: 0,
        transitionRiskPercent: 0,
        liabilityRiskPercent: 0
    },
    {
        name: "Cambodia",
        code: "KH",
        flagUrl: "flags/kh.png",
        coordinates: [12.565679, 104.990963],
        description: "Cambodia faces significant flooding risks due to the Mekong River system and monsoon patterns. The country experiences substantial physical risks from extreme weather and economic transition challenges.",
        states: ["Phnom Penh", "Siem Reap", "Battambang", "Sihanoukville", "Kampong Cham", "Takeo", "Kandal"],
        physicalRisk: 0,
        transitionRisk: 0,
        liabilityRisk: 0,
        physicalRiskPercent: 0,
        transitionRiskPercent: 0,
        liabilityRiskPercent: 0
    },
    {
        name: "Laos",
        code: "LA",
        flagUrl: "flags/la.png",
        coordinates: [19.855627, 102.495496],
        description: "Laos is a landlocked Southeast Asian country with mountainous terrain. It faces physical risks from flooding and erosion, particularly along the Mekong River, with limited economic resources for climate adaptation.",
        states: ["Vientiane", "Luang Prabang", "Savannakhet", "Pakse", "Thakhek", "Vang Vieng", "Luang Namtha"],
        physicalRisk: 0,
        transitionRisk: 0,
        liabilityRisk: 0,
        physicalRiskPercent: 0,
        transitionRiskPercent: 0,
        liabilityRiskPercent: 0
    },
    {
        name: "Brunei",
        code: "BN",
        flagUrl: "flags/bn.svg",
        coordinates: [4.535277, 114.727669],
        description: "Brunei is a small Southeast Asian nation with significant oil and gas resources. It faces physical risks from flooding and maritime impacts, with transition risks related to fossil fuel dependency.",
        states: ["Bandar Seri Begawan", "Kuala Belait", "Tutong", "Limbang"],
        physicalRisk: 0,
        transitionRisk: 0,
        liabilityRisk: 0,
        physicalRiskPercent: 0,
        transitionRiskPercent: 0,
        liabilityRiskPercent: 0
    },
    {
        name: "Timor-Leste",
        code: "TL",
        flagUrl: "flags/tl.png",
        coordinates: [-8.874217, 125.727539],
        description: "Timor-Leste (East Timor) is a Southeast Asian island nation facing high exposure to tropical cyclones, flooding, and droughts. It experiences significant physical climate risks with limited economic adaptation capacity.",
        states: ["Dili", "Baucau", "Maliana", "Suai", "Oecusse", "Lospalos"],
        physicalRisk: 0,
        transitionRisk: 0,
        liabilityRisk: 0,
        physicalRiskPercent: 0,
        transitionRiskPercent: 0,
        liabilityRiskPercent: 0
    }
];

const physicalRiskData = {
    Thailand: { frequency: 3.6, severity: 10.0, score: 7.4 },
    Philippines: { frequency: 10.0, severity: 5.5, score: 7.3 },
    Vietnam: { frequency: 5.1, severity: 7.2, score: 6.4 },
    Myanmar: { frequency: 2.0, severity: 8.0, score: 5.6 },
    Indonesia: { frequency: 5.9, severity: 4.1, score: 4.8 },
    Cambodia: { frequency: 1.6, severity: 4.5, score: 3.3 },
    Laos: { frequency: 1.7, severity: 3.2, score: 2.6 },
    Malaysia: { frequency: 2.4, severity: 2.5, score: 2.5 },
    'Timor-Leste': { frequency: 1.1, severity: 1.2, score: 1.2 },
    Brunei: { frequency: 1.0, severity: 1.1, score: 1.1 },
    Singapore: { frequency: 1.0, severity: 1.0, score: 1.0 }
};

const countryNameAliases = {
    'Brunei Darussalam': 'Brunei',
    'Lao PDR': 'Laos',
    'Viet Nam': 'Vietnam'
};

function normalizeCountryName(name) {
    return countryNameAliases[name] || name;
}

// Raw frequency dataset (id,country,year,count) provided by user — parsed below
const _physicalFrequencyRaw = `
207,Myanmar,2020,2
208,Myanmar,2021,1
209,Myanmar,2023,5
210,Myanmar,2024,3
211,Myanmar,2025,4
212,Philippines,1905,1
213,Philippines,1912,1
214,Philippines,1931,1
215,Philippines,1932,1
216,Philippines,1934,2
217,Philippines,1936,1
218,Philippines,1937,1
219,Philippines,1938,1
220,Philippines,1940,1
221,Philippines,1946,1
222,Philippines,1949,1
223,Philippines,1951,3
224,Philippines,1952,2
225,Philippines,1955,2
226,Philippines,1956,2
227,Philippines,1957,1
228,Philippines,1959,2
229,Philippines,1960,7
230,Philippines,1962,2
231,Philippines,1963,1
232,Philippines,1964,2
233,Philippines,1965,1
234,Philippines,1966,1
235,Philippines,1967,1
236,Philippines,1968,3
237,Philippines,1969,1
238,Philippines,1970,8
239,Philippines,1971,5
240,Philippines,1972,4
241,Philippines,1973,4
242,Philippines,1974,9
243,Philippines,1975,2
244,Philippines,1976,5
245,Philippines,1977,8
246,Philippines,1978,13
247,Philippines,1979,6
248,Philippines,1980,8
249,Philippines,1981,12
250,Philippines,1982,13
251,Philippines,1983,6
252,Philippines,1984,4
253,Philippines,1985,5
254,Philippines,1986,7
255,Philippines,1987,6
256,Philippines,1988,8
257,Philippines,1989,12
258,Philippines,1990,9
259,Philippines,1991,9
260,Philippines,1992,7
261,Philippines,1993,10
262,Philippines,1994,17
263,Philippines,1995,13
264,Philippines,1996,5
265,Philippines,1997,4
266,Philippines,1998,6
267,Philippines,1999,16
268,Philippines,2000,10
269,Philippines,2001,9
270,Philippines,2002,11
271,Philippines,2003,10
272,Philippines,2004,12
273,Philippines,2005,4
274,Philippines,2006,19
275,Philippines,2007,14
276,Philippines,2008,20
277,Philippines,2009,23
278,Philippines,2010,13
279,Philippines,2011,30
280,Philippines,2012,18
281,Philippines,2013,13
282,Philippines,2014,12
283,Philippines,2015,16
284,Philippines,2016,10
285,Philippines,2017,11
286,Philippines,2018,8
287,Philippines,2019,9
288,Philippines,2020,7
289,Philippines,2021,13
290,Philippines,2022,9
291,Philippines,2023,11
292,Philippines,2024,15
293,Philippines,2025,13
294,Philippines,2026,1
295,Thailand,1962,1
296,Thailand,1966,1
297,Thailand,1975,1
298,Thailand,1978,3
299,Thailand,1980,1
300,Thailand,1981,2
301,Thailand,1983,1
302,Thailand,1984,2
303,Thailand,1985,1
304,Thailand,1986,1
305,Thailand,1987,1
306,Thailand,1988,1
307,Thailand,1989,1
308,Thailand,1990,1
309,Thailand,1991,4
310,Thailand,1992,4
311,Thailand,1993,5
312,Thailand,1994,7
313,Thailand,1995,2
314,Thailand,1996,2
315,Thailand,1997,3
316,Thailand,1999,7
317,Thailand,2000,5
318,Thailand,2001,6
319,Thailand,2002,6
320,Thailand,2003,4
321,Thailand,2004,6
322,Thailand,2005,6
323,Thailand,2006,3
324,Thailand,2007,5
325,Thailand,2008,6
326,Thailand,2009,1
327,Thailand,2010,2
328,Thailand,2011,3
329,Thailand,2012,2
330,Thailand,2013,3
331,Thailand,2014,7
332,Thailand,2015,2
333,Thailand,2016,4
334,Thailand,2017,6
335,Thailand,2019,3
336,Thailand,2020,7
337,Thailand,2021,6
338,Thailand,2022,11
339,Thailand,2023,3
340,Thailand,2024,7
341,Thailand,2025,13
342,Timor-Leste,2001,1
343,Timor-Leste,2003,2
344,Timor-Leste,2006,1
345,Timor-Leste,2007,2
346,Timor-Leste,2016,1
347,Timor-Leste,2020,1
348,Timor-Leste,2021,1
349,Viet Nam,1952,1
350,Viet Nam,1953,1
351,Viet Nam,1956,1
352,Viet Nam,1964,2
353,Viet Nam,1966,1
354,Viet Nam,1970,2
355,Viet Nam,1971,2
356,Viet Nam,1973,1
357,Viet Nam,1977,1
358,Viet Nam,1978,1
359,Viet Nam,1980,3
360,Viet Nam,1982,1
361,Viet Nam,1983,3
362,Viet Nam,1984,3
363,Viet Nam,1985,2
364,Viet Nam,1986,2
365,Viet Nam,1987,3
366,Viet Nam,1988,2
367,Viet Nam,1989,3
368,Viet Nam,1990,4
369,Viet Nam,1991,7
370,Viet Nam,1992,5
371,Viet Nam,1993,4
372,Viet Nam,1994,3
373,Viet Nam,1995,2
374,Viet Nam,1996,6
375,Viet Nam,1997,4
376,Viet Nam,1998,4
377,Viet Nam,1999,5
378,Viet Nam,2000,11
379,Viet Nam,2001,7
380,Viet Nam,2002,6
381,Viet Nam,2003,4
382,Viet Nam,2004,6
383,Viet Nam,2005,10
384,Viet Nam,2006,11
385,Viet Nam,2007,6
386,Viet Nam,2008,10
387,Viet Nam,2009,6
388,Viet Nam,2010,7
389,Viet Nam,2011,5
390,Viet Nam,2012,4
391,Viet Nam,2013,10
392,Viet Nam,2014,3
393,Viet Nam,2015,5
394,Viet Nam,2016,8
395,Viet Nam,2017,9
396,Viet Nam,2018,7
397,Viet Nam,2019,8
398,Viet Nam,2020,11
399,Viet Nam,2021,8
400,Viet Nam,2022,8
401,Viet Nam,2023,6
402,Viet Nam,2024,10
403,Viet Nam,2025,14
404,Viet Nam,2026,1
`;

// Parse raw data into per-country timeseries
const physicalFrequencyData = {};
_physicalFrequencyRaw.trim().split('\n').forEach(line => {
    const parts = line.trim().split(',');
    if (parts.length < 4) return;
    const country = normalizeCountryName(parts[1].trim());
    const year = Number(parts[2]);
    const value = Number(parts[3]);
    if (!physicalFrequencyData[country]) {
        physicalFrequencyData[country] = { years: [], values: [] };
    }
    physicalFrequencyData[country].years.push(year);
    physicalFrequencyData[country].values.push(value);
});
window.physicalFrequencyData = physicalFrequencyData;
window.countryNameAliases = countryNameAliases;

// Load forecast data and populate transition risk scores
let forecastData = {};

fetch('forecast-data.json')
    .then(response => response.json())
    .then(data => {
        forecastData = data;
        window.forecastData = data;  // Make it globally accessible
        
        // Update transition risk scores for each country
        countriesData.forEach(country => {
            const physicalInfo = physicalRiskData[country.name];
            if (physicalInfo) {
                country.physicalFrequency = physicalInfo.frequency;
                country.physicalFrequencyPercent = physicalInfo.frequency * 10;
                country.physicalSeverity = physicalInfo.severity;
                country.physicalSeverityPercent = physicalInfo.severity * 10;
                country.physicalRisk = physicalInfo.score;
                country.physicalRiskPercent = physicalInfo.score * 10;
            }

            // Map country name to transition risk score with fallbacks
            let transitionRiskInfo = (data.transitionRiskScores || {})[country.name];

            // Fallback: use countryCodeMap to find a matching name for this country.code
            if (!transitionRiskInfo && data.countryCodeMap) {
                const entry = Object.values(data.countryCodeMap).find(e => e.code === country.code);
                if (entry && data.transitionRiskScores && data.transitionRiskScores[entry.name]) {
                    transitionRiskInfo = data.transitionRiskScores[entry.name];
                }
            }

            // Fallback: case-insensitive / spacing-insensitive name match
            if (!transitionRiskInfo && data.transitionRiskScores) {
                const normalizedTarget = country.name.toLowerCase().replace(/\s+/g, '');
                const foundKey = Object.keys(data.transitionRiskScores).find(k => k.toLowerCase().replace(/\s+/g, '') === normalizedTarget);
                if (foundKey) transitionRiskInfo = data.transitionRiskScores[foundKey];
            }

            if (transitionRiskInfo) {
                country.transitionRisk = transitionRiskInfo.score;
                country.transitionRiskPercent = transitionRiskInfo.percent;
            } else {
                // Ensure safe defaults so UI doesn't render NaN/-- for arithmetic
                country.transitionRisk = country.transitionRisk || 0;
                country.transitionRiskPercent = country.transitionRiskPercent || 0;
            }

            const averageRisk = (Number(country.physicalRisk || 0) * 0.8) + (Number(country.transitionRisk || 0) * 0.2);
            country.averageRisk = Number(averageRisk.toFixed(1));
            country.averageRiskPercent = country.averageRisk * 10;
            country.liabilityRisk = country.averageRisk;
            country.liabilityRiskPercent = country.averageRiskPercent;
        });

        // If a country is currently selected, refresh the displayed metrics (guard if function exists)
        if (window.currentCountry && typeof window.updateRiskMetrics === 'function') {
            // find matching country object by code/name and update UI
            const matched = countriesData.find(c => c.code === window.currentCountry.code || c.name === window.currentCountry.name);
            if (matched) window.updateRiskMetrics(matched);
        }
        
        console.log('Forecast data loaded successfully');
        console.log('Transition risk scores updated');
    })
    .catch(error => console.error('Error loading forecast data:', error));
