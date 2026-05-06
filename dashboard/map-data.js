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
    "Lao People's Democratic Republic": 'Laos',
    'Viet Nam': 'Vietnam'
};

function normalizeCountryName(name) {
    return countryNameAliases[name] || name;
}

// Raw frequency dataset (id,country,year,count) provided by user — parsed below
const _physicalFrequencyRaw = `
167,Myanmar,1902,1
168,Myanmar,1923,1
169,Myanmar,1926,1
170,Myanmar,1936,1
171,Myanmar,1963,1
172,Myanmar,1965,2
173,Myanmar,1967,2
174,Myanmar,1968,1
175,Myanmar,1970,1
176,Myanmar,1974,1
177,Myanmar,1975,1
178,Myanmar,1976,1
179,Myanmar,1977,1
180,Myanmar,1978,1
181,Myanmar,1979,2
182,Myanmar,1981,1
183,Myanmar,1982,1
184,Myanmar,1991,1
185,Myanmar,1992,1
186,Myanmar,1994,1
187,Myanmar,1995,1
188,Myanmar,1997,1
189,Myanmar,1999,1
190,Myanmar,2001,1
191,Myanmar,2002,1
192,Myanmar,2004,1
193,Myanmar,2005,1
194,Myanmar,2006,2
195,Myanmar,2007,4
196,Myanmar,2008,1
197,Myanmar,2009,1
198,Myanmar,2010,3
199,Myanmar,2011,1
200,Myanmar,2012,1
201,Myanmar,2013,2
202,Myanmar,2014,2
203,Myanmar,2015,6
204,Myanmar,2016,5
205,Myanmar,2017,2
206,Myanmar,2018,4
207,Myanmar,2019,3
208,Myanmar,2020,2
209,Myanmar,2021,1
210,Myanmar,2023,5
211,Myanmar,2024,3
212,Myanmar,2025,4
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
405,Cambodia,1987,1
406,Cambodia,1991,1
407,Cambodia,1994,2
408,Cambodia,1996,1
409,Cambodia,1997,1
410,Cambodia,1999,2
411,Cambodia,2000,1
412,Cambodia,2001,2
413,Cambodia,2002,2
414,Cambodia,2004,1
415,Cambodia,2005,2
416,Cambodia,2006,2
417,Cambodia,2007,1
418,Cambodia,2009,2
419,Cambodia,2010,1
420,Cambodia,2011,1
421,Cambodia,2012,1
422,Cambodia,2013,1
423,Cambodia,2014,1
424,Cambodia,2015,2
425,Cambodia,2016,1
426,Cambodia,2018,1
427,Cambodia,2019,1
428,Cambodia,2020,2
429,Cambodia,2021,4
430,Cambodia,2022,2
431,Cambodia,2023,1
432,Cambodia,2024,1
433,Cambodia,2025,1
434,Lao People's Democratic Republic,1966,1
435,Lao People's Democratic Republic,1968,1
436,Lao People's Democratic Republic,1969,1
437,Lao People's Democratic Republic,1971,1
438,Lao People's Democratic Republic,1977,1
439,Lao People's Democratic Republic,1978,1
440,Lao People's Democratic Republic,1981,1
441,Lao People's Democratic Republic,1984,1
442,Lao People's Democratic Republic,1987,1
443,Lao People's Democratic Republic,1988,1
444,Lao People's Democratic Republic,1991,2
445,Lao People's Democratic Republic,1992,2
446,Lao People's Democratic Republic,1993,1
447,Lao People's Democratic Republic,1994,1
448,Lao People's Democratic Republic,1995,3
449,Lao People's Democratic Republic,1996,1
450,Lao People's Democratic Republic,1999,1
451,Lao People's Democratic Republic,2000,1
452,Lao People's Democratic Republic,2001,1
453,Lao People's Democratic Republic,2002,1
454,Lao People's Democratic Republic,2008,1
455,Lao People's Democratic Republic,2009,2
456,Lao People's Democratic Republic,2011,2
457,Lao People's Democratic Republic,2013,2
458,Lao People's Democratic Republic,2014,1
459,Lao People's Democratic Republic,2015,2
460,Lao People's Democratic Republic,2016,1
461,Lao People's Democratic Republic,2017,1
462,Lao People's Democratic Republic,2018,3
463,Lao People's Democratic Republic,2019,2
464,Lao People's Democratic Republic,2020,2
465,Lao People's Democratic Republic,2021,1
466,Lao People's Democratic Republic,2022,1
467,Lao People's Democratic Republic,2023,1
468,Lao People's Democratic Republic,2024,4
469,Lao People's Democratic Republic,2025,2
470,Malaysia,1965,1
471,Malaysia,1967,1
472,Malaysia,1968,1
473,Malaysia,1970,1
474,Malaysia,1978,1
475,Malaysia,1983,1
476,Malaysia,1986,1
477,Malaysia,1987,1
478,Malaysia,1988,1
479,Malaysia,1993,1
480,Malaysia,1995,2
481,Malaysia,1996,3
482,Malaysia,1997,2
483,Malaysia,1998,3
484,Malaysia,1999,1
485,Malaysia,2000,2
486,Malaysia,2001,4
487,Malaysia,2002,2
488,Malaysia,2003,3
489,Malaysia,2004,5
490,Malaysia,2005,3
491,Malaysia,2006,4
492,Malaysia,2007,2
493,Malaysia,2008,2
494,Malaysia,2009,2
495,Malaysia,2011,2
496,Malaysia,2013,1
497,Malaysia,2014,2
498,Malaysia,2015,1
499,Malaysia,2016,4
500,Malaysia,2017,4
501,Malaysia,2018,2
502,Malaysia,2019,4
503,Malaysia,2020,5
504,Malaysia,2021,8
505,Malaysia,2022,6
506,Malaysia,2023,4
507,Malaysia,2025,7
508,Indonesia,1953,1
509,Indonesia,1955,1
510,Indonesia,1956,1
511,Indonesia,1966,2
512,Indonesia,1967,4
513,Indonesia,1968,1
514,Indonesia,1970,1
515,Indonesia,1972,1
516,Indonesia,1973,1
517,Indonesia,1974,1
518,Indonesia,1976,2
519,Indonesia,1977,4
520,Indonesia,1978,5
521,Indonesia,1979,4
522,Indonesia,1980,3
523,Indonesia,1981,4
524,Indonesia,1982,7
525,Indonesia,1983,4
526,Indonesia,1984,4
527,Indonesia,1985,4
528,Indonesia,1986,5
529,Indonesia,1987,6
530,Indonesia,1988,3
531,Indonesia,1989,3
532,Indonesia,1990,2
533,Indonesia,1991,4
534,Indonesia,1992,2
535,Indonesia,1993,2
536,Indonesia,1994,7
537,Indonesia,1995,5
538,Indonesia,1996,6
539,Indonesia,1997,2
540,Indonesia,1998,2
541,Indonesia,1999,4
542,Indonesia,2000,8
543,Indonesia,2001,7
544,Indonesia,2002,9
545,Indonesia,2003,13
546,Indonesia,2004,7
547,Indonesia,2005,6
548,Indonesia,2006,12
549,Indonesia,2007,10
550,Indonesia,2008,13
551,Indonesia,2009,7
552,Indonesia,2010,8
553,Indonesia,2011,8
554,Indonesia,2012,11
555,Indonesia,2013,12
556,Indonesia,2014,8
557,Indonesia,2015,11
558,Indonesia,2016,12
559,Indonesia,2017,10
560,Indonesia,2018,8
561,Indonesia,2019,14
562,Indonesia,2020,26
563,Indonesia,2021,21
564,Indonesia,2022,13
565,Indonesia,2023,10
566,Indonesia,2024,14
567,Indonesia,2025,15
568,Indonesia,2026,5
569,Brunei Darussalam,1998,1
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
        physicalFrequencyData[country] = { points: {} };
    }
    physicalFrequencyData[country].points[year] = value;
});

Object.keys(physicalFrequencyData).forEach(country => {
    const years = Object.keys(physicalFrequencyData[country].points).map(Number).sort((a, b) => a - b);
    if (years.length === 0) {
        physicalFrequencyData[country] = { years: [], values: [] };
        return;
    }

    const startYear = years[0];
    const endYear = years[years.length - 1];
    const denseYears = [];
    const denseValues = [];

    for (let year = startYear; year <= endYear; year += 1) {
        denseYears.push(year);
        denseValues.push(physicalFrequencyData[country].points[year] ?? 0);
    }

    physicalFrequencyData[country] = { years: denseYears, values: denseValues };
});

if (!physicalFrequencyData.Singapore) {
    const startYear = 1953;
    const endYear = 2026;
    physicalFrequencyData.Singapore = {
        years: Array.from({ length: endYear - startYear + 1 }, (_, i) => startYear + i),
        values: Array.from({ length: endYear - startYear + 1 }, () => 0)
    };
}

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
