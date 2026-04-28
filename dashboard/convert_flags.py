#!/usr/bin/env python3
"""Convert emoji flags to flagcdn URLs in map-data.js"""

flag_mapping = {
    "🇹🇭": "th",  # Thailand
    "🇻🇳": "vn",  # Vietnam
    "🇮🇩": "id",  # Indonesia
    "🇵🇭": "ph",  # Philippines
    "🇲🇾": "my",  # Malaysia
    "🇸🇬": "sg",  # Singapore
    "🇲🇲": "mm",  # Myanmar
    "🇰🇭": "kh",  # Cambodia
    "🇱🇦": "la",  # Laos
    "🇧🇳": "bn",  # Brunei
    "🇹🇱": "tl",  # Timor-Leste
}

file_path = "/workspaces/MASA-HACKATHON/dashboard/map-data.js"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace flag references
for emoji, code in flag_mapping.items():
    old = f'flag: "{emoji}",'
    new = f'flagUrl: "https://flagcdn.com/h80/{code}.svg",'
    content = content.replace(old, new)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Updated map-data.js with national flag URLs")
