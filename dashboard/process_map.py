#!/usr/bin/env python3
"""
Process Southeast Asia map and add water effect to empty areas
"""

from PIL import Image, ImageDraw, ImageFilter
import os

# Define the paths
avif_path = '/workspaces/MASA-HACKATHON/southeast-asia-country-map-map-southeast-asia-grey-color_1091279-2521.avif'
output_path = '/workspaces/MASA-HACKATHON/dashboard/custom-map.png'

try:
    # Try to open AVIF file
    print("Opening AVIF map file...")
    img = Image.open(avif_path)
    
    # Convert to RGBA for transparency
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Get image dimensions
    width, height = img.size
    print(f"Original image size: {width}x{height}")
    
    # Create a new image with water background
    water_map = Image.new('RGBA', (width, height), (70, 150, 220, 255))  # Ocean blue
    
    # Process: Replace white/grey areas with water blue gradient
    pixels = img.load()
    water_pixels = water_map.load()
    
    land_color = (100, 180, 100)  # Green for land
    water_blue = (70, 150, 220)   # Ocean blue
    
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            
            # If pixel is grey or white (likely background/sea)
            if (r > 200 and g > 200 and b > 200):
                # Keep as water blue
                water_pixels[x, y] = (*water_blue, 255)
            elif (abs(r - g) < 20 and abs(g - b) < 20 and 150 < r < 200):
                # Keep as water blue
                water_pixels[x, y] = (*water_blue, 255)
            else:
                # It's land - keep original or enhance
                if (100 < r < 200 and 100 < g < 200 and 100 < b < 150):
                    # Make land slightly darker/greener
                    water_pixels[x, y] = (int(r * 0.8), int(g * 1.1), int(b * 0.7), 255)
                else:
                    water_pixels[x, y] = (r, g, b, a)
    
    # Apply subtle blur for anti-aliasing
    water_map = water_map.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Resize to web-friendly size
    max_size = 2048
    if width > max_size or height > max_size:
        water_map.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        print(f"Resized to: {water_map.size}")
    
    # Save the processed image
    water_map.save(output_path, 'PNG', quality=95)
    print(f"✓ Custom map with water saved to: {output_path}")
    print(f"✓ Final size: {water_map.size}")
    print(f"✓ Water color: RGB{water_blue}")
    print(f"✓ Land color: RGB{land_color}")
    
except FileNotFoundError:
    print(f"Error: File not found at {avif_path}")
except Exception as e:
    print(f"Error processing image: {e}")
    print("\nTrying alternative method...")
    
    # If AVIF fails, try using the download.png file
    png_path = '/workspaces/MASA-HACKATHON/download.png'
    if os.path.exists(png_path):
        print("Found PNG file, using that instead...")
        img = Image.open(png_path)
        img.save(output_path, 'PNG')
        print(f"✓ Map saved to: {output_path}")
