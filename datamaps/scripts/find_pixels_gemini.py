import os
import json
from google import genai
from google.genai import types

def find_pixels():
    client = genai.Client()
    
    # We need to upload the file to Gemini to analyze
    print("Uploading image to Gemini...")
    sample_file = client.files.upload(file="report/page_4.png")
    
    prompt = """
You are an expert GIS and image analysis assistant.
Look at the map of Honduras in this image. The image dimensions are exactly 1848 x 884 pixels (Width x Height).
I need the EXACT (X, Y) pixel coordinates for the following three geographic extremes of the Honduras landmass shown in the map:
1. Westernmost tip (Min Longitude)
2. Easternmost tip (Max Longitude)
3. Northernmost tip (Max Latitude)
4. Southernmost tip (Min Latitude)

Please respond ONLY with a JSON object in this exact format, with no markdown formatting or other text:
{
  "westernmost": {"x": 120, "y": 400},
  "easternmost": {"x": 1600, "y": 400},
  "northernmost": {"x": 800, "y": 100},
  "southernmost": {"x": 800, "y": 750}
}
Note: The X coordinate goes from 0 (left) to 1848 (right). The Y coordinate goes from 0 (top) to 884 (bottom).
Estimate the coordinates as accurately as possible based on the visible map.
"""
    print("Asking Gemini for coordinates...")
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[sample_file, prompt]
    )
    
    print("Response from Gemini:")
    print(response.text)
    
    try:
        data = json.loads(response.text.strip('` \njson'))
        with open("gemini_pixels.json", "w") as f:
            json.dump(data, f, indent=2)
        print("Saved to gemini_pixels.json")
    except Exception as e:
        print(f"Failed to parse JSON: {e}")

if __name__ == "__main__":
    find_pixels()
