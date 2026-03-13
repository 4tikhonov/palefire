import os
import sys
import argparse
import time
from google import genai
from google.genai import types
from pathlib import Path

def transcribe_media(file_path, language=None, output_path=None, model_name="gemini-2.0-flash"):
    """
    Transcribes media file using Gemini API (via google-genai SDK).
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in environment.")
        return False
    
    client = genai.Client(api_key=api_key)
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return False
    
    try:
        print(f"Uploading {file_path} to Gemini...")
        # Upload the file
        media_file = client.files.upload(path=file_path)
        
        # Wait for processing
        while media_file.state == "PROCESSING":
            print("Processing file...")
            time.sleep(2)
            media_file = client.files.get(name=media_file.name)
            
        if media_file.state == "FAILED":
            print("File processing failed.")
            return False

        print(f"Transcribing {file_path} with {model_name}...")
        
        prompt = "Please provide a full, verbatim transcript of this recording."
        if language:
            prompt += f" The language is {language}."
            
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(file_uri=media_file.uri, mime_type=media_file.mime_type),
                        types.Part.from_text(text=prompt)
                    ]
                )
            ]
        )
        
        text_content = response.text
        
        if not output_path:
            lang_suffix = language if language else "en"
            os.makedirs("cache", exist_ok=True)
            output_path = f"cache/full_transcript_{lang_suffix}.txt"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        
        # Clean up the file from Gemini
        client.files.delete(name=media_file.name)
        
        print(f"Transcript saved to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Transcribe media files using Gemini.")
    parser.add_argument("file", help="Path to audio or video file")
    parser.add_argument("--lang", help="ISO 639-1 language code (e.g., en, fr)")
    parser.add_argument("--output", help="Custom output path")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model name")
    
    args = parser.parse_args()
    
    transcribe_media(args.file, args.lang, args.output, args.model)

if __name__ == "__main__":
    main()
