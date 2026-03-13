import os
import sys
import argparse
import json
import requests
from google import genai

def translate_text(text, target_lang, domain="scientific and data management", model_name="gemini-2.0-flash"):
    """
    Translates text using Gemini (via google-genai SDK) with domain-specific focus.
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    
    prompt = f"""You are a professional translator specialized in {domain}.
Translate the following text to {target_lang}.
Ensure technical terminology is preserved or translated accurately according to established {domain} standards.
Avoid literal translations of idioms; focus on semantics.

Text to translate:
{text}
"""

    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini translation error, falling back to Ollama: {str(e)}")

    # Fallback to Ollama
    try:
        url = f"{ollama_host}/api/generate"
        payload = {
            "model": "gpt-oss:latest",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3}
        }
        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"Ollama translation error: {str(e)}")
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Translate text or files using Gemini.")
    parser.add_argument("source", help="Path to text file or raw text")
    parser.add_argument("--lang", required=True, help="Target language code (e.g., en, fr)")
    parser.add_argument("--domain", default="scientific and data management", help="Domain context")
    parser.add_argument("--output", help="Custom output path")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model name")
    
    args = parser.parse_args()
    
    text = ""
    if os.path.exists(args.source):
        # Basic check for media files to guide the user
        media_exts = {'.mp3', '.mp4', '.wav', '.m4a', '.avi', '.mov'}
        if os.path.splitext(args.source)[1].lower() in media_exts:
            print(f"Error: '{args.source}' appears to be a media file.")
            print("Please use the Transcriber skill first to generate a text transcript:")
            print(f"python3 backend/.agent/skills/transcriber/scripts/transcribe.py {args.source}")
            sys.exit(1)
            
        try:
            with open(args.source, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            print(f"Error: Could not read '{args.source}' as a UTF-8 text file.")
            print("If this is an audio/video file, use the Transcriber skill first.")
            sys.exit(1)
    else:
        text = args.source
    
    translated = translate_text(text, args.lang, args.domain, args.model)
    
    if translated:
        output_path = args.output
        if not output_path:
            os.makedirs("cache", exist_ok=True)
            output_path = f"cache/full_transcript_{args.lang}.txt"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(translated)
        
        print(f"Translation saved to {output_path}")
    else:
        print("Translation failed.")

if __name__ == "__main__":
    main()
