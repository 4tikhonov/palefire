
import json
import os
import argparse
import re

def extract_video_id(url):
    # Regex for standard YouTube URLs
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return url # Return original if no match, might be just the ID

import json
import os
import argparse
import re
import subprocess
import sys

def extract_video_id(url):
    # Regex for standard YouTube URLs
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return url # Return original if no match, might be just the ID

def parse_vtt(vtt_path):
    """Parses a VTT file and returns a list of (timestamp, text) tuples."""
    entries = []
    
    # Regex to match VTT timestamps: 00:00:00.000 or 00:00.000
    timestamp_pattern = re.compile(r'((?:\d{2}:)?\d{2}:\d{2}\.\d{3})\s-->')
    
    try:
        with open(vtt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        current_timestamp = None
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                # End of an entry
                if current_timestamp and current_text:
                    text = " ".join(current_text)
                    # Clean up VTT formatting tags like <c> or <00:00:00.000>
                    text = re.sub(r'<[^>]+>', '', text)
                    if text:
                        entries.append((current_timestamp, text))
                
                current_timestamp = None
                current_text = []
                continue
            
            if line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
                continue
                
            # Check for timestamp line
            ts_match = timestamp_pattern.match(line)
            if ts_match:
                current_timestamp = ts_match.group(1).split('.')[0] # Take just the HH:MM:SS part
                continue
                
            # If we have a timestamp, this must be text content
            if current_timestamp:
                current_text.append(line)
                
        # Handle last entry
        if current_timestamp and current_text:
             text = " ".join(current_text)
             text = re.sub(r'<[^>]+>', '', text)
             if text:
                entries.append((current_timestamp, text))
                
    except Exception as e:
        print(f"Error parsing VTT: {e}")
        
    return entries

def get_transcript(video_url, output_format="text"):
    video_id = extract_video_id(video_url)
    print(f"Video ID: {video_id}")
    
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Construct the command
    # We use the yt-dlp installed in the same environment as this script
    yt_dlp_path = os.path.join(os.path.dirname(sys.executable), 'yt-dlp')
    if not os.path.exists(yt_dlp_path):
        # Fallback to just "yt-dlp" if not found in venv bin
        yt_dlp_path = "yt-dlp"

    # Command: yt-dlp --write-auto-sub --skip-download --sub-lang en --output "transcript_%(id)s" <url>
    cmd = [
        yt_dlp_path,
        '--write-sub',
        '--write-auto-sub',
        '--skip-download',
        '--sub-lang', 'en',
        '--output', f'transcript_{video_id}',
        url
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        
        # Check specifically for the .en.vtt file (yt-dlp standard naming with --output)
        # With --output "transcript_ID", it usually appends .en.vtt or .en.ext
        expected_filename = f"transcript_{video_id}.en.vtt"
        
        if os.path.exists(expected_filename):
            print(f"\nTranscript downloaded to: {expected_filename}")
            
            if output_format == "text":
                entries = parse_vtt(expected_filename)
                
                output_txt_file = f"transcript_{video_id}.txt"
                with open(output_txt_file, "w", encoding='utf-8') as f:
                    for timestamp, text in entries:
                        line = f"[{timestamp}] {text}"
                        print(line)
                        f.write(line + "\n")
                
                print(f"\nFull text transcript saved to: {output_txt_file}")
            
            else:
                # Raw VTT output
                print("\nFirst 10 lines of content:")
                try:
                    with open(expected_filename, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if i < 10:
                                print(line.rstrip())
                except Exception as e:
                    print(f"Could not read file: {e}")
        else:
            print("Transcript file not found after download.")
            # Debug listing
            print("Files in directory:")
            subprocess.run(['ls', '-F'], check=False)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running yt-dlp: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch YouTube transcript.')
    parser.add_argument('url', help='YouTube video URL or ID')
    parser.add_argument('--format', choices=['vtt', 'text'], default='text', help='Output format: vtt (raw) or text (timestamped lines)')
    args = parser.parse_args()
    
    get_transcript(args.url, output_format=args.format)
