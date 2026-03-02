#!/usr/bin/env python3
import sys

try:
    from pytubefix import YouTube
except ImportError:
    print("Error: pytubefix is not installed. Please run `python3 -m pip install pytubefix`")
    sys.exit(1)

def get_transcript(video_id):
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        yt = YouTube(url)
        
        caption = None
        # Try to find standard English captions, then auto-generated English
        if 'en' in yt.captions:
            caption = yt.captions['en']
        elif 'en-US' in yt.captions:
            caption = yt.captions['en-US']
        elif 'en-GB' in yt.captions:
            caption = yt.captions['en-GB']
        elif 'a.en' in yt.captions:
            caption = yt.captions['a.en']
        else:
            print("Error: No English transcript found for this video.")
            sys.exit(1)
            
        full_text = caption.generate_txt_captions()
        print(full_text)
        
    except Exception as e:
        print(f"Failed to retrieve transcript: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 get_transcript.py <VIDEO_ID>")
        sys.exit(1)
        
    v_id = sys.argv[1]
    get_transcript(v_id)
