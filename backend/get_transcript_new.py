from youtube_transcript_api import YouTubeTranscriptApi
import sys

video_id = sys.argv[1]
try:
    api = YouTubeTranscriptApi()
    transcript_data = api.fetch(video_id, languages=['en'])
    # transcript_data is a list of FetchedTranscriptSnippet objects: {text, start, duration}
    full_text = " ".join([t.text for t in transcript_data])
    print(full_text)
except Exception as e:
    print(f"Failed to retrieve transcript: {e}")
    sys.exit(1)
