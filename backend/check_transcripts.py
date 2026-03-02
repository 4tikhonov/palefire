from youtube_transcript_api import YouTubeTranscriptApi
import sys

video_id = sys.argv[1]
try:
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    print(f"Available transcripts for {video_id}:")
    for transcript in transcript_list:
        print(f"- {transcript.language} ({transcript.language_code})")
except Exception as e:
    print(f"Error checking transcripts: {e}")
