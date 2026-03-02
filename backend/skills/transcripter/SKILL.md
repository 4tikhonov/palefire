---
name: transcripter
description: Extract English transcripts from YouTube videos when requested.
---

When the user provides a YouTube URL, video ID, or asks you to summarize, extract, or get the transcript for a YouTube video, use this skill.

Follow these steps:
1. **Identify the Video ID**: Extract the YouTube video ID from the user's request (e.g., from `https://www.youtube.com/watch?v=dQw4w9WgXcQ`, the ID is `dQw4w9WgXcQ`).
2. **Fetch Transcript**: Use the provided script `skills/transcripter/scripts/get_transcript.py <VIDEO_ID>` to extract the English text of the video. 
3. **Format and Analyze**: Once you receive the transcript from the script:
   - Provide a brief summary of the video content.
   - Answer any specific questions the user had about the video based on the transcript.
   - If the transcript is very long, provide key highlights or timestamps if the script provides them.
4. **Error Handling**: If the script fails (e.g., no English transcript is available or subtitles are disabled), politely inform the user.
