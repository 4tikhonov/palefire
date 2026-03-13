# Transcriber Skill

Specialized in converting audio and video recordings into high-quality text transcripts using state-of-the-art multimodal models like Gemini.

## 1. Transcription Process
- **Input**: Path to a media file (mp3, mp4, wav, etc.).
- **Models**: Uses `gemini-2.0-flash` by default.
- **Output**: Full transcript stored in `cache/` directory.

## 2. Usage Examples

### Transcribe an English recording
```bash
python3 backend/.agent/skills/transcriber/scripts/transcribe.py data/meeting.mp3 --lang en
```
Results will be saved in `cache/full_transcript_en.txt`.

### Transcribe a French video
```bash
python3 backend/.agent/skills/transcriber/scripts/transcribe.py data/interview.mp4 --lang fr
```
Results will be saved in `cache/full_transcript_fr.txt`.

## 3. Policy and Best Practices
- Always ensure the `cache/` directory exists.
- Prefer specifying the language code to improve accuracy.
- For long recordings, ensure the environment has sufficient timeout settings for API calls.
