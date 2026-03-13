# Translator Skill

Specialized in high-fidelity translation between English and other languages, with a strict focus on preserving domain-specific terminology (scientific, technical, data management).

## 1. Features
- **Technical Precision**: Uses LLMs tuned for scientific and data management contexts.
- **Multilingual Support**: Can translate to English, French, Spanish, German, etc.
- **Transcript Integration**: Designed to work with the output of the Transcriber skill.

## 2. Usage Examples

### Translate a transcript to English
```bash
python3 backend/.agent/skills/translator/scripts/translate.py cache/full_transcript_fr.txt --lang en
```
Results will be saved in `cache/full_transcript_en.txt`.

### Translate raw scientific text to French
```bash
python3 backend/.agent/skills/translator/scripts/translate.py "The dataset follows the FDIR principle." --lang fr --domain "Data Science"
```
Results will be saved in `cache/full_transcript_fr.txt`.

## 3. Configuration
- Defaults to `cache/full_transcript_<lang>.txt` naming convention as requested.
- Automatically falls back to local Ollama if `GOOGLE_API_KEY` or `GEMINI_API_KEY` is not provided.
