import json
import os
import re
import glob
import time
from pathlib import Path

try:
    from google import genai
except ImportError:
    genai = None


TRANSLATION_SYSTEM_PROMPT = """
You are a specialist Spanish-to-English translator for meteorological and climate map data from Honduras. Your task is to translate the given Spanish text into English while preserving all numbers, units, percentages, and technical abbreviations exactly as they appear.

Translation rules:
- Translate map titles, legends, and OCR text accurately
- Preserve ALL numbers, percentages (%), units (mm, °C, W, etc.), and codes (e.g., ENSO, CENAOS, COPECO)
- Keep geographic names (departments, cities) as-is
- Keep dates and years as-is
- For OCR noise/garbled text, leave it unchanged or note "[unreadable]"
- "Pronostico de anomalia de lluvia" -> "Rainfall Anomaly Forecast"
- "Pronostico de anomalia de Temperatura" -> "Temperature Anomaly Forecast"
- "PRONOSTICO DE LLUVIA ACUMULADA" -> "ACCUMULATED RAINFALL FORECAST"
- "NORMA CLIMATICA" -> "CLIMATE NORM"
- "ANOMALIA" -> "ANOMALY"
- "ANOS ANALOGOS" -> "ANALOGOUS YEARS"
- "REPUBLICA DE HONDURAS" -> "REPUBLIC OF HONDURAS"

Respond ONLY with a JSON object: {"translation": "<translated text>", "confidence": <0.0-1.0>}
"""

BATCH_TRANSLATION_PROMPT = """
You are a specialist Spanish-to-English translator for meteorological and climate map data from Honduras.
Translate each of the following Spanish texts into English. Preserve all numbers, units, codes, and geographic names.

Return a JSON object where each key maps to its English translation:
{"<key>": {"translation": "...", "confidence": 0.0-1.0}}

Rules:
- "Pronostico de anomalia de lluvia" -> "Rainfall Anomaly Forecast"
- "Pronostico de anomalia de Temperatura" -> "Temperature Anomaly Forecast"
- "PRONOSTICO DE LLUVIA ACUMULADA" -> "ACCUMULATED RAINFALL FORECAST"
- "NORMA CLIMATICA" -> "CLIMATE NORM"
- "ANOMALIA" -> "ANOMALY"
- "ANOS ANALOGOS" -> "ANALOGOUS YEARS"
- "REPUBLICA DE HONDURAS" -> "REPUBLIC OF HONDURAS"
"""


def _load_env_api_key():
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("GEMINI_API_KEY", "")


def _get_client():
    api_key = _load_env_api_key()
    if not api_key or genai is None:
        return None
    return genai.Client(api_key=api_key)


_shared_client = None


def _client():
    global _shared_client
    if _shared_client is None:
        _shared_client = _get_client()
    return _shared_client


def translate_text(text: str, context: str = "", model: str = "gemini-2.5-flash") -> dict:
    client = _client()
    if client is None:
        return {"translation": text, "confidence": 0.0}

    prompt = TRANSLATION_SYSTEM_PROMPT.strip()
    if context:
        prompt += f"\n\nContext: {context}"
    prompt += f"\n\nSpanish text to translate:\n{text}"

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            raw = response.text.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            return json.loads(raw)
        except Exception as e:
            if attempt == 0:
                time.sleep(2)
                continue
            print(f"  Translation error: {e}")
            return {"translation": text, "confidence": 0.0}


def translate_batch(items: dict, context: str = "", model: str = "gemini-2.5-flash",
                    chunk_size: int = 0) -> dict:
    client = _client()
    if client is None:
        return {k: {"translation": v, "confidence": 0.0} for k, v in items.items()}

    if chunk_size and len(items) > chunk_size:
        keys = list(items.keys())
        result = {}
        for i in range(0, len(keys), chunk_size):
            chunk = {k: items[k] for k in keys[i:i + chunk_size]}
            print(f"    Processing chunk {i // chunk_size + 1}/{-(-len(keys) // chunk_size)} ({len(chunk)} items)...")
            result.update(_translate_batch_inner(chunk, context, model))
        return result
    return _translate_batch_inner(items, context, model)


def _translate_batch_inner(items: dict, context: str = "", model: str = "gemini-2.5-flash") -> dict:
    client = _client()
    prompt = BATCH_TRANSLATION_PROMPT.strip()
    if context:
        prompt += f"\n\nContext: {context}"
    prompt += "\n\nSpanish texts to translate:\n"
    for key, text in items.items():
        prompt += f"\n---\nkey: {key}\ntext: {text}"

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        raw = response.text.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        return json.loads(raw)
    except Exception as e:
        print(f"  Batch translation error: {e}. Falling back to individual translations.")
        return {k: translate_text(v, context, model) for k, v in items.items()}


def translate_titles(titles_path: str, output_dir: str) -> str:
    with open(titles_path, "r", encoding="utf-8") as f:
        titles = json.load(f)

    spanish_keys = {k: v for k, v in titles.items() if _looks_spanish(v)}

    all_translations = {}
    for key, title in titles.items():
        if key not in spanish_keys:
            all_translations[key] = {
                "original": title,
                "translation": title,
                "confidence": 1.0,
            }

    if spanish_keys:
        print(f"  Translating {len(spanish_keys)} titles in batch...")
        results = translate_batch(spanish_keys, context="Map titles from Honduran climate report",
                                  chunk_size=20)
        for key, title in spanish_keys.items():
            entry = results.get(key, {})
            all_translations[key] = {
                "original": title,
                "translation": entry.get("translation", title),
                "confidence": entry.get("confidence", 0.0),
            }
            if entry.get("translation") and entry["translation"] != title:
                print(f"    {title} -> {entry['translation']}")

    out_path = os.path.join(output_dir, "titles.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_translations, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {out_path}")
    return out_path


def translate_data_points(data_dir: str, output_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    title_batch = {}
    legend_batch = {}

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "variable_page_title" in data and _looks_spanish(data["variable_page_title"]):
            title_batch[fpath] = data["variable_page_title"]
        if "legend" in data:
            for line in data["legend"]:
                if _looks_spanish(line):
                    if fpath not in legend_batch:
                        legend_batch[fpath] = []
                    legend_batch[fpath].append(line)

    title_results = {}
    if title_batch:
        print(f"  Translating {len(title_batch)} data point titles in batch...")
        title_results = translate_batch(
            {k: v for k, v in title_batch.items()},
            context="Map panel titles from Honduran climate report"
        )

    legend_results = {}
    if legend_batch:
        flat = {}
        for fpath, lines in legend_batch.items():
            for i, line in enumerate(lines):
                flat[f"{fpath}||{i}"] = line
        print(f"  Translating {len(flat)} legend entries in batch...")
        raw_results = translate_batch(
            flat,
            context="Map legend entries from Honduran weather map",
            chunk_size=30,
        )
        for key, result in raw_results.items():
            legend_results[key] = result

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        changed = False
        if fpath in title_batch:
            entry = title_results.get(fpath, {})
            data["variable_page_title_en"] = entry.get("translation", data["variable_page_title"])
            data["variable_page_title_es"] = data["variable_page_title"]
            changed = True

        if "legend" in data:
            translated = []
            for i, line in enumerate(data["legend"]):
                key = f"{fpath}||{i}"
                if key in legend_results and legend_results[key].get("translation"):
                    translated.append(legend_results[key]["translation"])
                else:
                    translated.append(line)
            if translated != data["legend"]:
                data["legend_en"] = translated
                changed = True

        if changed:
            rel = os.path.relpath(fpath, data_dir)
            out_path = os.path.join(output_dir, rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  Wrote {out_path}")


def translate_raw_ocr(raw_dir: str, output_dir: str):
    page_dirs = sorted(glob.glob(os.path.join(raw_dir, "page-*")))
    batch = {}
    for page_dir in page_dirs:
        page_name = os.path.basename(page_dir)
        for fname in ["fullpage.txt", "left.txt", "right.txt", "legend.txt"]:
            src_path = os.path.join(page_dir, fname)
            if not os.path.exists(src_path):
                continue
            with open(src_path, "r", encoding="utf-8") as f:
                text = f.read()
            if _looks_spanish(text):
                batch[f"{page_name}/{fname}"] = (src_path, text)

    if not batch:
        print("  No Spanish OCR text found.")
        return

    print(f"  Translating {len(batch)} OCR files in chunks...")
    flat = {k: v[1] for k, v in batch.items()}
    results = translate_batch(
        flat,
        context="OCR text from Honduran climate report map pages",
        chunk_size=10,
    )

    for key, (src_path, original) in batch.items():
        entry = results.get(key, {})
        translated = entry.get("translation", original)

        page_name, fname = key.split("/", 1)
        out_dir = os.path.join(output_dir, page_name)
        os.makedirs(out_dir, exist_ok=True)

        name, ext = os.path.splitext(fname)
        out_path = os.path.join(out_dir, f"{name}_en{ext}")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(translated)
        print(f"  Wrote {out_path}")

        meta_path = os.path.join(out_dir, f"{name}_en_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "source": src_path,
                "original_length": len(original),
                "translated_length": len(translated),
                "confidence": entry.get("confidence", 0.0),
            }, f, indent=2)


def _looks_spanish(text: str) -> bool:
    if not text or len(text.strip()) < 3:
        return False
    spanish_words = [
        "de", "del", "la", "las", "los", "el", "en", "por", "con", "para",
        "pronostico", "anomalia", "lluvia", "temperatura", "honduras",
        "republica", "norma", "climatica", "acumulada", "agosto",
        "septiembre", "octubre", "noviembre", "julio", "junio",
        "municipios", "alerta", "sequia", "cenaos", "copeco",
        "gobierno", "anos", "analogos", "porcentaje",
    ]
    text_lower = text.lower()
    matches = sum(1 for w in spanish_words if w in text_lower)
    return matches >= 1


def run_translate(titles_path="report/titles.json",
                  data_dir="data_points",
                  raw_dir="raw",
                  output_dir="translations"):
    os.makedirs(output_dir, exist_ok=True)

    api_key = _load_env_api_key()
    if not api_key:
        print("ERROR: No GEMINI_API_KEY found in .env or environment.")
        print("Translation requires a valid Gemini API key.")
        return

    print("=" * 60)
    print("Translating Spanish content to English (The Minority Report style)")
    print("Using Gemini API for context-aware translation")
    print("=" * 60)

    print("\n--- Step 1/3: Map titles ---")
    if os.path.exists(titles_path):
        translate_titles(titles_path, output_dir)
    else:
        print(f"  Not found: {titles_path}")

    print("\n--- Step 2/3: Data point labels and legends ---")
    data_points_out = os.path.join(output_dir, "data_points")
    os.makedirs(data_points_out, exist_ok=True)
    if os.path.exists(data_dir):
        translate_data_points(data_dir, data_points_out)
    else:
        print(f"  Not found: {data_dir}")

    print("\n--- Step 3/3: Raw OCR text ---")
    ocr_out = os.path.join(output_dir, "ocr")
    os.makedirs(ocr_out, exist_ok=True)
    if os.path.exists(raw_dir):
        translate_raw_ocr(raw_dir, ocr_out)
    else:
        print(f"  Not found: {raw_dir}")

    print("\n" + "=" * 60)
    print(f"Translation complete. Output in: {output_dir}")
    print("=" * 60)
