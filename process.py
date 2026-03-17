"""CLI tool to reprocess raw transcripts with a different LLM model."""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from InquirerPy import inquirer

load_dotenv()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"
TERMS_FILE = Path(__file__).parent / "terms.txt"


def load_terms() -> str:
    if not TERMS_FILE.exists():
        return ""
    entries = []
    for line in TERMS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(f"- {line}")
    return "\n".join(entries)


def build_system_prompt() -> str:
    prompt = """\
You are a transcription corrector. Fix speech-to-text errors in the provided lines.

Rules:
- Fix misheard words, especially domain-specific terms listed below
- Fix punctuation and capitalization
- Do NOT rephrase, summarize, or add content
- Output exactly the same number of lines as input
- Output ONLY the corrected lines, nothing else"""

    terms = load_terms()
    if terms:
        prompt += f"""

Domain-specific terms (use these exact spellings and meanings):
{terms}"""
    return prompt


def find_raw_transcripts() -> list[Path]:
    if not TRANSCRIPTS_DIR.exists():
        return []
    files = sorted(TRANSCRIPTS_DIR.glob("raw_*.jsonl"), reverse=True)
    return files


def preview_transcript(path: Path, max_lines: int = 3) -> str:
    """Return a short preview of a transcript file."""
    lines = []
    for line in path.read_text().splitlines()[:max_lines]:
        entry = json.loads(line)
        text = entry.get("text", "")
        if text:
            lines.append(text[:60])
    return " | ".join(lines) if lines else "(empty)"


def format_transcript_choice(path: Path) -> str:
    """Format a transcript file as a menu choice."""
    stat = path.stat()
    line_count = sum(1 for _ in path.read_text().splitlines())
    size_kb = stat.st_size / 1024
    preview = preview_transcript(path)
    return f"{path.name}  ({line_count} lines, {size_kb:.1f}KB) — {preview}"


async def fetch_models() -> list[str]:
    """Fetch available models from Ollama."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"Warning: Could not fetch Ollama models: {e}")
        return []


async def correct_lines(client: httpx.AsyncClient, model: str, system_prompt: str, texts: list[str]) -> list[str]:
    """Send lines to Ollama for correction."""
    if not texts:
        return texts

    user_content = "\n".join(texts)
    try:
        resp = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        result = resp.json()["message"]["content"].strip()
        corrected = result.splitlines()

        if len(corrected) != len(texts):
            print(f"  Warning: model returned {len(corrected)} lines, expected {len(texts)}. Using originals.")
            return texts
        return corrected
    except Exception as e:
        print(f"  Error: {e}. Using originals.")
        return texts


async def process_transcript(path: Path, model: str, output_format: str) -> Path:
    """Process a raw transcript file with the specified model."""
    system_prompt = build_system_prompt()
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Derive name from original raw file timestamp
    raw_ts = path.stem.replace("raw_", "")

    if output_format == "text":
        out_path = TRANSCRIPTS_DIR / f"corrected_{raw_ts}_{model.replace(':', '-')}_{ts}.txt"
    else:
        out_path = TRANSCRIPTS_DIR / f"corrected_{raw_ts}_{model.replace(':', '-')}_{ts}.jsonl"

    batch_size = 10
    corrected_entries = []

    async with httpx.AsyncClient() as client:
        for i in range(0, len(entries), batch_size):
            batch = entries[i : i + batch_size]
            texts = [e["text"] for e in batch if e.get("text")]
            text_indices = [j for j, e in enumerate(batch) if e.get("text")]

            if texts:
                corrected = await correct_lines(client, model, system_prompt, texts)
                correction_map = dict(zip(text_indices, corrected))
            else:
                correction_map = {}

            for j, entry in enumerate(batch):
                corrected_text = correction_map.get(j, entry.get("text", ""))
                corrected_entries.append({**entry, "corrected": corrected_text})

            done = min(i + batch_size, len(entries))
            print(f"  Processed {done}/{len(entries)} segments", end="\r")

    print()

    with open(out_path, "w") as f:
        if output_format == "text":
            for entry in corrected_entries:
                start = format_time(entry.get("start"))
                end = format_time(entry.get("end"))
                speaker = entry.get("speaker")
                has_speaker = speaker is not None and speaker not in ("-1", "-2")
                prefix = f"Speaker {speaker}: " if has_speaker else ""
                f.write(f"[{start} - {end}] {prefix}{entry['corrected']}\n")
        else:
            for entry in corrected_entries:
                f.write(json.dumps(entry) + "\n")

    return out_path


def format_time(seconds) -> str:
    if seconds is None:
        return "?:??:??.??"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def main():
    # Find raw transcripts
    transcripts = find_raw_transcripts()
    if not transcripts:
        print("No raw transcripts found in transcripts/ directory.")
        print("Run the server first to record a transcript.")
        sys.exit(1)

    # Build choices with previews
    choices = []
    for path in transcripts:
        choices.append({"name": format_transcript_choice(path), "value": path})

    selected_path = inquirer.select(
        message="Select a transcript to process:",
        choices=choices,
        max_height="70%",
    ).execute()

    if selected_path is None:
        return

    # Fetch and select model
    print(f"Fetching models from {OLLAMA_URL}...")
    models = asyncio.run(fetch_models())

    if not models:
        model = inquirer.text(
            message="Enter Ollama model name:",
            default=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        ).execute()
    else:
        default_model = os.environ.get("OLLAMA_MODEL", "llama3.2")
        default_idx = None
        for i, m in enumerate(models):
            if m == default_model:
                default_idx = i
                break

        model = inquirer.select(
            message="Select Ollama model:",
            choices=models,
            default=models[default_idx] if default_idx is not None else None,
            max_height="70%",
        ).execute()

    # Select output format
    output_format = inquirer.select(
        message="Output format:",
        choices=[
            {"name": "Text  — human-readable with timestamps", "value": "text"},
            {"name": "JSONL — structured with raw + corrected text", "value": "json"},
        ],
    ).execute()

    # Process
    print(f"\nProcessing {selected_path.name} with {model}...")
    out_path = asyncio.run(process_transcript(selected_path, model, output_format))
    print(f"Done! Output saved to {out_path}")


if __name__ == "__main__":
    main()
