import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import compat  # noqa: E402, F401 — must run before pyannote/diart imports

import httpx  # noqa: E402
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402

from whisperlivekit import AudioProcessor, TranscriptionEngine, get_inline_ui_html, parse_args  # noqa: E402

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
TRANSCRIPT_FORMAT = os.environ.get("TRANSCRIPT_FORMAT", "none").lower()
TERMS_FILE = Path(__file__).parent / "terms.txt"

config = parse_args()


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


TERMS = load_terms()

SYSTEM_PROMPT = """\
You are a transcription corrector. Fix speech-to-text errors in the provided lines.

Rules:
- Fix misheard words, especially domain-specific terms listed below
- Fix punctuation and capitalization
- Do NOT rephrase, summarize, or add content
- Output exactly the same number of lines as input
- Output ONLY the corrected lines, nothing else"""

if TERMS:
    SYSTEM_PROMPT += f"""

Domain-specific terms (use these exact spellings and meanings):
{TERMS}"""


async def correct_lines(client: httpx.AsyncClient, texts: list[str]) -> list[str]:
    """Send lines to Ollama for correction."""
    if not texts:
        return texts

    user_content = "\n".join(texts)
    logger.debug("Ollama request: %d lines to correct: %s", len(texts), texts)

    try:
        resp = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        result = resp.json()["message"]["content"].strip()
        corrected = result.splitlines()
        logger.debug("Ollama response: %s", corrected)

        # Safety: if line count doesn't match, return originals
        if len(corrected) != len(texts):
            logger.warning(
                "Ollama returned %d lines, expected %d. Using originals.",
                len(corrected),
                len(texts),
            )
            return texts
        return corrected
    except Exception as e:
        logger.warning("Ollama correction failed: %s. Using originals.", e)
        return texts


class TranscriptWriter:
    """Continuously appends transcript segments to text and/or JSONL files."""

    def __init__(self, fmt: str):
        self.fmt = fmt
        self._text_file = None
        self._json_file = None
        if fmt == "none":
            return

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if fmt in ("text", "both"):
            path = Path(f"transcript_{ts}.txt")
            self._text_file = open(path, "a")
            logger.info("Saving text transcript to %s", path)
        if fmt in ("json", "both"):
            path = Path(f"transcript_{ts}.jsonl")
            self._json_file = open(path, "a")
            logger.info("Saving JSON transcript to %s", path)

    def write(self, segment, raw_text: str, corrected_text: str):
        if self.fmt == "none":
            return

        speaker = segment.speaker
        has_speaker = speaker is not None and str(speaker) not in ("-1", "-2")

        if self._text_file:
            start = self._format_time(segment.start)
            end = self._format_time(segment.end)
            prefix = f"Speaker {speaker}: " if has_speaker else ""
            self._text_file.write(f"[{start} - {end}] {prefix}{corrected_text}\n")
            self._text_file.flush()

        if self._json_file:
            entry = {
                "start": segment.start,
                "end": segment.end,
                "speaker": str(speaker) if has_speaker else None,
                "raw": raw_text,
                "corrected": corrected_text,
            }
            self._json_file.write(json.dumps(entry) + "\n")
            self._json_file.flush()

    def update_buffer(self, text: str):
        """Track the latest non-empty buffer text so it can be flushed on close."""
        if text and text.strip():
            self._pending_buffer = text
            logger.debug("Transcript buffer updated: %r", text[:80])

    def clear_buffer(self):
        """Clear pending buffer after its content has been finalized and written."""
        self._pending_buffer = ""

    def close(self):
        """Flush any remaining buffer text, then close files."""
        logger.debug("TranscriptWriter.close() called, pending_buffer=%r", getattr(self, "_pending_buffer", ""))
        if self.fmt != "none":
            buf = getattr(self, "_pending_buffer", "")
            if buf and buf.strip():
                logger.info("Flushing remaining buffer to transcript: %r", buf.strip()[:80])
                if self._text_file:
                    self._text_file.write(f"{buf.strip()}\n")
                if self._json_file:
                    entry = {"start": None, "end": None, "speaker": None, "raw": buf.strip(), "corrected": None}
                    self._json_file.write(json.dumps(entry) + "\n")
        if self._text_file:
            self._text_file.close()
            self._text_file = None
        if self._json_file:
            self._json_file.close()
            self._json_file = None

    @staticmethod
    def _format_time(seconds):
        if seconds is None:
            return "?:??:??.??"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h}:{m:02d}:{s:05.2f}"


async def corrected_results(results_gen, client: httpx.AsyncClient, writer: TranscriptWriter):
    """Wrap the results async generator to apply LLM correction to finalized lines."""
    seen = 0
    corrections: dict[int, str] = {}

    async for front_data in results_gen:
        writer.update_buffer(front_data.buffer_transcription or "")
        n = len(front_data.lines)

        # Correct newly finalized lines
        if n > seen:
            new_texts = []
            new_indices = []
            for i in range(seen, n):
                line = front_data.lines[i]
                if line.text and not line.is_silence():
                    new_texts.append(line.text)
                    new_indices.append(i)

            if new_texts:
                corrected = await correct_lines(client, new_texts)
                for idx, (raw, cor) in zip(new_indices, zip(new_texts, corrected)):
                    corrections[idx] = cor
                    writer.write(front_data.lines[idx], raw, cor)
                writer.clear_buffer()

            seen = n

        # Apply all corrections to current front_data
        for i, text in corrections.items():
            if i < len(front_data.lines):
                front_data.lines[i].text = text

        yield front_data


# --- FastAPI app ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.transcription_engine = TranscriptionEngine(config=config)
    app.state.http_client = httpx.AsyncClient()
    logger.info("LLM correction via Ollama model=%s at %s", OLLAMA_MODEL, OLLAMA_URL)
    if TERMS:
        logger.info("Loaded %d domain terms from %s", TERMS.count("\n") + 1, TERMS_FILE)
    else:
        logger.info("No domain terms loaded (edit %s to add terms)", TERMS_FILE)
    yield
    await app.state.http_client.aclose()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get():
    return HTMLResponse(get_inline_ui_html())


async def handle_websocket_results(websocket, results_generator, diff_tracker=None):
    try:
        async for response in results_generator:
            if diff_tracker is not None:
                await websocket.send_json(diff_tracker.to_message(response))
            else:
                await websocket.send_json(response.to_dict())
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("Error in results handler: %s", e)


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    session_language = websocket.query_params.get("language", None)
    mode = websocket.query_params.get("mode", "full")

    audio_processor = AudioProcessor(
        transcription_engine=app.state.transcription_engine,
        language=session_language,
    )
    await websocket.accept()

    diff_tracker = None
    if mode == "diff":
        from whisperlivekit.diff_protocol import DiffTracker

        diff_tracker = DiffTracker()

    await websocket.send_json(
        {"type": "config", "useAudioWorklet": bool(config.pcm_input), "mode": mode}
    )

    results_gen = await audio_processor.create_tasks()
    writer = TranscriptWriter(TRANSCRIPT_FORMAT)
    corrected_gen = corrected_results(results_gen, app.state.http_client, writer)

    websocket_task = asyncio.create_task(
        handle_websocket_results(websocket, corrected_gen, diff_tracker)
    )

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except (KeyError, WebSocketDisconnect):
        pass
    except Exception as e:
        logger.error("Error in websocket main loop: %s", e, exc_info=True)
    finally:
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except (asyncio.CancelledError, Exception):
            pass
        writer.close()
        await audio_processor.cleanup()


def main():
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        log_level="info",
        lifespan="on",
    )


if __name__ == "__main__":
    main()
