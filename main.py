import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import compat  # noqa: E402, F401 — must run before pyannote/diart imports

import httpx  # noqa: E402
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402

from whisperlivekit import AudioProcessor, TranscriptionEngine, get_inline_ui_html, parse_args  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
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


async def corrected_results(results_gen, client: httpx.AsyncClient):
    """Wrap the results async generator to apply LLM correction to finalized lines."""
    seen = 0
    corrections: dict[int, str] = {}

    async for front_data in results_gen:
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
                for idx, text in zip(new_indices, corrected):
                    corrections[idx] = text

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
    corrected_gen = corrected_results(results_gen, app.state.http_client)

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
