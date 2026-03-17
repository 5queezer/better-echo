import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- torchaudio compat shim for pyannote-audio 3.x + torchaudio 2.10+ ---
# torchaudio 2.10 removed torchaudio.info() and torchaudio.AudioMetaData.
# Patch them back in using soundfile so pyannote can load.
import torchaudio as _ta

if not hasattr(_ta, "AudioMetaData"):
    from dataclasses import dataclass
    from io import IOBase

    import soundfile as _sf

    @dataclass
    class _AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int = 0
        encoding: str = ""

    def _torchaudio_info(file, backend=None):
        if isinstance(file, IOBase):
            info = _sf.info(file)
            file.seek(0)
        else:
            info = _sf.info(file)
        return _AudioMetaData(
            sample_rate=info.samplerate,
            num_frames=info.frames,
            num_channels=info.channels,
        )

    _ta.AudioMetaData = _AudioMetaData
    _ta.info = _torchaudio_info

if not hasattr(_ta, "list_audio_backends"):
    _ta.list_audio_backends = lambda: ["soundfile"]

if not hasattr(_ta, "set_audio_backend"):
    _ta.set_audio_backend = lambda backend: None

import sys
import types

if not hasattr(_ta, "io"):
    _ta_io = types.ModuleType("torchaudio.io")
    _ta_io.StreamReader = None  # unused by whisperlivekit's diart integration
    _ta.io = _ta_io
    sys.modules["torchaudio.io"] = _ta_io

# huggingface_hub dropped use_auth_token in favor of token.
# Wrap the real function so any caller (even via `from X import hf_hub_download`)
# gets the fix, by patching the underlying function object's module reference.
import functools
import huggingface_hub as _hfh
import huggingface_hub.file_download as _hfh_fd

_orig_hf_hub_download = _hfh_fd.hf_hub_download

@functools.wraps(_orig_hf_hub_download)
def _patched_hf_hub_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _orig_hf_hub_download(*args, **kwargs)

_hfh_fd.hf_hub_download = _patched_hf_hub_download
_hfh.hf_hub_download = _patched_hf_hub_download

# PyTorch 2.6+ defaults to weights_only=True in torch.load, which rejects
# pyannote checkpoints that contain custom classes. Patch lightning's loader
# to use weights_only=False for local files (trusted HF-downloaded models).
import torch
import lightning_fabric.utilities.cloud_io as _lio

_orig_pl_load = _lio._load

@functools.wraps(_orig_pl_load)
def _patched_pl_load(path_or_url, map_location=None, weights_only=None):
    return _orig_pl_load(path_or_url, map_location=map_location, weights_only=False)

_lio._load = _patched_pl_load
# --- end compat shim ---

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from whisperlivekit import AudioProcessor, TranscriptionEngine, get_inline_ui_html, parse_args

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
TERMS_FILE = Path(__file__).parent / "terms.txt"

config = parse_args()
transcription_engine = None
http_client = None


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
    global transcription_engine, http_client
    transcription_engine = TranscriptionEngine(config=config)
    http_client = httpx.AsyncClient()
    logger.info("LLM correction via Ollama model=%s at %s", OLLAMA_MODEL, OLLAMA_URL)
    if TERMS:
        logger.info("Loaded %d domain terms from %s", TERMS.count("\n") + 1, TERMS_FILE)
    else:
        logger.info("No domain terms loaded (edit %s to add terms)", TERMS_FILE)
    yield
    await http_client.aclose()


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
    global transcription_engine, http_client

    session_language = websocket.query_params.get("language", None)
    mode = websocket.query_params.get("mode", "full")

    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
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
    corrected_gen = corrected_results(results_gen, http_client)

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
