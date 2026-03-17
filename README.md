# better-echo

Real-time speech-to-text with LLM-powered transcription correction. Wraps [whisperlivekit](https://github.com/QuentinFuworworklP/whisperlivekit) and sends finalized lines through a local Ollama model to fix domain-specific terminology.

## Setup

```bash
# Python 3.13, uses uv
uv sync

# Copy and fill in environment variables
cp .env.example .env
```

### Platform notes

- **Linux with NVIDIA GPU** — `uv sync` installs CUDA 12.9 binaries automatically. Make sure you have NVIDIA drivers installed.
- **macOS (Apple Silicon / M1+)** — `uv sync` installs CPU-only PyTorch. The M1/M2/M3 GPU is used automatically via PyTorch's MPS backend when available (`PYTORCH_ENABLE_MPS_FALLBACK=1` is set by the app). Expect slower inference than a dedicated NVIDIA GPU — use a smaller model size (e.g. `--model-size base` or `small`) if real-time performance is needed.
- **macOS (Intel)** — works the same as Apple Silicon but without MPS GPU acceleration (CPU only).

### Ollama (for transcription correction)

```bash
ollama pull llama3.2
ollama serve
```

### Speaker diarization (optional)

Requires a HuggingFace token with access to gated pyannote models.

1. Set `HF_TOKEN` in `.env`
2. Accept the model licenses:
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/embedding

## Usage

```bash
# Basic
uv run python main.py

# With speaker diarization
uv run python main.py --diarization --diarization-backend diart

# Common options (passed through to whisperlivekit)
uv run python main.py --model-size large-v3 --language de
```

Open the printed URL in your browser, allow microphone access, and start speaking.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace token (required for diarization) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Model used for transcription correction |
| `TRANSCRIPT_FORMAT` | `none` | Save transcripts: `text`, `json`, `both`, or `none` |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, etc.) |

## Transcript saving

Set `TRANSCRIPT_FORMAT` to continuously save transcripts to the working directory. Each session creates timestamped files (e.g. `transcript_2026-03-17_01-54-30`).

- **`text`** — human-readable with timestamps and speaker labels:
  ```
  [0:01:23.45 - 0:01:25.67] Speaker 1: Hello, how are you?
  ```
- **`json`** — JSONL with both raw Whisper output and corrected text:
  ```json
  {"start": 83.45, "end": 85.67, "speaker": "1", "raw": "hello how are you", "corrected": "Hello, how are you?"}
  ```
- **`both`** — saves `.txt` and `.jsonl` side by side

## Privacy

No telemetry, analytics, or tracking. All audio and text processing happens locally. The only network call is to your own Ollama instance for transcription correction.

## Domain vocabulary

Edit `terms.txt` to add domain-specific terms the LLM should know about:

```
Kubernetes
Celery: Python distributed task queue
```
