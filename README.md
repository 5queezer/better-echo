# better-echo

**Real-time speech-to-text that actually gets your words right.** Powered by Whisper and a local LLM, better-echo fixes domain-specific terminology, punctuation, and misheard words — all running on your machine, with zero data leaving your network.

## Why better-echo?

Standard speech-to-text tools stumble on technical jargon — "Kubernetes" becomes "Cooper Netties," "Celery" becomes "salary." better-echo solves this by piping Whisper's output through a local [Ollama](https://ollama.com) model that knows your vocabulary. You define your domain terms once, and every transcript comes back clean.

**Key features:**
- **Local & private** — all processing on your hardware, no cloud APIs, no telemetry
- **Domain-aware correction** — teach it your terminology via a simple `terms.txt` file
- **Speaker diarization** — identify who said what (via [pyannote](https://github.com/pyannote/pyannote-audio) + [diart](https://github.com/juanmc2005/diart))
- **Auto language detection** — detects and tracks language per speaker with voting stabilization
- **Transcript export** — save as human-readable text, structured JSONL, or both
- **Cross-platform** — Linux (NVIDIA CUDA) and macOS (Apple Silicon MPS)

## Quick start

```bash
# 1. Clone and install (Python 3.13, uv)
git clone https://github.com/5queezer/better-echo.git
cd better-echo
uv sync

# 2. Start Ollama with a correction model
ollama pull llama3.2 && ollama serve

# 3. Run
uv run python main.py
```

Open the printed URL in your browser, allow microphone access, and start speaking. That's it.

## Setup details

### Environment

```bash
cp .env.example .env   # then fill in your values
```

### Platform notes

- **Linux (NVIDIA GPU)** — `uv sync` installs CUDA 12.9 binaries automatically. Requires NVIDIA drivers.
- **macOS (Apple Silicon)** — uses PyTorch MPS backend automatically. For real-time performance, use a smaller model (`--model-size base` or `small`).
- **macOS (Intel)** — CPU only, no GPU acceleration.

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

### Makefile targets

```bash
make serve              # Run without diarization
make serve-diart        # Run with diart speaker diarization
make process            # Reprocess a raw transcript with a different model
```

### Reprocessing transcripts

Raw transcripts are always saved as JSONL. You can reprocess them later with a different LLM model:

```bash
uv run python process.py
```

This launches an interactive CLI that lets you pick a raw transcript, choose an Ollama model, and output corrected text or JSON.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace token (required for diarization) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Model used for transcription correction |
| `TRANSCRIPT_FORMAT` | `none` | Save transcripts: `text`, `json`, `both`, or `none` |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, etc.) |
| `ALLOWED_LANGUAGES` | — | Comma-separated language codes to restrict auto-detection (e.g. `en,de,fr`) |

### Domain vocabulary

Edit `terms.txt` to add domain-specific terms the LLM should know about:

```
Kubernetes
Celery: Python distributed task queue
FastAPI
pyannote
```

### Transcript saving

Set `TRANSCRIPT_FORMAT` to continuously save transcripts. Each session creates timestamped files (e.g. `transcript_2026-03-17_01-54-30`).

- **`text`** — human-readable with timestamps and speaker labels:
  ```
  [0:01:23.45 - 0:01:25.67] Speaker 1: Hello, how are you?
  ```
- **`json`** — JSONL with both raw Whisper output and corrected text:
  ```json
  {"start": 83.45, "end": 85.67, "speaker": "1", "raw": "hello how are you", "corrected": "Hello, how are you?"}
  ```
- **`both`** — saves `.txt` and `.jsonl` side by side

## How it works

```
Microphone → Whisper (speech-to-text) → Ollama LLM (correction) → Browser UI
                                    ↗
                          terms.txt (domain vocabulary)
```

better-echo wraps [whisperlivekit](https://github.com/QuentinFuworworklP/whisperlivekit) and streams audio from your browser over a WebSocket. Finalized transcript lines are sent to a local Ollama model that corrects terminology and punctuation using your custom term list. Results stream back to the browser in real time.

## Privacy

No telemetry, analytics, or tracking. All audio and text processing happens locally. The only network call is to your own Ollama instance for transcription correction.

## License

[MIT](LICENSE) — use it however you want.

## Changelog

### 2026-03-17
- Revamp README with hook, feature highlights, quick-start guide, and architecture overview (#11)
- Improved language auto-detection with voting stabilization and per-speaker tracking (#9)
- Updated uv.lock to include all declared dependencies

### 2026-03-16
- Auto-install mlx-whisper on macOS for Apple Silicon performance (#8)
- Add raw transcript storage and reprocessing CLI (#7)
- Fix TypeError in audio processor diarization: coerce string speakers to int (#6)
- Fix DiartDiarization kwarg mismatch in whisperlivekit 0.2.20 (#5)

### 2026-03-15
- Add privacy section to README (#4)
- Add MIT license (#3)
- Add Makefile with serve and serve-diart targets (#2)
- Add macOS (Apple Silicon) compatibility (#1)
- Transcript writer (initial)
