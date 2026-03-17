# better-echo

Real-time speech-to-text with LLM-powered transcription correction. Wraps [whisperlivekit](https://github.com/QuentinFuworworklP/whisperlivekit) and sends finalized lines through a local Ollama model to fix domain-specific terminology.

## Setup

```bash
# Python 3.13, uses uv
uv sync

# Copy and fill in environment variables
cp .env.example .env
```

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

## Domain vocabulary

Edit `terms.txt` to add domain-specific terms the LLM should know about:

```
Kubernetes
Celery: Python distributed task queue
```
