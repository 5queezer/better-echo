.PHONY: serve serve-diart process

# Default server (no diarization)
serve:
	uv run python main.py

# Server with diart speaker diarization enabled
serve-diart:
	uv run python main.py --diarization --diarization-backend diart

# Reprocess a raw transcript with a different model
process:
	uv run python process.py
