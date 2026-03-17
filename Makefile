.PHONY: serve serve-diart

# Default server (no diarization)
serve:
	uv run python main.py

# Server with diart speaker diarization enabled
serve-diart:
	uv run python main.py --diarization --diarization-backend diart
