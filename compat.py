"""Compatibility shims for torchaudio 2.10+, huggingface_hub, and PyTorch 2.6+.

Import this module before any pyannote/diart imports to patch missing APIs.
"""

import functools
import sys
import types

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

if not hasattr(_ta, "io"):
    _ta_io = types.ModuleType("torchaudio.io")
    _ta_io.StreamReader = None  # unused by whisperlivekit's diart integration
    _ta.io = _ta_io
    sys.modules["torchaudio.io"] = _ta_io

# --- huggingface_hub compat ---
# huggingface_hub dropped use_auth_token in favor of token.
# Wrap the real function so any caller gets the fix.
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

# --- PyTorch 2.6+ compat ---
# PyTorch 2.6+ defaults to weights_only=True in torch.load, which rejects
# pyannote checkpoints that contain custom classes. Patch lightning's loader
# to use weights_only=False for local files (trusted HF-downloaded models).
import torch  # noqa: F401
import lightning_fabric.utilities.cloud_io as _lio

_orig_pl_load = _lio._load


@functools.wraps(_orig_pl_load)
def _patched_pl_load(path_or_url, map_location=None, weights_only=None):
    return _orig_pl_load(path_or_url, map_location=map_location, weights_only=False)


_lio._load = _patched_pl_load
