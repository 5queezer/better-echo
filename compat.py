"""Compatibility shims for torchaudio 2.10+, huggingface_hub, PyTorch 2.6+,
whisperlivekit DiartDiarization parameter naming, and diarization speaker type.

Import this module before any pyannote/diart imports to patch missing APIs.
Also enables MPS (Metal) fallback on macOS for Apple Silicon GPU acceleration.
"""

import functools
import os
import sys
import types

# --- macOS MPS (Metal Performance Shaders) support ---
# Allow PyTorch to fall back to CPU for ops not yet implemented on MPS.
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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

# --- whisperlivekit DiartDiarization kwarg compat ---
# whisperlivekit core.py passes segmentation_model= and embedding_model= but
# DiartDiarization.__init__ expects segmentation_model_name= and
# embedding_model_name=. Patch _do_init to apply the fix lazily (avoids
# eagerly importing the heavy diart module).
import whisperlivekit.core as _wlk_core

_orig_do_init = _wlk_core.TranscriptionEngine._do_init


@functools.wraps(_orig_do_init)
def _patched_do_init(self, config=None, **kwargs):
    if config is not None and getattr(config, "diarization", False):
        if getattr(config, "diarization_backend", None) == "diart":
            from whisperlivekit.diarization import diart_backend as _db

            _real_init = _db.DiartDiarization.__init__
            if not getattr(_real_init, "_kwarg_patched", False):

                @functools.wraps(_real_init)
                def _fixed_init(*a, **kw):
                    if "segmentation_model" in kw:
                        kw["segmentation_model_name"] = kw.pop("segmentation_model")
                    if "embedding_model" in kw:
                        kw["embedding_model_name"] = kw.pop("embedding_model")
                    return _real_init(*a, **kw)

                _fixed_init._kwarg_patched = True
                _db.DiartDiarization.__init__ = _fixed_init

    return _orig_do_init(self, config, **kwargs)


_wlk_core.TranscriptionEngine._do_init = _patched_do_init

# --- whisperlivekit tokens_alignment speaker type fix ---
# get_lines_diarization() does `diarization_segment.speaker + 1` but diart
# returns string speakers like "SPEAKER_00".  Patch the method to coerce
# the speaker value to int via extract_number before the addition.
import re as _re
import whisperlivekit.tokens_alignment as _wlk_ta

_orig_get_lines_diarization = _wlk_ta.TokensAlignment.get_lines_diarization


def _patched_get_lines_diarization(self):
    # Coerce string speakers to ints on the diarization segments before
    # the original method tries arithmetic on them.
    for seg in self.all_diarization_segments:
        if isinstance(seg.speaker, str):
            m = _re.search(r'\d+', seg.speaker)
            seg.speaker = int(m.group()) if m else 0
    return _orig_get_lines_diarization(self)


_wlk_ta.TokensAlignment.get_lines_diarization = _patched_get_lines_diarization
