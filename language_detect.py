"""Per-session language detection with stabilization and per-speaker tracking.

Wraps a shared ASR backend so that language is auto-detected from actual
audio content rather than relying on Whisper's often-unreliable per-chunk
detection (which frequently guesses Korean on short segments).

Two modes:
1. Single-language stabilization: votes across initial chunks, locks in once
   confident.
2. Mixed-language with diarization: tracks language per speaker ID, allowing
   each participant to speak a different language.
"""

import logging
import threading
from collections import Counter
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Minimum probability to accept a single detection
CONFIDENCE_THRESHOLD = 0.6
# Number of agreeing votes needed to lock in a language
VOTES_TO_LOCK = 3
# Minimum audio duration (seconds) for a reliable detection
MIN_AUDIO_SECONDS = 1.0


class LanguageStabilizer:
    """Accumulates language votes and returns the most confident detection."""

    def __init__(self):
        self.votes: list[tuple[str, float]] = []
        self.locked_language: Optional[str] = None

    @property
    def is_locked(self) -> bool:
        return self.locked_language is not None

    def add_vote(self, language: str, probability: float) -> Optional[str]:
        """Add a language detection vote and return locked language if ready."""
        if self.locked_language:
            return self.locked_language

        self.votes.append((language, probability))

        # High-confidence single detection
        if probability >= 0.85:
            self.locked_language = language
            logger.info(
                "Language locked to '%s' (high confidence: %.2f)",
                language,
                probability,
            )
            return self.locked_language

        # Majority vote check
        lang_counts = Counter(lang for lang, _ in self.votes)
        top_lang, count = lang_counts.most_common(1)[0]
        if count >= VOTES_TO_LOCK:
            self.locked_language = top_lang
            avg_prob = sum(p for l, p in self.votes if l == top_lang) / count
            logger.info(
                "Language locked to '%s' after %d votes (avg prob: %.2f)",
                top_lang,
                count,
                avg_prob,
            )
            return self.locked_language

        return None

    def get_best_guess(self) -> Optional[str]:
        """Return the best guess so far, even if not locked."""
        if self.locked_language:
            return self.locked_language
        if not self.votes:
            return None
        lang_counts = Counter(lang for lang, _ in self.votes)
        return lang_counts.most_common(1)[0][0]


class LanguageDetectingASRProxy:
    """Wraps an ASR backend to auto-detect and stabilize the transcription language.

    On each transcribe() call where the language is not yet locked, this proxy
    uses the underlying faster-whisper model's detect_language() to vote, then
    sets the language for transcription once confident.

    For mixed-language groups with diarization, create one proxy per speaker
    using the per_speaker_proxy() factory.
    """

    def __init__(self, asr, allowed_languages: Optional[list[str]] = None):
        """
        Args:
            asr: The underlying ASR backend (e.g. FasterWhisperASR).
            allowed_languages: Optional whitelist of expected language codes.
                If set, only these languages are considered during detection.
        """
        object.__setattr__(self, "_asr", asr)
        object.__setattr__(self, "_stabilizer", LanguageStabilizer())
        object.__setattr__(self, "_allowed_languages", allowed_languages)
        # Reuse the session lock from the ASR if available
        if not hasattr(asr, "_session_lock"):
            asr._session_lock = threading.Lock()
        object.__setattr__(self, "_lock", asr._session_lock)

    def __getattr__(self, name):
        return getattr(self._asr, name)

    def _detect_language(self, audio: np.ndarray) -> Optional[tuple[str, float]]:
        """Run language detection on the audio using the underlying model."""
        try:
            model = self._asr.model
            if not hasattr(model, "detect_language"):
                return None

            language, probability, all_probs = model.detect_language(audio)

            if self._allowed_languages:
                # Filter to allowed languages and pick the best
                filtered = [
                    (lang, prob)
                    for lang, prob in all_probs
                    if lang in self._allowed_languages
                ]
                if filtered:
                    language, probability = filtered[0]

            return language, probability
        except Exception as e:
            logger.debug("Language detection failed: %s", e)
            return None

    def transcribe(self, audio, init_prompt=""):
        """Transcribe with auto-detected language stabilization."""
        with self._lock:
            stabilizer = self._stabilizer

            # Detect language if not yet locked and audio is long enough
            if not stabilizer.is_locked:
                duration = len(audio) / 16000
                if duration >= MIN_AUDIO_SECONDS:
                    result = self._detect_language(audio)
                    if result:
                        lang, prob = result
                        locked = stabilizer.add_vote(lang, prob)
                        if locked:
                            self._asr.original_language = locked
                        else:
                            # Use best guess for now
                            guess = stabilizer.get_best_guess()
                            if guess:
                                self._asr.original_language = guess

            saved = self._asr.original_language
            try:
                return self._asr.transcribe(audio, init_prompt=init_prompt)
            finally:
                # Restore if not locked (so next call can detect again)
                if not stabilizer.is_locked:
                    self._asr.original_language = saved


class PerSpeakerLanguageProxy:
    """Tracks language independently per speaker for mixed-language groups.

    Wraps an ASR backend and maintains a separate LanguageStabilizer per
    speaker ID. Each speaker's language is detected and locked independently.
    """

    def __init__(self, asr, allowed_languages: Optional[list[str]] = None):
        object.__setattr__(self, "_asr", asr)
        object.__setattr__(self, "_allowed_languages", allowed_languages)
        object.__setattr__(self, "_speaker_stabilizers", {})
        if not hasattr(asr, "_session_lock"):
            asr._session_lock = threading.Lock()
        object.__setattr__(self, "_lock", asr._session_lock)
        # Global fallback stabilizer for when no speaker is identified
        object.__setattr__(self, "_global_stabilizer", LanguageStabilizer())
        object.__setattr__(self, "_current_speaker", None)

    def __getattr__(self, name):
        return getattr(self._asr, name)

    def set_current_speaker(self, speaker_id):
        """Set the current speaker for the next transcribe call."""
        self._current_speaker = speaker_id

    def _get_stabilizer(self, speaker_id=None) -> LanguageStabilizer:
        if speaker_id is None:
            return self._global_stabilizer
        if speaker_id not in self._speaker_stabilizers:
            self._speaker_stabilizers[speaker_id] = LanguageStabilizer()
        return self._speaker_stabilizers[speaker_id]

    def _detect_language(self, audio: np.ndarray) -> Optional[tuple[str, float]]:
        try:
            model = self._asr.model
            if not hasattr(model, "detect_language"):
                return None
            language, probability, all_probs = model.detect_language(audio)
            if self._allowed_languages:
                filtered = [
                    (lang, prob)
                    for lang, prob in all_probs
                    if lang in self._allowed_languages
                ]
                if filtered:
                    language, probability = filtered[0]
            return language, probability
        except Exception as e:
            logger.debug("Language detection failed: %s", e)
            return None

    def transcribe(self, audio, init_prompt=""):
        with self._lock:
            speaker = self._current_speaker
            stabilizer = self._get_stabilizer(speaker)

            if not stabilizer.is_locked:
                duration = len(audio) / 16000
                if duration >= MIN_AUDIO_SECONDS:
                    result = self._detect_language(audio)
                    if result:
                        lang, prob = result
                        locked = stabilizer.add_vote(lang, prob)
                        if locked:
                            self._asr.original_language = locked
                            logger.info(
                                "Speaker %s language locked to '%s'",
                                speaker,
                                locked,
                            )
                        else:
                            guess = stabilizer.get_best_guess()
                            if guess:
                                self._asr.original_language = guess
            else:
                self._asr.original_language = stabilizer.locked_language

            saved_lang = self._asr.original_language
            try:
                return self._asr.transcribe(audio, init_prompt=init_prompt)
            finally:
                self._asr.original_language = None  # Reset to auto for next speaker

    def get_speaker_languages(self) -> dict:
        """Return a mapping of speaker_id -> detected language."""
        return {
            spk: stab.locked_language or stab.get_best_guess()
            for spk, stab in self._speaker_stabilizers.items()
        }
