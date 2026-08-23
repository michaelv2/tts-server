"""Kokoro-82M TTS backend."""

import logging

import numpy as np

from backends import TTSBackend

logger = logging.getLogger("tts-service")

BUILTIN_VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_liam", "am_michael", "am_onyx",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]


class KokoroBackend(TTSBackend):
    name = "kokoro-82m"
    sample_rate = 24000
    handles_speed = True

    def load(self) -> None:
        from kokoro import KPipeline

        logger.info("Loading Kokoro-82M pipeline...")
        self._pipeline = KPipeline(lang_code="a")
        logger.info("Kokoro ready")

    def generate(self, text: str, voice: str, speed: float = 1.0, **kwargs) -> np.ndarray:
        if voice == "default":
            voice = "af_heart"

        lang_code = kwargs.get("lang_code", "a")
        if lang_code != "a" and not hasattr(self, "_lang_warn_shown"):
            # Recreate pipeline for different language if requested
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code=lang_code)
            self._lang_warn_shown = True

        chunks = []
        for _gs, _ps, audio in self._pipeline(text, voice=voice, speed=speed):
            if audio is not None:
                chunks.append(audio)

        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def generate_stream(self, text: str, voice: str, speed: float = 1.0, **kwargs):
        if voice == "default":
            voice = "af_heart"
        lang_code = kwargs.get("lang_code", "a")
        if lang_code != "a" and not hasattr(self, "_lang_warn_shown"):
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code=lang_code)
            self._lang_warn_shown = True
        for _gs, _ps, audio in self._pipeline(text, voice=voice, speed=speed):
            if audio is not None:
                yield np.asarray(audio)

    def list_voices(self) -> dict[str, str]:
        return {v: v for v in BUILTIN_VOICES}

    def cleanup(self) -> None:
        if hasattr(self, "_pipeline"):
            del self._pipeline
