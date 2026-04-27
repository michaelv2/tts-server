"""Chatterbox Turbo TTS backend."""

import logging
import os
from pathlib import Path

import numpy as np
import torch

from backends import TTSBackend

logger = logging.getLogger("tts-service")

VOICES_DIR = Path(os.environ.get("VOICES_DIR", Path(__file__).resolve().parent.parent / "voices"))
CHATTERBOX_VOICES_DIR = VOICES_DIR / "chatterbox"
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "default")


class ChatterboxBackend(TTSBackend):
    name = "chatterbox-turbo"

    def load(self) -> None:
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        logger.info("Loading Chatterbox Turbo...")
        self._model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        self.sample_rate = self._model.sr
        logger.info("Chatterbox ready (sr=%d)", self.sample_rate)

    def generate(self, text: str, voice: str, speed: float = 1.0, **kwargs) -> np.ndarray:
        voice_path = self._find_voice(voice)
        exaggeration = kwargs.get("exaggeration", 0.5)

        wav = self._model.generate(
            text,
            audio_prompt_path=str(voice_path),
            exaggeration=exaggeration,
        )
        return wav.squeeze(0).cpu().numpy()

    def list_voices(self) -> dict[str, str]:
        if not CHATTERBOX_VOICES_DIR.exists():
            return {}
        return {
            p.stem: f"{p.stem} (reference: {p.name})"
            for p in sorted(CHATTERBOX_VOICES_DIR.glob("*.wav"))
        }

    def cleanup(self) -> None:
        if hasattr(self, "_model"):
            del self._model
            torch.cuda.empty_cache()

    def _find_voice(self, name: str) -> Path:
        path = CHATTERBOX_VOICES_DIR / f"{name}.wav"
        if path.exists():
            return path
        fallback = CHATTERBOX_VOICES_DIR / f"{DEFAULT_VOICE}.wav"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"Voice '{name}' not found in {CHATTERBOX_VOICES_DIR}")
