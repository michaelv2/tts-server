"""Qwen3-TTS backends (Base for voice cloning, VoiceDesign for text-described voices)."""

import logging
import os
from pathlib import Path

import numpy as np
import torch

from backends import TTSBackend

logger = logging.getLogger("tts-service")

VOICES_DIR = Path(os.environ.get("VOICES_DIR", Path(__file__).resolve().parent.parent / "voices"))
QWEN3_VOICES_DIR = VOICES_DIR / "qwen3"

CLONE_MODEL_IDS = {
    "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

DESIGN_MODEL_IDS = {
    "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign",
    "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}


class Qwen3Backend(TTSBackend):
    """Voice cloning backend — requires reference audio in voices/qwen3/<name>/."""

    sample_rate: int  # set after model loads

    def __init__(self, size: str = "1.7b") -> None:
        if size not in CLONE_MODEL_IDS:
            raise ValueError(f"Unknown Qwen3 size: {size!r}. Choose from: {list(CLONE_MODEL_IDS)}")
        self._size = size
        self._model_id = CLONE_MODEL_IDS[size]
        self.name = f"qwen3-tts-{size}"

    def load(self) -> None:
        from qwen_tts import Qwen3TTSModel

        logger.info("Loading %s from %s...", self.name, self._model_id)
        self._model = Qwen3TTSModel.from_pretrained(
            self._model_id,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        logger.info("%s ready", self.name)

    def generate(self, text: str, voice: str, speed: float = 1.0, **kwargs) -> np.ndarray:
        language = kwargs.get("language", "English")
        ref_audio, ref_text = self._load_voice_config(voice)

        wavs, sr = self._model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=str(ref_audio),
            ref_text=ref_text,
        )
        self.sample_rate = sr
        return wavs[0] if isinstance(wavs[0], np.ndarray) else np.array(wavs[0])

    def list_voices(self) -> dict[str, str]:
        if not QWEN3_VOICES_DIR.exists():
            return {}
        voices = {}
        for d in sorted(QWEN3_VOICES_DIR.iterdir()):
            if d.is_dir() and (d / "reference.wav").exists():
                has_transcript = (d / "transcript.txt").exists()
                status = "ready" if has_transcript else "no transcript"
                voices[d.name] = f"{d.name} ({status})"
        return voices

    def cleanup(self) -> None:
        if hasattr(self, "_model"):
            del self._model
            torch.cuda.empty_cache()

    def _load_voice_config(self, name: str) -> tuple[Path, str]:
        voice_dir = QWEN3_VOICES_DIR / name
        ref_audio = voice_dir / "reference.wav"
        transcript_file = voice_dir / "transcript.txt"

        if not ref_audio.exists():
            available = list(self.list_voices().keys())
            raise FileNotFoundError(
                f"Voice '{name}' not found. "
                f"Expected {ref_audio}. Available: {available}"
            )

        ref_text = ""
        if transcript_file.exists():
            ref_text = transcript_file.read_text().strip()

        return ref_audio, ref_text


class Qwen3DesignBackend(TTSBackend):
    """Voice design backend — voice field is a natural language description."""

    sample_rate: int  # set after model loads

    def __init__(self, size: str = "1.7b") -> None:
        if size not in DESIGN_MODEL_IDS:
            raise ValueError(f"Unknown Qwen3 size: {size!r}. Choose from: {list(DESIGN_MODEL_IDS)}")
        self._size = size
        self._model_id = DESIGN_MODEL_IDS[size]
        self.name = f"qwen3-tts-{size}-design"

    def load(self) -> None:
        from qwen_tts import Qwen3TTSModel

        logger.info("Loading %s from %s...", self.name, self._model_id)
        self._model = Qwen3TTSModel.from_pretrained(
            self._model_id,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        logger.info("%s ready", self.name)

    def generate(self, text: str, voice: str, speed: float = 1.0, **kwargs) -> np.ndarray:
        language = kwargs.get("language", "English")

        wavs, sr = self._model.generate_voice_design(
            text=text,
            language=language,
            instruct=voice,
        )
        self.sample_rate = sr
        return wavs[0] if isinstance(wavs[0], np.ndarray) else np.array(wavs[0])

    def list_voices(self) -> dict[str, str]:
        return {"(any)": "Pass a natural language voice description as the voice field"}

    def cleanup(self) -> None:
        if hasattr(self, "_model"):
            del self._model
            torch.cuda.empty_cache()
