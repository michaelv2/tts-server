"""TTS backend abstraction and registry."""

import asyncio
from abc import ABC, abstractmethod
from functools import partial

import numpy as np


class TTSBackend(ABC):
    """Base class for all TTS backends."""

    name: str
    sample_rate: int
    handles_speed: bool = False  # True if backend applies speed itself

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Called once at startup."""

    @abstractmethod
    def generate(self, text: str, voice: str, speed: float = 1.0, **kwargs) -> np.ndarray:
        """Synthesize speech. Returns a 1-D float numpy array at self.sample_rate."""

    def generate_stream(self, text: str, voice: str, speed: float = 1.0, **kwargs):
        """Yield audio chunks as they're produced. Override for streaming backends.

        Default yields the full result of generate() as a single chunk.
        """
        yield self.generate(text, voice, speed, **kwargs)

    async def async_generate(self, text: str, voice: str, speed: float = 1.0, **kwargs) -> np.ndarray:
        """Async version of generate. Override for backends with async engines.

        Default runs sync generate in a thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, partial(self.generate, text, voice, speed, **kwargs)
        )

    @abstractmethod
    def list_voices(self) -> dict[str, str]:
        """Return {voice_name: description} for available voices."""

    def cleanup(self) -> None:
        """Release resources. Called on shutdown."""


def get_backend(model_name: str) -> TTSBackend:
    """Instantiate a backend by CLI model name."""
    if model_name == "chatterbox":
        from backends.chatterbox import ChatterboxBackend
        return ChatterboxBackend()
    elif model_name == "kokoro":
        from backends.kokoro import KokoroBackend
        return KokoroBackend()
    elif model_name.startswith("qwen3-") and model_name.endswith("-design"):
        from backends.qwen3 import Qwen3DesignBackend
        size = model_name.removeprefix("qwen3-").removesuffix("-design")
        return Qwen3DesignBackend(size=size)
    elif model_name.startswith("qwen3-"):
        from backends.qwen3 import Qwen3Backend
        size = model_name.split("-", 1)[1]  # "0.6b" or "1.7b"
        return Qwen3Backend(size=size)
    elif model_name == "orpheus":
        from backends.orpheus import OrpheusBackend
        return OrpheusBackend()
    else:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Choose from: chatterbox, kokoro, qwen3-0.6b, qwen3-1.7b, "
            f"qwen3-0.6b-design, qwen3-1.7b-design, orpheus"
        )
