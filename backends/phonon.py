"""Gradium Phonon TTS backend (CPU, via the ptts Rust runtime)."""

import logging
import os
import threading

import numpy as np

from backends import TTSBackend

logger = logging.getLogger("tts-service")

MODELS = {
    "phonon": None,  # ptts.DEFAULT_MODEL: Phonon 31M English, Apache 2.0
    "phonon-40m": "hf://gradium/phonon-40m-en",  # CC BY-NC 4.0, gated: accept the terms on Hugging Face first
}


class PhononBackend(TTSBackend):
    """Phonon on CPU. Voices are checkpoint voices or catalog voices, by name or catalog ID.

    Environment overrides:
      PHONON_MODEL     hf://owner/name[@revision] or a local path (wins over the CLI model name)
      PHONON_LANG      text normalization language: en, fr, de, es, pt (default: en)
      PHONON_REWRITES  default, all (adds phones, times, dates), none, or a rule list
      PHONON_THREADS   CPU threads for tensor ops (default: 6)
      PHONON_QUANT     weight format, e.g. q8_0, q4k, f32 (default: q8_0)

    The thread and quant defaults come from a sweep on a Ryzen 9 5900XT (16 cores, 32 threads),
    first audio / real-time factor: ptts's own defaults (all 32 threads, f32) gave 222 ms / 0.69;
    6 threads with q8_0 gave 24 ms / 0.06. Past ~8 threads it gets slower, not faster.
    """

    sample_rate = 24000
    handles_speed = False  # Phonon has no speed control; the server resamples

    def __init__(self, model_name: str = "phonon"):
        self.name = model_name
        self._model = os.environ.get("PHONON_MODEL") or MODELS[model_name]
        self._lang = os.environ.get("PHONON_LANG", "en")
        self._rewrites = os.environ.get("PHONON_REWRITES") or None
        self._threads = int(os.environ.get("PHONON_THREADS", "6"))
        self._quant = os.environ.get("PHONON_QUANT", "q8_0")
        # One model instance serves every request; serialize access rather than rely on the
        # runtime being safe for concurrent generations.
        self._lock = threading.Lock()

    def load(self) -> None:
        import ptts

        ptts.set_num_threads(self._threads)  # sizes a global pool, so it must precede the model load

        logger.info("Loading Phonon %s (ptts %s)...", self._model or ptts.DEFAULT_MODEL, ptts.__version__)
        self._tts = ptts.TTS(
            model=self._model, quant=self._quant, lang=self._lang, rewrites=self._rewrites
        )
        self.sample_rate = self._tts.sample_rate

        voices = {v: "checkpoint voice" for v in self._tts.voices}
        for v in ptts.catalog_voices():
            if v["language"] == self._lang:
                voices[v["name"]] = v["description"] or ""
        self._voices = voices
        logger.info(
            "Phonon ready on %s (%s, %d threads), %d %s voices",
            self._tts.device, self._tts.quant, ptts.get_num_threads(), len(voices), self._lang,
        )

    @staticmethod
    def _voice(voice: str) -> str | None:
        # "default" is the server's placeholder; None selects the checkpoint's own voice.
        return None if voice == "default" else voice

    def generate(self, text: str, voice: str, speed: float = 1.0, **kwargs) -> np.ndarray:
        with self._lock:
            return np.asarray(self._tts.synth(text, voice=self._voice(voice)), dtype=np.float32)

    def generate_stream(self, text: str, voice: str, speed: float = 1.0, **kwargs):
        with self._lock, self._tts.stream(text, voice=self._voice(voice)) as stream:
            for chunk in stream:
                yield np.asarray(chunk, dtype=np.float32)

    def list_voices(self) -> dict[str, str]:
        return self._voices

    def cleanup(self) -> None:
        if hasattr(self, "_tts"):
            del self._tts
