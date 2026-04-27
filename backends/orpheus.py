"""Orpheus TTS backend."""

import logging
import struct

import numpy as np

from backends import TTSBackend

logger = logging.getLogger("tts-service")

BUILTIN_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]


class OrpheusBackend(TTSBackend):
    name = "orpheus-tts"
    sample_rate = 24000

    def load(self) -> None:
        from orpheus_tts import OrpheusModel
        from orpheus_tts.decoder import convert_to_audio, turn_token_into_id
        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

        # Patch _setup_engine to cap max_model_len (default 131072 is far too large for TTS)
        def _setup_engine(self_model):
            engine_args = AsyncEngineArgs(
                model=self_model.model_name,
                dtype=self_model.dtype,
                max_model_len=2048,
            )
            return AsyncLLMEngine.from_engine_args(engine_args)

        OrpheusModel._setup_engine = _setup_engine

        logger.info("Loading Orpheus TTS...")
        self._model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
        self._SamplingParams = SamplingParams
        self._convert_to_audio = convert_to_audio
        self._turn_token_into_id = turn_token_into_id
        logger.info("Orpheus ready")

    def generate(self, text: str, voice: str, speed: float = 1.0, **kwargs) -> np.ndarray:
        raise RuntimeError("OrpheusBackend requires async_generate; sync generate is not supported")

    async def async_generate(self, text: str, voice: str, speed: float = 1.0, **kwargs) -> np.ndarray:
        if voice == "default":
            voice = "tara"

        temperature = kwargs.get("temperature", 0.6)
        repetition_penalty = kwargs.get("repetition_penalty", 1.3)

        # Format prompt and build sampling params (same logic as OrpheusModel)
        prompt_string = self._model._format_prompt(text, voice)
        sampling_params = self._SamplingParams(
            temperature=temperature,
            top_p=0.8,
            max_tokens=1200,
            stop_token_ids=[49158],
            repetition_penalty=repetition_penalty,
        )

        # Drive the async vLLM engine directly on the current event loop
        token_buffer = []
        count = 0
        audio_buf = bytearray()

        async for result in self._model.engine.generate(
            prompt=prompt_string,
            sampling_params=sampling_params,
            request_id="tts-req",
        ):
            token_str = result.outputs[0].text
            token_id = self._turn_token_into_id(token_str, count)
            if token_id is not None and token_id > 0:
                token_buffer.append(token_id)
                count += 1

                if count % 7 == 0 and count > 27:
                    chunk = self._convert_to_audio(token_buffer[-28:], count)
                    if chunk is not None:
                        audio_buf.extend(chunk)

        if not audio_buf:
            return np.zeros(0, dtype=np.float32)

        # PCM int16 bytes → float32 numpy
        n_samples = len(audio_buf) // 2
        samples = struct.unpack(f"<{n_samples}h", bytes(audio_buf))
        return np.array(samples, dtype=np.float32) / 32768.0

    def list_voices(self) -> dict[str, str]:
        return {v: v for v in BUILTIN_VOICES}

    def cleanup(self) -> None:
        if hasattr(self, "_model"):
            del self._model
