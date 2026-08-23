"""Multi-model TTS HTTP server."""

import argparse
import asyncio
import io
import logging
from contextlib import asynccontextmanager

import numpy as np

import soundfile as sf
import torch
import torchaudio as ta
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backends import get_backend

logger = logging.getLogger("tts-service")

MODEL_CHOICES = [
    "chatterbox", "kokoro",
    "qwen3-0.6b", "qwen3-1.7b",
    "qwen3-0.6b-design", "qwen3-1.7b-design",
    "orpheus",
]

# Set by main() before uvicorn starts; read by lifespan().
_selected_model: str = "chatterbox"


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = get_backend(_selected_model)
    backend.load()
    app.state.backend = backend
    logger.info("Backend ready: %s", backend.name)
    yield
    backend.cleanup()
    torch.cuda.empty_cache()


app = FastAPI(title="TTS Service", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str
    voice: str = Field(default="default", description="Voice name")
    speed: float = Field(default=1.0, description="Speech speed multiplier")
    format: str = Field(default="wav", description="Output format: wav, flac, ogg")
    # Model-specific (backends ignore fields they don't use)
    exaggeration: float = Field(default=0.5, description="Chatterbox: emotion exaggeration")
    language: str = Field(default="English", description="Qwen3: target language")
    lang_code: str = Field(default="a", description="Kokoro: language code (a=American English)")
    temperature: float = Field(default=0.6, description="Orpheus: sampling temperature")
    repetition_penalty: float = Field(default=1.3, description="Orpheus: repetition penalty")


FORMATS = {
    "wav": ("audio/wav", "WAV"),
    "flac": ("audio/flac", "FLAC"),
    "ogg": ("audio/ogg", "OGG"),
}


@app.post("/tts")
async def synthesize(req: TTSRequest):
    backend = app.state.backend

    if req.format not in FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {req.format}")

    mime_type, sf_format = FORMATS[req.format]

    kwargs = {
        "exaggeration": req.exaggeration,
        "language": req.language,
        "lang_code": req.lang_code,
        "temperature": req.temperature,
        "repetition_penalty": req.repetition_penalty,
    }

    try:
        audio = await backend.async_generate(req.text, req.voice, req.speed, **kwargs)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))

    # Apply speed adjustment via resampling (skip if backend already handled it)
    if req.speed != 1.0 and not backend.handles_speed:
        sr = backend.sample_rate
        new_rate = int(sr * req.speed)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        audio_tensor = ta.functional.resample(audio_tensor, new_rate, sr)
        audio = audio_tensor.squeeze(0).numpy()

    buf = io.BytesIO()
    sf.write(buf, audio, backend.sample_rate, format=sf_format)
    buf.seek(0)

    return StreamingResponse(buf, media_type=mime_type)


@app.post("/tts/stream")
async def synthesize_stream(req: TTSRequest):
    """Stream raw PCM audio chunks as they're generated.

    Binary protocol: 8-byte header + length-prefixed chunks + zero terminator.
    Header: sample_rate (u32 LE) | channels (u16 LE) | bit_depth (u16 LE)
    Each chunk: length (u32 LE) | PCM float32 data (length bytes)
    Terminator: length = 0 (u32 LE)
    """
    import struct
    from functools import partial

    backend = app.state.backend
    kwargs = {
        "exaggeration": req.exaggeration,
        "language": req.language,
        "lang_code": req.lang_code,
        "temperature": req.temperature,
        "repetition_penalty": req.repetition_penalty,
    }

    async def pcm_chunk_generator():
        yield struct.pack("<IHH", backend.sample_rate, 1, 32)

        loop = asyncio.get_event_loop()
        gen = partial(backend.generate_stream, req.text, req.voice, req.speed, **kwargs)
        chunks = await loop.run_in_executor(None, lambda: list(gen()))

        for chunk in chunks:
            if req.speed != 1.0 and not backend.handles_speed:
                sr = backend.sample_rate
                new_rate = int(sr * req.speed)
                audio_tensor = torch.from_numpy(chunk).unsqueeze(0)
                audio_tensor = ta.functional.resample(audio_tensor, new_rate, sr)
                chunk = audio_tensor.squeeze(0).numpy()

            pcm_bytes = chunk.astype(np.float32).tobytes()
            yield struct.pack("<I", len(pcm_bytes)) + pcm_bytes

        yield struct.pack("<I", 0)

    return StreamingResponse(
        pcm_chunk_generator(),
        media_type="application/octet-stream",
    )


@app.get("/health")
async def health():
    backend = app.state.backend
    voices = backend.list_voices()
    return {
        "status": "ok",
        "model": backend.name,
        "voices": len(voices),
    }


@app.get("/voices")
async def list_voices():
    return app.state.backend.list_voices()


def main():
    global _selected_model

    parser = argparse.ArgumentParser(description="TTS HTTP Server")
    parser.add_argument(
        "--model", "-m",
        choices=MODEL_CHOICES,
        default="chatterbox",
        help="TTS backend to use (default: chatterbox)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    args = parser.parse_args()

    _selected_model = args.model

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting server with model: %s", _selected_model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
