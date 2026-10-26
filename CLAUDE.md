# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A multi-model TTS HTTP server behind a FastAPI API. Supports six backends selectable at startup: Chatterbox Turbo, Kokoro-82M, Qwen3-TTS (0.6B & 1.7B), Orpheus TTS, and Gradium Phonon (31M & 40M).

## Development Setup

```bash
# Python 3.11 via pyenv; venv already exists
source .venv/bin/activate
pip install -e ".[chatterbox]"   # or .[kokoro], .[qwen3], .[orpheus], .[phonon], .[all]
```

Requires a **CUDA GPU** for every backend except Phonon, which runs on CPU through the `ptts` Rust runtime (no PyTorch needed by the model itself).

Kokoro additionally requires system package `espeak-ng` (`apt-get install espeak-ng`).

## Running

```bash
python server.py --model chatterbox     # default
python server.py --model kokoro
python server.py --model qwen3-0.6b
python server.py --model qwen3-1.7b
python server.py --model orpheus
python server.py --model phonon        # Phonon 31M English (Apache 2.0)
python server.py --model phonon-40m    # CC BY-NC 4.0; accept the gated repo's terms on Hugging Face first

# Also available as entry point
tts-server --model kokoro
```

Listens on `0.0.0.0:9100` (override with `--host`/`--port`). The model loads during FastAPI lifespan startup.

## API Endpoints

- `POST /tts` — synthesize speech. Body: `{text, voice, speed, format, ...model-specific fields}`. Returns audio stream.
- `GET /voices` — list available voices for the active backend.
- `GET /health` — readiness check (includes model name and voice count).

Model-specific request fields (ignored by backends that don't use them):
- `exaggeration` (chatterbox), `language` (qwen3), `lang_code` (kokoro), `temperature`/`repetition_penalty` (orpheus)

## Environment Variables

- `VOICES_DIR` — path to voice files root (default: `./voices`). Each backend reads from its own subdirectory: `chatterbox/`, `qwen3/<name>/`.
- `DEFAULT_VOICE` — chatterbox fallback voice name (default: `"default"`)

## Architecture

### Backend Abstraction (`backends/`)

`TTSBackend` ABC in `backends/__init__.py` defines the interface: `load()`, `generate()`, `list_voices()`, `cleanup()`. Each backend lazily imports its model package inside `load()` so missing dependencies only error when that specific backend is selected.

`get_backend(model_name)` is the factory that maps CLI names to backend classes.

- `backends/chatterbox.py` — wraps `chatterbox-tts`. Voices: `.wav` files in `voices/chatterbox/`.
- `backends/kokoro.py` — wraps `kokoro` (KPipeline). Built-in voices; default `af_heart`. Handles speed natively (`handles_speed = True`), so server skips resampling.
- `backends/qwen3.py` — wraps `qwen-tts` (Qwen3TTSModel). Parameterized by size. Voice cloning from `voices/qwen3/<name>/reference.wav` + `transcript.txt`.
- `backends/orpheus.py` — wraps `orpheus-speech` (OrpheusModel). Built-in voices; default `tara`. Collects streaming PCM int16 chunks into numpy.
- `backends/phonon.py` — wraps `ptts` (pinned to 0.4.1, the audited version). CPU only. Voices: the checkpoint's own (`default`) plus Gradium's catalog, by name or catalog ID, downloaded on first use. No speed control, so the server resamples. One model instance serves all requests behind a lock. Env: `PHONON_MODEL`, `PHONON_LANG`, `PHONON_REWRITES`, `PHONON_THREADS` (default 6), `PHONON_QUANT` (default `q8_0`). The thread and quant defaults matter a lot: ptts's own defaults (all CPU threads, f32) were ~10x slower to first audio on a 16-core Ryzen.

### Server (`server.py`)

`argparse` in `main()` sets a module-level `_selected_model` read by the lifespan. The backend instance lives in `app.state.backend`. Speed adjustment via torchaudio resampling is shared across backends unless `backend.handles_speed` is True.

### Voice Directories

```
voices/
  chatterbox/               # Chatterbox .wav reference files
  qwen3/                    # Qwen3 voice cloning configs
    <voice_name>/
      reference.wav         # 3+ seconds of reference audio
      transcript.txt        # Transcript of the reference audio
```

## Notes

- **Tests:** `test_streaming.py` (stream protocol) and `test_phonon_backend.py` (Phonon backend, ptts mocked). Run with `pytest`.
- **No Dockerfile.** Deployment is run-from-source.
- Dependencies are optional groups in `pyproject.toml` — only install what you need.
