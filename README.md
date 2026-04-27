# tts-server

Multi-model TTS HTTP server behind a FastAPI API. Pick one backend at startup; all backends load on GPU.

Supported backends:

- **Chatterbox Turbo** (`chatterbox`)
- **Kokoro-82M** (`kokoro`)
- **Qwen3-TTS** (`qwen3-0.6b`, `qwen3-1.7b`)
- **Orpheus TTS** (`orpheus`)

## Requirements

- Python 3.11
- CUDA GPU (no CPU fallback)
- `espeak-ng` system package (Kokoro only): `sudo apt-get install espeak-ng`

## Setup

Each backend gets its own venv to avoid dependency conflicts:

```bash
./setup-backend.sh kokoro       # creates .venv-kokoro
./setup-backend.sh chatterbox   # creates .venv-chatterbox
./setup-backend.sh qwen3        # creates .venv-qwen3
./setup-backend.sh orpheus      # creates .venv-orpheus
```

## Running

```bash
source .venv-kokoro/bin/activate
python server.py --model kokoro
```

Other models:

```bash
python server.py --model chatterbox     # default
python server.py --model qwen3-0.6b
python server.py --model qwen3-1.7b
python server.py --model orpheus
```

Listens on `0.0.0.0:9100`. Override with `--host` / `--port`.

## API

- `POST /tts` — synthesize speech. Body: `{text, voice, speed, format, ...}`. Returns audio stream.
- `GET /voices` — list available voices for the active backend.
- `GET /health` — readiness check (model name + voice count).

Backend-specific request fields (ignored by backends that don't use them):

| Field | Used by |
|-------|---------|
| `exaggeration` | chatterbox |
| `language` | qwen3 |
| `lang_code` | kokoro |
| `temperature`, `repetition_penalty` | orpheus |

## Voices

```
voices/
  chatterbox/                 # .wav reference files
  qwen3/
    <voice_name>/
      reference.wav           # 3+ seconds of reference audio
      transcript.txt          # transcript of the reference
```

Kokoro and Orpheus ship with built-in voices (default `af_heart` and `tara`).

## Environment Variables

- `VOICES_DIR` — voice files root (default: `./voices`)
- `DEFAULT_VOICE` — chatterbox fallback voice (default: `default`)
