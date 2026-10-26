"""Tests for the Phonon backend, with the ptts runtime mocked out."""

import os
import sys
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from backends import get_backend  # noqa: E402
from backends.phonon import PhononBackend  # noqa: E402


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def __iter__(self):
        return iter(self._chunks)


@pytest.fixture
def fake_ptts(monkeypatch):
    ptts = MagicMock()
    ptts.__version__ = "0.4.1"
    ptts.DEFAULT_MODEL = "hf://gradium/phonon-31m-en@abc"
    tts = MagicMock()
    tts.sample_rate = 24000
    tts.voices = ["default"]
    tts.device = "cpu"
    tts.quant = "q8_0"
    tts.synth.return_value = np.ones(480, dtype=np.float64)
    ptts.TTS.return_value = tts
    ptts.catalog_voices.return_value = [
        {"name": "Zoey", "language": "en", "voice_id": "z1", "description": "upbeat"},
        {"name": "Léa", "language": "fr", "voice_id": "l1", "description": "french"},
    ]
    monkeypatch.setitem(sys.modules, "ptts", ptts)
    for var in ("PHONON_MODEL", "PHONON_LANG", "PHONON_REWRITES", "PHONON_THREADS", "PHONON_QUANT"):
        monkeypatch.delenv(var, raising=False)
    return ptts


def test_registry_maps_both_model_names():
    assert isinstance(get_backend("phonon"), PhononBackend)
    assert get_backend("phonon-40m")._model == "hf://gradium/phonon-40m-en"


def test_defaults_tuned_threads_and_quant(fake_ptts):
    backend = PhononBackend("phonon")
    backend.load()
    fake_ptts.set_num_threads.assert_called_once_with(6)
    fake_ptts.TTS.assert_called_once_with(model=None, quant="q8_0", lang="en", rewrites=None)


def test_threads_set_before_model_load(fake_ptts):
    calls = []
    fake_ptts.set_num_threads.side_effect = lambda n: calls.append("threads")
    fake_ptts.TTS.side_effect = lambda **kw: calls.append("load") or MagicMock(voices=[])
    PhononBackend("phonon").load()
    assert calls == ["threads", "load"]


def test_env_overrides(fake_ptts, monkeypatch):
    monkeypatch.setenv("PHONON_MODEL", "hf://gradium/phonon-40m-en@deadbeef")
    monkeypatch.setenv("PHONON_REWRITES", "all")
    monkeypatch.setenv("PHONON_THREADS", "4")
    monkeypatch.setenv("PHONON_QUANT", "q4k")
    backend = PhononBackend("phonon")
    backend.load()
    fake_ptts.set_num_threads.assert_called_once_with(4)
    fake_ptts.TTS.assert_called_once_with(
        model="hf://gradium/phonon-40m-en@deadbeef", quant="q4k", lang="en", rewrites="all"
    )


def test_voices_include_checkpoint_and_same_language_catalog(fake_ptts):
    backend = PhononBackend("phonon")
    backend.load()
    voices = backend.list_voices()
    assert set(voices) == {"default", "Zoey"}
    assert voices["Zoey"] == "upbeat"


def test_default_voice_means_checkpoint_voice(fake_ptts):
    backend = PhononBackend("phonon")
    backend.load()
    audio = backend.generate("hi", "default")
    backend._tts.synth.assert_called_once_with("hi", voice=None)
    assert audio.dtype == np.float32

    backend.generate("hi", "Zoey")
    backend._tts.synth.assert_called_with("hi", voice="Zoey")


def test_stream_yields_float32_chunks_and_closes(fake_ptts):
    backend = PhononBackend("phonon")
    backend.load()
    stream = _FakeStream([np.zeros(240, dtype=np.float64), np.ones(240, dtype=np.float64)])
    backend._tts.stream.return_value = stream

    chunks = list(backend.generate_stream("hi", "Zoey"))

    backend._tts.stream.assert_called_once_with("hi", voice="Zoey")
    assert [c.dtype for c in chunks] == [np.float32, np.float32]
    assert stream.closed


def test_generations_are_serialized(fake_ptts):
    backend = PhononBackend("phonon")
    backend.load()
    active, peak = 0, 0
    guard = threading.Lock()

    def slow_synth(text, voice=None):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        threading.Event().wait(0.05)
        with guard:
            active -= 1
        return np.zeros(10, dtype=np.float32)

    backend._tts.synth.side_effect = slow_synth
    threads = [threading.Thread(target=backend.generate, args=("hi", "default")) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 1
