"""Tests for TTS streaming endpoint."""

import io
import struct
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

# Mock heavy dependencies
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torchaudio", MagicMock())
sys.modules.setdefault("torchaudio.functional", MagicMock())


def _make_pcm_chunk(n_samples: int = 480, sample_rate: int = 24000) -> np.ndarray:
    """Generate a short sine wave chunk as float32 numpy array."""
    t = np.linspace(0, n_samples / sample_rate, n_samples, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestStreamProtocol:
    """Test the PCM stream binary protocol: header + chunks."""

    def test_header_format(self):
        """Stream header is 8 bytes: sample_rate (u32 LE) + channels (u16 LE) + bit_depth (u16 LE)."""
        sample_rate = 24000
        channels = 1
        bit_depth = 32
        header = struct.pack("<IHH", sample_rate, channels, bit_depth)
        assert len(header) == 8

        sr, ch, bd = struct.unpack("<IHH", header)
        assert sr == 24000
        assert ch == 1
        assert bd == 32

    def test_chunk_frame_format(self):
        """Each chunk is: length (u32 LE) + PCM data (length bytes)."""
        audio = _make_pcm_chunk(480)
        pcm_bytes = audio.tobytes()
        frame = struct.pack("<I", len(pcm_bytes)) + pcm_bytes
        assert len(frame) == 4 + len(pcm_bytes)

        # Parse it back
        length = struct.unpack("<I", frame[:4])[0]
        data = frame[4 : 4 + length]
        recovered = np.frombuffer(data, dtype=np.float32)
        np.testing.assert_array_equal(recovered, audio)

    def test_multiple_chunks_roundtrip(self):
        """Simulate reading a stream with header + N chunks + empty terminator."""
        chunks = [_make_pcm_chunk(480), _make_pcm_chunk(960), _make_pcm_chunk(240)]

        # Build the stream
        buf = io.BytesIO()
        buf.write(struct.pack("<IHH", 24000, 1, 32))  # header
        for chunk in chunks:
            pcm = chunk.tobytes()
            buf.write(struct.pack("<I", len(pcm)))
            buf.write(pcm)
        buf.write(struct.pack("<I", 0))  # terminator

        # Read it back
        buf.seek(0)
        sr, ch, bd = struct.unpack("<IHH", buf.read(8))
        assert sr == 24000

        recovered = []
        while True:
            length_bytes = buf.read(4)
            length = struct.unpack("<I", length_bytes)[0]
            if length == 0:
                break
            data = buf.read(length)
            recovered.append(np.frombuffer(data, dtype=np.float32))

        assert len(recovered) == 3
        for orig, rec in zip(chunks, recovered):
            np.testing.assert_array_equal(orig, rec)


class TestKokoroStreamGenerate:
    """Test that the Kokoro backend's generate_stream yields chunks."""

    def test_generate_stream_yields_chunks(self):
        """generate_stream should yield individual chunks, not concatenate."""
        from backends import TTSBackend

        class FakeKokoro(TTSBackend):
            name = "fake-kokoro"
            sample_rate = 24000

            def load(self):
                pass

            def generate(self, text, voice, speed=1.0, **kwargs):
                return np.concatenate([_make_pcm_chunk(480), _make_pcm_chunk(480)])

            def generate_stream(self, text, voice, speed=1.0, **kwargs):
                yield _make_pcm_chunk(480)
                yield _make_pcm_chunk(480)

            def list_voices(self):
                return {}

        backend = FakeKokoro()
        chunks = list(backend.generate_stream("test", "default"))
        assert len(chunks) == 2
        assert chunks[0].shape == (480,)
        assert chunks[1].shape == (480,)

    def test_base_generate_stream_falls_back_to_generate(self):
        """Backends without generate_stream should fall back to single-chunk."""
        from backends import TTSBackend

        class LegacyBackend(TTSBackend):
            name = "legacy"
            sample_rate = 24000

            def load(self):
                pass

            def generate(self, text, voice, speed=1.0, **kwargs):
                return np.concatenate([_make_pcm_chunk(480), _make_pcm_chunk(480)])

            def list_voices(self):
                return {}

        backend = LegacyBackend()
        chunks = list(backend.generate_stream("test", "default"))
        assert len(chunks) == 1
        assert chunks[0].shape == (960,)
