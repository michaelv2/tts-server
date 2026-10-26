"""Tests that /tts/stream forwards chunks as the backend produces them."""

import asyncio
import os
import struct
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

fastapi_testclient = pytest.importorskip("fastapi.testclient")

import server  # noqa: E402


class TwoChunkBackend:
    name = "two-chunk"
    sample_rate = 24000
    handles_speed = True

    def __init__(self):
        self.closed = threading.Event()

    def generate_stream(self, text, voice, speed=1.0, **kwargs):
        try:
            yield np.full(4, 0.25, dtype=np.float32)
            yield np.full(4, 0.5, dtype=np.float32)
        finally:
            self.closed.set()


class FailingBackend:
    name = "failing"
    sample_rate = 24000
    handles_speed = True

    def generate_stream(self, text, voice, speed=1.0, **kwargs):
        yield np.zeros(4, dtype=np.float32)
        raise RuntimeError("synthesis blew up")


def _read_frames(body: bytes):
    sr, channels, bits = struct.unpack("<IHH", body[:8])
    frames, pos = [], 8
    while True:
        (n,) = struct.unpack("<I", body[pos : pos + 4])
        pos += 4
        if n == 0:
            break
        frames.append(np.frombuffer(body[pos : pos + n], dtype=np.float32))
        pos += n
    return (sr, channels, bits), frames


@pytest.fixture
def client():
    # No `with`: skips the lifespan, so no real model loads; tests install their own backend.
    return fastapi_testclient.TestClient(server.app)


def test_stream_protocol_and_backend_closed(client):
    backend = TwoChunkBackend()
    server.app.state.backend = backend

    resp = client.post("/tts/stream", json={"text": "hi"})

    header, frames = _read_frames(resp.content)
    assert header == (24000, 1, 32)
    assert [f.tolist() for f in frames] == [[0.25] * 4, [0.5] * 4]
    assert backend.closed.wait(timeout=2)


# Starlette's TestClient buffers whole responses, so chunk timing is tested on the helper itself.


def test_first_item_arrives_while_producer_still_running():
    first_seen = threading.Event()
    timed_out = []

    def produce():
        yield 1
        # The old list() drain would block here until the timeout, because the consumer could
        # not see item 1 before the generator finished.
        if not first_seen.wait(timeout=5):
            timed_out.append(True)
        yield 2

    async def consume():
        got = []
        async for item in server.iterate_in_thread(produce):
            got.append(item)
            first_seen.set()
        return got

    assert asyncio.run(consume()) == [1, 2]
    assert not timed_out


def test_early_stop_closes_producer():
    closed = threading.Event()
    produced = []

    def produce():
        try:
            for i in range(1000):
                produced.append(i)
                yield i
                threading.Event().wait(0.001)
        finally:
            closed.set()

    async def consume_one():
        agen = server.iterate_in_thread(produce)
        item = await agen.__anext__()
        await agen.aclose()  # what Starlette does when the client disconnects
        return item

    assert asyncio.run(consume_one()) == 0
    assert closed.wait(timeout=2)
    assert len(produced) < 1000


def test_backend_error_surfaces(client):
    server.app.state.backend = FailingBackend()
    with pytest.raises(RuntimeError, match="synthesis blew up"):
        with client.stream("POST", "/tts/stream", json={"text": "hi"}) as resp:
            for _ in resp.iter_bytes():
                pass


def test_apply_speed_is_fast_and_shortens_audio():
    # 24000 * 1.15 is 27599.999...: truncating it gave a coprime rate and a ~6 s resample.
    audio = np.zeros(24000, dtype=np.float32)
    start = time.monotonic()
    out = server.apply_speed(audio, 24000, 1.15)
    assert time.monotonic() - start < 1.0
    assert abs(len(out) - 24000 / 1.15) < 2
