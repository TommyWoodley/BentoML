from __future__ import annotations

import asyncio
import typing as t

import numpy as np
import pytest

from bentoml._internal.marshal.dispatcher import CorkDispatcher
from bentoml._internal.marshal.dispatcher import Job


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_dispatcher(
    clock: t.Callable[[], float],
    *,
    max_latency_in_ms: int = 100,
    max_batch_size: int = 8,
    get_batch_size: t.Callable[[t.Any], int] = lambda _: 1,
) -> CorkDispatcher[t.Any, t.Any]:
    return CorkDispatcher(
        max_latency_in_ms=max_latency_in_ms,
        max_batch_size=max_batch_size,
        fallback=lambda: "fallback",
        get_batch_size=get_batch_size,
        batch_dim=(0, 0),
        clock=clock,
    )


@pytest.mark.asyncio
async def test_inbound_call_uses_explicit_deadline() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock)

    task = asyncio.create_task(dispatcher.inbound_call("request", max_latency_ms=25))
    await asyncio.sleep(0)

    job = dispatcher._queue[0]
    assert job.enqueue_time == 100.0
    assert job.deadline == pytest.approx(100.025)

    job.future.set_result("response")
    assert await task == "response"


@pytest.mark.asyncio
async def test_inbound_call_uses_constructor_default() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock, max_latency_in_ms=80)

    task = asyncio.create_task(dispatcher.inbound_call("request"))
    await asyncio.sleep(0)

    job = dispatcher._queue[0]
    assert job.deadline == pytest.approx(100.08)

    job.future.set_result("response")
    assert await task == "response"


@pytest.mark.asyncio
async def test_dispatcher_wrapper_forwards_explicit_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock)
    observed: list[int | None] = []

    async def fake_controller() -> None:
        await asyncio.Future()

    async def fake_inbound(data: str, *, max_latency_ms: int | None = None) -> str:
        observed.append(max_latency_ms)
        return data

    monkeypatch.setattr(dispatcher, "controller", fake_controller)
    monkeypatch.setattr(dispatcher, "inbound_call", fake_inbound)

    async def callback(values: t.Sequence[str]) -> t.Sequence[str]:
        return values

    wrapped = dispatcher(callback)
    try:
        assert await wrapped("request", max_latency_ms=12) == "request"
    finally:
        dispatcher.shutdown()

    assert observed == [12]


@pytest.mark.asyncio
async def test_split_job_preserves_deadline() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock)
    future: asyncio.Future[t.Any | Exception] = (
        asyncio.get_running_loop().create_future()
    )
    job = Job(
        enqueue_time=100.0,
        data=np.array([1, 2, 3, 4]),
        future=future,
        deadline=100.025,
    )

    first, second = dispatcher.split_job(job, [0, 2, 4])

    assert first.deadline == job.deadline
    assert second.deadline == job.deadline
