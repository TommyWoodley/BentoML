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


def queued_job(
    dispatcher: CorkDispatcher[t.Any, t.Any],
    *,
    data: tuple[str, int],
    enqueue_time: float,
    deadline: float,
) -> Job[t.Any, t.Any]:
    future: asyncio.Future[t.Any | Exception] = dispatcher._loop.create_future()
    return Job(
        enqueue_time=enqueue_time,
        data=data,
        future=future,
        deadline=deadline,
    )


def test_peek_candidate_matches_batch_size_without_mutating_queue() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(
        clock,
        max_batch_size=10,
        get_batch_size=lambda data: data[1],
    )
    jobs = [
        queued_job(dispatcher, data=("A", 7), enqueue_time=90, deadline=200),
        queued_job(dispatcher, data=("B", 7), enqueue_time=91, deadline=120),
        queued_job(dispatcher, data=("C", 6), enqueue_time=92, deadline=105),
    ]
    for job in jobs:
        dispatcher._append_job(job)

    candidate = dispatcher._peek_candidate()

    assert candidate is not None
    assert candidate.job_count == 2
    assert candidate.sample_size == 10
    assert candidate.earliest_deadline == 120
    assert candidate.last_enqueue_time == 91
    assert list(dispatcher._queue) == jobs


def test_candidate_cache_is_invalidated_by_tighter_enqueue() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock, max_batch_size=4)
    dispatcher._append_job(
        queued_job(dispatcher, data=("A", 1), enqueue_time=90, deadline=200)
    )
    first = dispatcher._peek_candidate()
    dispatcher._append_job(
        queued_job(dispatcher, data=("B", 1), enqueue_time=91, deadline=110)
    )
    second = dispatcher._peek_candidate()

    assert first is not None
    assert second is not None
    assert first.earliest_deadline == 200
    assert second.earliest_deadline == 110


def test_deadline_outside_candidate_does_not_change_candidate_deadline() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock, max_batch_size=2)
    for name, deadline in [("A", 200), ("B", 210), ("C", 105)]:
        dispatcher._append_job(
            queued_job(
                dispatcher,
                data=(name, 1),
                enqueue_time=90,
                deadline=deadline,
            )
        )

    candidate = dispatcher._peek_candidate()

    assert candidate is not None
    assert candidate.job_count == 2
    assert candidate.earliest_deadline == 200
    assert dispatcher._next_expiry == 105


def test_expire_jobs_removes_only_expired_jobs_and_preserves_fifo() -> None:
    clock = FakeClock(110)
    dispatcher = make_dispatcher(clock, max_batch_size=2)
    jobs = [
        queued_job(dispatcher, data=("A", 1), enqueue_time=90, deadline=200),
        queued_job(dispatcher, data=("B", 1), enqueue_time=91, deadline=105),
        queued_job(dispatcher, data=("C", 1), enqueue_time=92, deadline=210),
    ]
    for job in jobs:
        dispatcher._append_job(job)

    assert dispatcher._expire_jobs(clock()) == 1
    assert [job.data[0] for job in dispatcher._queue] == ["A", "C"]
    assert jobs[1].future.cancelled()
    assert not jobs[0].future.done()
    assert not jobs[2].future.done()
    assert dispatcher._next_expiry == 200


class CountingDeadline(float):
    comparisons = 0

    def __lt__(self, other: object) -> bool:
        type(self).comparisons += 1
        return super().__lt__(other)


def test_take_inputs_does_not_scan_every_remaining_deadline() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock, max_batch_size=1)
    for index in range(64):
        dispatcher._append_job(
            queued_job(
                dispatcher,
                data=(str(index), 1),
                enqueue_time=float(index),
                deadline=CountingDeadline(200 + index),
            )
        )

    CountingDeadline.comparisons = 0
    taken = dispatcher._take_inputs()

    assert len(taken) == 1
    assert CountingDeadline.comparisons < 16
    assert dispatcher._next_expiry == 201


def test_popleft_job_updates_candidate_and_expiry_state() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock, max_batch_size=2)
    first = queued_job(
        dispatcher,
        data=("A", 1),
        enqueue_time=90,
        deadline=105,
    )
    second = queued_job(
        dispatcher,
        data=("B", 1),
        enqueue_time=91,
        deadline=120,
    )
    dispatcher._append_job(first)
    dispatcher._append_job(second)
    assert dispatcher._peek_candidate() is not None

    assert dispatcher._popleft_job() is first

    candidate = dispatcher._peek_candidate()
    assert candidate is not None
    assert candidate.job_count == 1
    assert candidate.earliest_deadline == 120
    assert dispatcher._next_expiry == 120
