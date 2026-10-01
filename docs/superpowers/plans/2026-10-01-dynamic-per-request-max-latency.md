# Dynamic Per Request max_latency_ms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow a synchronous request resolver to assign an individual adaptive-batching deadline to each HTTP or ASGI request while preserving a shared FIFO dispatcher and legacy static callers.

**Architecture:** Resolve callable latency configuration in `ServiceAppFactory` after request decoding, pass only the validated integer through `batch_infer()`, and store an absolute monotonic deadline on each dispatcher `Job`. Keep the optimizer's existing job-count model, use a non-mutating cached view of the next FIFO candidate, and separate candidate release from stable full-queue expiration.

**Tech Stack:** Python 3.9+, asyncio, attrs, Starlette, BentoML new SDK, pytest, pytest-asyncio, PDM, Sphinx/reStructuredText.

**Design:** `docs/superpowers/specs/2026-10-01-dynamic-per-request-max-latency-design.md`

---

## File map

- Modify `src/_bentoml_sdk/method.py`: define the public latency union, validate static and callable configuration, and expose shared positive-integer validation.
- Modify `src/_bentoml_sdk/decorators.py`: type `@bentoml.api` with the latency union while keeping `@bentoml.task` integer-only.
- Modify `src/bentoml/_internal/marshal/dispatcher.py`: add job deadlines, monotonic timing, per-call latency, candidate caching, independent expiry, and pure scheduling decisions.
- Modify `src/_bentoml_impl/server/app.py`: seed dynamic dispatchers, invoke resolvers once, and forward resolved latency to the dispatcher.
- Create `tests/unit/_bentoml_sdk/test_max_latency.py`: configuration and resolver contract tests.
- Create `tests/unit/_internal/marshal/test_dispatcher.py`: deterministic dispatcher, candidate, expiration, split, and scheduling tests.
- Create `tests/e2e/fixtures/dynamic_batching/service.py`: dependency-free packaged service used by end-to-end tests.
- Create `tests/e2e/bento_new_sdk/test_dynamic_batching.py`: concurrent HTTP and packaged-Bento coverage.
- Modify `docs/source/get-started/adaptive-batching.rst`: document static and dynamic latency forms, scope, FIFO semantics, and input clamping.

## Task 1: Define and validate the SDK contract

**Files:**

- Create: `tests/unit/_bentoml_sdk/test_max_latency.py`
- Modify: `src/_bentoml_sdk/method.py:1-130`
- Modify: `src/_bentoml_sdk/decorators.py:1-210`

- [ ] **Step 1: Write configuration tests that currently fail**

Create `tests/unit/_bentoml_sdk/test_max_latency.py`:

```python
from __future__ import annotations

import functools
import typing as t

import pytest
from starlette.requests import Request

import bentoml


def resolver(_: Request) -> int:
    return 25


async def async_resolver(_: Request) -> int:
    return 25


class AsyncCallable:
    async def __call__(self, _: Request) -> int:
        return 25


def test_api_accepts_positive_static_latency() -> None:
    @bentoml.api(batchable=True, max_latency_ms=25)
    def predict(_: t.Any, values: list[int]) -> list[int]:
        return values

    assert predict.max_latency_ms == 25


def test_api_accepts_sync_latency_resolver() -> None:
    @bentoml.api(batchable=True, max_latency_ms=resolver)
    def predict(_: t.Any, values: list[int]) -> list[int]:
        return values

    assert predict.max_latency_ms is resolver


@pytest.mark.parametrize("value", [0, -1, 1.5, "10", None, True, False])
def test_api_rejects_invalid_static_latency(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer or a synchronous callable"):

        @bentoml.api(batchable=True, max_latency_ms=value)  # type: ignore[arg-type]
        def predict(_: t.Any, values: list[int]) -> list[int]:
            return values


def test_api_rejects_callable_when_not_batchable() -> None:
    with pytest.raises(ValueError, match="requires batchable=True"):

        @bentoml.api(max_latency_ms=resolver)
        def predict(_: t.Any, value: int) -> int:
            return value


@pytest.mark.parametrize(
    "value",
    [async_resolver, functools.partial(async_resolver), AsyncCallable()],
)
def test_api_rejects_async_latency_resolver(value: object) -> None:
    with pytest.raises(ValueError, match="must be synchronous"):

        @bentoml.api(batchable=True, max_latency_ms=value)  # type: ignore[arg-type]
        def predict(_: t.Any, values: list[int]) -> list[int]:
            return values


def test_task_rejects_latency_resolver() -> None:
    with pytest.raises(ValueError, match="not supported for tasks"):

        @bentoml.task(batchable=True, max_latency_ms=resolver)  # type: ignore[arg-type]
        def predict(_: t.Any, values: list[int]) -> list[int]:
            return values


def test_local_invocation_does_not_resolve_latency() -> None:
    calls = 0

    def counting_resolver(_: Request) -> int:
        nonlocal calls
        calls += 1
        return 25

    @bentoml.service
    class Service:
        @bentoml.api(batchable=True, max_latency_ms=counting_resolver)
        def predict(self, values: list[int]) -> list[int]:
            return values

    assert Service().predict([1]) == [1]
    assert calls == 0
```

- [ ] **Step 2: Run the new test module and verify it fails**

Run:

```bash
pdm run pytest tests/unit/_bentoml_sdk/test_max_latency.py -q
```

Expected: failures for invalid values, callable scope, and async resolver rejection because `APIMethod` currently accepts them.

- [ ] **Step 3: Add the shared type, default, and validators**

In `src/_bentoml_sdk/method.py`, add the imports and definitions near the existing type variables:

```python
from starlette.requests import Request

from bentoml._internal.utils import is_async_callable

DEFAULT_MAX_LATENCY_MS = 60_000
LatencyResolver = t.Callable[[Request], int]
# This assignment is evaluated at runtime, so use Union for Python 3.9 support.
MaxLatency = t.Union[int, LatencyResolver]


def validate_max_latency_value(value: t.Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("max_latency_ms must resolve to a positive integer")
    return value
```

Change the `APIMethod` field and extend `__attrs_post_init__()`:

```python
    max_latency_ms: MaxLatency = DEFAULT_MAX_LATENCY_MS

    def __attrs_post_init__(self) -> None:
        if self.is_stream and not self.output_spec.media_type:
            self.output_spec.media_type = DEFAULT_STREAM_MEDIA_TYPE

        configured = self.max_latency_ms
        if type(configured) is int:
            try:
                validate_max_latency_value(configured)
            except ValueError as exc:
                raise ValueError(
                    "max_latency_ms must be a positive integer or a synchronous callable"
                ) from exc
            return
        if not callable(configured):
            raise ValueError(
                "max_latency_ms must be a positive integer or a synchronous callable"
            )
        if self.is_task:
            raise ValueError("callable max_latency_ms is not supported for tasks")
        if not self.batchable:
            raise ValueError("callable max_latency_ms requires batchable=True")
        if is_async_callable(configured):
            raise ValueError("max_latency_ms resolver must be synchronous")
```

- [ ] **Step 4: Apply the union only to `@bentoml.api`**

In `src/_bentoml_sdk/decorators.py`, import the shared names:

```python
from .method import APIMethod
from .method import DEFAULT_MAX_LATENCY_MS
from .method import MaxLatency
```

Change both `api()` signatures and its default:

```python
    max_latency_ms: MaxLatency = ...,
```

and:

```python
    max_latency_ms: MaxLatency = DEFAULT_MAX_LATENCY_MS,
```

Keep both `task()` signatures typed as `int`, changing only their literal default to the shared constant:

```python
    max_latency_ms: int = DEFAULT_MAX_LATENCY_MS,
```

Update the `api()` docstring entry to say:

```python
        max_latency_ms: A positive static latency in milliseconds, or a synchronous
            request resolver returning a positive latency in milliseconds.
```

- [ ] **Step 5: Run SDK tests and type checking for the touched files**

Run:

```bash
pdm run pytest tests/unit/_bentoml_sdk/test_max_latency.py -q
pdm run pyright src/_bentoml_sdk/method.py src/_bentoml_sdk/decorators.py tests/unit/_bentoml_sdk/test_max_latency.py
```

Expected: both commands pass.

- [ ] **Step 6: Commit the SDK contract**

Before staging, invoke `superpowers:safe-pilot-committing`. Then commit:

```bash
git add src/_bentoml_sdk/method.py src/_bentoml_sdk/decorators.py tests/unit/_bentoml_sdk/test_max_latency.py
git commit -m "feat(sdk): accept request latency resolvers"
```

## Task 2: Add job deadlines and monotonic dispatcher plumbing

**Files:**

- Create: `tests/unit/_internal/marshal/test_dispatcher.py`
- Modify: `src/bentoml/_internal/marshal/dispatcher.py:99-198,327-371,400-438`

- [ ] **Step 1: Write deadline, default, clock, and split tests**

Create `tests/unit/_internal/marshal/test_dispatcher.py`:

```python
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

    task = asyncio.create_task(
        dispatcher.inbound_call("request", max_latency_ms=25)
    )
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

    async def fake_inbound(
        data: str, *, max_latency_ms: int | None = None
    ) -> str:
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
```

- [ ] **Step 2: Run the dispatcher tests and verify they fail**

Run:

```bash
pdm run pytest tests/unit/_internal/marshal/test_dispatcher.py -q
```

Expected: collection or construction failures because `Job.deadline`, the `clock` constructor argument, and per-call latency do not exist.

- [ ] **Step 3: Add the monotonic clock and deadline field**

In `src/bentoml/_internal/marshal/dispatcher.py`, change `Job` and the dispatcher constructor:

```python
@attr.define
class Job(t.Generic[T_IN, T_OUT]):
    enqueue_time: float
    data: T_IN
    future: asyncio.Future[T_OUT | Exception]
    deadline: float
    dispatch_time: float = 0
```

Add the keyword argument and state:

```python
        clock: t.Callable[[], float] = time.monotonic,
```

```python
        self._clock = clock
```

Replace every `time.time()` used for enqueue, wait, start, or duration measurement in this file with `self._clock()`.

- [ ] **Step 4: Add optional per-call latency without breaking static callers**

Change the wrapper and inbound method. Because the returned wrapper now accepts a
keyword-only argument, also widen `__call__()`'s return annotation from
`Callable[[T_IN], ...]` to `Callable[..., Coroutine[...]]`:

```python
    def __call__(
        self,
        callback: t.Callable[
            [t.Sequence[T_IN]], t.Coroutine[None, None, t.Sequence[T_OUT]]
        ],
    ) -> t.Callable[..., t.Coroutine[None, None, T_OUT | None]]:
```

```python
        @functools.wraps(callback)
        async def _func(
            data: T_IN, *, max_latency_ms: int | None = None
        ) -> T_OUT:
            if self._controller is None:
                self._controller = self._loop.create_task(self.controller())
            try:
                r = await self.inbound_call(data, max_latency_ms=max_latency_ms)
            except asyncio.CancelledError:
                return self.fallback()
            if isinstance(r, Exception):
                raise r
            return r
```

```python
    async def inbound_call(
        self, data: T_IN, *, max_latency_ms: int | None = None
    ) -> T_OUT | Exception:
        now = self._clock()
        latency = (
            self.max_latency
            if max_latency_ms is None
            else max_latency_ms / 1000.0
        )
        future: asyncio.Future[T_OUT | Exception] = self._loop.create_future()
        input_info = Job(
            enqueue_time=now,
            data=data,
            future=future,
            deadline=now + latency,
        )
        self._queue.append(input_info)
        async with self._wake_event:
            self._wake_event.notify_all()
        return await future
```

Keep `max_latency_ms=None` as the path used by the legacy Runner server.

- [ ] **Step 5: Run the focused tests**

Run:

```bash
pdm run pytest tests/unit/_internal/marshal/test_dispatcher.py -q
```

Expected: all four tests pass.

- [ ] **Step 6: Commit deadline plumbing**

Invoke `superpowers:safe-pilot-committing`, then commit:

```bash
git add src/bentoml/_internal/marshal/dispatcher.py tests/unit/_internal/marshal/test_dispatcher.py
git commit -m "feat(batch): add per-job latency deadlines"
```

## Task 3: Add candidate inspection, cache invalidation, and independent expiration

**Files:**

- Modify: `src/bentoml/_internal/marshal/dispatcher.py:99-438`
- Modify: `tests/unit/_internal/marshal/test_dispatcher.py`

- [ ] **Step 1: Add failing candidate and expiration tests**

Append to `tests/unit/_internal/marshal/test_dispatcher.py`:

```python
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
```

- [ ] **Step 2: Run the new cases and verify missing-helper failures**

Run:

```bash
pdm run pytest tests/unit/_internal/marshal/test_dispatcher.py -q
```

Expected: failures because `_append_job`, `_peek_candidate`, `_expire_jobs`, and candidate metadata do not exist.

- [ ] **Step 3: Add candidate metadata and queue-cache state**

Add next to `Job` in `dispatcher.py`:

```python
@attr.frozen
class CandidateInfo:
    job_count: int
    sample_size: int
    first_enqueue_time: float
    last_enqueue_time: float
    earliest_deadline: float
```

Initialize these fields in `CorkDispatcher.__init__()`:

```python
        self._queue_revision = 0
        self._candidate_cache: tuple[int, CandidateInfo] | None = None
        self._next_expiry: float | None = None
```

Add queue helpers:

```python
    def _invalidate_queue_state(self, *, recompute_expiry: bool) -> None:
        self._queue_revision += 1
        self._candidate_cache = None
        if recompute_expiry:
            self._next_expiry = min(
                (job.deadline for job in self._queue), default=None
            )

    def _append_job(self, job: Job[T_IN, T_OUT]) -> None:
        self._queue.append(job)
        self._invalidate_queue_state(recompute_expiry=False)
        if self._next_expiry is None or job.deadline < self._next_expiry:
            self._next_expiry = job.deadline

    def _get_batch_size(self, data: T_IN) -> int:
        try:
            return self.get_batch_size(data)
        except Exception:
            logger.exception(
                "error in batch-size aware batching, falling back to regular batching method"
            )
            return 1
```

Update `inbound_call()` to use `_append_job(input_info)` instead of appending directly.

- [ ] **Step 4: Implement non-mutating candidate inspection**

Add:

```python
    def _peek_candidate(self, num_batches: int | None = None) -> CandidateInfo | None:
        use_cache = num_batches is None
        if use_cache and self._candidate_cache is not None:
            revision, candidate = self._candidate_cache
            if revision == self._queue_revision:
                return candidate

        limit = self.max_batch_size if num_batches is None else num_batches
        sample_size = 0
        job_count = 0
        first_enqueue_time = 0.0
        last_enqueue_time = 0.0
        earliest_deadline = float("inf")

        for job in self._queue:
            remaining = limit - sample_size
            if remaining <= 0:
                break
            batch_size = self._get_batch_size(job.data)
            if job_count == 0:
                first_enqueue_time = job.enqueue_time
            job_count += 1
            sample_size += min(batch_size, remaining)
            last_enqueue_time = job.enqueue_time
            earliest_deadline = min(earliest_deadline, job.deadline)
            if batch_size >= remaining:
                break

        if job_count == 0:
            return None

        candidate = CandidateInfo(
            job_count=job_count,
            sample_size=sample_size,
            first_enqueue_time=first_enqueue_time,
            last_enqueue_time=last_enqueue_time,
            earliest_deadline=earliest_deadline,
        )
        if use_cache:
            self._candidate_cache = (self._queue_revision, candidate)
        return candidate
```

Use `_get_batch_size()` inside `_get_inputs()` to keep peek and dequeue behavior aligned.

- [ ] **Step 5: Add stable expiration and cache-aware dequeue**

Add:

```python
    def _expire_jobs(self, now: float) -> int:
        if self._next_expiry is None or now < self._next_expiry:
            return 0

        surviving: collections.deque[Job[T_IN, T_OUT]] = collections.deque()
        expired: list[Job[T_IN, T_OUT]] = []
        for job in self._queue:
            if job.deadline <= now:
                expired.append(job)
            else:
                surviving.append(job)

        self._queue = surviving
        self._invalidate_queue_state(recompute_expiry=True)
        for job in expired:
            if not job.future.done():
                job.future.cancel()
        return len(expired)

    def _take_inputs(
        self, num_batches: int | None = None
    ) -> tuple[Job[T_IN, T_OUT], ...]:
        inputs = tuple(self._get_inputs(num_batches))
        if inputs:
            self._invalidate_queue_state(recompute_expiry=True)
        return inputs
```

Replace both `tuple(self._get_inputs(...))` call sites with `_take_inputs(...)`.

- [ ] **Step 6: Run dispatcher tests**

Run:

```bash
pdm run pytest tests/unit/_internal/marshal/test_dispatcher.py -q
```

Expected: all candidate, cache, expiration, deadline, and split tests pass.

- [ ] **Step 7: Commit candidate and expiry support**

Invoke `superpowers:safe-pilot-committing`, then commit:

```bash
git add src/bentoml/_internal/marshal/dispatcher.py tests/unit/_internal/marshal/test_dispatcher.py
git commit -m "feat(batch): track candidate deadlines"
```

## Task 4: Make controller and optimizer training deadline-aware

**Files:**

- Modify: `src/bentoml/_internal/marshal/dispatcher.py:198-326`
- Modify: `tests/unit/_internal/marshal/test_dispatcher.py`

- [ ] **Step 1: Add failing pure scheduling tests**

Append these imports and tests to `test_dispatcher.py`:

```python
from bentoml._internal.marshal.dispatcher import CandidateInfo
from bentoml._internal.marshal.dispatcher import ScheduleDecision
from bentoml._internal.marshal.dispatcher import decide_next_action


def candidate(deadline: float = 110.0) -> CandidateInfo:
    return CandidateInfo(
        job_count=2,
        sample_size=2,
        first_enqueue_time=100.0,
        last_enqueue_time=101.0,
        earliest_deadline=deadline,
    )


def test_decision_waits_when_optimizer_and_deadline_allow() -> None:
    decision = decide_next_action(
        now=105.0,
        candidate=candidate(),
        predicted_duration=2.0,
        adaptive_should_wait=True,
        semaphore_locked=False,
        next_expiry=110.0,
        ticket_interval=1.0,
    )
    assert decision == ScheduleDecision(dispatch=False, wait_time=1.0)


def test_decision_caps_wait_at_latest_safe_start() -> None:
    decision = decide_next_action(
        now=107.5,
        candidate=candidate(),
        predicted_duration=2.0,
        adaptive_should_wait=True,
        semaphore_locked=False,
        next_expiry=110.0,
        ticket_interval=1.0,
    )
    assert decision == ScheduleDecision(dispatch=False, wait_time=0.5)


def test_decision_dispatches_at_latest_safe_start() -> None:
    decision = decide_next_action(
        now=108.0,
        candidate=candidate(),
        predicted_duration=2.0,
        adaptive_should_wait=True,
        semaphore_locked=False,
        next_expiry=110.0,
        ticket_interval=1.0,
    )
    assert decision == ScheduleDecision(dispatch=True, wait_time=0.0)


def test_impossible_deadline_dispatches_when_capacity_is_free() -> None:
    decision = decide_next_action(
        now=105.0,
        candidate=candidate(deadline=106.0),
        predicted_duration=2.0,
        adaptive_should_wait=True,
        semaphore_locked=False,
        next_expiry=106.0,
        ticket_interval=1.0,
    )
    assert decision.dispatch


def test_locked_semaphore_waits_until_ticket_or_expiry() -> None:
    decision = decide_next_action(
        now=105.0,
        candidate=candidate(deadline=106.0),
        predicted_duration=2.0,
        adaptive_should_wait=False,
        semaphore_locked=True,
        next_expiry=105.25,
        ticket_interval=1.0,
    )
    assert decision == ScheduleDecision(dispatch=False, wait_time=0.25)


def test_optimizer_can_release_before_deadline() -> None:
    decision = decide_next_action(
        now=105.0,
        candidate=candidate(),
        predicted_duration=2.0,
        adaptive_should_wait=False,
        semaphore_locked=False,
        next_expiry=110.0,
        ticket_interval=1.0,
    )
    assert decision.dispatch
```

- [ ] **Step 2: Run the scheduling tests and verify import failures**

Run:

```bash
pdm run pytest tests/unit/_internal/marshal/test_dispatcher.py -q
```

Expected: import errors for `ScheduleDecision` and `decide_next_action`.

- [ ] **Step 3: Implement the pure decision helper**

Add beside `CandidateInfo`:

```python
@attr.frozen
class ScheduleDecision:
    dispatch: bool
    wait_time: float


def decide_next_action(
    *,
    now: float,
    candidate: CandidateInfo,
    predicted_duration: float,
    adaptive_should_wait: bool,
    semaphore_locked: bool,
    next_expiry: float | None,
    ticket_interval: float,
) -> ScheduleDecision:
    if semaphore_locked:
        wake_at = now + ticket_interval
        if next_expiry is not None:
            wake_at = min(wake_at, next_expiry)
        return ScheduleDecision(False, max(0.0, wake_at - now))

    latest_safe_start = candidate.earliest_deadline - predicted_duration
    if now >= latest_safe_start or not adaptive_should_wait:
        return ScheduleDecision(True, 0.0)

    wake_at = min(now + ticket_interval, latest_safe_start)
    if next_expiry is not None:
        wake_at = min(wake_at, next_expiry)
    return ScheduleDecision(False, max(0.0, wake_at - now))
```

- [ ] **Step 4: Replace the normal controller's global timeout logic**

Inside the controller loop:

1. Read `now = self._clock()`.
2. Call `_expire_jobs(now)` before candidate selection.
3. Continue to the condition wait if the queue became empty.
4. Read `_peek_candidate()`.
5. Compute `predicted_duration` from `candidate.job_count`.
6. Compute the existing adaptive heuristic from candidate sample and job counts.
7. Call `decide_next_action()`.
8. Sleep for `decision.wait_time` or dispatch `_take_inputs()`.

Use this complete loop body:

```python
        while True:
            try:
                async with self._wake_event:
                    await self._wake_event.wait_for(self._queue.__len__)

                now = self._clock()
                self._expire_jobs(now)
                if not self._queue:
                    continue

                candidate = self._peek_candidate()
                assert candidate is not None
                a = self.optimizer.o_a
                b = self.optimizer.o_b
                predicted_duration = a * candidate.job_count + b
                newest_wait = now - candidate.last_enqueue_time
                adaptive_should_wait = (
                    candidate.sample_size < self.max_batch_size
                    and candidate.job_count
                    * (newest_wait + self.TICKET_INTERVAL + (a or 0))
                    <= self.optimizer.wait * 0.95
                )
                decision = decide_next_action(
                    now=now,
                    candidate=candidate,
                    predicted_duration=predicted_duration,
                    adaptive_should_wait=adaptive_should_wait,
                    semaphore_locked=self._sema.is_locked(),
                    next_expiry=self._next_expiry,
                    ticket_interval=self.TICKET_INTERVAL,
                )
                if not decision.dispatch:
                    await asyncio.sleep(decision.wait_time)
                    continue

                acquired = self._sema.acquire()
                assert acquired
                inputs_info = self._take_inputs()
                self._loop.create_task(self.outbound_call(inputs_info))
            except Exception:
                logger.exception("Error processing batch requests")
```

Delete the old `latency_0`, global cancellation, and queue-length duration branches.

- [ ] **Step 5: Apply the same deadline rules to optimizer training**

Change `train_optimizer()` to remove `num_required_reqs`; update the four calls in `controller()` accordingly. Use this method:

```python
    async def train_optimizer(
        self,
        num_reqs_to_train: int,
        training_batch_size: int,
    ) -> None:
        target_job_count = min(self.max_batch_size, training_batch_size)
        req_count = 0
        try:
            while req_count < num_reqs_to_train:
                async with self._wake_event:
                    await self._wake_event.wait_for(self._queue.__len__)

                now = self._clock()
                self._expire_jobs(now)
                if not self._queue:
                    continue

                # Training still observes the actual next outbound batch. Do not
                # change the optimizer from its existing job-count model.
                candidate = self._peek_candidate()
                assert candidate is not None
                a = self.optimizer.o_a
                b = self.optimizer.o_b
                predicted_duration = a * candidate.job_count + b
                oldest_wait = now - candidate.first_enqueue_time
                training_wait = (target_job_count * 2 + 1) * a + b
                adaptive_should_wait = (
                    candidate.job_count < target_job_count
                    and (target_job_count * a + b) + oldest_wait
                    <= training_wait
                )
                decision = decide_next_action(
                    now=now,
                    candidate=candidate,
                    predicted_duration=predicted_duration,
                    adaptive_should_wait=adaptive_should_wait,
                    semaphore_locked=self._sema.is_locked(),
                    next_expiry=self._next_expiry,
                    ticket_interval=self.TICKET_INTERVAL,
                )
                if not decision.dispatch:
                    await asyncio.sleep(decision.wait_time)
                    continue

                acquired = self._sema.acquire()
                assert acquired
                inputs_info = self._take_inputs()
                req_count += 1
                self._loop.create_task(
                    self.outbound_call(inputs_info, training=True)
                )
        except Exception:
            logger.exception("Error in training optimizer")
```

Update the calls to:

```python
            await self.train_optimizer(self.optimizer.N_SKIPPED_SAMPLE + 6, 1)
            await self.train_optimizer(1, 1)
            await self.train_optimizer(1, 2)
            await self.train_optimizer(1, 3)
```

- [ ] **Step 6: Add mixed-deadline, fallback, and result-mapping regressions**

Append:

```python
def test_late_tight_deadline_changes_release_without_reordering() -> None:
    clock = FakeClock(105.0)
    dispatcher = make_dispatcher(clock, max_batch_size=4)
    for name, deadline in [("A", 200.0), ("B", 201.0)]:
        dispatcher._append_job(
            queued_job(
                dispatcher,
                data=(name, 1),
                enqueue_time=100.0,
                deadline=deadline,
            )
        )

    before = dispatcher._peek_candidate()
    assert before is not None
    assert not decide_next_action(
        now=clock(),
        candidate=before,
        predicted_duration=0.002,
        adaptive_should_wait=True,
        semaphore_locked=False,
        next_expiry=dispatcher._next_expiry,
        ticket_interval=0.001,
    ).dispatch

    dispatcher._append_job(
        queued_job(
            dispatcher,
            data=("C", 1),
            enqueue_time=105.0,
            deadline=105.001,
        )
    )
    after = dispatcher._peek_candidate()
    assert after is not None
    assert decide_next_action(
        now=clock(),
        candidate=after,
        predicted_duration=0.002,
        adaptive_should_wait=True,
        semaphore_locked=False,
        next_expiry=dispatcher._next_expiry,
        ticket_interval=0.001,
    ).dispatch
    assert [job.data[0] for job in dispatcher._take_inputs()] == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_expired_job_uses_existing_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock, max_latency_in_ms=10)

    async def fake_controller() -> None:
        await asyncio.Future()

    monkeypatch.setattr(dispatcher, "controller", fake_controller)

    async def callback(values: t.Sequence[str]) -> t.Sequence[str]:
        return values

    wrapped = dispatcher(callback)
    task = asyncio.create_task(wrapped("expires"))
    await asyncio.sleep(0)
    try:
        clock.advance(0.011)
        assert dispatcher._expire_jobs(clock()) == 1
        assert await task == "fallback"
    finally:
        dispatcher.shutdown()


@pytest.mark.asyncio
async def test_cancelled_future_does_not_shift_other_results() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock)
    jobs = [
        queued_job(
            dispatcher,
            data=(name, 1),
            enqueue_time=100.0,
            deadline=101.0,
        )
        for name in ("A", "B", "C")
    ]
    for job in jobs:
        dispatcher._append_job(job)

    async def callback(
        values: t.Sequence[tuple[str, int]],
    ) -> t.Sequence[tuple[str, int]]:
        return [(f"{name}-result", size) for name, size in values]

    dispatcher.callback = callback
    assert dispatcher._sema.acquire()
    inputs = dispatcher._take_inputs()
    jobs[1].future.cancel()

    await dispatcher.outbound_call(inputs)

    assert jobs[0].future.result() == ("A-result", 1)
    assert jobs[1].future.cancelled()
    assert jobs[2].future.result() == ("C-result", 1)
```

This covers a late tight request invalidating the candidate cache and changing
the release decision, FIFO membership of the shared batch, the existing
fallback path, and stable output-to-future mapping when one caller cancels.

- [ ] **Step 7: Add a legacy default regression test**

Append:

```python
@pytest.mark.asyncio
async def test_static_wrapper_keeps_constructor_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock, max_latency_in_ms=40)

    async def fake_controller() -> None:
        await asyncio.Future()

    monkeypatch.setattr(dispatcher, "controller", fake_controller)

    async def callback(values: t.Sequence[str]) -> t.Sequence[str]:
        return values

    wrapped = dispatcher(callback)
    task = asyncio.create_task(wrapped("legacy"))
    await asyncio.sleep(0)
    try:
        assert dispatcher._queue[0].deadline == pytest.approx(100.04)
        dispatcher._queue[0].future.set_result("legacy")
        assert await task == "legacy"
    finally:
        dispatcher.shutdown()
```

- [ ] **Step 8: Run dispatcher tests and type checks**

Run:

```bash
pdm run pytest tests/unit/_internal/marshal/test_dispatcher.py -q
pdm run pyright src/bentoml/_internal/marshal/dispatcher.py tests/unit/_internal/marshal/test_dispatcher.py
```

Expected: all tests and type checks pass.

- [ ] **Step 9: Commit controller behavior**

Invoke `superpowers:safe-pilot-committing`, then commit:

```bash
git add src/bentoml/_internal/marshal/dispatcher.py tests/unit/_internal/marshal/test_dispatcher.py
git commit -m "feat(batch): enforce per-job wait limits"
```

## Task 5: Resolve request latency in the new-style server

**Files:**

- Modify: `src/_bentoml_impl/server/app.py:117-135,668-713,760-806`
- Modify: `tests/unit/_bentoml_sdk/test_max_latency.py`

- [ ] **Step 1: Add failing server-resolution tests**

Append to `tests/unit/_bentoml_sdk/test_max_latency.py`:

```python
from unittest.mock import AsyncMock

from starlette.testclient import TestClient

from _bentoml_impl.server.app import ServiceAppFactory


def test_http_resolver_runs_once_and_reaches_batch_infer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[Request] = []
    observed: list[int] = []

    def request_latency(request: Request) -> int:
        requests.append(request)
        return int(request.headers["X-Latency"])

    async def fake_batch_infer(
        self: ServiceAppFactory,
        name: str,
        input_args: tuple[t.Any, ...],
        input_kwargs: dict[str, t.Any],
        *,
        max_latency_ms: int,
    ) -> t.Any:
        observed.append(max_latency_ms)
        return input_args[0]

    monkeypatch.setattr(ServiceAppFactory, "batch_infer", fake_batch_infer)

    @bentoml.service(metrics={"enabled": False})
    class Service:
        @bentoml.api(batchable=True, max_latency_ms=request_latency)
        def predict(self, values: list[int]) -> list[int]:
            return values

    with TestClient(Service.to_asgi()) as client:
        response = client.post(
            "/predict",
            json=[1],
            headers={"X-Latency": "17"},
        )

    assert response.status_code == 200
    assert response.json() == [1]
    assert observed == [17]
    assert len(requests) == 1
    assert requests[0].headers["X-Latency"] == "17"


@pytest.mark.parametrize("resolved", [0, -1, 1.5, "10", None, True, False])
def test_invalid_resolver_output_returns_500_without_batching(
    monkeypatch: pytest.MonkeyPatch,
    resolved: object,
) -> None:
    batch_infer = AsyncMock()
    monkeypatch.setattr(ServiceAppFactory, "batch_infer", batch_infer)

    @bentoml.service(metrics={"enabled": False})
    class Service:
        @bentoml.api(batchable=True, max_latency_ms=lambda _: resolved)  # type: ignore[arg-type,return-value]
        def predict(self, values: list[int]) -> list[int]:
            return values

    with TestClient(Service.to_asgi()) as client:
        response = client.post("/predict", json=[1])

    assert response.status_code == 500
    batch_infer.assert_not_awaited()


def test_resolver_exception_returns_500_without_batching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_infer = AsyncMock()
    monkeypatch.setattr(ServiceAppFactory, "batch_infer", batch_infer)

    def broken(_: Request) -> int:
        raise RuntimeError("invalid priority")

    @bentoml.service(metrics={"enabled": False})
    class Service:
        @bentoml.api(batchable=True, max_latency_ms=broken)
        def predict(self, values: list[int]) -> list[int]:
            return values

    with TestClient(Service.to_asgi()) as client:
        response = client.post("/predict", json=[1])

    assert response.status_code == 500
    batch_infer.assert_not_awaited()
```

- [ ] **Step 2: Run the tests and verify the callable breaks dispatcher construction**

Run:

```bash
pdm run pytest tests/unit/_bentoml_sdk/test_max_latency.py -q
```

Expected: HTTP resolver tests fail because `ServiceAppFactory` passes the callable to `CorkDispatcher` and `batch_infer()` has no latency keyword.

- [ ] **Step 3: Seed dynamic dispatchers with the existing default**

In `src/_bentoml_impl/server/app.py`, import:

```python
from _bentoml_sdk.method import DEFAULT_MAX_LATENCY_MS
from _bentoml_sdk.method import validate_max_latency_value
```

Change dispatcher construction:

```python
            configured_latency = method.max_latency_ms
            dispatcher_latency = (
                configured_latency
                if type(configured_latency) is int
                else DEFAULT_MAX_LATENCY_MS
            )
            self.dispatchers[name] = CorkDispatcher(
                max_latency_in_ms=dispatcher_latency,
                max_batch_size=method.max_batch_size,
                fallback=fallback,
                get_batch_size=functools.partial(
                    AutoContainer.get_batch_size, batch_dim=method.batch_dim[0]
                ),
                batch_dim=method.batch_dim,
            )
```

- [ ] **Step 4: Resolve after decoding and pass the integer through `batch_infer()`**

Change the signature:

```python
    async def batch_infer(
        self,
        name: str,
        input_args: tuple[t.Any, ...],
        input_kwargs: dict[str, t.Any],
        *,
        max_latency_ms: int,
    ) -> t.Any:
```

Change the final dispatcher call:

```python
        return await self.dispatchers[name](
            functools.partial(inner_infer, **input_kwargs)
        )(value, max_latency_ms=max_latency_ms)
```

In `api_endpoint()`, immediately before the batchable branch, resolve and validate:

```python
        if method.batchable:
            configured_latency = method.max_latency_ms
            if type(configured_latency) is int:
                resolved_latency = configured_latency
            else:
                resolver = t.cast(
                    t.Callable[[Request], int], configured_latency
                )
                resolved_latency = resolver(request)
            max_latency_ms = validate_max_latency_value(resolved_latency)
            output = await self.batch_infer(
                name,
                call_args,
                call_kwargs,
                max_latency_ms=max_latency_ms,
            )
```

Leave non-batchable branches unchanged. Do not catch resolver exceptions here; `api_endpoint_wrapper()` already maps them through existing error handling.

- [ ] **Step 5: Run server and dispatcher regression tests**

Run:

```bash
pdm run pytest tests/unit/_bentoml_sdk/test_max_latency.py tests/unit/_internal/marshal/test_dispatcher.py -q
pdm run pytest tests/unit/bentoml_io/test_decorators.py tests/e2e/bento_new_sdk/test_asgi.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit server resolution**

Invoke `superpowers:safe-pilot-committing`, then commit:

```bash
git add src/_bentoml_impl/server/app.py tests/unit/_bentoml_sdk/test_max_latency.py
git commit -m "feat(server): resolve request latency budgets"
```

## Task 6: Add packaged HTTP integration coverage

**Files:**

- Create: `tests/e2e/fixtures/dynamic_batching/service.py`
- Create: `tests/e2e/bento_new_sdk/test_dynamic_batching.py`

- [ ] **Step 1: Create a dependency-free batching fixture**

Create `tests/e2e/fixtures/dynamic_batching/service.py`:

```python
from __future__ import annotations

import threading
from typing import Any

import bentoml
from starlette.requests import Request

RESOLVED_LATENCIES: list[int] = []
LOCK = threading.Lock()


def resolve_latency(request: Request) -> int:
    value = int(request.headers.get("X-Latency", "100"))
    with LOCK:
        RESOLVED_LATENCIES.append(value)
    return value


@bentoml.service(metrics={"enabled": False}, workers=1)
class DynamicBatching:
    def __init__(self) -> None:
        self.batches: list[list[int]] = []

    @bentoml.api(
        batchable=True,
        max_batch_size=8,
        max_latency_ms=resolve_latency,
    )
    def predict(self, values: list[int]) -> list[int]:
        with LOCK:
            self.batches.append(list(values))
        return values

    @bentoml.api
    def reset(self) -> bool:
        with LOCK:
            self.batches.clear()
            RESOLVED_LATENCIES.clear()
        return True

    @bentoml.api
    def stats(self) -> dict[str, Any]:
        with LOCK:
            return {
                "batches": list(self.batches),
                "latencies": list(RESOLVED_LATENCIES),
            }
```

- [ ] **Step 2: Write the packaged concurrent HTTP test**

Create `tests/e2e/bento_new_sdk/test_dynamic_batching.py`:

```python
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bentoml
import httpx

PORT = 35681


def post_prediction(
    base_url: str,
    value: int,
    latency: int,
) -> list[int]:
    with httpx.Client(base_url=base_url, timeout=30) as client:
        response = client.post(
            "/predict",
            json=[value],
            headers={"X-Latency": str(latency)},
        )
        response.raise_for_status()
        return response.json()


def test_dynamic_latency_survives_build_and_shares_batches(examples: Path) -> None:
    fixture = examples / "dynamic_batching"
    bento = bentoml.build(
        "service.py:DynamicBatching",
        build_ctx=str(fixture),
    )

    with bentoml.serve(bento, port=PORT) as server:
        with bentoml.SyncHTTPClient(
            server.url,
            server_ready_timeout=100,
        ) as client:
            with ThreadPoolExecutor(max_workers=16) as executor:
                warmup = [
                    executor.submit(post_prediction, server.url, value, 500)
                    for value in range(16)
                ]
                assert [future.result() for future in warmup] == [
                    [value] for value in range(16)
                ]

            client.reset()

            values = [100, 101, 102, 103]
            latencies = [500, 500, 50, 500]
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(
                        post_prediction, server.url, value, latency
                    )
                    for value, latency in zip(values, latencies)
                ]
                assert sorted(future.result()[0] for future in futures) == values

            stats = client.stats()
            assert sorted(stats["latencies"]) == sorted(latencies)
            assert any(
                len(set(values).intersection(batch)) >= 2
                for batch in stats["batches"]
            )
```

The warmup requests satisfy the dispatcher's existing optimizer-training phase. Each worker creates its own HTTP client, avoiding undocumented cross-thread client sharing. The assertion proves that requests with different resolved budgets can share a model batch without requiring all four requests to arrive within a flaky wall-clock window; exact release timing and full four-job batching are covered by deterministic unit tests.

- [ ] **Step 3: Run the packaged end-to-end test**

Run:

```bash
pdm run pytest tests/e2e/bento_new_sdk/test_dynamic_batching.py -q
```

Expected: the Bento builds, the service starts, every response succeeds, the resolver records all four requested values, and at least two mixed-budget requests appear in one model batch.

- [ ] **Step 4: Run the whole new-SDK end-to-end suite**

Run:

```bash
pdm run pytest tests/e2e/bento_new_sdk -q
```

Expected: all new-SDK end-to-end tests pass.

- [ ] **Step 5: Commit integration coverage**

Invoke `superpowers:safe-pilot-committing`, then commit:

```bash
git add tests/e2e/fixtures/dynamic_batching/service.py tests/e2e/bento_new_sdk/test_dynamic_batching.py
git commit -m "test(batch): cover packaged dynamic latency"
```

## Task 7: Document dynamic request latency

**Files:**

- Modify: `docs/source/get-started/adaptive-batching.rst:90-125`

- [ ] **Step 1: Add the dynamic form after the adaptive batching parameter list**

In `docs/source/get-started/adaptive-batching.rst`, replace the `max_latency_ms` bullet and following paragraph with:

```rst
- ``max_latency_ms``: A positive latency target in milliseconds. It can be a
  static integer for all requests or a synchronous callable that receives the
  incoming Starlette ``Request`` and returns a positive integer for that request.

With a static value, every request submitted to the API uses the same target:

.. code-block:: python

    @bentoml.api(
        batchable=True,
        max_batch_size=32,
        max_latency_ms=50,
    )
    def predict(self, inputs: list[str]) -> list[str]:
        return self.model(inputs)

For request-specific targets, provide a synchronous resolver:

.. code-block:: python

    from starlette.requests import Request


    def latency_budget(request: Request) -> int:
        priority = request.headers.get("X-Priority")
        return {
            "realtime": 5,
            "interactive": 20,
            "background": 100,
        }.get(priority, 20)


    @bentoml.api(
        batchable=True,
        max_batch_size=32,
        max_latency_ms=latency_budget,
    )
    def predict(self, inputs: list[str]) -> list[str]:
        return self.model(inputs)

Requests with different latency budgets still share one adaptive batching queue
within each Service worker. A tighter deadline in the next FIFO candidate can
release the batch earlier, allowing looser-deadline requests already in that
candidate to use the same model invocation. The latency changes when BentoML
releases work; it does not give the request priority or move it ahead of earlier
requests.

The resolver runs once after request decoding and before queueing. It must be
synchronous and return a positive integer without coercion. Dynamic resolution
applies to request-backed ``@bentoml.api`` calls. Local Python invocation,
``@bentoml.task``, legacy Runner configuration, and legacy gRPC APIs do not use
the resolver.

.. warning::

    Treat client-provided latency metadata as untrusted input. Authenticate and
    clamp requested values so a client cannot force arbitrarily small batches.

    .. code-block:: python

        def latency_budget(request: Request) -> int:
            requested = int(request.headers.get("X-Latency", "50"))
            return min(100, max(5, requested))

The adaptive batching algorithm continues to optimize throughput within the
configured batch-size limit. Predicted execution time caps additional waiting,
but BentoML does not reject an immediately dispatchable request solely because
the prediction indicates that its target may be missed.
```

- [ ] **Step 2: Update the error-handling wording**

Replace the final error-handling paragraph with:

```rst
If a queued request reaches its resolved ``max_latency_ms`` before dispatch,
BentoML returns HTTP 503 Service Unavailable through the existing overload
fallback. Increase the latency target or improve available capacity if this
occurs regularly. Once a batch has been dispatched, BentoML does not preempt the
model invocation if an estimate proves inaccurate.
```

- [ ] **Step 3: Build the documentation page**

Run:

```bash
pdm run sphinx-build -W -b html docs/source docs/build/html
```

Expected: Sphinx exits successfully with no warnings for the updated page.

- [ ] **Step 4: Commit documentation**

Invoke `superpowers:safe-pilot-committing`, then commit:

```bash
git add docs/source/get-started/adaptive-batching.rst
git commit -m "docs: explain request-specific batch latency"
```

## Task 8: Run final verification and capture regression evidence

**Files:**

- Verify all files changed by Tasks 1-7.
- Do not create benchmark output, coverage HTML, or log files in the repository.

- [ ] **Step 1: Run the focused feature tests**

Run:

```bash
pdm run pytest \
  tests/unit/_bentoml_sdk/test_max_latency.py \
  tests/unit/_internal/marshal/test_dispatcher.py \
  tests/unit/bentoml_io/test_decorators.py \
  tests/e2e/bento_new_sdk/test_asgi.py \
  tests/e2e/bento_new_sdk/test_dynamic_batching.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run static analysis and repository formatting**

Run:

```bash
make type
pre-commit run --all-files
```

Expected: both commands complete successfully. If formatting modifies files, inspect the diff, rerun the focused tests, and commit only the related formatting with the feature commit it belongs to.

- [ ] **Step 3: Run the broader test suite**

Run:

```bash
pdm run nox -s unit-3.12
pdm run nox -s "e2e-testing-3.12(suite='bento_new_sdk')"
```

Expected: the unit and new-SDK end-to-end sessions pass.

- [ ] **Step 4: Compare static-path controller overhead**

Temporarily check out `main` in a separate worktree:

```bash
git worktree add ../bentoml-main-benchmark main
git -C ../bentoml-main-benchmark status --short
```

Create `/tmp/bentoml_static_dispatch_benchmark.py` outside the repository with
this exact static-path benchmark:

```python
from __future__ import annotations

import asyncio
import statistics
import time
import typing as t

from bentoml._internal.marshal.dispatcher import CorkDispatcher


async def measure(requests: int) -> float:
    dispatcher: CorkDispatcher[int, int] = CorkDispatcher(
        max_latency_in_ms=60_000,
        max_batch_size=1,
        get_batch_size=lambda _: 1,
    )

    async def callback(values: t.Sequence[int]) -> t.Sequence[int]:
        return values

    infer = dispatcher(callback)
    started = time.perf_counter()
    try:
        for value in range(requests):
            assert await infer(value) == value
    finally:
        dispatcher.shutdown()
    return time.perf_counter() - started


async def main() -> None:
    samples = [await measure(2_000) for _ in range(7)]
    print(f"median_seconds={statistics.median(samples):.6f}")


asyncio.run(main())
```

Run the same interpreter and script against each source tree, five times each
to distinguish repeatable changes from host noise:

```bash
for run in 1 2 3 4 5; do
  PYTHONPATH="$PWD/src" .venv/bin/python /tmp/bentoml_static_dispatch_benchmark.py
done
for run in 1 2 3 4 5; do
  PYTHONPATH="$PWD/../bentoml-main-benchmark/src" .venv/bin/python /tmp/bentoml_static_dispatch_benchmark.py
done
```

Record the ten printed medians outside the repository in the PR description.
Do not commit the script or generated benchmark data. Remove the temporary
worktree and script after recording results:

```bash
git worktree remove ../bentoml-main-benchmark
rm /tmp/bentoml_static_dispatch_benchmark.py
```

Acceptance: no material static-path regression. Treat a repeatable regression above 5 percent as a reason to profile candidate-cache invalidation before requesting review.

- [ ] **Step 5: Inspect the final diff and commit any verification-only fixes**

Invoke `superpowers:safe-pilot-committing` before staging. Inspect:

```bash
git status --short
git diff --check
git diff main...HEAD --stat
git log --oneline main..HEAD
```

Expected: no secrets, logs, build output, temporary worktrees, or unrelated changes. If verification required a code or test fix, commit it using the smallest relevant conventional commit rather than a generic cleanup commit.

- [ ] **Step 6: Request code review**

Invoke `superpowers:requesting-code-review` against `main`. Address any correctness findings, rerun the affected focused tests, and then use `superpowers:finishing-a-development-branch` to choose whether to push and open the upstream pull request.
