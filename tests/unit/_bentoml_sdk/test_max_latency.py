from __future__ import annotations

import functools

import pytest
from starlette.requests import Request

import bentoml


def test_api_accepts_positive_integer_max_latency() -> None:
    @bentoml.api(batchable=True, max_latency_ms=25)
    def predict(_, values: list[int]) -> list[int]:
        return values

    assert predict.max_latency_ms == 25


def test_batchable_api_accepts_synchronous_latency_resolver() -> None:
    def resolver(request: Request) -> int:
        return 25

    @bentoml.api(batchable=True, max_latency_ms=resolver)
    def predict(_, values: list[int]) -> list[int]:
        return values

    assert predict.max_latency_ms is resolver


@pytest.mark.parametrize("max_latency_ms", [0, -1, 1.5, "10", None, True, False])
def test_api_rejects_invalid_max_latency(max_latency_ms: object) -> None:
    with pytest.raises(ValueError, match="positive integer or a synchronous callable"):

        @bentoml.api(batchable=True, max_latency_ms=max_latency_ms)  # type: ignore[arg-type]
        def predict(_, values: list[int]) -> list[int]:
            return values


def test_non_batchable_api_rejects_latency_resolver() -> None:
    def resolver(request: Request) -> int:
        return 25

    with pytest.raises(ValueError, match="requires batchable=True"):

        @bentoml.api(max_latency_ms=resolver)
        def predict(_, value: int) -> int:
            return value


async def async_resolver(request: Request) -> int:
    return 25


class AsyncResolver:
    async def __call__(self, request: Request) -> int:
        return 25


@pytest.mark.parametrize(
    "resolver",
    [async_resolver, functools.partial(async_resolver), AsyncResolver()],
)
def test_api_rejects_async_latency_resolver(resolver: object) -> None:
    with pytest.raises(ValueError, match="must be synchronous"):

        @bentoml.api(batchable=True, max_latency_ms=resolver)  # type: ignore[arg-type]
        def predict(_, values: list[int]) -> list[int]:
            return values


def test_task_rejects_latency_resolver() -> None:
    def resolver(request: Request) -> int:
        return 25

    with pytest.raises(ValueError, match="not supported for tasks"):

        @bentoml.task(batchable=True, max_latency_ms=resolver)  # type: ignore[arg-type]
        def predict(_, values: list[int]) -> list[int]:
            return values


def test_local_invocation_does_not_call_latency_resolver() -> None:
    calls = 0

    def resolver(request: Request) -> int:
        nonlocal calls
        calls += 1
        return 25

    @bentoml.service
    class TestService:
        @bentoml.api(batchable=True, max_latency_ms=resolver)
        def predict(self, values: list[int]) -> list[int]:
            return values

    service = TestService()

    assert service.predict([1, 2]) == [1, 2]
    assert calls == 0
