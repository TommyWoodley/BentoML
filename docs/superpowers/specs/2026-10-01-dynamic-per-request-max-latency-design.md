# BentoML Dynamic Per Request max_latency_ms

## Summary

Extend BentoML adaptive batching so that `max_latency_ms` may be resolved independently for each incoming request handled by a new-style batchable API.

Today, one integer value belongs to the API's `CorkDispatcher` and applies to every queued request. This change allows `@bentoml.api` to accept either:

```python
int
```

or:

```python
Callable[[starlette.requests.Request], int]
```

The callable is evaluated once after request deserialization and validation and immediately before the request enters the batching queue. BentoML validates the returned value and stores the resulting latency constraint on the queued `Job` as an absolute monotonic deadline.

Requests with different latency budgets continue to use the same API, FIFO queue, dispatcher, model invocation, `max_batch_size`, `batch_dim`, and adaptive batching optimizer. A tighter deadline limits how long the optimizer may continue waiting. Looser-deadline jobs already in the candidate batch may join the earlier dispatch.

This is a best-effort batching latency target, consistent with the existing feature. Predicted execution time limits further queueing, but it is not used as an admission-control rule. BentoML does not preempt an inference call after dispatch.

The first implementation is intentionally limited to request-backed new-style APIs served through BentoML's current HTTP or ASGI server path. Local Python calls, `@bentoml.task`, legacy Runner configuration, and legacy gRPC APIs remain unchanged.

## Motivation

Adaptive batching trades request latency for throughput. BentoML currently exposes this trade-off through a static API configuration:

```python
@bentoml.api(
    batchable=True,
    max_batch_size=32,
    max_latency_ms=100,
)
def predict(self, inputs):
    ...
```

Every request using `predict` receives the same batching latency target. Real services often receive traffic with different latency requirements while executing the same model operation:

```text
interactive request       5 ms
normal request           20 ms
background request      100 ms
```

Using `max_latency_ms=5` for the whole API meets the strict workload's target but reduces batching efficiency for background traffic. Using `max_latency_ms=100` improves batching efficiency but cannot react to the interactive workload. Separate APIs or queues fragment traffic and prevent otherwise compatible requests from sharing an inference batch.

The desired behavior is:

```text
A 100 ms --+
B 100 ms --+
C   5 ms --+-- shared adaptive queue --> one inference batch
D 100 ms --+
```

If C is eligible for the next batch, its tighter deadline limits further waiting. A, B, and D retain FIFO order and may be processed in the same earlier batch. If no tight request arrives, looser traffic may continue accumulating when the existing adaptive heuristic considers that useful.

## Goals

The implementation must:

1. Allow requests submitted to the same new-style batchable API to have different `max_latency_ms` values.
2. Preserve batching across requests with different latency requirements.
3. Preserve the existing public integer form for valid positive integer configurations.
4. Keep one FIFO queue and avoid priority scheduling or request reordering.
5. Keep `max_batch_size` and `batch_dim` static at the API and dispatcher level.
6. Continue using the existing adaptive optimizer and its current job-count model.
7. Treat each deadline as an upper bound on additional batching wait.
8. Include predicted execution time when deciding whether further waiting is safe.
9. Dispatch immediately when the latest useful start time has been reached and capacity is available.
10. Expire queued jobs independently when their actual deadline passes.
11. Preserve the existing fallback result for jobs that expire before dispatch.
12. Preserve legacy Runner uses of `CorkDispatcher` without adding a request resolver to that API.
13. Add deterministic scheduler tests without relying on narrow wall-clock assertions.
14. Keep the implementation contained within the existing decorator, server, dispatcher, and job architecture.

## Non goals

This change does not introduce:

- custom scheduler plugins;
- arbitrary batching strategies;
- priority queues or request reordering;
- weighted fairness or per-tenant QoS;
- batch groups or separate latency classes;
- dynamic `max_batch_size` or `batch_dim`;
- preemption of an executing model invocation;
- model routing;
- per-request concurrency settings;
- batch-affinity metadata;
- asynchronous latency resolvers;
- automatic propagation of latency metadata between services;
- external lookups during latency resolution;
- dynamic latency for `@bentoml.task`;
- dynamic latency for legacy Runner configuration or legacy gRPC APIs;
- a change from job-count-based to sample-count-based optimizer training.

## Public API

### Static form

The existing form remains valid:

```python
@bentoml.api(
    batchable=True,
    max_batch_size=32,
    max_latency_ms=50,
)
def predict(self, inputs):
    return self.model(inputs)
```

All requests receive a 50 ms batching latency budget.

Backward compatibility applies to existing valid values where `type(max_latency_ms) is int` and the value is greater than zero. Values such as booleans, zero, and negative integers were not previously validated consistently; the new validation deliberately rejects them.

### Dynamic form

`max_latency_ms` may additionally be a synchronous request resolver:

```python
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
def predict(self, inputs):
    return self.model(inputs)
```

The public type is conceptually:

```python
from collections.abc import Callable
from typing import TypeAlias

from starlette.requests import Request

MaxLatency: TypeAlias = int | Callable[[Request], int]
```

Starlette's `Request` is the narrowest honest type for the first implementation. BentoML's current new-style service request context is also backed by a Starlette request, so introducing a new wrapper would not provide transport independence today.

### API scope

The union type applies to `@bentoml.api` only.

A callable requires `batchable=True`. Service construction rejects a callable configured on a non-batchable API rather than silently accepting a resolver that can never run.

`@bentoml.task` retains `max_latency_ms: int`. A task submission responds before its background model execution, so request latency has different semantics and should be designed separately. If a callable is passed to `@bentoml.task`, service construction must reject it with a clear configuration error.

A callable is meaningful only for request-backed server invocation:

- HTTP and in-process ASGI requests resolve the callable normally.
- Remote service dependency calls use the receiving service's HTTP or ASGI request. BentoML does not promise to propagate arbitrary caller headers or latency metadata.
- Direct local Python calls continue to invoke the method directly and bypass adaptive batching, as they do today. They do not invoke the resolver.
- Spark batch execution currently starts a Bento HTTP server. A resolver used with that path must provide an application-appropriate default when its preferred header or metadata is absent.
- Legacy gRPC service APIs do not enter this new-style adaptive batching path and are unchanged.
- The internal legacy Runner server continues using the dispatcher's static default latency and does not receive the public resolver feature.

## Resolver contract

### Invocation point

For a valid request to a batchable API, the server performs these steps:

1. Route the request and run middleware, including authentication middleware.
2. Deserialize and validate the API input.
3. Populate the normal request context.
4. Resolve and validate `max_latency_ms`.
5. Submit the request to `batch_infer()` and the dispatcher.

The resolver therefore has access to request headers, path information, query parameters, and in-memory state populated by middleware. It is not invoked for a request that fails input deserialization, and it is never invoked after the job has entered the queue.

The resolver is called exactly once per enqueued request. Queue processing, retries inside the controller, job splitting, and outbound execution must not call it again.

### Validation

Static values are validated during service construction. Resolver functions are checked during service construction to reject known asynchronous callables. Resolved values are validated immediately before enqueue.

Every resolved value must satisfy:

```python
type(value) is int and value > 0
```

Using `type(value) is int` avoids accepting booleans. BentoML does not coerce values.

The following are invalid:

```text
0
-1
None
1.5
"10"
True
False
```

Async functions, async callable objects, and synchronous callables that return an awaitable are unsupported. Known async callables should be rejected during service construction. An unexpected awaitable returned at runtime fails the same validation as any other non-integer result.

### Resolver errors

If the resolver raises or returns an invalid value:

- the request is not enqueued;
- BentoML does not substitute a default latency;
- the exception follows the existing request error-handling path;
- ordinary exceptions and invalid outputs produce an application or server error;
- applications may raise an appropriate BentoML exception if they intentionally want a specific client-error response.

The resolver should be cheap, deterministic, and side-effect free. Typical inputs include headers, the request path, authenticated state, or tenant metadata already in memory. Database and network calls are outside the supported design.

## Latency semantics

### Deadline definition

The resolved value starts when the request is enqueued for adaptive batching, matching the existing dispatcher timing boundary. It does not include request-body upload or input deserialization.

The dispatcher uses one monotonic clock for all related values:

```python
enqueue_time = clock()
deadline = enqueue_time + max_latency_ms / 1000.0
```

`clock` defaults to `time.monotonic`. Queue wait, deadline comparisons, dispatch timing, and optimizer duration measurements use that same clock. The dispatcher must not compare monotonic and wall-clock timestamps.

### Predicted execution time

For the next candidate batch:

```python
predicted_duration = (
    optimizer.o_a * candidate_job_count
    + optimizer.o_b
)

earliest_deadline = min(job.deadline for job in candidate_jobs)
latest_safe_start = earliest_deadline - predicted_duration
```

The optimizer remains advisory. Its prediction limits additional batching wait, but it is not sufficiently authoritative to reject a request that can be dispatched immediately.

### Scheduling decision table

The controller applies the following behavior after removing jobs whose actual deadline has passed:

| State | Semaphore available | Semaphore occupied |
| --- | --- | --- |
| `now < latest_safe_start` | Existing adaptive heuristic decides whether to wait or dispatch. Any wait is capped at `latest_safe_start`. | Continue waiting and re-evaluating. |
| `latest_safe_start <= now < earliest_deadline` | Dispatch the candidate immediately. | Continue waiting for capacity; do not expire solely because the prediction says completion may be late. |
| `now >= a job's deadline` | Expire that queued job through the existing fallback path before selecting the next candidate. | Expire that queued job through the same path. |

This makes `max_latency_ms` a best-effort batching latency target rather than a hard execution timeout. Once a job has left the queue for outbound execution, the dispatcher does not cancel or preempt it if its deadline passes.

### Impossible deadlines

If predicted execution duration already exceeds a newly enqueued job's entire budget:

- dispatch it immediately if capacity is available;
- otherwise retain it until capacity becomes available or its actual deadline expires;
- do not wait intentionally for a larger batch;
- do not reject it solely because an optimizer estimate predicts a miss.

This preserves the useful existing single-request behavior and avoids turning a noisy performance estimate into admission control.

## Dispatcher design

### Job

`Job` gains an absolute deadline:

```python
@attr.define
class Job:
    enqueue_time: float
    data: T_IN
    future: asyncio.Future[T_OUT | Exception]
    deadline: float
    dispatch_time: float = 0
```

`split_job()` already uses `attr.evolve()`. Both child jobs therefore retain the parent's `enqueue_time` and `deadline`; only their data and futures change. Tests must lock in that behavior.

### Dispatcher compatibility

`CorkDispatcher` remains usable by static internal callers such as the legacy Runner server.

Its constructor retains a numeric default:

```python
CorkDispatcher(
    max_latency_in_ms: int,
    max_batch_size: int,
    ...,
)
```

The returned dispatcher function and `inbound_call()` additionally accept an optional explicit value:

```python
async def inbound_call(
    self,
    data: T_IN,
    *,
    max_latency_ms: int | None = None,
) -> T_OUT | Exception:
    ...
```

If `max_latency_ms` is omitted, the job uses the constructor's static default. This preserves existing legacy Runner and internal callers. New-style dynamic APIs always provide the resolved value.

The wrapper returned by `CorkDispatcher.__call__()` must expose the same optional keyword and forward it to `inbound_call()`; changing `inbound_call()` alone is insufficient because `batch_infer()` currently invokes the wrapper.

### Optimizer initialization

The optimizer keeps its current job-count regression model and initialization behavior.

- For a static API, the dispatcher constructor receives the configured static value, as it does today.
- For a dynamic API, `ServiceAppFactory` passes the existing 60,000 ms default as an internal initialization value.
- That initialization value seeds `optimizer.o_a` and `optimizer.o_b` and remains the fallback for callers that omit an explicit per-call latency.
- It is not authoritative for jobs submitted with an explicit latency.

This choice may make the earliest dynamic requests dispatch conservatively while optimizer training converges, but it does not cause them to wait beyond their own deadline. It avoids adding a second public fallback parameter or redesigning the optimizer in the first change.

### Optimizer units

The optimizer currently learns duration as a function of the number of dispatched jobs or chunks. This PR preserves that unit.

Candidate inspection therefore produces two different counts:

- `candidate_sample_size` is the number of input samples that fit under `max_batch_size` according to `get_batch_size()`.
- `candidate_job_count` is the number of `Job` objects or chunks that `_get_inputs()` would yield for that candidate.

`candidate_sample_size` controls capacity. `candidate_job_count` is passed to the optimizer's duration model. Changing optimizer training to use true sample counts is sensible future work but is not part of this feature.

### Candidate batch

The candidate is the FIFO prefix that `_get_inputs()` could dispatch next without exceeding `max_batch_size`. Candidate inspection must not mutate the queue or create split futures.

For example:

```text
job A sample size 7
job B sample size 7
job C sample size 6
max_batch_size 10
```

The candidate contains all of A and the first chunk of B. Candidate metadata therefore includes A's and B's deadlines, a sample size of 10, and a job count of 2. C's deadline cannot affect release of this candidate.

If a job will be split, the peek operation treats the dispatchable first chunk as one candidate job without performing the split. `_get_inputs()` remains responsible for the actual split after the controller decides to dispatch.

### Cached candidate metadata

Scanning the candidate prefix on every one-millisecond controller tick would add avoidable overhead. The dispatcher caches candidate metadata against a queue revision:

```text
queue revision
candidate job count
candidate sample size
earliest candidate deadline
next queued expiry
```

Enqueue, dequeue, split, expiration, or any other queue mutation increments the revision and invalidates affected metadata. Time passing alone does not require candidate membership to be recomputed.

For a static FIFO workload, the head job necessarily has the earliest candidate deadline because every later job has the same budget and a later enqueue time. The implementation may use this as an O(1) fast path, provided both paths use the same scheduling semantics.

### Per-job expiration

Candidate deadlines control release timing. Expiration applies to the entire queue independently of candidate membership.

The dispatcher maintains the earliest deadline among all queued jobs. It does not scan the full queue on every controller tick. When `now` reaches the cached next expiry, it performs one stable queue sweep:

1. Cancel every queued job whose `deadline <= now`.
2. Retain all surviving jobs in their existing order.
3. Recompute the next expiry and invalidate candidate metadata.
4. Continue controller processing if jobs remain.

This lets a tight-deadline job outside the next candidate expire on time without forcing an unrelated earlier batch to dispatch. A stable filter preserves FIFO ordering among surviving jobs and avoids introducing a scheduling priority queue.

Expired jobs must be removed before `_get_inputs()` runs so their data is never sent to the model after their future has been cancelled.

### Adaptive waiting

The existing adaptive heuristic remains responsible for answering:

> Would waiting longer probably improve batching efficiency?

Per-job deadlines answer:

> Is the dispatcher still allowed to wait?

When the adaptive heuristic chooses to wait, the controller caps that wait at the candidate's latest safe start and the next queued expiry. A new request still wakes or invalidates the controller state so that a later tight-deadline request can shorten the wait for an existing queue.

### Warmup and training

Optimizer warmup continues to collect execution duration and queue-wait observations, but it must use the same job-aware scheduling rules as the normal controller:

- target training batch sizes remain advisory;
- the training path may dispatch a smaller batch when a candidate reaches its latest safe start;
- actual expiration uses each job's deadline;
- training must not cancel one job according to another job's latency;
- the constructor's static default must not override an explicit job deadline.

Training remains concerned with execution-time estimation. Deadline enforcement remains a dispatcher responsibility.

### Semaphore contention

When the outbound semaphore is occupied, deadlines continue advancing.

- Jobs are not cancelled merely because predicted completion is now later than their deadline.
- Jobs remain eligible if capacity becomes available before their actual deadline.
- A queued job is cancelled independently when its actual deadline passes.
- Other queued jobs remain valid and in FIFO order.
- An outbound job is never preempted.

This is deliberately best effort. The dispatcher cannot predict when an occupied model invocation will release the semaphore accurately enough to use that estimate as an admission decision.

## Server integration

### Decorator and APIMethod

`@bentoml.api` accepts `MaxLatency`. `APIMethod.max_latency_ms` stores the union because the decorator metadata must retain either the static value or resolver.

Decorator or service-construction validation performs these checks:

- a static value is an exact positive integer;
- any other accepted value is callable;
- known async callables are rejected;
- a callable requires `batchable=True`;
- a callable is rejected when `is_task` is true;
- non-integer, non-callable values are rejected.

The static `@bentoml.task` overload remains typed as `int`.

### ServiceAppFactory

For each batchable API, `ServiceAppFactory` constructs one dispatcher as today.

- Static API: pass the configured integer to `CorkDispatcher`.
- Dynamic API: pass 60,000 ms as the internal dispatcher and optimizer default.

During `api_endpoint()` processing, after successful input deserialization and before calling `batch_infer()`:

```python
configured = method.max_latency_ms
if type(configured) is int:
    resolved_max_latency_ms = configured
else:
    resolved_max_latency_ms = configured(request)

validate_resolved_latency(resolved_max_latency_ms)
```

`batch_infer()` receives the resolved integer and passes it through the dispatcher's wrapper:

```python
output = await self.batch_infer(
    name,
    call_args,
    call_kwargs,
    max_latency_ms=resolved_max_latency_ms,
)
```

The resolver is not passed into `batch_infer()` or `CorkDispatcher`; only its validated integer result crosses that boundary.

## FIFO and result semantics

Dynamic latency affects when the next FIFO candidate is released, not which request goes first.

Given:

```text
A 100 ms
B 100 ms
C   5 ms
D 100 ms
```

the dispatcher may release `[A, B, C, D]` earlier. C does not jump ahead of A and B.

If C expires while waiting behind work that cannot yet be dispatched, C is removed without changing the relative ordering of A, B, and D. This is expiration, not priority scheduling.

The existing output splitting and child-future callbacks continue mapping model results to the correct original futures. Mixed deadlines must not affect output order or request-result association.

## Backward compatibility

The change is source compatible for existing APIs configured with valid positive integer latency values. It retains the same public parameter, default value, FIFO queue, fallback result, optimizer model, maximum batch size, batch dimensions, and legacy Runner default path.

It does not promise instruction-for-instruction scheduler equivalence near a deadline. The existing controller can cancel a multi-job head request when an execution estimate crosses the global latency value. The new controller instead uses that estimate to stop waiting and dispatches when capacity is available. This is an intentional correction required for a tight request to cause early shared-batch release. Actual queued-job expiration still uses the existing cancellation and fallback mechanism.

The validation change also intentionally rejects invalid static values such as booleans, zero, and negative integers. These cases are outside the compatibility guarantee.

## Packaging behavior

New-style Bento services are reconstructed through their service module import string, so decorator execution recreates the resolver when a packaged Bento starts. The resolver is not serialized into the public service schema.

The feature must still include a packaged-Bento test because module imports, closure placement, and service startup validation can differ from an in-process development server. Documentation should recommend module-level resolver functions for clarity, without imposing an unnecessary importability rule on callables that already work with normal service-module reconstruction.

## Observability

Existing adaptive batching metrics must continue working. No new metric is required for the first implementation.

Debug logging may include deadline-related scheduling decisions, but it must not log sensitive request metadata. Raw latency values must not become metric labels because arbitrary resolver output would create high cardinality.

Possible future metrics include:

- a resolved latency histogram;
- deadline misses before dispatch;
- jobs expired while queued;
- queue time grouped into bounded latency buckets.

## Security and abuse

A resolver may derive latency from client-controlled metadata. An untrusted client requesting extremely small budgets can force frequent small batches and reduce throughput. BentoML does not add fairness or admission control in this feature.

Applications should authenticate, validate, and clamp untrusted values:

```python
def resolve_latency(request: Request) -> int:
    requested = int(request.headers.get("X-Latency", "50"))
    return min(100, max(5, requested))
```

A sustained stream of tight-budget requests causing earlier dispatch is expected behavior under the configured application policy.

## Testing strategy

### Pure scheduler tests

Extract a small internal scheduling-decision helper. It receives already-inspected state and returns a decision such as wait, dispatch, or expire. It must not sleep, mutate the queue, or read wall-clock time itself.

Test at least:

- optimizer permits waiting before `latest_safe_start`;
- waiting is capped at `latest_safe_start`;
- capacity available at or after `latest_safe_start` returns dispatch;
- an impossible predicted deadline returns immediate dispatch when capacity is available;
- semaphore contention does not expire a job before its actual deadline;
- a job at its actual deadline expires;
- a later deadline outside the candidate does not affect candidate release;
- candidate duration uses job count rather than sample count;
- partial inclusion of a pre-batched job counts as one candidate job;
- static equal-deadline behavior follows the same decision rules.

These tests use explicit `now`, optimizer coefficients, candidate metadata, and semaphore state. They require no real sleeps.

### Dispatcher tests

Use an injected monotonic clock, controlled futures, and explicit events. Bypass or deterministically seed optimizer warmup when testing normal controller behavior.

Required cases:

- valid static latency preserves the existing public behavior;
- mixed deadlines share one dispatcher and one batch;
- a short candidate deadline causes earlier release;
- loose-only traffic may continue accumulating;
- a late short deadline invalidates cached candidate state and shortens the wait;
- FIFO order is preserved;
- a tight deadline outside the candidate does not release the earlier candidate;
- a non-candidate job still expires at its own deadline;
- expiration removes the job before model invocation;
- one expired job does not expire surviving jobs;
- expiration while the semaphore is occupied uses the existing fallback;
- an impossible deadline dispatches immediately when capacity is free;
- `max_batch_size` remains enforced using sample count;
- `batch_dim` behavior is unchanged;
- split children inherit the parent's deadline;
- a split child's expiration resolves the parent consistently;
- results map to the correct original futures;
- cancelled futures do not corrupt a later batch;
- legacy Runner-style calls that omit explicit latency retain the constructor default;
- cache invalidation occurs after enqueue, dequeue, split, and expiration.

### Resolver and API tests

Required cases:

- `@bentoml.api` accepts a valid static latency;
- `@bentoml.api` accepts a synchronous callable;
- a callable on a non-batchable API is rejected;
- the resolver receives the Starlette request;
- resolution happens after input validation and before enqueue;
- the resolver is called once per successfully enqueued request;
- the resolved integer reaches the `Job.deadline` calculation;
- zero, negative, float, string, boolean, and `None` outputs are rejected;
- invalid static values are rejected during service construction;
- resolver exceptions prevent enqueue and follow existing error handling;
- async functions and async callable objects are rejected;
- a returned awaitable is rejected;
- `@bentoml.task` rejects a callable;
- local invocation bypasses the resolver and dispatcher as it does today.

### Integration test

Create one real batchable service with a header-based resolver and no ML dependency. Submit concurrent requests with 100 ms, 100 ms, 5 ms, and 100 ms budgets.

Verify that:

- the resolver runs for every accepted HTTP request;
- all requests enter the same dispatcher;
- compatible requests may form one model batch;
- the 5 ms request limits further waiting when it belongs to the candidate;
- outputs remain associated with the correct requests;
- a control run containing only loose budgets waits longer when the optimizer permits it.

The integration test may use generous timing bounds only to verify the end-to-end effect. Exact scheduling thresholds belong in pure deterministic tests.

### Packaged Bento test

Build and serve a Bento whose service module defines a dynamic resolver. Submit a request and verify that:

- service startup succeeds;
- the resolver is reconstructed by module import;
- the resolver executes;
- its value reaches the dispatcher;
- the response is correct.

### Performance regression check

Run a static workload where all requests use `max_latency_ms=20` before and after the change. Compare throughput, p50, p95, and batch-size distribution. The goal is to detect a meaningful regression, not to prove an improvement.

Also run a mixed workload such as:

```text
20%   5 ms
30%  20 ms
50% 100 ms
```

Expected qualitative behavior is:

- tight candidate deadlines produce earlier release;
- looser jobs already in the candidate piggyback;
- loose-only traffic produces larger batches when the optimizer permits;
- static-path controller overhead remains negligible because candidate metadata is cached.

## Implementation sequence

### Phase 1 Dispatcher support

1. Add the monotonic clock dependency and convert all comparable dispatcher timing to it.
2. Add `Job.deadline`.
3. Add optional per-call latency to the dispatcher wrapper and `inbound_call()` while retaining the constructor default.
4. Add non-mutating candidate inspection with separate job and sample counts.
5. Add cached candidate and next-expiry metadata.
6. Add stable per-job expiration across the full queue when the next expiry is reached.
7. Extract and test the pure scheduling decision.
8. Apply the same deadline rules to optimizer warmup and normal controller operation.
9. Verify unchanged legacy Runner call behavior.

### Phase 2 Request resolver

1. Add the `MaxLatency` type to `@bentoml.api` and `APIMethod`.
2. Keep the `@bentoml.task` public type static and reject callable task configuration.
3. Validate static configuration and known async callables at service construction.
4. Resolve and validate the latency in `api_endpoint()` after input validation.
5. Pass only the resolved integer through `batch_infer()` to the dispatcher wrapper.

### Phase 3 Integration and packaging

1. Add the HTTP mixed-deadline integration service.
2. Add packaged-Bento validation.
3. Run the static and mixed performance checks.

### Phase 4 Documentation

Update adaptive batching documentation with static and dynamic examples. Explain that mixed budgets share one queue, tight candidate deadlines may release looser jobs earlier, and client-derived budgets should be authenticated and clamped.

## CI and quality checks

Before requesting review, run:

```bash
pre-commit run --all-files
make type
pytest <dispatcher tests>
pytest <API tests>
pytest <integration test>
pdm run nox
```

The targeted tests should run first. Run the broader `pdm run nox` suite where practical. New scheduler branches require direct coverage, especially static configuration, dynamic resolution, candidate boundaries, semaphore contention, expiration, split jobs, and cache invalidation.

## Documentation example

```python
from starlette.requests import Request

import bentoml


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
def predict(self, inputs):
    return self.model(inputs)
```

Documentation must state:

> Requests with different latency budgets still share one adaptive batching queue within each service worker. A tighter deadline in the next candidate batch can cause earlier release, allowing looser-deadline requests already in that candidate to be processed in the same model invocation. Latency changes release timing, not FIFO ordering.

## Acceptance criteria

The implementation is complete when:

1. Existing valid positive-integer `max_latency_ms` configurations remain supported.
2. Invalid static values are rejected during service construction.
3. `@bentoml.api` accepts a synchronous `Callable[[Request], int]`.
4. Callable configuration requires a batchable API.
5. `@bentoml.task` remains integer-only.
6. The callable runs exactly once for each valid request that is about to be enqueued.
7. Resolver output is validated without coercion.
8. Resolver failures and invalid values never enqueue a job.
9. Every queued job has an absolute monotonic deadline.
10. Requests with different budgets share one dispatcher and may share one inference batch.
11. Candidate selection matches `_get_inputs()` capacity and FIFO semantics without mutating the queue.
12. Candidate capacity uses sample count while predicted duration uses the optimizer's existing job-count unit.
13. A tighter candidate deadline limits further adaptive waiting.
14. A job with an impossible predicted deadline dispatches immediately when capacity is available.
15. A queued job expires independently when its actual deadline passes.
16. A deadline outside the next candidate does not force that candidate to release.
17. Expired jobs outside the candidate are still removed on time.
18. Expiration preserves the relative FIFO order of surviving jobs.
19. `max_batch_size` and `batch_dim` behavior remain unchanged.
20. Split children inherit the original deadline.
21. Results remain mapped to the correct request futures.
22. Legacy Runner calls that omit explicit latency retain their static dispatcher behavior.
23. Local calls, legacy gRPC APIs, and tasks remain outside dynamic resolution.
24. The resolver works after building and serving a packaged Bento.
25. Static workloads show no meaningful performance regression.
26. Scheduler behavior is covered primarily by deterministic tests rather than narrow real-time assertions.
27. Documentation explains scope, shared batching, FIFO behavior, best-effort semantics, and abuse mitigation.

## Upstream contribution framing

Suggested issue title:

```text
Feature: support per-request max_latency_ms in adaptive batching
```

Suggested PR title:

```text
feat: support per-request max latency in adaptive batching
```

The issue should focus on the user-visible problem, the `int | Callable[[Request], int]` API, best-effort deadline semantics, and the request-backed transport scope. The implementation details and complete test matrix belong in the PR or linked design document.

The contribution should be presented as a contained extension of the existing adaptive batching model. It does not attempt to solve general custom batching or scheduling.

## Resolved design decisions

The design makes the following explicit choices:

1. Use `starlette.requests.Request` for the first public resolver type.
2. Support request-backed new-style APIs served through the current HTTP or ASGI path.
3. Leave direct local invocation outside batching and resolver execution.
4. Keep `@bentoml.task` integer-only.
5. Reject known async resolvers at service construction.
6. Use one monotonic clock throughout the dispatcher.
7. Preserve the optimizer's current job-count model.
8. Use 60,000 ms only as an internal optimizer and dispatcher default for dynamically configured APIs.
9. Use predictions to cap waiting, not as an admission-control decision.
10. Expire queued jobs only when their actual deadline passes.
11. Preserve FIFO order among all surviving jobs.
12. Retain the dispatcher's static default for legacy Runner compatibility.
13. Cache candidate metadata and scan the full queue only when expiration or queue mutation requires it.
14. Return resolver configuration failures through existing request error handling without fallback or coercion.
