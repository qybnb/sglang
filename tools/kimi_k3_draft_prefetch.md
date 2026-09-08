# Kimi-K3 DSPark draft prefetch on Ascend NPU

This port adds opt-in draft prefetch to the speculative overlap path and adapts
Kimi-K3 DSPark to Ascend attention, fixed-width TorchAir GE draft replay, and
the existing NPU Target graph runner. It does not implement DCP or change the
external `sgl-kernel-npu` repository.

## Serving configuration

Keep the existing working Kimi-K3 DSPark model paths, TP/DP topology, cache and
graph sizes. Add `--enable-draft-prefetch` to the server arguments. For the
conservative NPU configuration used by the profiled snapshot, set:

```bash
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_TORCHAIR_FIA=1
export SGLANG_ENABLE_WAR_BARRIER=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export SGLANG_DSPARK_TARGET_GRAPH_REUSE_GUARD=1
export SGLANG_DSPARK_DEFER_TARGET_METADATA=0
export SGLANG_DSPARK_DIAG_DIR=
export SGLANG_DSPARK_DIAG_SYNC_PHASES=
export SGLANG_LOG_DECODE_GRAPH_KEY=0
```

The reproduced topology was TP=64, DP=4, attention-TP=16, EP=64 across four
16-device nodes; CP/DCP were disabled. The DSPark block size was 7. Target
graphs resolved to batch tiers 2 and 16; the draft GE tier was 16. Radix cache
remained enabled, static memory fraction was 0.75, and both KV caches used bf16.
These are reproduction details, not universal deployment defaults.

Do **not** use `--skip-draft-prefetch-seq-lens-cpu-sync` to bypass K3 Target
metadata requirements. Draft GE can consume device sequence lengths; Target
FIA still uses an exact pinned CPU length snapshot. The NPU path issues that
snapshot on the existing private D2H stream after accept and waits only for
its completion event when the next Target requires it.

The steady-state prefetch path supports static greedy verification. Bootstrap,
new/mixed requests and non-greedy cases retain proposer/fallback handling.
The draft is submitted on the existing forward stream, not a background
draft thread or an additional draft compute stream.

## Reuse protection is not diagnostic instrumentation

`SGLANG_DSPARK_TARGET_GRAPH_REUSE_GUARD=1` is independent of diagnostics. For NPU
DSPark with prefetch enabled, every Target runner retains one completion event,
including idle DP participation. Before loading the next graph's input slots
or updating host attributes it waits for the previous execution. The event is
re-recorded only after that wait. Graph bucket changes share the same guard.

The diagnostic `target_reuse` experiment allowed the full model to complete
requests where an unprotected run stalled and later reported a
`MoeLowLatencyDispatchV2` failure. This supports a conservative reuse fence;
it does not prove which internal resource or ordering caused that failure.
The fence's synchronization cost is part of serving latency, not a zero-cost
performance claim. Draft and non-prefetch runners do not use this guard.

Leave diagnostic flags empty for latency measurements. See
[DSPark diagnostics](dspark_diagnostics.md) for isolated debugging. Cluster IPs,
container-specific launchers, credentials, raw logs, weights, traces and kernel
dumps are intentionally not included in this contribution.

## Rank0 decode profiling

Start the service with `SGLANG_PROFILE_V2=0` and `SGLANG_PROFILE_RANKS=0`, then:

```bash
SERVER_URL=http://127.0.0.1:15010 PROFILE_STEPS=5 ./profile_kimi_k3_decode_rank0.sh start
```

Send the workload after arming the profiler. All ranks retain the same
scheduler/collective behavior, while only the selected global TP rank creates
the profiler. Do not compare instrumented request latency with a normal
benchmark. Exclude capture/startup and profiler-stop boundaries when measuring
steady-state device steps.

## Profile observations and validation limits

The measured snapshot is based on `ff2f4caf6` plus this port, with diagnostics
disabled and the independent reuse guard enabled. On rank0, eight interior
Target-start-to-Target-start intervals averaged **82.765ms**:

| Phase | Mean device span |
| --- | ---: |
| Target model compute span | 74.751ms |
| Target tail, accept/state commit and draft preparation | 3.519ms |
| Draft GE, including entry update | 2.813ms |
| Candidate generation and next Target preparation | 1.682ms |

The latter phases contain actual compute/communication, not wholly idle gaps.
CPU scheduler/Gloo work of approximately 19ms overlapped Target execution.
The rank0 reuse guard's CPU event wait averaged approximately 0.040ms; other
ranks were not profiled. No continuous >=0.5ms gap across the recorded
Core/Vector/Aiv-communication streams appeared in those eight intervals.
Smaller AICPU, graph-control and memory-transfer intervals remained.

Historical traces had approximately 89.003ms baseline steps and 97.785ms steps
in an earlier prefetch implementation. These are **not** a controlled A/B on
the current upstream revision and do not establish a corresponding TPOT gain,
acceptance-length equivalence, or long-running multi-request stability.

The PR also integrates target-branch updates through `6b712e962`, including its
persistent NPU graph-update worker. The 82.765ms profile predates that merge.
Full four-node correctness/acceptance and performance validation of the merged
upstream combination remains required; the running profiled checkout is not
modified by preparing this PR.
