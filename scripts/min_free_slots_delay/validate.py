#!/usr/bin/env python3
"""Manual min-free-slots-delay functional check on the CURRENT checkout.

No engine files or kernels are changed. sitecustomize.py only wraps the two
scheduler methods to record their original decisions/results. There is no GDN
fallback: missing kernels or other model startup errors fail the run explicitly.
This is NOT an accuracy/performance benchmark; graphs and speculation are off
by default. Run in the Linux environment where SGLang is already installed.

NPU example (choose free logical device IDs, not physical card IDs):
  python scripts/min_free_slots_delay/validate.py \\
    --model-path /home/weights/Qwen3.6-27B --device npu --devices 0,1 \\
    --tp-size 2 --output /tmp/min-slots-run-01 \\
    --server-args --attention-backend ascend --dtype bfloat16 \\
    --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 64 \\
    --mamba-radix-cache-strategy extra_buffer

GPU example:
  python scripts/min_free_slots_delay/validate.py \\
    --model-path /models/your-model --device cuda --devices 0 \\
    --tp-size 1 --output /tmp/min-slots-run-02

By default, run normal/overlap x thresholds 1/4 sequentially (19 requests each).
Use --mode normal|overlap and --threshold 1|4 to select one case. --dry-run
prints commands without starting a server. Model/backend-specific environment
variables are inherited; the runner does not impose CANN/GDN tuning variables.
Pass extra SGLang arguments LAST, after --server-args. Each case directory must
be new. Results, commands, logs, and per-process JSONL traces remain in --output.

A zero exit code requires all cases to PASS. Startup/dependency/request failures
are FAILED; insufficient workload coverage is INCONCLUSIVE (also nonzero), not
evidence that the feature is broken. Trace event counts aggregate TP ranks, not
unique requests. Read result.json for the exact free-slot values observed.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


class Inconclusive(RuntimeError):
    """The run did not exercise enough of the target behavior."""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def repository(path=None):
    candidates = [Path(path).resolve()] if path else Path(__file__).resolve().parents
    for candidate in candidates:
        if (candidate / "python/sglang/launch_server.py").is_file():
            return candidate
    raise ValueError("Cannot locate SGLang checkout; pass --repo /path/to/sglang")


def git_revision(repo):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def command_for(args, mode, threshold, port):
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        args.device,
        "--tp-size",
        str(args.tp_size),
        "--max-running-requests",
        "8",
        "--context-length",
        "4096",
        "--chunked-prefill-size",
        "256",
        "--max-prefill-tokens",
        "2048",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--min-free-slots-delay",
        str(threshold),
    ]
    if not args.enable_graphs:
        command += [
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
        ]
    if mode == "normal":
        command += ["--disable-overlap-schedule"]
    return command + args.server_args


def environment_for(args, repo, case):
    env = os.environ.copy()
    parts = [str(Path(__file__).resolve().parent), str(repo / "python")]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["SGLANG_MIN_SLOTS_TEST_TRACE_DIR"] = str(case)
    if args.devices:
        variable = (
            "ASCEND_RT_VISIBLE_DEVICES"
            if args.device == "npu"
            else "CUDA_VISIBLE_DEVICES"
        )
        env[variable] = args.devices
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def wait_ready(process, base_url, timeout):
    import requests

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Server exited with {process.returncode}; see server.log"
            )
        try:
            if requests.get(base_url + "/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"Startup timed out after {timeout}s; see server.log")


def generate(base_url, payload, timeout):
    import requests

    response = requests.post(base_url + "/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    rows = response.json()
    rows = rows if isinstance(rows, list) else [rows]
    expected_count = len(payload["text"]) if isinstance(payload["text"], list) else 1
    require(
        len(rows) == expected_count,
        f"Expected {expected_count} responses; got {len(rows)}",
    )
    expected = payload["sampling_params"]
    if not isinstance(expected, list):
        expected = [expected] * expected_count
    for row, params in zip(rows, expected):
        require("error" not in row, f"Generation returned error: {row}")
        actual = row["meta_info"]["completion_tokens"]
        require(
            actual == params["max_new_tokens"], f"Unexpected completion length: {row}"
        )
    return [
        {
            "completion_tokens": row["meta_info"]["completion_tokens"],
            "finish_reason": row["meta_info"].get("finish_reason"),
            "text_sample": row.get("text", "")[:160],
        }
        for row in rows
    ]


def workload(base_url, timeout):
    def params(n):
        return {"temperature": 0, "max_new_tokens": n, "ignore_eos": True}

    lengths = [32, 64, 96, 128, 160, 192, 224, 256] + [64] * 8
    payloads = {
        "idle": {"text": "The capital of France is", "sampling_params": params(16)},
        "saturation": {
            "text": [
                f"Task {i}: List natural numbers starting from {i + 1}:"
                for i in range(16)
            ],
            "sampling_params": [params(n) for n in lengths],
        },
        "chunked": {
            "text": [
                f"Document {i}: " + "This is a test sentence about science. " * 90
                for i in range(2)
            ],
            "sampling_params": params(16),
        },
    }
    result = {}
    for name, payload in payloads.items():
        start = time.time()
        responses = generate(base_url, payload, timeout)
        result[name] = {"start": start, "end": time.time(), "responses": responses}
    return result


def summarize(rows, threshold):
    prefills = [r for r in rows if r["event"] == "prefill"]
    active = [
        r for r in prefills if not r["chunk"] and r["running"] > 0 and r["waiting"] > 0
    ]
    decisions = [r for r in rows if r["event"] == "decision"]
    # Compare both arms against the enabled-arm threshold (4).
    return {
        "aggregation": "events across all scheduler processes / TP ranks",
        "prefill_events": len(prefills),
        "scheduler_modes": sorted({r["overlap"] for r in prefills}),
        "effective_max_running_requests": sorted(
            {r["max_running_requests"] for r in prefills}
        ),
        "effective_thresholds": sorted({r["threshold"] for r in prefills}, key=str),
        "delay_decisions": sum(r["delayed"] for r in decisions),
        "delayed_free_slots": sorted({r["slots"] for r in decisions if r["delayed"]}),
        "admissions_below_4_free_slots": sum(
            0 < r["slots"] < 4 and r["admitted"] > 0 for r in active
        ),
        "admissions_at_least_4_free_slots": sum(
            r["slots"] >= 4 and r["admitted"] > 0 for r in active
        ),
        "idle_admissions": sum(
            r["running"] == 0 and r["admitted"] > 0 for r in prefills
        ),
        "chunk_continuations": sum(r["chunk"] and r["admitted"] > 0 for r in prefills),
        "threshold_requested": threshold,
    }


def check_observations(rows, summary, threshold, mode):
    require(
        summary["prefill_events"] > 0,
        "No prefill traces: observer is missing/incompatible",
    )
    require(
        summary["scheduler_modes"] == [mode == "overlap"],
        "Actual scheduler mode differs from requested mode",
    )
    require(
        summary["effective_thresholds"] == [None if threshold == 1 else 4],
        "Effective delay threshold differs from requested value",
    )
    if summary["effective_max_running_requests"] != [8]:
        raise Inconclusive(
            "Actual request limit is not 8; for hybrid models increase --max-mamba-cache-size via --server-args"
        )
    prefills = [r for r in rows if r["event"] == "prefill"]
    require(
        not any(r["delayed"] and r["admitted"] > 0 for r in prefills),
        "A delayed pass admitted a prefill",
    )
    require(
        not any(r["chunk"] and r["decision_called"] for r in prefills),
        "Chunk continuation incorrectly consulted the slot gate",
    )
    for row in (r for r in rows if r["event"] == "decision"):
        expected = row["running"] > 0 and row["slots"] < row["threshold"]
        require(row["delayed"] == expected, f"Incorrect delay decision: {row}")
    if threshold == 4:
        require(
            summary["admissions_below_4_free_slots"] == 0,
            "Fresh prefill admitted below the threshold",
        )
        coverage = any(slot in (1, 2, 3) for slot in summary["delayed_free_slots"])
        coverage = coverage and summary["admissions_at_least_4_free_slots"] > 0
    else:
        require(summary["delay_decisions"] == 0, "Disabled gate delayed a request")
        coverage = summary["admissions_below_4_free_slots"] > 0
    coverage = (
        coverage
        and summary["idle_admissions"] > 0
        and summary["chunk_continuations"] > 0
    )
    if not coverage:
        raise Inconclusive(
            "Required slot/idle/chunk scenarios were not observed; inspect trace and workload before interpreting the result"
        )


def stop_owned_server(process):
    import psutil

    try:
        parent = psutil.Process(process.pid)
        descendants = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    # Ask the server to shut down before terminating remaining owned children.
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    for child in descendants:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(descendants, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)
    _, alive = psutil.wait_procs(descendants, timeout=5)
    if alive:
        raise RuntimeError(
            f"Owned child processes still alive: {[p.pid for p in alive]}"
        )


def run_case(args, repo, mode, threshold):
    import requests

    case = args.output / f"{mode}-delay{threshold}"
    case.mkdir(parents=True, exist_ok=False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    command = command_for(args, mode, threshold, port)
    env = environment_for(args, repo, case)
    report = {
        "name": case.name,
        "status": "starting",
        "repo": str(repo),
        "revision": git_revision(repo),
        "command": command,
        "device": args.device,
        "devices": args.devices,
        "kernel_fallback": False,
        "observation_wrappers": True,
    }
    write_json(case / "command.json", command)
    process = None
    interrupted = False
    print(f"START {case.name}: {case}", flush=True)
    try:
        with (case / "server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=repo,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            (case / "server.pid").write_text(str(process.pid), encoding="utf-8")
            wait_ready(process, base_url, args.startup_timeout)
            print(f"READY {case.name}", flush=True)
            report["workload"] = workload(base_url, args.request_timeout)
            records = []
            for path in sorted(case.glob("trace-*.jsonl")):
                with path.open(encoding="utf-8") as stream:
                    records.extend(json.loads(line) for line in stream if line.strip())
            require(
                not any(r["event"] == "observer_error" for r in records),
                "Observer failed; see trace",
            )
            sources = {r["source"] for r in records if r["event"] == "instrumented"}
            expected_source = (
                repo / "python/sglang/srt/managers/scheduler.py"
            ).resolve()
            require(
                bool(sources)
                and all(Path(p).resolve() == expected_source for p in sources),
                "Server imported a different SGLang checkout",
            )
            windows = [
                (stage["start"], stage["end"]) for stage in report["workload"].values()
            ]
            rows = [
                r
                for r in records
                if any(start <= r["time"] <= end for start, end in windows)
            ]
            summary = summarize(rows, threshold)
            summary["requests_completed"] = sum(
                len(stage["responses"]) for stage in report["workload"].values()
            )
            report["summary"] = summary
            require(
                requests.get(base_url + "/health", timeout=5).status_code == 200,
                "Health check failed after workload",
            )
            require(process.poll() is None, "Server exited after workload")
            check_observations(rows, summary, threshold, mode)
            report["status"] = "passed"
    except KeyboardInterrupt:
        interrupted = True
        report.update(status="interrupted", error="Interrupted by user")
    except Inconclusive as exc:
        report.update(status="inconclusive", error=str(exc))
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
    finally:
        if process is not None:
            try:
                stop_owned_server(process)
            except Exception as exc:
                report.update(status="failed", cleanup_error=repr(exc))
        write_json(case / "result.json", report)
    print(
        f"{report['status'].upper()} {case.name}: {json.dumps(report.get('summary', report.get('error')))}",
        flush=True,
    )
    if interrupted:
        raise KeyboardInterrupt
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--repo", help="Checkout to test; defaults to this script's checkout"
    )
    parser.add_argument("--device", choices=["npu", "cuda"], default="npu")
    parser.add_argument(
        "--devices",
        help="Comma-separated logical device IDs; otherwise inherit visibility",
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["normal", "overlap", "both"], default="both")
    parser.add_argument("--threshold", choices=["1", "4", "both"], default="both")
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument(
        "--enable-graphs", action="store_true", help="Leave graph selection to SGLang"
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--server-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.tp_size < 1 or not 0 < args.mem_fraction_static < 1:
        parser.error(
            "tp-size must be positive and mem-fraction-static must be between 0 and 1"
        )
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        parser.error("timeouts must be positive")
    if args.devices:
        devices = args.devices.split(",")
        if (
            any(not part.strip() for part in devices)
            or len(set(devices)) != len(devices)
            or len(devices) < args.tp_size
        ):
            parser.error(
                "devices must contain distinct IDs and at least tp-size entries"
            )
    protected = {
        "--model-path",
        "--model",
        "--host",
        "--port",
        "--device",
        "--tp-size",
        "--tp",
        "--pp-size",
        "--pp",
        "--pipeline-parallel-size",
        "--max-running-requests",
        "--min-free-slots-delay",
        "--disable-overlap-schedule",
        "--enable-dp-attention",
        "--dp-size",
        "--dp",
        "--nnodes",
        "--disaggregation-mode",
        "--config",
        "--config-file",
        "--enable-prefill-delayer",
        "--speculative-algorithm",
        "--enable-mixed-chunk",
        "--chunked-prefill-size",
        "--max-prefill-tokens",
        "--context-length",
        "--mem-fraction-static",
    }
    if any(part.split("=", 1)[0] in protected for part in args.server_args):
        parser.error(
            "server-args overrides a controlled workload/scheduling option; use the runner options instead"
        )
    args.output = args.output.resolve()
    return args


def main():
    args = parse_args()
    repo = repository(args.repo)
    modes = ["normal", "overlap"] if args.mode == "both" else [args.mode]
    thresholds = [1, 4] if args.threshold == "both" else [int(args.threshold)]
    if args.dry_run:
        for mode in modes:
            for threshold in thresholds:
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "threshold": threshold,
                            "command": command_for(args, mode, threshold, 30000),
                            "repo": str(repo),
                            "kernel_fallback": False,
                        }
                    )
                )
        return 0
    if os.name != "posix":
        raise RuntimeError("Run this script inside the Linux GPU/NPU environment")
    import psutil  # noqa: F401 -- check cleanup dependency before starting a server
    import requests  # noqa: F401

    # Reject reused directories before starting any server in the matrix.
    for mode in modes:
        for threshold in thresholds:
            if (args.output / f"{mode}-delay{threshold}").exists():
                raise FileExistsError(
                    "A case directory already exists; choose a fresh --output"
                )
    results = []
    for mode in modes:
        for threshold in thresholds:
            report = run_case(args, repo, mode, threshold)
            results.append({"name": report["name"], "status": report["status"]})
            write_json(args.output / "summary.json", results)
            if report["status"] != "passed":
                return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
