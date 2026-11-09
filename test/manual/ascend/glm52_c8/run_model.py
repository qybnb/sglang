"""Launch single-host GLM-5.2 C8 cases or probe an existing /generate endpoint."""

import argparse
import hashlib
import json
import math
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DRAFTS = {"off": None, "nextn-bf16": "bf16", "nextn-fp8": "fp8_e4m3"}
CASES = tuple(
    topology if draft == "off" else f"{topology}-{draft}"
    for topology in ("base", "pcp", "dcp")
    for draft in DRAFTS
)
REQUESTS = ("short", "long_fresh", "long_prefix_hit", "batch")


def launch_command(args, case):
    topology, _, draft = case.partition("-")
    if topology != "base" and not 1 < args.cp_size <= args.tp:
        raise ValueError("--cp-size must be greater than 1 and no larger than --tp")
    if topology != "base" and args.tp % args.cp_size:
        raise ValueError("--tp must be divisible by --cp-size")
    extra = json.loads(args.extra_args_json)
    if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
        raise ValueError("--extra-args-json must contain an array of strings")
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        *extra,
        "--model-path",
        args.model,
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--tp-size",
        str(args.tp),
        "--attn-dp-size",
        "1",
        "--attn-cp-size",
        str(args.cp_size if topology == "pcp" else 1),
        "--dcp-size",
        str(args.cp_size if topology == "dcp" else 1),
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--dtype",
        "bfloat16",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--context-length",
        str(args.context_length),
        "--max-running-requests",
        "8",
        "--chunked-prefill-size",
        str(args.chunk_size),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--enable-metrics",
        "--enable-cache-report",
        "--random-seed",
        "531",
        "--disable-prefill-cuda-graph",
    ]
    if args.quantization != "auto":
        command += ["--quantization", args.quantization]
    if args.graph == "eager":
        command += ["--disable-decode-cuda-graph"]
    else:
        command += ["--cuda-graph-bs-decode", "1", "2", "4", "8"]
    if topology == "pcp":
        command += ["--enable-prefill-cp", "--cp-strategy", "interleave"]
    if args.diagnose_indexer:
        command += ["--skip-server-warmup"]
    if draft:
        command += [
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-num-steps",
            str(args.spec_steps),
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            str(args.spec_steps + 1),
            "--speculative-draft-kv-cache-dtype",
            DRAFTS[draft],
        ]
    return command


def dump(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def http(url, timeout, payload=None, *, text=False):
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("GLM_C8_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None if payload is None else json.dumps(payload).encode()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, data, headers), timeout=timeout
        ) as response:
            body = response.read()
        if text:
            return body.decode()
        return json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        detail = exc.read(4000).decode(errors="replace")
        raise RuntimeError(f"{url}: HTTP {exc.code}: {detail}") from exc


def workload(args, tokenizer_path, *, seed=531):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    text = (
        "Context parallelism distributes a long sequence across ranks. "
        "Sparse attention uses the indexer to select relevant positions in the KV cache. "
        "Compare the numerical result with an independent reference across page boundaries. "
        "请解释为什么测试应覆盖普通解码、投机验证和前缀缓存。"
    )
    safe_ids = list(dict.fromkeys(tokenizer.encode(text, add_special_tokens=False)))
    safe_ids = [token for token in safe_ids if token not in tokenizer.all_special_ids]
    if len(safe_ids) < 2:
        raise ValueError("Tokenizer did not produce enough ordinary token IDs")
    rng = random.Random(seed)

    def tokens(count):
        return rng.choices(safe_ids, k=count)

    long_prompt = tokens(args.long_tokens)
    return [
        ("short", tokens(97)),
        ("long_fresh", long_prompt),
        ("long_prefix_hit", long_prompt + tokens(257)),
        ("batch", [tokens(193), tokens(387)]),
    ]


def check_response(name, response, prompt, args, expect_spec):
    prompts = prompt if isinstance(prompt[0], list) else [prompt]
    rows = response if isinstance(response, list) else [response]
    if len(rows) != len(prompts):
        raise AssertionError(
            f"{name}: expected {len(prompts)} responses, got {len(rows)}"
        )
    for index, (row, ids) in enumerate(zip(rows, prompts)):
        label = f"{name}[{index}]"
        meta = row["meta_info"]
        if meta.get("prompt_tokens") != len(ids):
            raise AssertionError(
                f"{label}: unexpected prompt length: {meta.get('prompt_tokens')}"
            )
        output_ids = row["output_ids"]
        if (
            len(output_ids) != args.new_tokens
            or meta.get("completion_tokens") != args.new_tokens
        ):
            raise AssertionError(
                f"{label}: unexpected completion length: {len(output_ids)}"
            )
        if not args.skip_logprobs:
            logprobs = meta["output_token_logprobs"]
            if len(logprobs) != len(output_ids) or any(
                entry[1] != token or not math.isfinite(entry[0])
                for entry, token in zip(logprobs, output_ids)
            ):
                raise AssertionError(
                    f"{label}: missing, mismatched or non-finite output logprobs"
                )
        if expect_spec and meta.get("spec_verify_ct", 0) <= 0:
            raise AssertionError(f"{label}: no NEXTN verify steps were reported")
        if (
            name == "long_prefix_hit"
            and args.require_cache_hit
            and meta.get("cached_tokens", 0) <= 0
        ):
            raise AssertionError(f"{label}: the repeated prefix did not hit the cache")
        print(
            f"{label}: prompt={len(ids)} cached={meta.get('cached_tokens')} "
            f"output={len(output_ids)} verify={meta.get('spec_verify_ct')} "
            f"accept_length={meta.get('spec_accept_length')}",
            flush=True,
        )


def probe(url, args, prompts, out, *, expect_spec):
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, prompt in prompts:
        print(f"REQUEST {out.name}/{name}", flush=True)
        payload = {
            "input_ids": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": not args.skip_logprobs,
            "logprob_start_len": -1,
            "stream": False,
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        start = time.monotonic()
        response = http(f"{url}/generate", args.request_timeout, payload)
        results[name] = {
            "request_sha256": fingerprint,
            "elapsed_s": time.monotonic() - start,
            "response": response,
        }
        try:
            check_response(name, response, prompt, args, expect_spec)
            results[name]["validated"] = True
        finally:
            dump(out / "results.json", results)
    print(f"PASS functional smoke: {out / 'results.json'}", flush=True)


def wait_ready(proc, url, timeout, log_path):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited with {proc.returncode}. See {log_path}")
        try:
            http(f"{url}/health", 5, text=True)
            info = http(f"{url}/server_info", 5)
            if isinstance(info, dict):
                return info
        except (RuntimeError, OSError, ValueError):
            pass
        time.sleep(2)
    raise TimeoutError(f"Server did not become ready within {timeout}s. See {log_path}")


def stop_owned_process_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def stream_server_logs(stream, log):
    for line in stream:
        log.write(line)
        log.flush()
        print(line, end="", flush=True)


def snapshot_graph_counter(url, path):
    metrics = http(f"{url}/metrics", 10, text=True)
    path.write_text(metrics, encoding="utf-8")
    return sum(
        float(line.rsplit("}", 1)[1].split()[0])
        for line in metrics.splitlines()
        if line.startswith("sglang:cuda_graph_passes_total{")
        and 'mode="decode_cuda_graph"' in line
    )


def run_case(args, case, prompts, command, revision):
    out = args.out_dir / f"{case}-{args.graph}"
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "command.json", {"git_sha": revision, "argv": command})
    url = f"http://127.0.0.1:{args.port}"
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", args.port))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT / "python"), env.get("PYTHONPATH")))
    )
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("SGLANG_ENABLE_SPEC_V2", "1")
    env["PYTHONUNBUFFERED"] = "1"
    if args.diagnose_indexer:
        diag = out.resolve() / "indexer_diag"
        diag.mkdir()
        capture = diag / "capture"
        capture.touch()
        env["SGLANG_NPU_INDEXER_DIAG_DIR"] = str(diag)
        env["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] = "0"
        env["ASCEND_LAUNCH_BLOCKING"] = "0"
        env["PYTHONFAULTHANDLER"] = "1"
    log_path = out / "server.log"
    print(f"START {case} {args.graph}, log={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        log_thread = threading.Thread(
            target=stream_server_logs, args=(proc.stdout, log)
        )
        try:
            log_thread.start()
            info = wait_ready(proc, url, args.startup_timeout, log_path)
            dump(out / "server_info.json", info)
            topology, _, draft = case.partition("-")
            expected = {
                "tp_size": args.tp,
                "attn_cp_size": args.cp_size if topology == "pcp" else 1,
                "dcp_size": args.cp_size if topology == "dcp" else 1,
                "kv_cache_dtype": args.kv_cache_dtype,
            }
            if draft:
                expected["speculative_draft_kv_cache_dtype"] = DRAFTS[draft]
            for key, value in expected.items():
                if info.get(key) != value:
                    raise AssertionError(
                        f"{case}: resolved {key}={info.get(key)!r}, expected {value!r}"
                    )
            before = snapshot_graph_counter(url, out / "metrics-before.prom")
            if args.diagnose_indexer:
                probe(url, args, prompts[:2], out / "synchronized", expect_spec=False)
                capture.unlink()
                fresh = workload(args, args.tokenizer or args.model, seed=532)
                probe(url, args, fresh[:2], out / "async", expect_spec=False)
            else:
                probe(url, args, prompts, out, expect_spec=bool(draft))
            graph_passes = (
                snapshot_graph_counter(url, out / "metrics-after.prom") - before
            )
            print(
                f"{case}: decode graph pass counter delta={graph_passes:g}", flush=True
            )
            if (args.graph == "decode") != (graph_passes > 0):
                raise AssertionError(
                    f"{case}: unexpected decode graph usage for {args.graph}"
                )
        finally:
            stop_owned_process_group(proc)
            if log_thread.ident is not None:
                log_thread.join()
            proc.stdout.close()
            if args.diagnose_indexer:
                for snapshot in sorted(diag.glob("rank-*.pt")):
                    with (diag / f"{snapshot.stem}-replay.log").open("w") as replay_log:
                        print(f"REPLAY {snapshot}", flush=True)
                        try:
                            result = subprocess.run(
                                [
                                    sys.executable,
                                    "-m",
                                    "sglang.srt.layers.attention.dsa.dsa_npu_indexer_diagnostics",
                                    str(snapshot),
                                ],
                                cwd=ROOT,
                                env=env,
                                stdout=replay_log,
                                stderr=subprocess.STDOUT,
                                timeout=90,
                                check=False,
                            )
                            print(
                                f"REPLAY {snapshot.stem} exit={result.returncode}",
                                flush=True,
                            )
                        except subprocess.TimeoutExpired:
                            print(f"REPLAY {snapshot.stem} timed out", flush=True)


def compare(args):
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    review = False
    for candidate in args.candidates:
        actual = json.loads(candidate.read_text(encoding="utf-8"))
        for path, results in ((args.baseline, baseline), (candidate, actual)):
            if results.keys() != set(REQUESTS) or not all(
                entry.get("validated") for entry in results.values()
            ):
                raise ValueError(
                    f"{path}: functional smoke did not finish successfully"
                )
        print(candidate)
        for name, entry in baseline.items():
            other = actual[name]
            if entry["request_sha256"] != other["request_sha256"]:
                raise ValueError(
                    f"{candidate}: {name} request differs from the baseline"
                )
            lhs, rhs = entry["response"], other["response"]
            lhs = lhs if isinstance(lhs, list) else [lhs]
            rhs = rhs if isinstance(rhs, list) else [rhs]
            if len(lhs) != len(rhs):
                raise ValueError(f"{candidate}: {name} batch sizes differ")
            for index, (ref, got) in enumerate(zip(lhs, rhs)):
                a, b = ref["output_ids"], got["output_ids"]
                n = next(
                    (i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]),
                    min(len(a), len(b)),
                )
                logs_a = ref["meta_info"].get("output_token_logprobs")
                logs_b = got["meta_info"].get("output_token_logprobs")
                if logs_a and logs_b:
                    delta = max(
                        (abs(x[0] - y[0]) for x, y in zip(logs_a[:n], logs_b[:n])),
                        default=0,
                    )
                    logprob_summary = f"{delta:.6g}" if n else "no common token"
                else:
                    delta = 0
                    logprob_summary = "not collected"
                verdict = (
                    "PASS"
                    if n == len(a) == len(b) and delta <= args.logprob_atol
                    else "REVIEW"
                )
                review |= verdict == "REVIEW"
                print(
                    f"  {verdict} {name}[{index}]: common_tokens={n}/{len(a)} max_same_prefix_logprob_delta={logprob_summary}"
                )
    return 2 if review else 0


def add_probe_args(parser, *, require_cache_hit):
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--long-tokens", type=int, default=4097)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument(
        "--require-cache-hit",
        action=argparse.BooleanOptionalAction,
        default=require_cache_hit,
    )
    parser.add_argument(
        "--skip-logprobs", action="store_true", help="Use token-only checks"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    run = subs.add_parser(
        "run", help="Launch and probe single-host A5 servers in sequence"
    )
    run.add_argument("--model", required=True)
    run.add_argument("--tokenizer")
    run.add_argument("--tp", type=int, default=8)
    run.add_argument("--cp-size", type=int, default=2)
    run.add_argument(
        "--cases", nargs="+", choices=CASES, default=["base", "pcp", "dcp"]
    )
    run.add_argument("--graph", choices=("eager", "decode"), default="eager")
    run.add_argument(
        "--kv-cache-dtype", choices=("fp8_e4m3", "bf16"), default="fp8_e4m3"
    )
    run.add_argument("--quantization", default="modelslim")
    run.add_argument("--port", type=int, default=31265)
    run.add_argument("--context-length", type=int, default=16384)
    run.add_argument("--chunk-size", type=int, default=8192)
    run.add_argument("--mem-fraction-static", type=float, default=0.80)
    run.add_argument("--startup-timeout", type=int, default=2400)
    run.add_argument("--spec-steps", type=int, default=3)
    run.add_argument("--extra-args-json", default="[]")
    run.add_argument("--print-command", action="store_true")
    run.add_argument(
        "--diagnose-indexer",
        action="store_true",
        help="Capture PCP C8 indexer inputs, then compare asynchronous requests",
    )
    add_probe_args(run, require_cache_hit=True)
    client = subs.add_parser(
        "client", help="Probe an already running worker or PD router"
    )
    client.add_argument("--url", required=True)
    client.add_argument("--tokenizer", required=True)
    client.add_argument("--expect-spec", action="store_true")
    add_probe_args(client, require_cache_hit=False)
    diff = subs.add_parser(
        "compare", help="Compare deterministic output IDs and same-prefix logprobs"
    )
    diff.add_argument("baseline", type=Path)
    diff.add_argument("candidates", nargs="+", type=Path)
    diff.add_argument("--logprob-atol", type=float, default=0.15)
    args = parser.parse_args()
    if args.action == "compare":
        return compare(args)
    diagnose_indexer = args.action == "run" and args.diagnose_indexer
    min_new_tokens = 2 if diagnose_indexer else 8
    if args.long_tokens <= 2048 or args.new_tokens < min_new_tokens:
        parser.error(f"use --long-tokens > 2048 and --new-tokens >= {min_new_tokens}")
    if diagnose_indexer and (
        args.cases != ["pcp"]
        or args.graph != "eager"
        or args.kv_cache_dtype != "fp8_e4m3"
    ):
        parser.error(
            "--diagnose-indexer requires --cases pcp --graph eager with FP8 KV"
        )
    if args.action == "client":
        probe(
            args.url.rstrip("/"),
            args,
            workload(args, args.tokenizer),
            args.out_dir,
            expect_spec=args.expect_spec,
        )
        return 0
    if args.long_tokens + 257 + args.new_tokens >= args.context_length:
        parser.error("--context-length must exceed long prompt + suffix + completion")
    commands = [(case, launch_command(args, case)) for case in args.cases]
    if args.print_command:
        for _, command in commands:
            print(shlex.join(command))
        return 0
    if sys.platform != "linux":
        parser.error(
            "run requires Linux with Ascend NPU; use --print-command to inspect arguments"
        )
    if not (Path(args.model) / "config.json").is_file():
        parser.error("--model must be a local checkpoint directory with config.json")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    prompts = workload(args, args.tokenizer or args.model)
    for case, command in commands:
        run_case(args, case, prompts, command, revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
