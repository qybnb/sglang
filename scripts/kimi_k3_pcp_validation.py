#!/usr/bin/env python3
"""Kimi K3 PCP/PD functional, numerical, and performance validation.

The script deliberately uses the PD router's native ``/generate`` endpoint for
functional and numerical checks.  Supplying exact token ids avoids chat-template
and tokenizer-length drift between PCP configurations.  Performance runs reuse
SGLang's serving benchmark instead of maintaining a second timing client.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "logs" / "kimi_k3_pcp_validation"


def _csv_positive_ints(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers: {value}"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return values


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer: {value}"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer: {value}")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative number: {value}"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative number: {value}")
    return parsed


def _ratio(value: str) -> float:
    parsed = _nonnegative_float(value)
    if parsed > 1:
        raise argparse.ArgumentTypeError(f"expected a ratio in [0, 1]: {value}")
    return parsed


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_output(prefix: str) -> Path:
    return DEFAULT_RESULTS_DIR / f"{prefix}_{_timestamp()}.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _auth_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if key := os.getenv("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    elif key := os.getenv("API_KEY"):
        headers["Authorization"] = key
    return headers


def _post_json(
    url: str, payload: dict[str, Any], timeout: float
) -> tuple[int, Any, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=_auth_headers(),
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        status = int(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {"transport_error": repr(exc)}, time.perf_counter() - start
    elapsed = time.perf_counter() - start
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = {"raw_body": body}
    return status, parsed, elapsed


def _load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _make_input_ids(tokenizer, target_len: int, case_id: int) -> list[int]:
    marker = tokenizer.encode(
        f" Kimi K3 PCP validation request {case_id}. ",
        add_special_tokens=False,
    )
    filler = tokenizer.encode(
        " Context parallel prefill must preserve every numerical result while "
        "the decode stage remains context-parallel free. ",
        add_special_tokens=False,
    )
    if not filler:
        raise RuntimeError("tokenizer produced an empty validation filler")

    repeats = (target_len + len(filler) - 1) // len(filler)
    input_ids = (filler * repeats)[:target_len]
    if marker:
        suffix = marker[-min(len(marker), target_len) :]
        input_ids[-len(suffix) :] = suffix
    if len(input_ids) != target_len:
        raise AssertionError(f"generated {len(input_ids)} ids, expected {target_len}")
    return [int(token_id) for token_id in input_ids]


def _prompt_hash(input_ids: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for token_id in input_ids:
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def _single_response(response: Any) -> dict[str, Any]:
    if isinstance(response, list):
        if len(response) != 1:
            raise ValueError(f"expected one response, got {len(response)}")
        response = response[0]
    if not isinstance(response, dict):
        raise ValueError(f"expected an object response, got {type(response).__name__}")
    return response


def _normalise_logprob_row(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        logprob = row.get("logprob")
        token_id = row.get("token_id", row.get("id"))
        text = row.get("text", row.get("token"))
    elif isinstance(row, (list, tuple)):
        logprob = row[0] if len(row) > 0 else None
        token_id = row[1] if len(row) > 1 else None
        text = row[2] if len(row) > 2 else None
    else:
        return {"logprob": None, "token_id": None, "text": None}
    return {
        "logprob": float(logprob) if logprob is not None else None,
        "token_id": int(token_id) if token_id is not None else None,
        "text": text,
    }


def _normalise_logprob_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    return [_normalise_logprob_row(row) for row in rows]


def _normalise_top_logprobs(rows: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(rows, list):
        return []
    return [_normalise_logprob_rows(row) for row in rows]


def _extract_accuracy_response(response: dict[str, Any]) -> dict[str, Any]:
    meta = response.get("meta_info") or {}
    return {
        "text": response.get("text", ""),
        "finish_reason": meta.get("finish_reason"),
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "input_token_logprobs": _normalise_logprob_rows(
            meta.get("input_token_logprobs")
        ),
        "output_token_logprobs": _normalise_logprob_rows(
            meta.get("output_token_logprobs")
        ),
        "input_top_logprobs": _normalise_top_logprobs(
            meta.get("input_top_logprobs")
        ),
        "output_top_logprobs": _normalise_top_logprobs(
            meta.get("output_top_logprobs")
        ),
    }


def _generate_payload(
    input_ids: list[int],
    output_len: int,
    *,
    return_logprob: bool,
    logprob_tail: int = 0,
    top_logprobs: int = 0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": False,
    }
    if return_logprob:
        payload.update(
            {
                "return_logprob": True,
                "return_text_in_logprobs": False,
                "logprob_start_len": max(0, len(input_ids) - logprob_tail),
                "top_logprobs_num": top_logprobs,
            }
        )
    return payload


def command_smoke(args: argparse.Namespace) -> int:
    tokenizer = _load_tokenizer(args.tokenizer)
    url = args.base_url.rstrip("/") + "/generate"

    def run_one(case_id: int) -> dict[str, Any]:
        input_ids = _make_input_ids(tokenizer, args.input_len, case_id)
        status, response, elapsed = _post_json(
            url,
            _generate_payload(
                input_ids, args.output_len, return_logprob=False
            ),
            args.timeout,
        )
        error = None
        prompt_tokens = None
        completion_tokens = None
        finish_reason = None
        if status != 200:
            error = response
        else:
            try:
                single = _single_response(response)
                if single.get("error"):
                    error = single["error"]
                else:
                    meta = single.get("meta_info") or {}
                    prompt_tokens = meta.get("prompt_tokens")
                    completion_tokens = meta.get("completion_tokens")
                    finish_reason = meta.get("finish_reason")
                    if prompt_tokens != args.input_len:
                        error = {
                            "validation_error": "prompt_token_count_mismatch",
                            "expected": args.input_len,
                            "actual": prompt_tokens,
                        }
                    elif completion_tokens != args.output_len:
                        error = {
                            "validation_error": "completion_token_count_mismatch",
                            "expected": args.output_len,
                            "actual": completion_tokens,
                        }
            except (TypeError, ValueError) as exc:
                error = {"validation_error": str(exc)}
        return {
            "case_id": case_id,
            "status": status,
            "success": error is None,
            "elapsed_s": elapsed,
            "prompt_hash": _prompt_hash(input_ids),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "error": error,
        }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        results = list(executor.map(run_one, range(args.num_requests)))

    succeeded = sum(bool(result["success"]) for result in results)
    latencies = [result["elapsed_s"] for result in results if result["success"]]
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": "smoke",
        "tag": args.tag,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_url": args.base_url,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_requests": args.num_requests,
        "concurrency": args.concurrency,
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "mean_latency_s": statistics.mean(latencies) if latencies else None,
        "max_latency_s": max(latencies) if latencies else None,
        "results": results,
    }
    output = args.output or _default_output(f"{args.tag}_smoke")
    _write_json(output, artifact)
    print(
        f"smoke {args.tag}: {succeeded}/{len(results)} succeeded; artifact={output}"
    )
    for result in results:
        if not result["success"]:
            print(f"  case {result['case_id']} failed: {result['error']}")
    return 0 if succeeded == len(results) else 1


def command_accuracy(args: argparse.Namespace) -> int:
    tokenizer = _load_tokenizer(args.tokenizer)
    url = args.base_url.rstrip("/") + "/generate"
    cases = []
    case_id = 0
    for input_len in args.input_lens:
        for repeat in range(args.repeats):
            input_ids = _make_input_ids(tokenizer, input_len, case_id)
            payload = _generate_payload(
                input_ids,
                args.output_len,
                return_logprob=True,
                logprob_tail=args.prefill_logprob_tokens,
                top_logprobs=args.top_logprobs,
            )
            status, response, elapsed = _post_json(url, payload, args.timeout)
            error = None
            extracted = None
            try:
                single = _single_response(response)
                if status != 200:
                    error = single.get("error") or single
                elif single.get("error"):
                    error = single["error"]
                else:
                    extracted = _extract_accuracy_response(single)
                    if extracted["prompt_tokens"] != input_len:
                        error = "prompt_token_count_mismatch"
                    elif extracted["completion_tokens"] != args.output_len:
                        error = "completion_token_count_mismatch"
                    elif not extracted["input_token_logprobs"]:
                        error = "missing_input_token_logprobs"
                    elif not extracted["output_token_logprobs"]:
                        error = "missing_output_token_logprobs"
            except (TypeError, ValueError) as exc:
                error = str(exc)
            cases.append(
                {
                    "case_id": case_id,
                    "repeat": repeat,
                    "target_input_len": input_len,
                    "output_len": args.output_len,
                    "prompt_hash": _prompt_hash(input_ids),
                    "status": status,
                    "success": error is None,
                    "elapsed_s": elapsed,
                    "error": error,
                    "result": extracted,
                }
            )
            state = "ok" if error is None else "failed"
            print(
                f"accuracy {args.tag}: input={input_len}, repeat={repeat}, "
                f"status={status}, result={state}, elapsed={elapsed:.3f}s"
            )
            case_id += 1

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": "accuracy",
        "tag": args.tag,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_url": args.base_url,
        "tokenizer": args.tokenizer,
        "input_lens": args.input_lens,
        "output_len": args.output_len,
        "prefill_logprob_tokens": args.prefill_logprob_tokens,
        "top_logprobs": args.top_logprobs,
        "cases": cases,
    }
    output = args.output or _default_output(f"{args.tag}_accuracy")
    _write_json(output, artifact)
    failures = sum(not case["success"] for case in cases)
    print(f"accuracy artifact={output}; failed_cases={failures}")
    return 0 if failures == 0 else 1


def _token_ids(rows: list[dict[str, Any]]) -> list[int | None]:
    return [row.get("token_id") for row in rows]


def _common_prefix_len(left: list[Any], right: list[Any]) -> int:
    length = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        length += 1
    return length


def _matching_logprob_diffs(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[float]:
    diffs = []
    for left_row, right_row in zip(left, right):
        if left_row.get("token_id") != right_row.get("token_id"):
            continue
        left_value = left_row.get("logprob")
        right_value = right_row.get("logprob")
        if left_value is None or right_value is None:
            continue
        if not (math.isfinite(left_value) and math.isfinite(right_value)):
            continue
        diffs.append(abs(left_value - right_value))
    return diffs


def _case_key(case: dict[str, Any]) -> tuple[int, int]:
    return int(case["target_input_len"]), int(case["repeat"])


def command_compare_accuracy(args: argparse.Namespace) -> int:
    baseline = _load_json(args.baseline)
    candidate = _load_json(args.candidate)
    baseline_cases = {_case_key(case): case for case in baseline.get("cases", [])}
    candidate_cases = {_case_key(case): case for case in candidate.get("cases", [])}
    keys = sorted(set(baseline_cases) | set(candidate_cases))
    comparisons = []
    overall_pass = True

    print(
        "input repeat first_token token_match common_prefix "
        "prefill_max_diff output_max_diff verdict"
    )
    for key in keys:
        left = baseline_cases.get(key)
        right = candidate_cases.get(key)
        errors = []
        if left is None or right is None:
            errors.append("case_missing")
        elif not left.get("success") or not right.get("success"):
            errors.append("request_failed")
        elif left.get("prompt_hash") != right.get("prompt_hash"):
            errors.append("prompt_mismatch")

        input_diffs: list[float] = []
        output_diffs: list[float] = []
        output_match = 0.0
        common_prefix = 0
        first_token_same = False
        left_output_ids: list[int | None] = []
        right_output_ids: list[int | None] = []
        if not errors:
            left_result = left["result"]
            right_result = right["result"]
            left_input = left_result.get("input_token_logprobs", [])
            right_input = right_result.get("input_token_logprobs", [])
            left_output = left_result.get("output_token_logprobs", [])
            right_output = right_result.get("output_token_logprobs", [])
            input_diffs = _matching_logprob_diffs(left_input, right_input)
            output_diffs = _matching_logprob_diffs(left_output, right_output)
            left_output_ids = _token_ids(left_output)
            right_output_ids = _token_ids(right_output)
            denominator = max(len(left_output_ids), len(right_output_ids), 1)
            output_match = sum(
                left_id == right_id
                for left_id, right_id in zip(left_output_ids, right_output_ids)
            ) / denominator
            common_prefix = _common_prefix_len(left_output_ids, right_output_ids)
            first_token_same = bool(
                left_output_ids
                and right_output_ids
                and left_output_ids[0] == right_output_ids[0]
            )
            if not input_diffs:
                errors.append("missing_prefill_logprobs")
            elif max(input_diffs) > args.max_prefill_logprob_diff:
                errors.append("prefill_logprob_diff")
            if not first_token_same:
                errors.append("first_token_mismatch")
            if output_match < args.min_output_token_match:
                errors.append("output_token_match")
            if output_diffs and max(output_diffs) > args.max_output_logprob_diff:
                errors.append("output_logprob_diff")

        case_pass = not errors
        overall_pass = overall_pass and case_pass
        comparison = {
            "target_input_len": key[0],
            "repeat": key[1],
            "pass": case_pass,
            "errors": errors,
            "first_token_same": first_token_same,
            "output_token_match": output_match,
            "output_common_prefix": common_prefix,
            "output_token_count_baseline": len(left_output_ids),
            "output_token_count_candidate": len(right_output_ids),
            "prefill_logprob_compared": len(input_diffs),
            "prefill_logprob_mean_abs_diff": (
                statistics.mean(input_diffs) if input_diffs else None
            ),
            "prefill_logprob_max_abs_diff": max(input_diffs) if input_diffs else None,
            "output_logprob_compared": len(output_diffs),
            "output_logprob_mean_abs_diff": (
                statistics.mean(output_diffs) if output_diffs else None
            ),
            "output_logprob_max_abs_diff": max(output_diffs) if output_diffs else None,
        }
        comparisons.append(comparison)
        print(
            f"{key[0]:5d} {key[1]:6d} {str(first_token_same):11s} "
            f"{output_match:11.4f} {common_prefix:13d} "
            f"{comparison['prefill_logprob_max_abs_diff']!s:16s} "
            f"{comparison['output_logprob_max_abs_diff']!s:15s} "
            f"{'PASS' if case_pass else 'FAIL:' + ','.join(errors)}"
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "accuracy_comparison",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_tag": baseline.get("tag"),
        "candidate_tag": candidate.get("tag"),
        "thresholds": {
            "max_prefill_logprob_diff": args.max_prefill_logprob_diff,
            "max_output_logprob_diff": args.max_output_logprob_diff,
            "min_output_token_match": args.min_output_token_match,
        },
        "pass": overall_pass,
        "comparisons": comparisons,
    }
    if args.output:
        _write_json(args.output, summary)
        print(f"comparison artifact={args.output}")
    print(f"accuracy comparison verdict: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


def _stream_subprocess(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n$ " + " ".join(command) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return process.wait()


def command_perf(args: argparse.Namespace) -> int:
    output_file = args.output_file
    if output_file is None:
        output_file = DEFAULT_RESULTS_DIR / f"{args.tag}_perf_{_timestamp()}.jsonl"
    output_file = output_file.resolve()
    if output_file.exists() and not args.append:
        print(
            "refusing to append to existing result file without --append: "
            f"{output_file}",
            file=sys.stderr,
        )
        return 2
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log_file = output_file.with_suffix(output_file.suffix + ".log")

    env = os.environ.copy()
    local_python = str(REPO_ROOT / "python")
    env["PYTHONPATH"] = (
        local_python
        if not env.get("PYTHONPATH")
        else local_python + os.pathsep + env["PYTHONPATH"]
    )

    for input_len in args.input_lens:
        for output_len in args.output_lens:
            for concurrency in args.concurrencies:
                num_prompts = max(args.num_prompts, concurrency * 2)
                case_tag = (
                    f"{args.tag}_in{input_len}_out{output_len}_c{concurrency}"
                )
                command = [
                    sys.executable,
                    "-m",
                    "sglang.benchmark.serving",
                    "--backend",
                    "sglang",
                    "--base-url",
                    args.base_url,
                    "--dataset-name",
                    "random-ids",
                    "--model",
                    args.model,
                    "--tokenizer",
                    args.tokenizer,
                    "--tokenize-prompt",
                    "--num-prompts",
                    str(num_prompts),
                    "--random-input-len",
                    str(input_len),
                    "--random-output-len",
                    str(output_len),
                    "--random-range-ratio",
                    "0",
                    "--request-rate",
                    "inf",
                    "--max-concurrency",
                    str(concurrency),
                    "--warmup-requests",
                    str(args.warmup_requests),
                    "--flush-cache",
                    "--disable-tqdm",
                    "--seed",
                    str(args.seed),
                    "--tag",
                    case_tag,
                    "--output-details",
                    "--output-file",
                    str(output_file),
                ]
                print(
                    f"\n=== perf {case_tag}: prompts={num_prompts}, "
                    f"result={output_file} ==="
                )
                ret = _stream_subprocess(command, log_file, env)
                if ret != 0:
                    print(f"benchmark case {case_tag} failed with exit code {ret}")
                    return ret
    print(f"performance result={output_file}; console_log={log_file}")
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def _aggregate_perf(
    records: list[dict[str, Any]],
) -> dict[tuple[int, int, int], dict[str, float]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            int(record["random_input_len"]),
            int(record["random_output_len"]),
            int(record["max_concurrency"]),
        )
        grouped[key].append(record)

    metrics = (
        "completed",
        "request_throughput",
        "input_throughput",
        "median_ttft_ms",
        "p90_ttft_ms",
        "median_tpot_ms",
        "median_e2e_latency_ms",
    )
    aggregated = {}
    for key, group in grouped.items():
        aggregated[key] = {
            metric: statistics.median(float(record[metric]) for record in group)
            for metric in metrics
        }
        aggregated[key]["failed_requests"] = sum(
            sum(bool(error) for error in (record.get("errors") or []))
            for record in group
        )
    return aggregated


def _gain(candidate: float, baseline: float, *, lower_is_better: bool) -> float | None:
    if baseline == 0:
        return None
    if lower_is_better:
        return (baseline - candidate) / baseline * 100.0
    return (candidate - baseline) / baseline * 100.0


def command_compare_perf(args: argparse.Namespace) -> int:
    baseline = _aggregate_perf(_read_jsonl(args.baseline))
    candidate = _aggregate_perf(_read_jsonl(args.candidate))
    keys = sorted(set(baseline) | set(candidate))
    missing = False
    failed_requests = False
    comparisons = []
    print(
        "input output conc base_ttft cand_ttft ttft_gain% "
        "input_tps_gain% req_tps_gain% tpot_gain%"
    )
    for key in keys:
        left = baseline.get(key)
        right = candidate.get(key)
        if left is None or right is None:
            missing = True
            print(f"{key[0]:5d} {key[1]:6d} {key[2]:4d} MISSING_CASE")
            continue
        if left["failed_requests"] or right["failed_requests"]:
            failed_requests = True
            print(
                f"{key[0]:5d} {key[1]:6d} {key[2]:4d} FAILED_REQUESTS "
                f"baseline={left['failed_requests']:.0f} "
                f"candidate={right['failed_requests']:.0f}"
            )
            continue
        comparison = {
            "input_len": key[0],
            "output_len": key[1],
            "concurrency": key[2],
            "baseline": left,
            "candidate": right,
            "ttft_gain_pct": _gain(
                right["median_ttft_ms"], left["median_ttft_ms"], lower_is_better=True
            ),
            "p90_ttft_gain_pct": _gain(
                right["p90_ttft_ms"], left["p90_ttft_ms"], lower_is_better=True
            ),
            "input_throughput_gain_pct": _gain(
                right["input_throughput"],
                left["input_throughput"],
                lower_is_better=False,
            ),
            "request_throughput_gain_pct": _gain(
                right["request_throughput"],
                left["request_throughput"],
                lower_is_better=False,
            ),
            "tpot_gain_pct": _gain(
                right["median_tpot_ms"], left["median_tpot_ms"], lower_is_better=True
            ),
        }
        comparisons.append(comparison)
        print(
            f"{key[0]:5d} {key[1]:6d} {key[2]:4d} "
            f"{left['median_ttft_ms']:9.2f} {right['median_ttft_ms']:9.2f} "
            f"{comparison['ttft_gain_pct']!s:10s} "
            f"{comparison['input_throughput_gain_pct']!s:15s} "
            f"{comparison['request_throughput_gain_pct']!s:13s} "
            f"{comparison['tpot_gain_pct']!s}"
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "performance_comparison",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "missing_cases": missing,
        "failed_requests": failed_requests,
        "comparisons": comparisons,
    }
    if args.output:
        _write_json(args.output, summary)
        print(f"performance comparison artifact={args.output}")
    return 1 if missing or failed_requests else 0


def command_diag(args: argparse.Namespace) -> int:
    files = sorted(args.diag_dir.rglob("*.jsonl"))
    if not files:
        print(f"no JSONL diagnostics found under {args.diag_dir}", file=sys.stderr)
        return 2
    records = []
    invalid_lines = 0
    for path in files:
        try:
            records.extend(_read_jsonl(path))
        except ValueError as exc:
            print(exc, file=sys.stderr)
            invalid_lines += 1

    counts = Counter(record.get("event", "<missing>") for record in records)
    managers = [
        record
        for record in records
        if record.get("event") == "manager_register_start"
    ]
    plans = counts["send_kvcache_plan"]
    failure_names = {
        "send_kvcache_missing_registration",
        "send_kvcache_registration_length_mismatch",
        "send_kvcache_pp_metadata_mismatch",
        "invalid_item_length",
        "item_length_mismatch",
        "source_range_out_of_bounds",
        "destination_range_out_of_bounds",
        "memfabric_transfer_exception",
        "combined_transfer_failed",
        "per_buffer_retry_failed",
    }

    def has_nonzero_ret(record: dict[str, Any]) -> bool:
        try:
            return int(record.get("ret", 0)) != 0
        except (TypeError, ValueError):
            return True

    failures = [
        record
        for record in records
        if record.get("event") in failure_names
        or str(record.get("event", "")).endswith("_exception")
        or (
            str(record.get("event", "")).endswith("_result")
            and has_nonzero_ret(record)
        )
    ]
    manager_bad = [
        record
        for record in managers
        if record.get("is_mla_backend") is not True
        or record.get("kv_buf_groups") != 2
    ]

    service_log_hits = []
    patterns = (
        "out of range",
        "ret=-2000",
        "Decode transfer failed",
        "Prefill transfer failed",
    )
    for path in args.service_log:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), 1):
            if any(pattern in line for pattern in patterns) or (
                "Session " in line and " failed" in line
            ):
                service_log_hits.append(
                    {"path": str(path), "line": line_number, "text": line[:500]}
                )

    errors = []
    if invalid_lines:
        errors.append("invalid_jsonl")
    if not managers:
        errors.append("missing_manager_registration")
    if manager_bad:
        errors.append("invalid_mla_layout")
    if args.role == "prefill" and plans == 0:
        errors.append("missing_send_kvcache_plan")
    if failures:
        errors.append("transfer_failure")
    if service_log_hits:
        errors.append("service_log_failure")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "diagnostic_check",
        "diag_dir": str(args.diag_dir),
        "role": args.role,
        "files": [str(path) for path in files],
        "records": len(records),
        "event_counts": dict(counts),
        "manager_count": len(managers),
        "invalid_manager_count": len(manager_bad),
        "send_kvcache_plan_count": plans,
        "failure_count": len(failures),
        "failures": failures[:50],
        "service_log_hits": service_log_hits[:50],
        "pass": not errors,
        "errors": errors,
    }
    print(
        f"diag role={args.role}: files={len(files)}, records={len(records)}, "
        f"managers={len(managers)}, plans={plans}, failures={len(failures)}, "
        f"service_log_hits={len(service_log_hits)}, verdict="
        f"{'PASS' if not errors else 'FAIL:' + ','.join(errors)}"
    )
    for record in manager_bad[:10]:
        print(
            "  bad manager: "
            f"is_mla_backend={record.get('is_mla_backend')}, "
            f"kv_buf_groups={record.get('kv_buf_groups')}, pid={record.get('pid')}"
        )
    for record in failures[:10]:
        print(
            f"  transfer failure: event={record.get('event')}, "
            f"ret={record.get('ret')}, transfer_id={record.get('transfer_id')}"
        )
    if args.output:
        _write_json(args.output, summary)
        print(f"diagnostic artifact={args.output}")
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "smoke", help="send concurrent exact-length requests through the PD router"
    )
    smoke.add_argument("--base-url", default="http://127.0.0.1:6688")
    smoke.add_argument("--tokenizer", required=True)
    smoke.add_argument("--tag", required=True)
    smoke.add_argument("--input-len", type=_positive_int, default=2048)
    smoke.add_argument("--output-len", type=_positive_int, default=32)
    smoke.add_argument("--num-requests", type=_positive_int, default=8)
    smoke.add_argument("--concurrency", type=_positive_int, default=4)
    smoke.add_argument("--timeout", type=float, default=1800)
    smoke.add_argument("--output", type=Path)
    smoke.set_defaults(func=command_smoke)

    accuracy = subparsers.add_parser(
        "accuracy", help="collect deterministic token and logprob artifacts"
    )
    accuracy.add_argument("--base-url", default="http://127.0.0.1:6688")
    accuracy.add_argument("--tokenizer", required=True)
    accuracy.add_argument("--tag", required=True)
    accuracy.add_argument(
        "--input-lens", type=_csv_positive_ints, default=[2048, 8192]
    )
    accuracy.add_argument("--output-len", type=_positive_int, default=32)
    accuracy.add_argument("--repeats", type=_positive_int, default=1)
    accuracy.add_argument("--prefill-logprob-tokens", type=_positive_int, default=64)
    accuracy.add_argument("--top-logprobs", type=_positive_int, default=5)
    accuracy.add_argument("--timeout", type=float, default=1800)
    accuracy.add_argument("--output", type=Path)
    accuracy.set_defaults(func=command_accuracy)

    compare_accuracy = subparsers.add_parser(
        "compare-accuracy", help="compare two deterministic accuracy artifacts"
    )
    compare_accuracy.add_argument("baseline", type=Path)
    compare_accuracy.add_argument("candidate", type=Path)
    compare_accuracy.add_argument(
        "--max-prefill-logprob-diff", type=_nonnegative_float, default=0.15
    )
    compare_accuracy.add_argument(
        "--max-output-logprob-diff", type=_nonnegative_float, default=0.15
    )
    compare_accuracy.add_argument(
        "--min-output-token-match", type=_ratio, default=0.95
    )
    compare_accuracy.add_argument("--output", type=Path)
    compare_accuracy.set_defaults(func=command_compare_accuracy)

    perf = subparsers.add_parser(
        "perf", help="run an exact-token serving benchmark matrix"
    )
    perf.add_argument("--base-url", default="http://127.0.0.1:6688")
    perf.add_argument("--model", required=True)
    perf.add_argument("--tokenizer", required=True)
    perf.add_argument("--tag", required=True)
    perf.add_argument(
        "--input-lens", type=_csv_positive_ints, default=[1024, 4096, 8192]
    )
    perf.add_argument("--output-lens", type=_csv_positive_ints, default=[1, 32])
    perf.add_argument("--concurrencies", type=_csv_positive_ints, default=[1, 4])
    perf.add_argument("--num-prompts", type=_positive_int, default=8)
    perf.add_argument("--warmup-requests", type=_positive_int, default=1)
    perf.add_argument("--seed", type=int, default=42)
    perf.add_argument("--output-file", type=Path)
    perf.add_argument("--append", action="store_true")
    perf.set_defaults(func=command_perf)

    compare_perf = subparsers.add_parser(
        "compare-perf", help="compare two serving benchmark JSONL files"
    )
    compare_perf.add_argument("baseline", type=Path)
    compare_perf.add_argument("candidate", type=Path)
    compare_perf.add_argument("--output", type=Path)
    compare_perf.set_defaults(func=command_compare_perf)

    diag = subparsers.add_parser(
        "diag", help="validate persisted Ascend KV transfer diagnostics"
    )
    diag.add_argument("--diag-dir", type=Path, required=True)
    diag.add_argument("--role", choices=["prefill", "decode"], required=True)
    diag.add_argument("--service-log", type=Path, action="append", default=[])
    diag.add_argument("--output", type=Path)
    diag.set_defaults(func=command_diag)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
