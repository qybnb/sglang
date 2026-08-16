"""
Usage:
python3 -m sglang.test.run_eval --port 30000 --eval-name mmlu --num-examples 10
"""

import argparse
import json
import os
import threading
import time

from sglang.test.simple_eval_common import (
    ChatCompletionSampler,
    CompletionSampler,
    Eval,
    make_report,
    set_ulimit,
)


class _IncrementalResultWriter:
    """Persist completed examples so an interrupted evaluation remains usable."""

    def __init__(self, raw_result_file: str, total: int):
        self.raw_result_file = os.path.abspath(raw_result_file)
        self.summary_file = os.path.join(
            os.path.dirname(self.raw_result_file), "partial_summary.json"
        )
        self.total = int(total)
        self.completed = 0
        self.scored = 0
        self.score_sum = 0.0
        self.status = "running"
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.raw_result_file), exist_ok=True)
        with open(self.raw_result_file, "w"):
            pass
        self._write_summary_locked("running")

    @staticmethod
    def _serialize_result(result) -> dict:
        record = dict(result.record or {})
        record.setdefault("score", result.score)
        record.setdefault("metrics", result.metrics)
        record.setdefault("conversation", result.convo)
        return record

    def __call__(self, result) -> None:
        record = self._serialize_result(result)
        line = json.dumps(record) + "\n"
        with self._lock:
            with open(self.raw_result_file, "a") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            self.completed += 1
            if result.score is not None:
                self.scored += 1
                self.score_sum += float(result.score)
            self._write_summary_locked(self.status)

    def mark_interrupted(self) -> None:
        with self._lock:
            self.status = "interrupted"
            self._write_summary_locked(self.status)

    def mark_complete(self) -> None:
        with self._lock:
            self.status = "complete"
            self._write_summary_locked(self.status)

    def _write_summary_locked(self, status: str) -> None:
        score = self.score_sum / self.scored if self.scored else None
        summary = {
            "status": status,
            "completed": self.completed,
            "total": self.total,
            "score_on_completed": score,
            "raw_result_file": self.raw_result_file,
            "updated_at_unix": time.time(),
        }
        tmp_file = f"{self.summary_file}.tmp"
        with open(tmp_file, "w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, self.summary_file)


def get_thinking_kwargs(args):
    thinking_mode = getattr(args, "thinking_mode", None)
    if thinking_mode in THINKING_MODE_CHOICES:
        if thinking_mode in ["deepseek-v3", "kimi-k2"]:
            thinking_param = "thinking"
        else:
            # All models other than dpsk v3/kimi_k2
            thinking_param = "enable_thinking"
        return {thinking_param: True}
    return {}


def parse_json_object(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError("must be a valid JSON object string") from e

    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")

    return parsed


def run_eval_once(args, base_url: str, eval_obj: Eval) -> dict:
    chat_template_kwargs = getattr(args, "chat_template_kwargs", None)
    if isinstance(chat_template_kwargs, str):
        chat_template_kwargs = parse_json_object(chat_template_kwargs)
    elif chat_template_kwargs is None:
        chat_template_kwargs = {}
    elif not isinstance(chat_template_kwargs, dict):
        raise ValueError("chat_template_kwargs must be a dict or a JSON object string")

    chat_template_kwargs = {**get_thinking_kwargs(args), **chat_template_kwargs}

    extra_body = {}
    if chat_template_kwargs:
        extra_body["chat_template_kwargs"] = chat_template_kwargs

    for param_name in ("top_k", "min_p"):
        value = getattr(args, param_name, None)
        if value is not None:
            extra_body[param_name] = value

    common_kwargs = dict(
        model=getattr(args, "model", None),
        max_tokens=getattr(args, "max_tokens", 2048),
        top_p=getattr(args, "top_p", 1.0),
        base_url=base_url,
        temperature=getattr(args, "temperature", 0.0),
    )

    api_mode = getattr(args, "api", "chat")
    if api_mode == "completion":
        # Default stop tokens for completion API (matches few_shot_gsm8k behavior)
        stop = getattr(args, "stop", ["Question", "Assistant:", "<|separator|>"])
        sampler = CompletionSampler(
            **common_kwargs,
            stop=stop,
        )
    else:
        sampler = ChatCompletionSampler(
            **common_kwargs,
            reasoning_effort=getattr(args, "reasoning_effort", None),
            extra_body=extra_body if extra_body else None,
        )

    # Run eval. GPQA may attach a durable per-example writer so Ctrl+C keeps
    # every response that finished before the interruption.
    tic = time.perf_counter()
    try:
        result = eval_obj(sampler)
    except KeyboardInterrupt:
        result_callback = getattr(eval_obj, "result_callback", None)
        if hasattr(result_callback, "mark_interrupted"):
            result_callback.mark_interrupted()
            print(
                "GPQA interrupted; completed examples were preserved at "
                f"{result_callback.raw_result_file}"
            )
        raise
    latency = time.perf_counter() - tic

    return result, latency, sampler


def run_eval(args):
    # Lazy import to avoid circular dependency with test_utils
    from sglang.test.test_utils import dump_metric

    set_ulimit()

    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "EMPTY"

    base_url = (
        f"{args.base_url}/v1" if args.base_url else f"http://{args.host}:{args.port}/v1"
    )

    incremental_result_writer = None

    if args.eval_name == "mmlu":
        from sglang.test.simple_eval_mmlu import MMLUEval

        filename = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
        eval_obj = MMLUEval(filename, args.num_examples, args.num_threads)
    elif args.eval_name == "math":
        from sglang.test.simple_eval_math import MathEval

        equality_checker = ChatCompletionSampler(model="gpt-4-turbo")

        filename = (
            "https://openaipublic.blob.core.windows.net/simple-evals/math_test.csv"
        )
        eval_obj = MathEval(
            filename, equality_checker, args.num_examples, args.num_threads
        )
    elif args.eval_name == "mgsm":
        from sglang.test.simple_eval_mgsm import MGSMEval

        eval_obj = MGSMEval(args.num_examples, args.num_threads)
    elif args.eval_name == "mgsm_en":
        from sglang.test.simple_eval_mgsm import MGSMEval

        eval_obj = MGSMEval(args.num_examples, args.num_threads, languages=["en"])
    elif args.eval_name == "gpqa":
        from sglang.test.simple_eval_gpqa import GPQAEval

        filename = getattr(args, "gpqa_data_path", None) or (
            "https://openaipublic.blob.core.windows.net/simple-evals/"
            "gpqa_diamond.csv"
        )
        eval_obj = GPQAEval(filename, args.num_examples, args.num_threads)
        raw_result_file = getattr(args, "raw_result_file", None)
        if raw_result_file and getattr(args, "repeat", 1) == 1:
            incremental_result_writer = _IncrementalResultWriter(
                raw_result_file=raw_result_file,
                total=len(eval_obj.examples),
            )
            eval_obj.result_callback = incremental_result_writer
    elif args.eval_name == "humaneval":
        from sglang.test.simple_eval_humaneval import HumanEval

        eval_obj = HumanEval(args.num_examples, args.num_threads)
    elif args.eval_name == "longbench_v2":
        from sglang.test.simple_eval_longbench_v2 import LongBenchV2Eval

        # Default to HuggingFace dataset, can be overridden with --dataset-path
        data_source = args.dataset_path
        categories = args.categories.split(",") if args.categories else None

        eval_obj = LongBenchV2Eval(
            model=getattr(args, "model", None),
            data_source=data_source,
            num_examples=args.num_examples,
            num_threads=args.num_threads,
            categories=categories,
            max_context_length=getattr(args, "max_context_length", None),
            min_context_length=getattr(args, "min_context_length", None),
        )
    elif args.eval_name == "mmmu":
        # VLM MMMU evaluation with fixed 100 examples by default
        from sglang.test.simple_eval_mmmu_vlm import MMMUVLMEval

        eval_obj = MMMUVLMEval(
            args.num_examples,
            args.num_threads,
            response_answer_regex=getattr(args, "response_answer_regex", None),
        )
    elif args.eval_name == "aime25":
        from sglang.test.simple_eval_aime25 import AIME25Eval

        eval_obj = AIME25Eval(args.num_examples, args.num_threads)
    elif args.eval_name == "gsm8k":
        from sglang.test.simple_eval_gsm8k import GSM8KEval

        eval_obj = GSM8KEval(
            num_examples=args.num_examples,
            num_threads=args.num_threads,
            num_shots=getattr(args, "num_shots", 5),
            data_path=getattr(args, "gsm8k_data_path", None),
        )
    elif args.eval_name == "mixed_prefix_gsm8k":
        from sglang.test.simple_eval_mixed_prefix_gsm8k import MixedPrefixGSM8KEval

        eval_obj = MixedPrefixGSM8KEval(
            num_examples=args.num_examples,
            num_threads=args.num_threads,
            num_shots=args.num_shots,
            secondary_pool_size=args.mixed_prefix_gsm8k_secondary_pool_size,
            data_path=args.gsm8k_data_path,
            seed=args.mixed_prefix_gsm8k_seed,
        )
    else:
        raise ValueError(f"Invalid eval name: {args.eval_name}")

    if getattr(args, "repeat", 1) == 1:
        result, latency, sampler = run_eval_once(args, base_url, eval_obj)
        metrics = result.metrics | {"score": result.score}
        metrics["latency"] = latency
        print(f"Total latency: {latency:.3f} s")
        print(f"Score: {metrics['score']:.3f}")

        # Compute output throughput from accumulated completion tokens
        total_completion_tokens = sum(sampler._completion_tokens)
        if total_completion_tokens > 0 and latency > 0:
            metrics["output_throughput"] = total_completion_tokens / latency
            print(f"Output throughput: {metrics['output_throughput']:.3f} token/s")

        # Report metrics to unified collection framework
        dump_metric(
            f"{args.eval_name}_score",
            metrics["score"],
            labels={"model": sampler.model, "eval": args.eval_name},
        )
        dump_metric(
            f"{args.eval_name}_latency",
            latency,
            labels={"model": sampler.model, "eval": args.eval_name},
        )
    else:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=args.repeat)

        futures = [
            executor.submit(run_eval_once, args, base_url, eval_obj)
            for _ in range(args.repeat)
        ]

        scores_repeat = []
        latencies = []
        total_completion_tokens = 0

        for f in futures:
            result, latency, sampler = f.result()
            scores_repeat.append(result.score)
            latencies.append(latency)
            total_completion_tokens += sum(sampler._completion_tokens)

        mean_score = sum(scores_repeat) / len(scores_repeat)
        mean_latency = sum(latencies) / len(latencies)
        total_latency = sum(latencies)
        scores_repeat = [f"{s:.3f}" for s in scores_repeat]
        print("=" * 20)
        print(f"Repeat: {args.repeat}, mean: {mean_score:.3f}")
        print(f"Scores: {scores_repeat}")
        print(f"Mean latency: {mean_latency:.3f} s")
        print("=" * 20)
        metrics = result.metrics | {"scores": scores_repeat}
        metrics = metrics | {"mean_score": mean_score}
        metrics["latency"] = mean_latency

        if total_completion_tokens > 0 and total_latency > 0:
            metrics["output_throughput"] = total_completion_tokens / total_latency
            print(f"Output throughput: {metrics['output_throughput']:.3f} token/s")

        # Report metrics to unified collection framework
        dump_metric(
            f"{args.eval_name}_mean_score",
            mean_score,
            labels={
                "model": sampler.model,
                "eval": args.eval_name,
                "repeat": args.repeat,
            },
        )

        executor.shutdown()

    # Dump reports
    file_stem = f"{args.eval_name}_{sampler.model.replace('/', '_')}"
    output_dir = os.path.abspath(getattr(args, "output_dir", None) or "/tmp")
    os.makedirs(output_dir, exist_ok=True)
    report_filename = os.path.join(output_dir, f"{file_stem}.html")
    print(f"Writing report to {report_filename}")
    with open(report_filename, "w") as fh:
        fh.write(make_report(result))
    print(metrics)
    result_filename = os.path.join(output_dir, f"{file_stem}.json")
    with open(result_filename, "w") as f:
        f.write(json.dumps(metrics, indent=2))
    print(f"Writing results to {result_filename}")

    raw_result_file = getattr(args, "raw_result_file", None)
    if raw_result_file:
        raw_result_file = os.path.abspath(raw_result_file)
        raw_parent = os.path.dirname(raw_result_file)
        if raw_parent:
            os.makedirs(raw_parent, exist_ok=True)
        with open(raw_result_file, "w") as f:
            for index, record in enumerate(result.records):
                f.write(json.dumps({"index": index, **record}) + "\n")
        print(f"Writing raw results to {raw_result_file}")

    if incremental_result_writer is not None:
        incremental_result_writer.mark_complete()

    if getattr(args, "return_latency", False):
        return metrics, latency
    return metrics


THINKING_MODE_CHOICES = ["deepseek-v3", "qwen-3", "glm-45", "kimi-k2"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Default host is 0.0.0.0."
    )
    parser.add_argument(
        "--port",
        type=int,
        help="If not set, the default port is configured according to its default value for different LLM Inference Engines.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name or path of the model. If not set, the default model will request /v1/models for conf.",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="repeat the evaluation n times"
    )
    parser.add_argument("--eval-name", type=str, default="mmlu")
    parser.add_argument(
        "--api",
        type=str,
        default="chat",
        choices=["chat", "completion"],
        help="API mode: 'chat' for /v1/chat/completions, 'completion' for /v1/completions",
    )
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--num-threads", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--top-k", type=int, default=None, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--min-p", type=float, default=None, help="Min-p sampling parameter"
    )
    parser.add_argument(
        "--chat-template-kwargs",
        type=parse_json_object,
        default=None,
        help="JSON object string for chat_template_kwargs, e.g. '{\"enable_thinking\": true}'",
    )
    parser.add_argument("--reasoning-effort", type=str)
    parser.add_argument(
        "--thinking-mode",
        default=None,
        type=str,
        choices=THINKING_MODE_CHOICES,
        help="Enable thinking mode in Deepseek V3.1/3.2, or Qwen3.--reasoning-parser must be set when launching the server.",
    )

    # LongBench-v2 specific arguments
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="THUDM/LongBench-v2",
        help="Path to dataset file or HuggingFace dataset name for LongBench-v2",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list of categories to evaluate for LongBench-v2",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        help="Maximum context length in characters for LongBench-v2",
    )
    parser.add_argument(
        "--min-context-length",
        type=int,
        help="Minimum context length in characters for LongBench-v2",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=5,
        help="Number of few-shot examples for GSM8K (default: 5)",
    )
    parser.add_argument(
        "--gsm8k-data-path",
        type=str,
        default=None,
        help="Path to GSM8K data file (e.g., test.jsonl)",
    )
    parser.add_argument(
        "--gpqa-data-path",
        type=str,
        default=None,
        help="Path to a local GPQA Diamond CSV; defaults to the public URL.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the HTML report and summary JSON (default: /tmp).",
    )
    parser.add_argument(
        "--raw-result-file",
        type=str,
        default=None,
        help="Optional JSONL file for machine-readable per-example results.",
    )
    parser.add_argument(
        "--mixed-prefix-gsm8k-secondary-pool-size",
        type=int,
        default=15,
        help="Size of secondary example pool for eval_name=mixed_prefix_gsm8k (default: 15)",
    )
    parser.add_argument(
        "--mixed-prefix-gsm8k-seed",
        type=int,
        default=42,
        help="Seed for per-question random sampling in mixed_prefix_gsm8k (default: 42)",
    )

    args = parser.parse_args()

    run_eval(args)
