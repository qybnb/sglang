import argparse
import ast
import json
import os
import re
import time

import numpy as np
from datasets import load_dataset

from sglang.lang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import download_and_cache_file, dump_state_text, read_jsonl

INVALID = -9999999
SPECULATIVE_META_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "spec_accept_rate",
    "spec_accept_length",
    "spec_num_correct_drafts",
    "spec_num_proposed_drafts",
    "spec_verify_ct",
    "spec_correct_drafts_histogram",
)


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_evaluation_examples(lines, num_shots, num_questions):
    """Return held-out examples, excluding the demonstrations from scoring."""
    if num_shots < 0:
        raise ValueError(f"num_shots must be non-negative, got {num_shots}")
    if num_questions < 0:
        raise ValueError(
            f"num_questions must be non-negative, got {num_questions}"
        )
    return lines[num_shots : num_shots + num_questions]


def collect_speculative_metrics(states):
    per_request = []
    for state in states:
        try:
            meta_info = state.get_meta_info("answer") or {}
        except Exception:
            # A transport/server failure may leave no generated variable.  It
            # must still be represented in the raw artifact instead of making
            # the entire benchmark fail while summarizing metadata.
            meta_info = {}
        per_request.append(
            {key: meta_info[key] for key in SPECULATIVE_META_KEYS if key in meta_info}
        )

    active = [row for row in per_request if row.get("spec_verify_ct", 0) > 0]
    total_verify_ct = sum(row.get("spec_verify_ct", 0) for row in active)
    total_correct_drafts = sum(
        row.get("spec_num_correct_drafts", 0) for row in active
    )
    total_proposed_drafts = sum(
        row.get("spec_num_proposed_drafts", 0) for row in active
    )
    total_completion_tokens = sum(row.get("completion_tokens", 0) for row in active)
    summary = {
        "requests_with_spec_verify": len(active),
        "total_spec_verify_ct": total_verify_ct,
        "spec_accept_rate": (
            total_correct_drafts / total_proposed_drafts
            if total_proposed_drafts > 0
            else None
        ),
        "spec_accept_length": (
            total_completion_tokens / total_verify_ct if total_verify_ct > 0 else None
        ),
    }
    return per_request, summary


def collect_state_outputs(states):
    """Extract completed outputs while retaining failed request diagnostics."""
    answers = []
    errors = []
    for state in states:
        try:
            answers.append(state["answer"])
            errors.append(None)
        except Exception as exc:
            answers.append("")
            try:
                state_error = state.error()
            except Exception:
                state_error = None
            errors.append(str(state_error or exc))
    return answers, errors


def dump_gsm8k_raw_result(
    path, states, answers, preds, labels, per_request_spec_metrics, errors
):
    """Write one row per requested question, including failed requests."""
    if not path:
        return

    rows = []
    for i, state in enumerate(states):
        try:
            full_text = state.text()
        except Exception:
            full_text = ""
        answer = answers[i]
        prompt = (
            full_text.removesuffix(answer)
            if answer and full_text.endswith(answer)
            else full_text
        )
        rows.append(
            {
                "prompt_id": i,
                "prompt": prompt,
                "output": answer,
                "label": labels[i],
                "prediction": preds[i],
                "correct": bool(errors[i] is None and preds[i] == labels[i]),
                "error": errors[i],
                "meta_info": per_request_spec_metrics[i],
            }
        )

    print(f"GSM8K raw results saved to {path}")
    with open(path, "w") as fout:
        fout.write("\n".join(json.dumps(row) for row in rows))


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def main(args):
    # Select backend
    set_default_backend(select_sglang_backend(args))

    # Load tokenizer if enable_thinking is set
    tokenizer = None
    if args.enable_thinking:
        from transformers import AutoTokenizer

        assert (
            args.tokenizer_path is not None
        ), "--tokenizer-path is required when --enable-thinking is set"
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )

    # Read data
    if args.platinum:
        print("Loading GSM8K Platinum dataset from HuggingFace...")
        dataset = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
        lines = [
            {"question": item["question"], "answer": item["answer"]} for item in dataset
        ]
    else:
        data_path = args.data_path
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        if not os.path.isfile(data_path):
            data_path = download_and_cache_file(url)
        lines = list(read_jsonl(data_path))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)
    evaluation_lines = get_evaluation_examples(lines, num_shots, num_questions)

    questions = []
    labels = []
    for i, example in enumerate(evaluation_lines):
        raw_question = few_shot_examples + get_one_example(
            evaluation_lines, i, False
        )
        if tokenizer is not None:
            messages = [{"role": "user", "content": raw_question}]
            raw_question = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        questions.append(raw_question)
        labels.append(get_answer_value(example["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def few_shot_gsm8k(s, question):
        s += question
        s += sgl.gen(
            "answer",
            max_tokens=args.max_new_tokens,
            stop=["Question", "Assistant:", "<|separator|>"],
        )

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    tic = time.perf_counter()
    states = few_shot_gsm8k.run_batch(
        arguments,
        temperature=args.temperature,
        top_p=args.top_p,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    answers, errors = collect_state_outputs(states)
    preds = [get_answer_value(answer) for answer in answers]

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Compute speed
    per_request_spec_metrics, spec_summary = collect_speculative_metrics(states)
    num_output_tokens = sum(
        row.get("completion_tokens", 0) for row in per_request_spec_metrics
    )
    output_throughput = num_output_tokens / latency
    num_failed_requests = sum(error is not None for error in errors)

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")
    print(f"Failed requests: {num_failed_requests}/{len(states)}")
    print(
        "Speculative decode: "
        f"requests={spec_summary['requests_with_spec_verify']}/{len(states)}, "
        f"verify_ct={spec_summary['total_spec_verify_ct']}, "
        f"accept_rate={spec_summary['spec_accept_rate']}, "
        f"accept_length={spec_summary['spec_accept_length']}"
    )

    # Dump results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)
    dump_gsm8k_raw_result(
        args.raw_result_file,
        states,
        answers,
        preds,
        labels,
        per_request_spec_metrics,
        errors,
    )

    with open(args.result_file, "a") as fout:
        value = {
            "task": "gsm8k-platinum" if args.platinum else "gsm8k",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "accuracy": round(acc, 3),
            "num_requests": len(evaluation_lines),
            "other": {
                "num_questions": len(evaluation_lines),
                "parallel": args.parallel,
                "num_shots": num_shots,
                "evaluation_start_index": num_shots,
                "failed_requests": num_failed_requests,
                "speculative_decode": spec_summary,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode by wrapping prompts with chat template",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer (required when --enable-thinking is set)",
    )
    parser.add_argument(
        "--platinum",
        action="store_true",
        help="Use GSM8K Platinum dataset (drop-in replacement with corrected labels)",
    )
    args = add_common_sglang_args_and_parse(parser)
    main(args)
