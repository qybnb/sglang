import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "kimi_k3_pcp_validation.py"
)
SPEC = importlib.util.spec_from_file_location("kimi_k3_pcp_validation", SCRIPT_PATH)
VALIDATION = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VALIDATION)


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) % 251 for character in text]


def _accuracy_case(token_offset=0, logprob_offset=0.0):
    rows = [
        {"logprob": -0.1 - logprob_offset, "token_id": 10 + token_offset},
        {"logprob": -0.2 - logprob_offset, "token_id": 11 + token_offset},
    ]
    return {
        "case_id": 0,
        "repeat": 0,
        "target_input_len": 2048,
        "output_len": 2,
        "prompt_hash": "same",
        "success": True,
        "result": {
            "input_token_logprobs": rows,
            "output_token_logprobs": rows,
        },
    }


class TestKimiK3PCPValidation(unittest.TestCase):
    def test_exact_length_prompts_are_case_specific(self):
        tokenizer = _FakeTokenizer()
        first = VALIDATION._make_input_ids(tokenizer, 2048, 1)
        second = VALIDATION._make_input_ids(tokenizer, 2048, 2)

        self.assertEqual(len(first), 2048)
        self.assertEqual(len(second), 2048)
        self.assertNotEqual(
            VALIDATION._prompt_hash(first), VALIDATION._prompt_hash(second)
        )

    def test_normalises_tuple_and_dict_logprobs(self):
        rows = VALIDATION._normalise_logprob_rows(
            [[-0.5, 7, "x"], {"logprob": -0.25, "token_id": 8, "text": "y"}]
        )

        self.assertEqual(rows[0], {"logprob": -0.5, "token_id": 7, "text": "x"})
        self.assertEqual(rows[1], {"logprob": -0.25, "token_id": 8, "text": "y"})

    def test_accuracy_repeats_reuse_the_same_prompt(self):
        response = {
            "text": "ok",
            "meta_info": {
                "prompt_tokens": 32,
                "completion_tokens": 2,
                "input_token_logprobs": [[None, 1], [-0.1, 2]],
                "output_token_logprobs": [[-0.2, 3], [-0.3, 4]],
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "accuracy.json"
            args = argparse.Namespace(
                base_url="http://127.0.0.1:6688",
                tokenizer="unused",
                tag="A1",
                input_lens=[32],
                output_len=2,
                repeats=3,
                prefill_logprob_tokens=2,
                top_logprobs=1,
                timeout=1,
                output=output,
            )
            with patch.object(
                VALIDATION, "_load_tokenizer", return_value=_FakeTokenizer()
            ), patch.object(
                VALIDATION, "_post_json", return_value=(200, response, 0.01)
            ):
                self.assertEqual(VALIDATION.command_accuracy(args), 0)

            cases = VALIDATION._load_json(output)["cases"]
            self.assertEqual(len(cases), 3)
            self.assertEqual(len({case["prompt_hash"] for case in cases}), 1)

    def test_accuracy_comparison_accepts_small_numeric_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            baseline = temp / "baseline.json"
            candidate = temp / "candidate.json"
            VALIDATION._write_json(baseline, {"tag": "A", "cases": [_accuracy_case()]})
            VALIDATION._write_json(
                candidate,
                {"tag": "C", "cases": [_accuracy_case(logprob_offset=0.01)]},
            )
            args = argparse.Namespace(
                baseline=baseline,
                candidate=candidate,
                max_prefill_logprob_diff=0.15,
                max_output_logprob_diff=0.15,
                min_output_token_match=0.95,
                output=None,
            )

            self.assertEqual(VALIDATION.command_compare_accuracy(args), 0)

    def test_accuracy_comparison_rejects_token_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            baseline = temp / "baseline.json"
            candidate = temp / "candidate.json"
            VALIDATION._write_json(baseline, {"tag": "A", "cases": [_accuracy_case()]})
            VALIDATION._write_json(
                candidate,
                {"tag": "C", "cases": [_accuracy_case(token_offset=1)]},
            )
            args = argparse.Namespace(
                baseline=baseline,
                candidate=candidate,
                max_prefill_logprob_diff=0.15,
                max_output_logprob_diff=0.15,
                min_output_token_match=0.95,
                output=None,
            )

            self.assertEqual(VALIDATION.command_compare_accuracy(args), 1)

    def test_accuracy_stability_compares_repeated_identical_prompts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            artifact = temp / "accuracy.json"
            cases = []
            for repeat, offset in enumerate((0.0, 0.01, 0.02)):
                case = _accuracy_case(logprob_offset=offset)
                case["repeat"] = repeat
                cases.append(case)
            VALIDATION._write_json(artifact, {"tag": "A1", "cases": cases})
            args = argparse.Namespace(
                artifact=artifact,
                min_repeats=3,
                max_prefill_logprob_diff=0.15,
                max_output_logprob_diff=0.15,
                min_output_token_match=0.95,
                output=None,
            )

            self.assertEqual(VALIDATION.command_accuracy_stability(args), 0)

    def test_prefill_diagnostics_require_mla_groups_and_send_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            records = [
                {
                    "event": "manager_register_start",
                    "is_mla_backend": True,
                    "kv_buf_groups": 2,
                },
                {"event": "send_kvcache_plan"},
                {"event": "memfabric_transfer_result", "ret": 0},
            ]
            (temp / "diag.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            service_log = temp / "prefill.log"
            service_log.write_text(
                "Session 10.0.0.1:1234 is alive\n", encoding="utf-8"
            )
            args = argparse.Namespace(
                diag_dir=temp,
                role="prefill",
                service_log=[service_log],
                output=None,
            )

            self.assertEqual(VALIDATION.command_diag(args), 0)

    def test_performance_aggregation_counts_failed_requests(self):
        record = {
            "random_input_len": 2048,
            "random_output_len": 32,
            "random_range_ratio": 1.0,
            "max_concurrency": 4,
            "completed": 7,
            "request_throughput": 1.0,
            "input_throughput": 2000.0,
            "median_ttft_ms": 100.0,
            "p90_ttft_ms": 120.0,
            "median_tpot_ms": 10.0,
            "median_e2e_latency_ms": 500.0,
            "input_lens": [2048] * 8,
            "output_lens": [32] * 8,
            "errors": ["", "transfer failed"],
        }

        aggregate = VALIDATION._aggregate_perf([record])[(2048, 32, 4)]

        self.assertEqual(aggregate["failed_requests"], 1)

    def test_performance_collection_uses_fixed_lengths_and_round_tags(self):
        commands = []

        def fake_benchmark(command, log_path, env):
            del log_path, env
            commands.append(command)

            def value(flag):
                return command[command.index(flag) + 1]

            input_len = int(value("--random-input-len"))
            output_len = int(value("--random-output-len"))
            num_prompts = int(value("--num-prompts"))
            record = {
                "tag": value("--tag"),
                "random_input_len": input_len,
                "random_output_len": output_len,
                "random_range_ratio": float(value("--random-range-ratio")),
                "max_concurrency": int(value("--max-concurrency")),
                "completed": num_prompts,
                "request_throughput": 2.0,
                "input_throughput": 4096.0,
                "median_ttft_ms": 100.0,
                "p90_ttft_ms": 120.0,
                "median_tpot_ms": 10.0,
                "median_e2e_latency_ms": 500.0,
                "input_lens": [input_len] * num_prompts,
                "output_lens": [output_len] * num_prompts,
                "errors": [""] * num_prompts,
            }
            with Path(value("--output-file")).open("a", encoding="utf-8") as output:
                output.write(json.dumps(record) + "\n")
            return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "perf.jsonl"
            args = argparse.Namespace(
                output_file=output,
                tag="A1",
                append=False,
                base_url="http://127.0.0.1:6688",
                model="model",
                tokenizer="model",
                input_lens=[2048],
                output_lens=[32],
                concurrencies=[4],
                num_prompts=8,
                rounds=3,
                warmup_requests=4,
                seed=42,
            )
            with patch.object(
                VALIDATION, "_stream_subprocess", side_effect=fake_benchmark
            ):
                self.assertEqual(VALIDATION.command_perf(args), 0)

            records = VALIDATION._read_jsonl(output)
            self.assertEqual(len(records), 3)
            self.assertEqual(
                [record["tag"] for record in records],
                [
                    "A1_in2048_out32_c4_r1",
                    "A1_in2048_out32_c4_r2",
                    "A1_in2048_out32_c4_r3",
                ],
            )
            self.assertTrue(
                all(
                    command[command.index("--random-range-ratio") + 1] == "1"
                    for command in commands
                )
            )

    def test_performance_validation_requires_fixed_lengths(self):
        record = {
            "random_input_len": 2048,
            "random_output_len": 32,
            "random_range_ratio": 1.0,
            "completed": 3,
            "input_lens": [2048, 2048, 2048],
            "output_lens": [32, 32, 32],
            "errors": ["", "", ""],
        }
        self.assertEqual(VALIDATION._perf_record_errors(record), [])

        record["random_range_ratio"] = 0.0
        record["input_lens"][-1] = 861
        record["output_lens"][-1] = 7
        errors = VALIDATION._perf_record_errors(record)
        self.assertTrue(any("random_range_ratio" in error for error in errors))
        self.assertTrue(any("input_lens" in error for error in errors))
        self.assertTrue(any("output_lens" in error for error in errors))

    def test_performance_aggregation_uses_round_median_and_mad(self):
        records = []
        for ttft in (100.0, 120.0, 500.0):
            records.append(
                {
                    "random_input_len": 2048,
                    "random_output_len": 32,
                    "random_range_ratio": 1.0,
                    "max_concurrency": 4,
                    "completed": 4,
                    "request_throughput": 2.0,
                    "input_throughput": 4096.0,
                    "median_ttft_ms": ttft,
                    "p90_ttft_ms": ttft + 10,
                    "median_tpot_ms": 10.0,
                    "median_e2e_latency_ms": ttft + 320,
                    "input_lens": [2048] * 4,
                    "output_lens": [32] * 4,
                    "errors": [""] * 4,
                }
            )

        aggregate = VALIDATION._aggregate_perf(records)[(2048, 32, 4)]
        self.assertEqual(aggregate["rounds"], 3)
        self.assertEqual(aggregate["median_ttft_ms"], 120.0)
        self.assertEqual(aggregate["median_ttft_ms_mad"], 20.0)
        self.assertEqual(aggregate["invalid_records"], 0)


if __name__ == "__main__":
    unittest.main()
