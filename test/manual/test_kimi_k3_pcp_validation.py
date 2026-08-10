import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


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
            "max_concurrency": 4,
            "completed": 7,
            "request_throughput": 1.0,
            "input_throughput": 2000.0,
            "median_ttft_ms": 100.0,
            "p90_ttft_ms": 120.0,
            "median_tpot_ms": 10.0,
            "median_e2e_latency_ms": 500.0,
            "errors": ["", "transfer failed"],
        }

        aggregate = VALIDATION._aggregate_perf([record])[(2048, 32, 4)]

        self.assertEqual(aggregate["failed_requests"], 1)


if __name__ == "__main__":
    unittest.main()
