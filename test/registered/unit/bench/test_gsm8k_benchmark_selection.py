import json
import tempfile
import unittest

from benchmark.gsm8k.bench_sglang import (
    collect_state_outputs,
    collect_speculative_metrics,
    dump_gsm8k_raw_result,
    get_evaluation_examples,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import dump_bench_raw_result

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGSM8KBenchmarkSelection(unittest.TestCase):
    def test_few_shot_demonstrations_are_not_scored(self):
        lines = [{"question": f"q{i}", "answer": str(i)} for i in range(20)]

        selected = get_evaluation_examples(lines, num_shots=5, num_questions=10)

        self.assertEqual(
            [item["question"] for item in selected],
            [f"q{i}" for i in range(5, 15)],
        )

    def test_selection_caps_at_remaining_dataset(self):
        lines = list(range(7))
        self.assertEqual(
            get_evaluation_examples(lines, num_shots=5, num_questions=10),
            [5, 6],
        )

    def test_negative_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            get_evaluation_examples([], num_shots=-1, num_questions=10)
        with self.assertRaises(ValueError):
            get_evaluation_examples([], num_shots=5, num_questions=-1)

    def test_speculative_metrics_are_aggregated_by_work(self):
        class State:
            def __init__(self, meta_info):
                self.meta_info = meta_info

            def get_meta_info(self, name):
                self.assert_name = name
                return self.meta_info

        per_request, summary = collect_speculative_metrics(
            [
                State(
                    {
                        "completion_tokens": 10,
                        "spec_num_correct_drafts": 6,
                        "spec_num_proposed_drafts": 14,
                        "spec_verify_ct": 2,
                    }
                ),
                State(
                    {
                        "completion_tokens": 8,
                        "spec_num_correct_drafts": 7,
                        "spec_num_proposed_drafts": 7,
                        "spec_verify_ct": 1,
                    }
                ),
                State({"completion_tokens": 1}),
            ]
        )

        self.assertEqual(len(per_request), 3)
        self.assertEqual(summary["requests_with_spec_verify"], 2)
        self.assertEqual(summary["total_spec_verify_ct"], 3)
        self.assertAlmostEqual(summary["spec_accept_rate"], 13 / 21)
        self.assertAlmostEqual(summary["spec_accept_length"], 6.0)

    def test_raw_dump_includes_per_request_metadata(self):
        class State:
            def __getitem__(self, name):
                self.name = name
                return "answer=42"

            def text(self):
                return "question answer=42"

        with tempfile.NamedTemporaryFile(mode="r+", suffix=".jsonl") as output:
            dump_bench_raw_result(
                path=output.name,
                states=[State()],
                preds=[42],
                labels=[42],
                per_state_extra_fields=[{"meta_info": {"spec_verify_ct": 3}}],
            )
            row = json.loads(output.read())

        self.assertTrue(row["correct"])
        self.assertEqual(row["meta_info"]["spec_verify_ct"], 3)

    def test_failed_request_is_retained_in_raw_artifact(self):
        class FailedState:
            def __getitem__(self, name):
                raise KeyError(name)

            def get_meta_info(self, name):
                raise KeyError(name)

            def error(self):
                return "connection refused"

            def text(self):
                return "question"

        state = FailedState()
        answers, errors = collect_state_outputs([state])
        per_request, summary = collect_speculative_metrics([state])

        self.assertEqual(answers, [""])
        self.assertEqual(errors, ["connection refused"])
        self.assertEqual(summary["requests_with_spec_verify"], 0)
        with tempfile.NamedTemporaryFile(mode="r+", suffix=".jsonl") as output:
            dump_gsm8k_raw_result(
                output.name,
                [state],
                answers,
                [-9999999],
                [42],
                per_request,
                errors,
            )
            output.seek(0)
            row = json.loads(output.read())

        self.assertFalse(row["correct"])
        self.assertEqual(row["error"], "connection refused")


if __name__ == "__main__":
    unittest.main()
