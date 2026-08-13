import unittest

from benchmark.gsm8k.bench_sglang import get_evaluation_examples
from sglang.test.ci.ci_register import register_cpu_ci

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


if __name__ == "__main__":
    unittest.main()
