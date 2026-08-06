"""Unit tests for nested Hugging Face model config overrides."""

import unittest

from transformers import PretrainedConfig

from sglang.srt.utils.hf_transformers.config import _apply_model_override_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNestedModelOverrides(CustomTestCase):
    def test_nested_config_object_is_preserved(self):
        text_config = PretrainedConfig()
        text_config.num_hidden_layers = 93
        text_config.linear_attn_config = {
            "kda_layers": [1, 2, 3],
            "full_attn_layers": [4],
        }
        config = PretrainedConfig()
        config.text_config = text_config

        _apply_model_override_args(
            config, {"text_config": {"num_hidden_layers": 24}}
        )

        self.assertIs(config.text_config, text_config)
        self.assertEqual(config.text_config.num_hidden_layers, 24)
        self.assertEqual(
            config.text_config.linear_attn_config["full_attn_layers"], [4]
        )

    def test_nested_dictionary_is_merged(self):
        config = PretrainedConfig()
        config.runtime = {"loader": {"threads": 8, "mmap": True}}

        _apply_model_override_args(
            config, {"runtime": {"loader": {"threads": 4}}}
        )

        self.assertEqual(
            config.runtime, {"loader": {"threads": 4, "mmap": True}}
        )


if __name__ == "__main__":
    unittest.main()
