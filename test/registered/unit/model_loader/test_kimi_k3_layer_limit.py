"""Unit tests for runtime Kimi-K3 layer-limited weight loading."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import sglang.srt.model_loader.weight_utils as weight_utils
from sglang.srt.model_loader.weight_utils import (
    filter_kimi_k3_safetensors_files_for_layer_limit,
    is_kimi_k3_weight_within_layer_limit,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKimiK3LayerLimit(CustomTestCase):
    def test_skips_shards_containing_only_removed_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            index = {
                "weight_map": {
                    "language_model.model.embed_tokens.weight": "global.safetensors",
                    "language_model.model.layers.0.mlp.weight": "layer0.safetensors",
                    "language_model.model.layers.1.mlp.weight": "layer1.safetensors",
                    "language_model.model.layers.2.mlp.weight": "layer2.safetensors",
                    # A mixed shard must remain because it contains the final norm.
                    "language_model.model.norm.weight": "layer2-mixed.safetensors",
                    "language_model.model.layers.2.self_attn.weight": (
                        "layer2-mixed.safetensors"
                    ),
                    "vision_tower.blocks.0.weight": "vision.safetensors",
                }
            }
            # ModelSlim checkpoints use this non-standard index name. The
            # runtime filter must discover it automatically.
            (model_dir / "quant_model_weights.safetensors.index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )
            files = [
                str(model_dir / name)
                for name in (
                    "global.safetensors",
                    "layer0.safetensors",
                    "layer1.safetensors",
                    "layer2.safetensors",
                    "layer2-mixed.safetensors",
                    "vision.safetensors",
                )
            ]

            filtered = filter_kimi_k3_safetensors_files_for_layer_limit(
                files,
                str(model_dir),
                "model.safetensors.index.json",
                2,
            )

            self.assertEqual(
                set(filtered),
                {
                    str(model_dir / "global.safetensors"),
                    str(model_dir / "layer0.safetensors"),
                    str(model_dir / "layer1.safetensors"),
                    str(model_dir / "layer2-mixed.safetensors"),
                    str(model_dir / "vision.safetensors"),
                },
            )

    def test_filters_tensor_names_before_materialization(self):
        self.assertTrue(
            is_kimi_k3_weight_within_layer_limit(
                "language_model.model.layers.23.mlp.weight", 24
            )
        )
        self.assertFalse(
            is_kimi_k3_weight_within_layer_limit(
                "language_model.model.layers.24.mlp.weight", 24
            )
        )
        self.assertTrue(
            is_kimi_k3_weight_within_layer_limit(
                "language_model.model.norm.weight", 24
            )
        )

    def test_safetensors_iterator_filters_before_get_tensor(self):
        class FakeSafeOpen:
            requested_names = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def keys(self):
                return ["keep.weight", "removed.weight"]

            def get_tensor(self, name):
                self.requested_names.append(name)
                if name == "removed.weight":
                    raise AssertionError("removed tensor was materialized")
                return torch.tensor([1])

        fake_file = FakeSafeOpen()
        with patch.object(
            weight_utils.safetensors, "safe_open", return_value=fake_file
        ):
            weights = list(
                weight_utils.safetensors_weights_iterator(
                    ["unused.safetensors"],
                    weight_name_filter=lambda name: name == "keep.weight",
                )
            )

        self.assertEqual([name for name, _ in weights], ["keep.weight"])
        self.assertEqual(fake_file.requested_names, ["keep.weight"])

    def test_rejects_limit_above_checkpoint_layer_count(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            index = {
                "weight_map": {
                    "language_model.model.layers.0.mlp.weight": "layer0.safetensors"
                }
            }
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, r"must be in \[1, 1\]"):
                filter_kimi_k3_safetensors_files_for_layer_limit(
                    [str(model_dir / "layer0.safetensors")],
                    str(model_dir),
                    "model.safetensors.index.json",
                    2,
                )


if __name__ == "__main__":
    unittest.main()
