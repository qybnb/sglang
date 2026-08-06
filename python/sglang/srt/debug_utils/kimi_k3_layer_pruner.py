"""Create a prefix-layer-pruned Kimi-K3 checkpoint without loading tensors.

Kimi-K3 stores the language backbone below ``text_config`` and uses tensor
names such as ``language_model.model.layers.23.*``.  The generic debug model
truncator does not handle either detail and materializes every safetensors
shard in host memory.  This utility instead filters the safetensors index and
links only the referenced shards into a new local checkpoint directory.

The official Kimi-K3 checkpoint places each language layer in its own shard,
so prefix pruning reduces both disk and model-loading I/O without rewriting
the roughly 17 GB layer shards.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


_LAYER_PATTERNS = (
    re.compile(r"^language_model\.model\.layers\.(\d+)\."),
    re.compile(r"^model\.layers\.(\d+)\."),
)
_INDEX_NAME = "model.safetensors.index.json"


def _layer_id(tensor_name: str) -> int | None:
    for pattern in _LAYER_PATTERNS:
        match = pattern.match(tensor_name)
        if match is not None:
            return int(match.group(1))
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _update_config(config: dict[str, Any], keep_num_layers: int) -> tuple[int, str]:
    if config.get("model_type") == "kimi_k3":
        if not isinstance(config.get("text_config"), dict):
            raise ValueError("Kimi-K3 config.json is missing the text_config object")
        text_config = config["text_config"]
        config_location = "text_config"
    else:
        # Also accept text-only Kimi Linear compatible checkpoints.
        text_config = config
        config_location = "top-level"

    original_num_layers = int(text_config["num_hidden_layers"])
    if not 1 <= keep_num_layers <= original_num_layers:
        raise ValueError(
            f"--keep-num-layers must be in [1, {original_num_layers}], "
            f"got {keep_num_layers}"
        )

    text_config["num_hidden_layers"] = keep_num_layers
    linear_config = text_config.get("linear_attn_config")
    if isinstance(linear_config, dict):
        for key in ("kda_layers", "full_attn_layers"):
            layer_ids = linear_config.get(key)
            if not isinstance(layer_ids, list):
                raise ValueError(f"linear_attn_config.{key} must be a list")
            # Kimi config uses one-based layer ids.
            linear_config[key] = [
                int(layer_id)
                for layer_id in layer_ids
                if int(layer_id) <= keep_num_layers
            ]

        covered = sorted(
            linear_config["kda_layers"] + linear_config["full_attn_layers"]
        )
        expected = list(range(1, keep_num_layers + 1))
        if covered != expected:
            raise ValueError(
                "The truncated KDA/MLA layout does not cover every retained layer: "
                f"covered={covered}, expected={expected}"
            )

    return original_num_layers, config_location


def _copy_metadata(source: Path, output: Path) -> None:
    excluded = {"config.json", _INDEX_NAME, ".cache", ".git"}
    for path in source.iterdir():
        if path.name in excluded:
            continue
        if path.suffix == ".safetensors":
            continue

        target = output / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        elif path.is_file():
            shutil.copy2(path, target)


def _link_or_copy(source: Path, target: Path, mode: str) -> None:
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError as error:
            raise OSError(
                f"Hardlink failed for {source.name}: {error}. Put input and "
                "output on the same filesystem, or explicitly use "
                "--link-mode symlink/copy."
            ) from error
    if mode == "symlink":
        target.symlink_to(source.resolve())
        return
    if mode == "copy":
        shutil.copy2(source, target)
        return
    raise ValueError(f"Unsupported link mode: {mode}")


def create_pruned_checkpoint(
    source: Path,
    output: Path,
    keep_num_layers: int,
    link_mode: str,
) -> None:
    source = source.resolve()
    output = output.resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"Input model directory does not exist: {source}")
    if output == source or source in output.parents:
        raise ValueError("Output must not be the input directory or inside it")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    existing_output_parent = output
    while not existing_output_parent.exists():
        existing_output_parent = existing_output_parent.parent
    if (
        link_mode == "hardlink"
        and source.stat().st_dev != existing_output_parent.stat().st_dev
    ):
        raise ValueError(
            "--link-mode hardlink requires input and output on the same "
            "filesystem. Use --link-mode symlink or --link-mode copy explicitly."
        )

    config_path = source / "config.json"
    index_path = source / _INDEX_NAME
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            f"{source} must contain config.json and {_INDEX_NAME}"
        )

    config = _read_json(config_path)
    original_num_layers, config_location = _update_config(
        config, keep_num_layers
    )

    index = _read_json(index_path)
    original_weight_map = index.get("weight_map")
    if not isinstance(original_weight_map, dict):
        raise ValueError(f"{index_path} does not contain a weight_map object")

    kept_weight_map: dict[str, str] = {}
    removed_weight_map: dict[str, str] = {}
    discovered_layer_ids: set[int] = set()
    for tensor_name, shard_name in original_weight_map.items():
        layer_id = _layer_id(tensor_name)
        if layer_id is not None:
            discovered_layer_ids.add(layer_id)
        if layer_id is None or layer_id < keep_num_layers:
            kept_weight_map[tensor_name] = shard_name
        else:
            removed_weight_map[tensor_name] = shard_name

    if not discovered_layer_ids:
        raise ValueError(
            "No Kimi language layer tensors were found. Expected names like "
            "'language_model.model.layers.0.*'"
        )
    missing_layer_ids = set(range(keep_num_layers)) - discovered_layer_ids
    if missing_layer_ids:
        raise ValueError(
            f"Checkpoint index is missing retained layers: {sorted(missing_layer_ids)}"
        )

    kept_shards = set(kept_weight_map.values())
    removed_shards = set(removed_weight_map.values())
    mixed_shards = kept_shards & removed_shards

    missing_shards = sorted(
        shard_name
        for shard_name in kept_shards
        if not (source / shard_name).is_file()
    )
    if missing_shards:
        raise FileNotFoundError(
            "Referenced safetensors shards are missing: " + ", ".join(missing_shards)
        )

    output.mkdir(parents=True, exist_ok=True)
    _copy_metadata(source, output)
    _write_json(output / "config.json", config)

    index["weight_map"] = kept_weight_map
    # This field is informational. File sizes are a safe upper bound and avoid
    # opening multi-GB shards just to reconstruct tensor byte counts.
    index.setdefault("metadata", {})["total_size"] = sum(
        (source / shard_name).stat().st_size for shard_name in kept_shards
    )
    _write_json(output / _INDEX_NAME, index)

    for shard_name in sorted(kept_shards):
        _link_or_copy(source / shard_name, output / shard_name, link_mode)

    size_gib = sum(
        (source / shard_name).stat().st_size for shard_name in kept_shards
    ) / 1024**3
    print(f"Input:                 {source}")
    print(f"Output:                {output}")
    print(f"Config location:       {config_location}")
    print(f"Language layers:       {original_num_layers} -> {keep_num_layers}")
    print(f"Retained tensors:      {len(kept_weight_map):,}")
    print(f"Retained shards:       {len(kept_shards):,}")
    print(f"Referenced size:       {size_gib:.2f} GiB")
    print(f"Shard materialization: {link_mode}")
    if mixed_shards:
        print(
            "WARNING: Some source shards mix retained and removed layers, so their "
            "full files must remain present: " + ", ".join(sorted(mixed_shards))
        )
    if keep_num_layers % 4 != 0:
        print(
            "WARNING: Kimi-K3 normally repeats 3 KDA + 1 MLA. A multiple of 4 "
            "is recommended for prefix pruning."
        )
    if keep_num_layers % 12 == 0:
        print("Attention-Residual groups are complete (12-layer boundary).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a prefix-layer-pruned Kimi-K3 checkpoint by filtering its "
            "safetensors index without loading tensor data."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-num-layers", type=int, default=24)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help=(
            "How retained shards are materialized. hardlink saves space and remains "
            "valid if the original path is renamed; copy creates an independent copy."
        ),
    )
    args = parser.parse_args()
    create_pruned_checkpoint(
        source=args.input,
        output=args.output,
        keep_num_layers=args.keep_num_layers,
        link_mode=args.link_mode,
    )


if __name__ == "__main__":
    main()
