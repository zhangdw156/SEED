# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create local placeholder datasets for environment-driven agent training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import datasets
from PIL import Image

from verl.utils.hdfs_io import copy, makedirs


def _dataset_features(mode: str) -> datasets.Features:
    features = {
        "data_source": datasets.Value("string"),
        "prompt": [
            {
                "role": datasets.Value("string"),
                "content": datasets.Value("string"),
            }
        ],
        "ability": datasets.Value("string"),
        "extra_info": {
            "split": datasets.Value("string"),
            "index": datasets.Value("int64"),
        },
    }
    if mode == "visual":
        features["images"] = datasets.Sequence(datasets.Image())
    return datasets.Features(features)


def build_placeholder_dataset(
    *,
    mode: str,
    size: int,
    split: str,
) -> datasets.Dataset:
    """Build rows whose real observations will be supplied by the environment."""

    if mode not in {"visual", "text"}:
        raise ValueError("mode must be visual or text")
    if size < 0:
        raise ValueError("size must be non-negative")

    prompt = "<image>" if mode == "visual" else ""
    columns = {
        "data_source": [mode for _ in range(size)],
        "prompt": [
            [{"role": "user", "content": prompt}]
            for _ in range(size)
        ],
        "ability": ["agent" for _ in range(size)],
        "extra_info": [
            {"split": split, "index": index}
            for index in range(size)
        ],
    }
    if mode == "visual":
        placeholder = Image.new("RGB", (8, 8), color=(0, 0, 0))
        columns["images"] = [
            [placeholder.copy()]
            for _ in range(size)
        ]
    return datasets.Dataset.from_dict(
        columns,
        features=_dataset_features(mode),
    )


def prepare_datasets(
    *,
    mode: str,
    local_dir: str,
    train_data_size: int,
    val_data_size: int,
    hdfs_dir: str | None = None,
) -> tuple[Path, Path]:
    """Write train and validation placeholder parquet files."""

    output_dir = Path(os.path.expanduser(local_dir)) / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"

    build_placeholder_dataset(
        mode=mode,
        size=train_data_size,
        split="train",
    ).to_parquet(str(train_path))
    build_placeholder_dataset(
        mode=mode,
        size=val_data_size,
        split="test",
    ).to_parquet(str(test_path))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=str(output_dir), dst=hdfs_dir)
    return train_path, test_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="visual",
        choices=["visual", "text"],
    )
    parser.add_argument(
        "--local_dir",
        default="~/data/verl-agent/",
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--train_data_size",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--val_data_size",
        default=256,
        type=int,
    )
    args = parser.parse_args()

    train_data_size_override = os.environ.get(
        "VERL_AGENT_TRAIN_DATA_SIZE"
    )
    if train_data_size_override is not None:
        args.train_data_size = int(train_data_size_override)

    print(f"processing data for mode: {args.mode}")
    prepare_datasets(
        mode=args.mode,
        local_dir=args.local_dir,
        train_data_size=args.train_data_size,
        val_data_size=args.val_data_size,
        hdfs_dir=args.hdfs_dir,
    )


if __name__ == "__main__":
    main()
