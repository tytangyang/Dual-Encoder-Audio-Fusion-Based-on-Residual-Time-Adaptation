# Copyright 2025 Horizon Team, MiLM Plus, Xiaomi Inc.
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

# =============================================================================
# MUST BE FIRST: HF mirror + timeout (before importing transformers/huggingface_hub)
# =============================================================================
import os

# 强制走镜像（不要 setdefault，必须覆盖）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

# 可选：减少 hub/transformers 的提示（不影响功能）
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def _force_hf_mirror_and_timeout(endpoint: str = "https://hf-mirror.com", timeout_s: int = 120):
    """
    关键：即使 huggingface_hub/transformers 已经被间接 import 过，也尽可能强制 endpoint/timeout 生效
    """
    endpoint = endpoint.rstrip("/")

    # 1) 强制改 huggingface_hub 的 ENDPOINT（避免仍然指向 huggingface.co）
    try:
        from huggingface_hub import constants as hf_constants
        hf_constants.ENDPOINT = endpoint
    except Exception:
        pass

    # 2) 强制设置 HTTP backend 的 timeout（把 connect timeout 也拉长）
    try:
        import requests
        from huggingface_hub import configure_http_backend

        def backend_factory():
            s = requests.Session()
            old_request = s.request

            def request(method, url, **kwargs):
                # timeout 支持 float 或 (connect, read)
                kwargs.setdefault("timeout", (timeout_s, timeout_s))
                return old_request(method, url, **kwargs)

            s.request = request
            return s

        configure_http_backend(backend_factory=backend_factory)
    except Exception:
        pass


_force_hf_mirror_and_timeout()
# =============================================================================
# END PATCH
# =============================================================================

import argparse
import json
from typing import Dict, Any, List

# >>> ADDED: suppress transformers repeated warnings/logs early (NOW safe to import transformers)
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
# <<< ADDED

import pandas as pd
from loguru import logger
from pathlib import Path

from xares_llm.task import (
    XaresLLMTask,
    XaresLLMTrainConfig,
    XaresLLMEvaluationConfig,
    AVAILABLE_EVALUATION_CONFIGS,
    AVAILABLE_TRAINING_CONFIGS,
)


def main(args):
    # Training
    train_config = XaresLLMTrainConfig.from_file_or_key(
        args.train_config,
        encoder_path=args.encoder_path,
        model_kwargs=args.model_args,
        overwrite_kwargs=args.args,
    )
    eval_configs = XaresLLMEvaluationConfig.configs_from_file_or_key(args.eval_configs)

    if args.benchmark:
        logger.info("Using deterministic mode.")
        import torch

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Training with Train config \n{train_config}\n Eval config: {eval_configs}")
    runner = XaresLLMTask(train_config)
    scores: List[Dict[str, Any]] = runner.run(eval_configs)

    logger.info("Scoring completed: All tasks scored.")

    # Print results
    df = pd.DataFrame(scores)
    df.sort_values(by="Task", inplace=True)

    new_row = pd.DataFrame(
        [
            {
                "Task": "Overall",
                "score": (df["score"] * df["weight"]).sum() / df["weight"].sum(),
                "weight": df["weight"].sum(),
            }
        ]
    )
    df = pd.concat((df, new_row), ignore_index=True)
    logger.info(f"\nResults:\n{df.to_string(index=False, float_format='%.3f')}")
    df.to_csv(Path(runner.output_dir) / "scores.tsv", sep="\t", index=False, float_format="%.3f")
    logger.info(f"\nFile saved: {Path(runner.output_dir) / 'scores.tsv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XARES-LLM")
    parser.add_argument(
        "encoder_path",
        type=str,
        help="Encoder path. e.g.: example/dummy/dummyencoder.py or path to the module (including class) e.g., example.dummy.dummyencoder.DummyEncoder",
    )
    parser.add_argument(
        "train_config",
        type=str,
        help=f"Tasks .yaml or predefined dataset. Datasets are: {list(AVAILABLE_TRAINING_CONFIGS.keys())}",
        nargs="?",
        default="all",
    )
    parser.add_argument(
        "eval_configs",
        type=str,
        nargs="?",
        help=f"Evaluation Task .yaml. One Yaml can specify multiple datasets. By default we use the XARES-LLM datasets. Datasets are : {list(AVAILABLE_EVALUATION_CONFIGS.keys())} ",
        default="all",
    )
    parser.add_argument(
        "--model_args",
        type=lambda arg: json.loads(arg),
        help="Additional args passed to the encoder model. Format is JSON like: --model_args {'my_paramter1':2, 'my_paramter2':30}",
        default={},
    )
    parser.add_argument(
        "--args",
        type=lambda arg: json.loads(arg),
        help="Additional training args. Format is JSON like: --args {'per_device_train_batch_size':16, 'save_steps':30}",
        default={},
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Using deterministic mode for training/evaluation. Slows down training, but is reproducible",
        default=True,
    )
    args = parser.parse_args()
    main(args)
