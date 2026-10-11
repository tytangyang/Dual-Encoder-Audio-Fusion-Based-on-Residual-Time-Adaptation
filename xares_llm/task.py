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


from __future__ import annotations
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import torch
from pathlib import Path
from transformers import AutoTokenizer, TrainingArguments
import pandas as pd
import yaml
from dataclasses import dataclass, field, asdict
from loguru import logger
from typing import Any, Dict, List

from xares_llm.utils import seed_everything, setup_global_logger
from xares_llm.audiowebdataset import AudioTextDataType, AudioTextTokenWebdataset
from xares_llm.trainer import XaresLLMTrainerEvaluator
from xares_llm.modeling_audiollm import XaresLLMModel, XaresLLMModelConfig
from xares_llm.metrics import get_metric, RegisteredMetricsLiteral, TokenDecoder
import importlib
import pprint


def _ensure_tokenizer_pad_token(tokenizer):
    """
    真正解决 generate() 重复打印：
      Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.

    根因：tokenizer/model/generation_config 没有 pad_token_id，
    generate() 每次会临时设置并打印。
    """
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def _ensure_model_pad_token(model, pad_token_id: int):
    """
    让 model.config / model.generation_config 都有 pad_token_id，
    避免 train/eval 任一路径调用 generate() 时刷屏。

    ✅ 修复点：
    - 你当前报错的根因是 model.generation_config 可能是 None
      之前代码直接访问 model.generation_config.pad_token_id -> AttributeError
    - 这里会在 generation_config 为 None 时创建一个 GenerationConfig 并挂回 model
    """
    if model is None:
        return model

    # 1) model.config.pad_token_id
    cfg = getattr(model, "config", None)
    if cfg is not None:
        if getattr(cfg, "pad_token_id", None) is None:
            cfg.pad_token_id = int(pad_token_id)

    # 2) model.generation_config.pad_token_id
    # generation_config 可能不存在/为 None/或不可写
    try:
        from transformers import GenerationConfig
    except Exception:
        GenerationConfig = None

    if hasattr(model, "generation_config"):
        gen_cfg = getattr(model, "generation_config", None)

        # generation_config == None -> 创建
        if gen_cfg is None and GenerationConfig is not None:
            try:
                # 优先从 model.config 构建，保持 eos/bos 等一致
                if cfg is not None:
                    gen_cfg = GenerationConfig.from_model_config(cfg)
                else:
                    gen_cfg = GenerationConfig()
            except Exception:
                gen_cfg = GenerationConfig()

            # 尝试挂回 model（有些模型可能不允许 set）
            try:
                model.generation_config = gen_cfg
            except Exception:
                # 挂不回就不强行挂，至少保证 model.config 已有 pad_token_id
                gen_cfg = None

        # gen_cfg 不为 None 才能写 pad_token_id
        if gen_cfg is not None:
            if getattr(gen_cfg, "pad_token_id", None) is None:
                gen_cfg.pad_token_id = int(pad_token_id)

            # 可选：如果 gen_cfg 没 eos_token_id，而 cfg 有，就补齐
            if getattr(gen_cfg, "eos_token_id", None) is None and cfg is not None and getattr(cfg, "eos_token_id", None) is not None:
                gen_cfg.eos_token_id = cfg.eos_token_id

    return model


def _ensure_tie_weights(model):
    """
    真正处理：lm_head.weight newly initialized
    - 很多 CausalLM 模型的 lm_head 与 embedding 是 tie 的
    - 有时 checkpoint 不显式存 lm_head.weight 或 tie 策略不同会提示新初始化
    这里强制 tie，避免留下随机初始化的头。
    """
    if model is None:
        return model
    try:
        model.tie_weights()
    except Exception:
        pass
    return model


# Mappings from config.yaml -> Path to the config. By default we store most configs in the package tree, but users can also provide their own
AVAILABLE_TRAINING_CONFIGS = {
    "all": next(importlib.resources.files("xares_llm.tasks.all.train").iterdir()),
    "task1": next(importlib.resources.files("xares_llm.tasks.task1.train").iterdir()),
    "task2": next(importlib.resources.files("xares_llm.tasks.task2.train").iterdir()),
} | {
    str(Path(_).stem).replace("_config", ""): _
    for _ in importlib.resources.files("xares_llm.tasks.single.train").iterdir()
}
AVAILABLE_EVALUATION_CONFIGS = {
    "all": next(importlib.resources.files("xares_llm.tasks.all.eval").iterdir()),
    "task1": next(importlib.resources.files("xares_llm.tasks.task1.eval").iterdir()),
    "task2": next(importlib.resources.files("xares_llm.tasks.task2.eval").iterdir()),
} | {
    str(Path(_).stem).replace("_test_config", ""): _
    for _ in importlib.resources.files("xares_llm.tasks.single.eval").iterdir()
}


@dataclass
class XaresLLMTrainConfig:
    audio_encoder_module_path: str  # path to the audio encoder
    audio_encoder_kwargs: Dict[str, Any] = field(default_factory=lambda: dict())
    output_dir: str = "experiments/"
    config_name: str = "default"  # Will be set if loaded from a .yaml

    # General
    torch_num_threads: int = 1  # Do not use too many otherwise slows down
    seed: int = 42  # manual seed for all experiments

    train_data: List[AudioTextDataType] | None = None

    # decoder
    decoder_model_name: str = "HuggingFaceTB/SmolLM2-135M"

    # Dataloader/dataset arguments
    seed: int = field(default=42)
    crop_audio_length: float = 30  # Cropping all audio to at most 30s
    save_total_limit: int | None = field(default=1)
    save_steps: float = field(default=200)  # TrainingArguments is float ....
    warmup_steps: int = field(default=200)
    max_steps: int = field(
        default=10000,
        metadata={"help": "Total number of training steps to perform (default: 10k)."},
    )
    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Batch size per device during training (default: 4)."}
    )

    # Optimizer
    optimizer: str = "adamw_torch"  # adamw_bnb_8bit
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.01)
    seed: int = field(default=42)
    max_grad_norm: float = field(default=1.0)
    logging_dir: str = "log"
    logging_steps: int = 100
    num_training_workers: int = 0
    sort_by_length: int = 128  # Sort 128 samples by length

    def __post_init__(self):
        if isinstance(self.train_data, dict):
            self.train_data = [AudioTextDataType(name=k, **val) for k, val in self.train_data.items()]
        torch.set_num_threads(self.torch_num_threads)
        setup_global_logger()
        seed_everything(self.seed)

    def __repr__(self):
        return pprint.pformat(asdict(self))

    @classmethod
    def from_file(
        cls,
        config_file: str,
        encoder_path: str,
        model_kwargs: Dict[str, Any] | None = None,
        overwrite_kwargs: Dict[str, Any] | None = None,
    ) -> "XaresLLMTrainConfig":
        with open(config_file) as con_read:
            yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
        yaml_config["config_name"] = Path(config_file).stem
        yaml_config["audio_encoder_module_path"] = encoder_path
        if model_kwargs is None:
            model_kwargs = dict()
        yaml_config["audio_encoder_kwargs"] = model_kwargs
        if overwrite_kwargs is None:
            overwrite_kwargs = dict()
        yaml_config = dict(**yaml_config, **overwrite_kwargs)
        return cls(**yaml_config)

    @classmethod
    def from_file_or_key(
        cls,
        config_identifier: str,
        encoder_path: str,
        model_kwargs: Dict[str, Any] | None = None,
        overwrite_kwargs: Dict[str, Any] | None = None,
    ) -> "XaresLLMTrainConfig":
        if config_identifier in AVAILABLE_TRAINING_CONFIGS:
            return cls.from_file(
                AVAILABLE_TRAINING_CONFIGS[config_identifier],
                encoder_path=encoder_path,
                model_kwargs=model_kwargs,
                overwrite_kwargs=overwrite_kwargs,
            )
        path_obj = Path(config_identifier)
        if path_obj.is_file():
            return cls.from_file(
                config_identifier,
                encoder_path=encoder_path,
                model_kwargs=model_kwargs,
                overwrite_kwargs=overwrite_kwargs,
            )
        raise ValueError(f"Unknown config identifier {config_identifier}")


@dataclass
class XaresLLMEvaluationConfig:
    data: AudioTextDataType
    metric: RegisteredMetricsLiteral
    metric_args: Dict[str, Any] = field(default_factory=lambda: dict())
    batch_size: int = 32
    num_workers: int = 0
    weight: float = 1

    @classmethod
    def configs_from_file(cls, yaml_config_file: str) -> List["XaresLLMEvaluationConfig"]:
        with open(yaml_config_file) as con_read:
            yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
        evaluation_configs = []
        for k, values in yaml_config.items():
            data_kwargs = values.pop("data")
            metric = values.pop("metric")
            evaluation_configs.append(cls(data=AudioTextDataType(name=k, **data_kwargs), metric=metric, **values))
        return evaluation_configs

    def __repr__(self):
        return pprint.pformat(asdict(self))

    @classmethod
    def configs_from_file_or_key(cls, config_identifier: str) -> List["XaresLLMEvaluationConfig"]:
        if config_identifier in AVAILABLE_EVALUATION_CONFIGS:
            return cls.configs_from_file(AVAILABLE_EVALUATION_CONFIGS[config_identifier])
        path_obj = Path(config_identifier)
        if path_obj.is_file():
            return cls.configs_from_file(config_identifier)
        raise ValueError(f"Unknown config identifier {config_identifier}")


class XaresLLMTask:
    def __init__(self, train_config: XaresLLMTrainConfig):
        self.train_config = train_config
        if Path(self.train_config.audio_encoder_module_path).is_file():
            model_name = str(Path(self.train_config.audio_encoder_module_path).stem)
        else:
            model_name = self.train_config.audio_encoder_module_path.split(".")[-1]
        self.output_dir = Path(train_config.output_dir) / train_config.config_name / model_name
        logger.add(
            self.output_dir / "log.txt",
            enqueue=True,
            level="INFO",
            format="[{level} {time:YYYY-MM-DD HH:mm:ss}] {message}",
        )
        logger.info(f"Experiment output path set to {self.output_dir}")
        logger.info(f"Loading {train_config.decoder_model_name} tokenizer")

        # ---- tokenizer: 真正补齐 pad_token_id ----
        self.tokenizer = AutoTokenizer.from_pretrained(train_config.decoder_model_name)
        self.tokenizer = _ensure_tokenizer_pad_token(self.tokenizer)
        self._pad_token_id = int(self.tokenizer.pad_token_id)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            learning_rate=self.train_config.learning_rate,
            per_device_train_batch_size=self.train_config.per_device_train_batch_size,
            save_total_limit=self.train_config.save_total_limit,
            save_steps=self.train_config.save_steps,
            warmup_steps=self.train_config.warmup_steps,
            max_grad_norm=self.train_config.max_grad_norm,
            max_steps=self.train_config.max_steps,
            optim=self.train_config.optimizer,
            weight_decay=self.train_config.weight_decay,
            seed=self.train_config.seed,
            logging_steps=self.train_config.logging_steps,
            logging_dir=Path(self.output_dir) / self.train_config.logging_dir,
        )

        # ---- model_init: lazy init 路径也必须补齐 pad_token_id / tie_weights ----
        def model_init_function():
            m = XaresLLMModel(
                config=XaresLLMModelConfig(
                    decoder_type=self.train_config.decoder_model_name,
                    audio_encoder_name=self.train_config.audio_encoder_module_path,
                    audio_encoder_params=self.train_config.audio_encoder_kwargs,
                ),
            )
            m = _ensure_model_pad_token(m, self._pad_token_id)
            m = _ensure_tie_weights(m)
            return m

        self.model = None

        # Glob for checkpoint directories (e.g., 'checkpoint-1000', 'checkpoint-2000')
        checkpoint_dirs = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if checkpoint_dirs:
            logger.info(f"Found pretrained model, loading from {checkpoint_dirs[0]}")
            self.model = XaresLLMModel.from_pretrained(checkpoint_dirs[0])

            # ---- checkpoint 加载路径也必须补齐 pad_token_id / tie_weights ----
            self.model = _ensure_model_pad_token(self.model, self._pad_token_id)
            self.model = _ensure_tie_weights(self.model)

        self.trainer = XaresLLMTrainerEvaluator(model=None, model_init=model_init_function, args=training_args)

    def run_mlp(self, eval_configs: List[XaresLLMEvaluationConfig]) -> List[Dict[str, Any]]:
        if not isinstance(eval_configs, list):
            eval_configs = [eval_configs]

        result = []
        model = self.train_mlp()
        for eval_config in eval_configs:
            dataset_name = eval_config.data.name
            score_file_cache = self.output_dir / f"score_{dataset_name}.yaml"

            if score_file_cache.exists() and score_file_cache.stat().st_size > 0:
                with open(score_file_cache, "r") as rp:
                    score = yaml.load(rp, Loader=yaml.SafeLoader)["score"]
                logger.debug(
                    f"Found cached result [{score_file_cache}] for {dataset_name}: "
                    f"Skipping Evaluation, Score: {score:.2f}"
                )
            else:
                score, output_df = self.evaluate_mlp(trained_model=model, eval_config=eval_config)
                logger.info(f"{dataset_name}: [{eval_config.metric}]: {score:.2f}")
                output_df.to_csv(self.output_dir / f"predictions_{dataset_name}.csv", index=False)

                with open(score_file_cache, "w") as wp:
                    yaml.dump({"score": score}, wp, default_flow_style=False)

            result.append({"Task": dataset_name, "score": score, "weight": eval_config.weight})
            logger.debug(f"Model outputs can be seen in {self.output_dir / f'predictions_{dataset_name}.csv'}")
        return result

    def train_mlp(self) -> XaresLLMModel:
        if self.model is not None:
            self.trainer.model = self.model
            return self.model  # Already trained

        train_data_object = AudioTextTokenWebdataset(
            data_urls=self.train_config.train_data,
            tokenizer=self.tokenizer,
            training=True,
            batch_size=self.train_config.per_device_train_batch_size,
            resample=True,
            sort_by_length=self.train_config.sort_by_length,
            num_workers=self.train_config.num_training_workers,
            crop_audio_length=self.train_config.crop_audio_length,
        )
        self.trainer.train_data_object = train_data_object
        self.trainer.train()
        logger.info(f"Finished training: {self.output_dir}")

        # ---- 训练后 trainer.model 也补齐 pad_token_id / tie ----
        if self.trainer.model is not None:
            self.trainer.model = _ensure_model_pad_token(self.trainer.model, self._pad_token_id)
            self.trainer.model = _ensure_tie_weights(self.trainer.model)

        return self.trainer.model

    def evaluate_mlp(
        self,
        eval_config: XaresLLMEvaluationConfig,
        trained_model: XaresLLMModel | None = None,
        chpt_path: str | Path | None = None,
    ) -> tuple[Dict[RegisteredMetricsLiteral, float], pd.DataFrame]:
        if trained_model is not None:
            model = trained_model
        elif chpt_path is not None:
            logger.info(f"Loaded model parameters from {chpt_path}")
            model = XaresLLMModel.from_pretrained(chpt_path)
        else:
            model = self.trainer.model

        # ---- eval 路径也补齐（防止 predict_with_generate 刷屏）----
        model = _ensure_model_pad_token(model, self._pad_token_id)
        model = _ensure_tie_weights(model)

        self.trainer.model = model

        metrics_compute_function = get_metric(eval_config.metric, tokenizer=self.tokenizer, **eval_config.metric_args)

        data_object_eval = AudioTextTokenWebdataset(
            data_urls=eval_config.data,
            tokenizer=self.tokenizer,
            training=False,
            batch_size=eval_config.batch_size,
            sort_by_length=256,
            num_workers=eval_config.num_workers,
        )
        self.trainer.compute_metrics = metrics_compute_function

        logger.info("Starting evaluation")
        result = self.trainer.predict(test_dataset=data_object_eval)

        decoder = TokenDecoder(self.tokenizer)
        prediced_text, targets = decoder.decode_predictions(result)

        prediction_df = pd.DataFrame({"predict": prediced_text, "labels": targets})
        return result.metrics[f"test_{eval_config.metric}"], prediction_df

    def run(self, eval_configs: List[XaresLLMEvaluationConfig]):
        scores = self.run_mlp(eval_configs=eval_configs)
        return scores
