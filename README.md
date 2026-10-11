# EchoBridge

**Dual-Encoder Audio Fusion Based on Residual Time Adaptation**

Submission to the **Interspeech 2026 Audio Encoder Capacity Challenge (AECC)**.

[![Report](https://img.shields.io/badge/Technical%20Report-PDF-red)](docs/technical-report.pdf)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![Encoders](https://img.shields.io/badge/Encoders-Whisper%20%2B%20CED--mini-blue)
![Tasks](https://img.shields.io/badge/Tasks-Speech%20%C2%B7%20Sound%20%C2%B7%20Music-green)

## Quick Links

- [Technical Report](docs/technical-report.pdf)
- [Challenge Website](https://dataoceanai.github.io/Interspeech2026-Audio-Encoder-Challenge/)
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Reproducibility Notes](#reproducibility-notes)
- [Citation](#citation)

## Overview

EchoBridge combines the complementary representations of **Whisper-base**, a speech-oriented encoder, and **CED-mini**, a general acoustic-event encoder, for audio classification and understanding across speech, environmental sounds, and music.

The competition system aligns the two feature sequences, projects them into a shared embedding space, and applies concatenation followed by linear fusion. A residual multi-scale temporal convolutional adapter with channel-wise scaling is designed to refine the fused representation while preserving baseline behavior at initialization.

The submitted local evaluations report an **unweighted average score of 0.712 on Task 1** and **0.401 on Task 2**, covering **20 datasets**. On Task 1, the reported score exceeds CED-mini by **0.114** and Whisper by **0.062** in absolute score units. These are local evaluation results, not official leaderboard placements.

### Encoder Versions

| Version | Class | Adapter | Return value |
|---|---|---|---|
| Root `echobridge.py` | `EchoBridgeUp2Encoder` | MLP residual adapter; scalar scaling | Tensor `[B, T, D]` |
| Submission `example/echobridge/echobridge.py` | `EchobridgeUp2Encoder` | Multi-scale temporal convolution; channel-wise scaling | `(features, attention_mask)` |
| Submission `example/echobridgeAPI/echobridge.py` | `EchobridgeUp2Encoder` | Multi-scale temporal convolution; channel-wise scaling | Tensor `[B, T, D]` |

Class names are case-sensitive. The root implementation also provides `forward_with_mask()` and stores `last_attention_mask`. The included XARES-LLM wrapper expects the tuple-returning interface.

## Repository Structure

### Complete Source Project

```text
encoder/
├── echobridge.py                       # Root encoder: MLP residual adapter, tensor API
├── check.py                            # Root encoder interface-check entry
├── audio_encoder_checker.py            # Tensor-returning encoder checker
├── models/
│   ├── audiotransformer.py             # CED model definitions and pretrained entries
│   ├── checkpoints.py                  # Checkpoint loading and model registry
│   ├── ensemble.py                     # Ensemble utilities
│   └── layers.py                       # Transformer and audio patch layers
├── scripts/
│   ├── audio_encoder_checker.py
│   └── generate_configs.py
├── xares_llm/
│   ├── run.py                          # Training/evaluation command-line entry
│   ├── task.py                         # Task setup, checkpoint and score handling
│   ├── trainer.py                      # Trainer and generation-based evaluation
│   ├── audiowebdataset.py              # Audio/text WebDataset loading
│   ├── download_data.py                # Dataset download utility
│   ├── metrics.py                      # Classification and understanding metrics
│   ├── modeling_audiollm/              # Audio projector and LoRA language-model wrapper
│   └── tasks/                          # Task 1, Task 2, joint and single-dataset YAMLs
├── docs/
│   ├── technical-report.pdf            # Report copied from the submission archive
│   └── requirements-submission.txt     # Original submitted dependency list
├── results/
│   ├── task1-scores.tsv                # Original submitted Task 1 score table
│   └── task2-scores.tsv                # Original submitted Task 2 score table
└── README.md
```
### Competition Submission Archive

The separately supplied `echobridge.zip` contains:

```text
echobridge/
├── example/
│   ├── echobridge/                     # Tuple-returning competition encoder + models/
│   └── echobridgeAPI/                  # Tensor-returning competition encoder + models/
├── experiments/
│   ├── Task1/echobridge/
│   │   ├── checkpoint-100000/
│   │   ├── predictions_eval_*.csv
│   │   ├── score_eval_*.yaml
│   │   └── scores.tsv
│   └── Task2/echobridge/
│       ├── checkpoint-50000/
│       ├── predictions_eval_*.csv
│       ├── score_eval_*.yaml
│       └── scores.tsv
├── Interspeech2026_Audio_Encoder_Challenge.pdf
├── requirements.txt
└── Readme.md
```

The archive contains submission artifacts; the complete source project supplies the XARES-LLM harness. Large checkpoints, prediction files, and the submission encoder directories are not duplicated by this documentation package.
## Getting Started

### 1. Environment

Create an environment from the repository root:

```bash
conda create -n echobridge python=3.12
conda activate echobridge
```

The submitted dependency list records these principal versions:

| Package | Submitted version |
|---|---|
| PyTorch | 2.6.0+cu124 |
| Torchaudio | 2.6.0+cu124 |
| Transformers | 4.56.2 |
| PEFT | 0.18.1 |
| Hugging Face Hub | 0.35.1 |
| WebDataset | 1.0.2 |

For the submitted CUDA 12.4 package variants:

```bash
python -m pip install -r docs/requirements-submission.txt --extra-index-url https://download.pytorch.org/whl/cu124
python -m pip check
```

The dependency file is preserved from the competition submission; a clean installation has not been validated during this documentation pass. CPU or other CUDA environments need matching PyTorch and Torchaudio pins. Run commands from the repository root: this source snapshot does not include a `pyproject.toml` or `setup.py` for `pip install .`.

### 2. Pretrained Models

The code initializes the following components:

| Component | Model identifier / source |
|---|---|
| Speech encoder | `openai/whisper-base` |
| Acoustic encoder | `ced_mini`, with a checkpoint URL defined in `models/audiotransformer.py` |
| Language decoder | `HuggingFaceTB/SmolLM2-135M` |

Initial construction may download pretrained assets. Instantiating an encoder loads its pretrained backbones but does **not** load trained EchoBridge fusion weights from the competition checkpoints.

### 3. Datasets

The included YAML configurations reference the Hugging Face dataset repositories `mispeech/xares_llm_data` and `mispeech/MECAT-Caption`. Data is cached under `./xares_data` by default.

```bash
python -m xares_llm.download_data
```

To select a cache directory, set `XARES_DATA_HOME` before running the download or training entry. For example, in Bash:

```bash
export XARES_DATA_HOME=/path/to/xares_data
```

In PowerShell:

```powershell
$env:XARES_DATA_HOME = 'D:/datasets/xares_data'
```

Dataset paths, task prompts, sampling probabilities, and evaluation metrics are defined in `xares_llm/tasks/`.

## Project Workflow

### 1. Check the Root Encoder Interface

```bash
python check.py
```

The checker inspects the class name, required public attributes, and output tensor shape using an input of shape `[2, 50000]`. For an explicit success assertion:

```python
from audio_encoder_checker import check_audio_encoder
from echobridge import EchoBridgeUp2Encoder

encoder = EchoBridgeUp2Encoder()
assert check_audio_encoder(encoder), "Encoder interface check failed"
```

This checks the root tensor API; it does not evaluate recognition quality or verify the submission results.

### 2. Extract Audio Features

```python
import torch
import torchaudio
from echobridge import EchoBridgeUp2Encoder

encoder = EchoBridgeUp2Encoder().eval()
audio, sample_rate = torchaudio.load("path/to/audio.wav")
audio = audio.mean(dim=0, keepdim=True)  # Mono batch: [1, samples]
if sample_rate != encoder.sampling_rate:
    audio = torchaudio.functional.resample(
        audio, sample_rate, encoder.sampling_rate
    )

with torch.no_grad():
    features, feature_mask = encoder.forward_with_mask(audio)

print(features.shape)      # [B, T, 1024] with the default configuration
print(feature_mask.shape)  # [B, T]; 1 = valid, 0 = padding
```

The default encoder exposes a 16 kHz input sample rate and an output dimension of 1024. Use short, appropriately segmented inputs; the harness's default training crop is 30 seconds. The example produces features from the initialized root variant, not the trained competition system.

### 3. Run the Submission Encoder with XARES-LLM

Extract the submission archive and copy its `example/` directory into the complete project's root. This creates `example/echobridge/echobridge.py` alongside `xares_llm/` without replacing the root encoder.

Use the **tuple-returning standard submission implementation** with the included harness:

```bash
# Task 1: audio classification across 15 datasets
python -m xares_llm.run example/echobridge/echobridge.py task1 task1

# Task 2: audio understanding across 5 datasets
python -m xares_llm.run example/echobridge/echobridge.py task2 task2

# Single-dataset example
python -m xares_llm.run example/echobridge/echobridge.py clotho clotho
```

The tensor-only root/API versions cannot be substituted directly in these commands: `XaresLLMModel` unpacks `(features, attention_mask)`. These commands describe the supplied harness workflow; exact report reproduction additionally depends on the implementation and checkpoint issues documented below.

### 4. Training Configuration

The task YAMLs and runner defaults specify:

| Setting | Value |
|---|---|
| Task 1 training steps | 100,000 |
| Task 2 training steps | 50,000 |
| Default per-device training batch size | 4 |
| Default learning rate | 1e-4 |
| Default optimizer | AdamW (`adamw_torch`) |
| Default weight decay | 0.01 |
| Default warmup steps | 200 |
| Seed | 42 |
| Default training audio crop | 30 seconds |
| Decoder | SmolLM2-135M |
| Decoder LoRA | rank 8, alpha 32, dropout 0.1; all linear layers |

Use one GPU and FP32 to follow the reproducibility guidance retained in the supplied project. The table combines code defaults and task YAML settings; it is not an independently recovered record of every submitted run.

### 5. Outputs and Checkpoints

The local harness builds output paths from the training YAML stem and encoder filename:

```text
experiments/
├── train_task1_config/echobridge/
└── train_task2_config/echobridge/
```

Each experiment may contain checkpoints, `log.txt`, prediction CSVs, per-dataset score YAMLs, and `scores.tsv`. The submission archive uses the different directory labels `Task1` and `Task2`.

When an output directory already contains checkpoints, the runner selects the most recently modified checkpoint and skips training. Existing per-dataset score YAMLs also cause evaluation to be skipped. Use a fresh output directory for a new experiment to avoid reusing cached results.

Submission checkpoints contain the **full XARES-LLM system**, not just a standalone audio encoder. Their configuration also records the encoder's import path. Task 1 points to `example/echobridge/echobridge.py`; Task 2 points to `example/enchobridgeup4/echobridgeup4.py`, which is absent from the supplied archive. Restore the matching source and verify parameter compatibility before reusing that checkpoint.

## Results

The following scores come from the technical report and the submitted `scores.tsv` files. No training or benchmark evaluation was rerun for this README. Higher scores are better under the configured metrics; an aggregate score is not a single classification-accuracy percentage.

### Overall Comparison

The report calls the tasks **Track A** and **Track B**; the code calls them **Task 1** and **Task 2**.

| Model / configuration | Task 1: Classification | Task 2: Understanding |
|---|---:|---:|
| Dasheng-base | 0.614 | 0.229 |
| CED-mini | 0.598 | 0.286 |
| Whisper | 0.650 | 0.399 |
| + Fusion | 0.702 | 0.384 |
| + Concat-linear | 0.708 | 0.398 |
| **EchoBridge (+ Adapter)** | **0.712** | **0.401** |

Baseline and ablation aggregates are quoted as reported in Tables 1–2. In particular, the report's rounded per-dataset CED-mini values in Table 1 do not reproduce its stated 0.598 aggregate; the original baseline logs were not supplied for reconciliation. EchoBridge's scores below agree with the submitted TSV files.

### Task 1: Audio Classification

Source: [submitted Task 1 scores](results/task1-scores.tsv).

| Dataset | Domain | Metric | Score |
|---|---|---|---:|
| ASVspoof2015 | Speech | Accuracy | 0.909 |
| CREMA-D | Speech | Accuracy | 0.490 |
| ESC-50 | Environmental sound | Accuracy | 0.877 |
| Fluent Speech Commands | Speech | Accuracy | 0.771 |
| Free Music Archive | Music | Accuracy | 0.646 |
| FSD50K | Environmental sound | mAP | 0.198 |
| FSDKaggle2018 | Environmental sound | mAP | 0.683 |
| GTZAN | Music | Accuracy | 0.899 |
| LibriCount | Speech | Accuracy | 0.470 |
| NSynth | Music | Accuracy | 0.693 |
| Speech Commands V1 | Speech | Accuracy | 0.632 |
| UrbanSound8K | Environmental sound | Accuracy | 0.830 |
| VocalSound | Vocal sounds | Accuracy | 0.913 |
| VoxCeleb1 | Speech | Accuracy | 0.886 |
| VoxLingua33 | Speech | Accuracy | 0.779 |
| **Unweighted average** | | | **0.712** |

### Task 2: Audio Understanding

Source: [submitted Task 2 scores](results/task2-scores.tsv). Metric names follow the included evaluation YAMLs.

| Dataset | Task | Metric | Score |
|---|---|---|---:|
| AISHELL-1 | Chinese speech transcription | iCER | 0.236 |
| Clotho | Audio captioning | FENSE | 0.422 |
| LibriSpeech | English speech transcription | iWER | 0.302 |
| MECAT | General audio captioning | DATE | 0.569 |
| Song Describer | Music captioning | FENSE | 0.477 |
| **Unweighted average** | | | **0.401** |

## Reproducibility Notes

- **Version selection:** the root MLP adapter and submitted temporal-convolution adapter are different implementations. The reported results belong to the submission artifacts; they should not be attributed to an untrained instance of the root encoder.
- **Encoder optimization:** the included XARES-LLM wrapper runs the entire encoder under `torch.no_grad()`. Consequently, that path does not optimize the encoder's internal fusion or adapter parameters, even though the report describes trainable fusion/adaptation. The submission's `freeze_base=True` also freezes its projection and fusion layers. A matching training configuration is needed to establish the report's optimization procedure.
- **Residual initialization:** the submission initializes both its pointwise adapter projection and residual scale to zero. By inspection, this makes both factors receive zero gradient through that residual product at initialization. Any claim that this branch learns from scratch requires a verified initialization or checkpoint history.
- **Time alignment:** the implementation interpolates shorter CED sequences and truncates longer ones. It also derives Whisper feature lengths using two reductions and advertises a 40 ms hop. Frame timing should be checked against the actual backbone before timestamp-sensitive use; the supplied checker validates shapes, not timing correctness.
- **Download endpoint:** `xares_llm/run.py` explicitly forces `HF_ENDPOINT` to `https://hf-mirror.com`. Change that override if your environment requires a different endpoint; setting an environment variable alone will not override this assignment.

These notes document the supplied source versions. The README preparation did not modify model code, run training, download model weights, or verify checkpoint inference.

## Citation

Please cite the accompanying technical report when using this work. It is identified here as a challenge report, without asserting a proceedings publication or DOI.

```bibtex
@techreport{tang2026echobridge,
  title       = {Echobridge: Dual-Encoder Audio Fusion Based on Residual Time Adaptation},
  author      = {Yang Tang and Hongqing Liu and Liming Shi and Lu Gan},
  year        = {2026},
  institution = {Chongqing University of Posts and Telecommunications and Brunel University},
  type        = {Technical Report},
  note        = {Submission to the Interspeech 2026 Audio Encoder Capacity Challenge}
}
```

## Acknowledgements

This project builds on [XARES-LLM](https://github.com/xiaomi-research/xares-llm), [Whisper](https://github.com/openai/whisper), CED, and the [audio encoder interface template](https://github.com/jimbozhang/xares-llm-template). References for the pretrained encoders and adaptation methods are included in the [technical report](docs/technical-report.pdf).

## License

The included XARES-LLM source carries Apache-2.0 notices. This project snapshot does not contain a root-level license covering all EchoBridge additions and assets. Retain upstream notices and refer to the applicable licenses for source code, pretrained weights, and datasets.
