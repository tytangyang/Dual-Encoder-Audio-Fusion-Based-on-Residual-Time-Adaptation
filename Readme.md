# Echobridge: Dual-encoder audio fusion based on residual time adaptation

Echobridge is a robust dual-encoder audio front-end that fuses **Whisper** (speech-centric) and **CED-mini** (environment-centric) using a **zero-initialised multi-scale temporal convolutional adapter** for stable and adaptive feature fusion.

## Installation

### 1) Create environment
```bash
conda create -n echobridge python=3.12 -y
conda activate echobridge
```
### 2) Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

We provide two implementations of Echobridge.

First, the standard Echobridge model outputs a tensor of shape [B, T′, D] together with an attention mask of shape [B, T′]. This implementation is compatible with the Xiaomi Research xares-llm project (https://github.com/xiaomi-research/xares-llm ）and its result store in experiments file.

Second, the Echobridge API implementation is validated by passing audio_encoder_check.py in the xares-llm-template repository（ https://github.com/jimbozhang/xares-llm-template ）. It likewise produces an output tensor of shape [B, T′, D].

### Task 1 ( Classification )
```bash
python3.12 -m xares_llm.run example/echobridge/echobridge.py task1 task1
python3.12 -m xares_llm.run example/echobridgeAPI/echobridge.py task1 task1
```

### Task 2 ( Understanding )
```bash
python3.12 -m xares_llm.run example/echobridge/echobridge.py task2 task2
python3.12 -m xares_llm.run example/echobridgeAPI/echobridge.py task2 task2
```
## Project Structure
├── Echobridge
|   ├── example
|       ├── echobridge
|           ├── models
|               ├── _init_.py
|               ├── audiotransformer.py
|               ├── checkpoints.py
|               ├── ensemble.py
|               └── layers.py
|           └── echobridge.py
|       └── echobridgeAPI
|           ├── models
|               ├── _init_.py
|               ├── audiotransformer.py
|               ├── checkpoints.py
|               ├── ensemble.py
|               └── layers.py
|           └── echobridge.py
|   ├──experiments
|   ├──Readme.md
|   └──requirement.txt

## Results


### Task1

| Task | echobridge Score |
| :--- | :---: |
| `eval_asvspoof2015` | 0.909 |
| `eval_cremad` | 0.490 |
| `eval_esc-50` | 0.877 |
| `eval_fluentspeechcommands` | 0.771 |
| `eval_freemusicarchive` | 0.646 |
| `eval_fsd50k` | 0.198 |
| `eval_fsdkaggle2018` | 0.683 |
| `eval_gtzan` | 0.899 |
| `eval_libricount` | 0.470 |
| `eval_nsynth` | 0.693 |
| `eval_speechcommandsv1` | 0.632 |
| `eval_urbansound8k` | 0.830 |
| `eval_vocalsound` | 0.913 |
| `eval_voxceleb1` | 0.886 |
| `eval_voxlingua33` | 0.779 |
| Overall | 0.712 | 


### Task2

| Task | echobridge Score |
| :--- | :---: | :---: |
| `eval_aishell-1` | 0.236 |
| `eval_clotho` | 0.422 |
| `eval_librispeech` | 0.302 |
| `eval_mecat` | 0.569 |
| `eval_songdescriber` | 0.477 |
| Overall | 0.401 | 