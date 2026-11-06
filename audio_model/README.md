---
license: mit
tags:
  - audio-classification
  - anti-spoofing
  - asvspoof
  - deepfake-detection
datasets:
  - asvspoof2019
pipeline_tag: audio-classification
metrics:
  - eer
  - t-dcf
---

# wav2vec2-v2-unfrozen — Improved Wav2Vec 2.0 for ASVspoof 2019 LA

Best improved Wav2Vec 2.0 checkpoint from a comparative study of three neural anti-spoofing architectures. Achieves the **best eval EER (1.55%) and best tandem min t-DCF (0.3966)** across all models in the study. Supersedes [caa-speech-detection-asvspoof2019/wav2vec2](https://huggingface.co/caa-speech-detection-asvspoof2019/wav2vec2).

**Version:** wav2vec2_v2 (top-4 transformer blocks unfrozen)

## Architecture

`facebook/wav2vec2-base` with the top 4 transformer blocks unfrozen and a lightweight classification head.

| Component | Value |
|---|---|
| Base model | `facebook/wav2vec2-base` |
| Encoder | Top 4 transformer blocks trainable (`freeze_last_n=4`); rest frozen |
| Classification head | Linear(768→256) → GELU → Dropout(0.1) → Linear(256→2) |
| Input | Raw waveform, 16 kHz, padded/truncated to 64 000 samples (4 s) |
| Parameters | ~95 M total; top-4 blocks + head trainable |
| Checkpoint size | ~362 MB |

Encoder masking (`mask_time_prob`, `mask_feature_prob`) is disabled at inference, following Tak et al. (2022).

Reference: Baevski et al., *"wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"*, NeurIPS 2020.

## Training

| Hyperparameter | Value |
|---|---|
| Epochs | 40 (stopped early at 35) |
| Batch size | 8 |
| Learning rate | 1e-5, cosine schedule |
| Weight decay | 1e-4 |
| Gradient clip norm | 1.0 |
| Early stopping patience | 12 |
| Class weights | `[3.0, 1.0]` |

Dataset: ASVspoof 2019 LA train split (~25k utterances). No data augmentation.

## Results

Baseline to beat: **EER 8.09%** (LFCC+GMM).

| Split | EER | tandem min t-DCF | In-the-Wild EER |
|---|---|---|---|
| Dev | 0.197% | — | — |
| **Eval** | **1.55%** | **0.3966** | **27.68%** |

Dev EER improved from 4.199% (baseline, fully-frozen encoder) to 0.197%. Eval EER improved from 7.53% to 1.55% — an 79% relative reduction. This is the best result across all three models (LCNN, RawNet2, Wav2Vec 2.0) in the study, both in EER and t-DCF.

See `learning_curves/wav2vec2_baseline_vs_improved.png` for the training trajectory.

## Usage

Install dependencies from the source repository, then:

```python
import torch
from src.models.wav2vec2.model import Wav2Vec2Model

config = {
    "pretrained_model": "facebook/wav2vec2-base",
    "freeze_encoder": True,
    "freeze_last_n": 4,
    "hidden_dim": 256,
    "dropout": 0.1,
    "target_samples": 64000,
    "class_weights": [3.0, 1.0],
}

model = Wav2Vec2Model(config)
state = torch.load("best.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

# waveform: (B, T) float32 at 16 kHz, T = 64000
with torch.no_grad():
    logits = model({"frames": waveform})["logits"]
    probs = torch.softmax(logits, dim=-1)  # [:, 0] = bonafide, [:, 1] = spoof
```

Source: [github.com/sebastiaoteixeira/caa-ai-generated-speech-detector](https://github.com/sebastiaoteixeira/caa-ai-generated-speech-detector)

## Limitations

- Trained on ASVspoof 2019 LA only. Not validated on physical access or other corpora.
- In-the-Wild EER of 27.68% indicates limited generalisation beyond the ASVspoof 2019 attack pool.
- The ~362 MB checkpoint size makes this unsuitable for edge or real-time deployment. For resource-constrained scenarios, `lcnn_v7_cqt` (~3.5 MB, 3.26% eval EER) is recommended.
- No data augmentation was applied; domain shift may degrade performance in real deployments.

## Citation

```bibtex
@inproceedings{wang2020asvspoof,
  title     = {{ASVspoof} 2019: A Large-Scale Public Database of Synthesized, Converted and Replayed Speech},
  author    = {Wang, Xin and others},
  booktitle = {Computer Speech \& Language},
  volume    = {64},
  year      = {2020}
}
```

```bibtex
@inproceedings{baevski2020wav2vec,
  title     = {wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations},
  author    = {Baevski, Alexei and Zhou, Henry and Mohamed, Abdelrahman and Auli, Michael},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2020}
}
```
