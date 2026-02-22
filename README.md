## [WACV 2026] Empowering Source-Free Domain Adaptation via MLLM-Guided Reliability-Based Curriculum Learning

[Dongjie Chen*](https://www.linkedin.com/in/dongjie-chen-94b21a165), 
[Kartik Patwari*](https://kartikp7.github.io/), 
[Zhengfeng Lai](https://zjujefflai.github.io/), 
[Xiaoguang Zhu](https://www.linkedin.com/in/xiaoguang-zhu-4b21bb26b/), 
[Sen-ching Cheung](https://sites.google.com/view/dr-cheung), 
[Chen-Nee Chuah](https://www.ece.ucdavis.edu/~chuah/rubinet/people/chuah/bio.html)

<div align="center">

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://github.com/Dong-Jie-Chen/RCL)
[![arXiv](https://img.shields.io/badge/arXiv-2405.18376-b31b1b.svg)](https://arxiv.org/abs/2405.18376)

<p align="center">
  <img src="images/RCL.jpg" width="500">
</p>

</div>

---

## Overview

This repository provides the official implementation of **RCL**, a three-stage source-free domain adaptation (SFDA) pipeline. Three MLLMs (CLIP ViT-L/14, LLaVA-v1.6-7B, ShareGPT4V-7B) are used to generate pseudo-labels for the target domain offline. Their agreement pattern partitions the target data by reliability, and a curriculum progressively trains the student model from high-confidence to low-confidence samples:

1. **Stage 1 – RKT (Reliable Knowledge Transfer):** Train exclusively on the samples where all MLLMs agree (fully reliable subset), using cross-entropy loss to give the student a clean start.
2. **Stage 2 – SMKE (Self-correcting and MLLM-guided Knowledge Expansion):** Expand to partially-agreed samples. When the student's confidence exceeds a threshold τ, it uses its own prediction (self-correction); otherwise the majority vote across MLLMs is used.
3. **Stage 3 – MMR (Multi-hot Masking Refinement):** Incorporate all remaining samples (full MLLM disagreement). A multi-hot mask—the union of all MLLM-suggested classes—constrains the student's logit space, with weak/strong augmentation consistency regularisation applied to uncertain samples.

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- scikit-learn
- numpy
- Pillow

For MLLM inference (pseudo-label generation, not included in this release):

- `transformers>=4.35` (BLIP-2, InstructBLIP, CLIP)
- `sentence-transformers` (class-name matching)
- LLaVA: `pip install git+https://github.com/haotian-liu/LLaVA.git`
- ShareGPT4V: `pip install git+https://github.com/ShareGPT4Omni/ShareGPT4V.git`

---

## Dataset Preparation

Organize datasets under `./data/` as follows:

```
data/
├── office_home/
│   ├── Art/
│   │   ├── Alarm_Clock/
│   │   │   ├── 00001.jpg
│   │   │   └── ...
│   │   └── ...
│   ├── Clipart/
│   ├── Product/
│   └── Real_World/
├── domainnet/
│   ├── clipart/
│   ├── infograph/
│   ├── painting/
│   ├── quickdraw/
│   ├── real/
│   └── sketch/
└── visda/
    ├── train/         (source domain)
    └── validation/    (target domain)
```

Each dataset also requires per-domain `.txt` index files (one `image_path label` pair per line), e.g. `data/office_home/Art_new_key.txt`. These follow the same format as [SHOT](https://github.com/tim-learn/SHOT).

### MLLM Pseudo-label Files

The training scripts expect pre-computed MLLM pseudo-label files under `data/<dataset>_<adapt_strategy>/`. Each file lists one `image_path label` pair per line.

**Training files** (one per stage):

| File | Used by |
|---|---|
| `all_agree_voting_<domain>.txt` | Stage 1 (RKT) |
| `majority_vote_no_disagree_<domain>.txt` | Stage 2 (SMKE) |
| `majority_vote_no_disagree_ind_vote_<domain>.txt` | Stage 3 (MMR, `--ind_vote`); 3 labels per line |

**Validation files** (optional, loaded every run for diagnostic logging):

| File | Description |
|---|---|
| `all_agree_gt_<domain>.txt` | Samples where all MLLMs agree |
| `only_two_agree_gt_<domain>.txt` | Samples where exactly two MLLMs agree |
| `all_disagree_gt_<domain>.txt` | Samples where no two MLLMs agree |
| `incorrect_vote_gt_<domain>.txt` | Samples where the majority vote is wrong |

These files are produced by running MLLM inference (CLIP, LLaVA-v1.6-7B, ShareGPT4V-7B) on the target domain images and aggregating the per-model predictions.

---

## Source Model Preparation

We use source-pretrained models from [SHOT](https://github.com/tim-learn/SHOT). Train or download SHOT source models and place them as:

```
source_models_from_SHOT/seed2021/
├── office_home/
│   ├── A/
│   │   ├── source_F.pt
│   │   ├── source_B.pt
│   │   └── source_C.pt
│   ├── C/
│   ├── P/
│   └── R/
├── domainnet/
│   ├── clipart/
│   └── ...
└── visda/
    └── T/
```

---

## Usage

Set `$i` (source domain index) and `$t` (target domain index).

- **Office-Home:** 0=Art, 1=Clipart, 2=Product, 3=Real\_World
- **DomainNet:** 0=clipart, 1=infograph, 2=painting, 3=quickdraw, 4=real, 5=sketch
- **VisDA:** 0=train (source), 1=validation (target)

### Stage 1 – RKT

```bash
python -u target_finetune.py \
  --dataset office_home \
  --pretrain SHOT \
  --work_dir office_home_stage1_all_agree_voting_mllm_clip_llava7b_sharegpt7b_1e-4 \
  --adapt_step 3000 \
  --batch_size 128 \
  --bAcc \
  --adapt mllm_clip_llava7b_sharegpt7b \
  --mllm_strategy all_agree_voting \
  --lr 1e-4 \
  --seed 0 \
  --source $i \
  --target $t \
  --ckpt_dir ./source_models_from_SHOT/seed2021
```

### Stage 2 – SMKE

```bash
python -u target_finetune.py \
  --dataset office_home \
  --pretrain PREV_STAGE \
  --work_dir office_home_stage2_majority_vote_no_disagree_mllm_clip_llava7b_sharegpt7b_clself070_1e-5 \
  --adapt_step 5000 \
  --batch_size 128 \
  --bAcc \
  --adapt mllm_clip_llava7b_sharegpt7b \
  --mllm_strategy majority_vote_no_disagree \
  --cl curriculum_2and3_self_correct \
  --p_th 0.7 \
  --lr 1e-5 \
  --seed 0 \
  --source $i \
  --target $t \
  --ckpt_dir ./logs/office_home/office_home_stage1_all_agree_voting_mllm_clip_llava7b_sharegpt7b_1e-4
```

### Stage 3 – MMR

```bash
python -u target_finetune.py \
  --dataset office_home \
  --pretrain PREV_STAGE \
  --work_dir office_home_stage3_majority_vote_mllm_clip_llava7b_sharegpt7b_cltcf_conf_th090_1e-5 \
  --adapt_step 8000 \
  --batch_size 128 \
  --bAcc \
  --adapt mllm_clip_llava7b_sharegpt7b \
  --mllm_strategy majority_vote \
  --cl curriculum_teacher_class_filter_conf_th_ssl \
  --p_th 0.9 \
  --ind_vote \
  --ssl fixmatch \
  --lr 1e-5 \
  --seed 0 \
  --source $i \
  --target $t \
  --ckpt_dir ./logs/office_home/office_home_stage2_majority_vote_no_disagree_mllm_clip_llava7b_sharegpt7b_clself070_1e-5
```

Outputs (checkpoints and logs) are saved under `./logs/<dataset>/<work_dir>/source<i>/target<t>/`.

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `office_home` | Dataset: `office_home`, `domainnet`, `visda` |
| `--pretrain` | `SHOT` | Source model: `SHOT` (from SHOT checkpoints) or `PREV_STAGE` (from a prior RCL stage) |
| `--adapt` | `None` | Adaptation mode (`mllm_clip_llava7b_sharegpt7b` for the paper's main method) |
| `--mllm_strategy` | `None` | MLLM voting strategy: `all_agree_voting`, `majority_vote_no_disagree`, `majority_vote` |
| `--cl` | `None` | Curriculum learning: `curriculum_2and3_self_correct` (Stage 2) or `curriculum_teacher_class_filter_conf_th_ssl` (Stage 3) |
| `--p_th` | `None` | Confidence threshold for curriculum filtering (0.7 in Stage 2, 0.9 in Stage 3) |
| `--ind_vote` | `False` | Use per-MLLM individual vote labels (required for Stage 3) |
| `--ssl` | `None` | Semi-supervised method (`fixmatch`, used in Stage 3) |
| `--lambda_u` | `0.5` | Weight for unlabeled FixMatch loss |
| `--bAcc` | `False` | Use balanced accuracy as the primary checkpoint-selection metric |
| `--lr` | `0.001` | Learning rate |
| `--seed` | `0` | Random seed |

---

## Acknowledgements & Credits

This codebase builds on several open-source works. We thank their authors:

- **SHOT** – Source model architecture (`ResBase`, `feat_bottleneck`, `feat_classifier`) and training protocol:
  [https://github.com/tim-learn/SHOT](https://github.com/tim-learn/SHOT)
  (Liang et al., "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation", ICML 2020)

- **fewshot-SFDA** – Overall training framework structure:
  [https://github.com/daintlab/fewshot-SFDA](https://github.com/daintlab/fewshot-SFDA)

- **DomainBed** – `InfiniteDataLoader` in `data_loader.py`:
  [https://github.com/facebookresearch/DomainBed](https://github.com/facebookresearch/DomainBed)

- **sentence-transformers** – Class-name matching in MLLM inference:
  [https://www.sbert.net](https://www.sbert.net)

---

## Citation

If you find RCL useful in your research or applications, please consider giving us a star 🌟 and citing our work:

```bibtex
@article{chen2024empowering,
  title={Empowering Source-Free Domain Adaptation via MLLM-Guided Reliability-Based Curriculum Learning},
  author={Chen, Dongjie and Patwari, Kartik and Lai, Zhengfeng and Zhu, Xiaoguang and Cheung, Sen-ching and Chuah, Chen-Nee},
  journal={arXiv preprint arXiv:2405.18376},
  year={2024}
}
```
