# BLIP-2 + LLaMA-3 on VQA-v2

Fine-tuning & evaluation scripts for swapping the **OPT-2.7B** language head in BLIP-2 with **LLaMA-3-8B** and training **only the Q-Former** on the Visual-Question-Answering v2 benchmark.

> **TL; DR**
> `main.py` trains the bridge module (full or LoRA).
> `evaluate.py` measures VQA accuracy with a tiny helper normaliser.
> Re-running the workflow exactly as in the accompanying term paper reproduces the **0 % validation accuracy** negative result.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Expected results & logs](#expected-results--logs)
7. [FAQ & limitations](#faq--limitations)
8. [Citation](#citation)
9. [Licence](#licence)

---

## Features<a name="features"></a>

| Component                          | What it does                                                                                                                                                                                                        | Where |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| **`main.py`**                      | Loads *ViT-L/14* + *LLaMA-3-8B*, freezes both back-bones, then tunes the 32-token **Q-Former** (full or LoRA) on 490 px crops from **VQA-v2**. Training loop is driven by `trl.SFTTrainer`, saving every 1 k steps. |       |
| **`evaluate.py`**                  | Greedy-decodes answers, normalises with a lightweight article & punctuation stripper, and reports the official VQA accuracy metric with a live `tqdm` progress bar.                                                 |       |
| **Paper (`712_Term_Paper-4.pdf`)** | Full experimental write-up explaining why *Q-Former-only* fine-tuning fails and plateaus at 7.7 loss → 0 % acc.                                                                                                     |       |

---

## Installation<a name="installation"></a>

```bash
# 🐧 Tested on Ubuntu 22.04 + Python 3.10
conda create -n blip2-llama3 python=3.10 -y
conda activate blip2-llama3

# CUDA-enabled PyTorch (adjust cu* tag to your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Core deps
pip install transformers==4.41.2 accelerate datasets peft==0.11.1 trl==0.8.6 \
            aiohttp tqdm pillow

# Optional: evaluation only
pip install scikit-learn
```

> **GPU** A single NVIDIA H100/80 GB (or ≥48 GB A/H-class) is required for the default batch sizes. Reduce `--batch_size` if memory-constrained.

---

## Quick start<a name="quick-start"></a>

1. **Authenticate LLaMA-3 weights**

```bash
huggingface-cli login      # must have accepted the Meta licence
```

2. **Train Q-Former (LoRA, 5 epochs)**

```bash
python main.py \
  --llama_name meta-llama/Llama-3.1-8B-Instruct \
  --blip2_opt_name Salesforce/blip2-opt-2.7b \
  --output_dir ./checkpoints/blip2-llama3-vqa \
  --tuning_mode lora \
  --batch_size 128 --epochs 5 --lr 1e-5
```

3. **Evaluate**

```bash
python evaluate.py \
  --model_path ./checkpoints/blip2-llama3-vqa \
  --split validation \
  --batch_size 32
```

Logs summarise **Q / GT / PRED** every 100 items and end with overall accuracy.

---

## Training<a name="training"></a>

`main.py --help` shows every flag, the most relevant being:

| Flag               | Default                            | Description                                           |
| ------------------ | ---------------------------------- | ----------------------------------------------------- |
| `--llama_name`     | `meta-llama/Llama-3.1-8B-Instruct` | Frozen decoder-only LLM.                              |
| `--blip2_opt_name` | `Salesforce/blip2-opt-2.7b`        | Supplies vision encoder + Q-Former init.              |
| `--tuning_mode`    | `full` \| `lora`                   | Update all Q-Former weights or inject LoRA (r = 8).   |
| `--epochs`         | `5`                                | Passes over the 443 k-pair training split.            |
| `--lr`             | `1e-5`                             | AdamW learning rate (linear decay after 1 k warm-up). |
| `--batch_size`     | `128`                              | Images / GPU step; halve on 24 GB cards.              |

Collation masks the prompt tokens (`labels = -100`) so that only the answer portion contributes to the cross-entropy loss.&#x20;

---

## Evaluation<a name="evaluation"></a>

`evaluate.py`:

```text
Evaluating: 100%|████████████████████████| 167/167 [02:13<00:00,  1.25batch/s]

Q: what color is the cat ? 
GT: ['white'] 
PRED: slim                                   

✅  validation accuracy: 0.00% (21 714 questions)
```

The script implements the official normalisation rule (article removal + punctuation stripping) and caps per-answer credit at 1/3 as per VQA-v2.&#x20;

---

## Expected results & logs<a name="expected-results--logs"></a>

| Setting                       | Val Acc (%) | Loss plateau |
| ----------------------------- | ----------- | ------------ |
| **Full** Q-Former, 5 ep, 1e-5 | **0.0**     | 7.7          |
| **LoRA** (r = 8), same        | **0.0**     | 7.7          |

See IV of the term paper for complete sweeps (epochs ∈ {1, 5, 10}, lr ∈ {1e-5, 2e-5, 5e-5}) and qualitative outputs.&#x20;

---

## FAQ & limitations<a name="faq--limitations"></a>

* **Why zero accuracy?**
  BLIP-2’s Q-Former was *pre-trained on \~129 M* image–caption pairs. Skipping that and relying on 400 k VQA pairs leaves the bridge hopelessly under-fit.

* **Can I unfreeze LLaMA layers?**
  Yes—extend `freeze_but_qformer` to leave the top *N* decoder blocks trainable, or swap to full-model LoRA.

* **Different resolution?**
  Change `--img_size` inside `load_blip2_llama`. Keep it ≥384 to match ViT patching.

* **Dataset download stalls behind a firewall**
  Set `HF_ENDPOINT=https://hf-mirror.com` or pre-download the `HuggingFaceM4/VQAv2` dataset.

---

## Citation<a name="citation"></a>

If you build on this code or the accompanying negative-results study, please cite:

```bibtex
@misc{kolukisa2025qformerfail,
  title   = {Frozen LLaMA-3 in a BLIP-2 Pipeline: Why Q-Former Fine-Tuning Alone Fails for Visual Question Answering},
  author  = {Oğuz Kolukısa},
  year    = {2025},
  note    = {Project code and reproduction scripts at https://https://github.com/oguz-kolukisa/712}
}
```

---

## Licence<a name="licence"></a>

* Code: MIT
* LLaMA-3 weights: Meta LLM Licence
* VQA-v2 dataset: CC BY 4.0

> “Look on my works, ye Mighty, and despair!” – *Ozymandias* 🏺
