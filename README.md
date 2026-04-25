<div align="center">

# 🧠 LLMMEMORY

### Differentiable Neural Computer Augmented Language Model

*Augmenting GPT-2 with learnable external memory for improved sequential reasoning*

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-4.x-FFD21E?style=for-the-badge)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-WikiText--2-blue?style=for-the-badge)](https://huggingface.co/datasets/wikitext)

</div>

---

## Overview

**DNC-GPT2** integrates a [Differentiable Neural Computer](https://www.nature.com/articles/nature20101) (Graves et al., 2016) external memory module into GPT-2, enabling the model to learn *when* and *what* to write to an external memory bank during language modelling. The repository provides a full training and evaluation pipeline with side-by-side comparison against a vanilla GPT-2 baseline.

<div align="center">

| | GPT-2 Baseline | GPT-2 + DNC |
|:---|:---:|:---:|
| Architecture | Transformer | Transformer + External Memory |
| Memory | Implicit (weights) | Explicit (N × W matrix) |
| Read heads | — | 4 |
| Extra parameters | — | ~2M |
| Loss | CE | CE + Routing + Entropy |

</div>

---

## Architecture

### DNC Memory Module

At each token step `t`, the GPT-2 hidden state `h_t` acts as the controller and interfaces with an `N × W` external memory matrix via soft, differentiable attention:

```
Input token sequence
        │
   ┌────▼─────┐
   │  GPT-2   │   (n_layer=6, n_head=8, n_embd=768)
   └────┬─────┘
        │  h_t  (per-token hidden state)
        │
   ┌────▼──────────────────────────────────┐
   │             DNC Memory                │
   │                                       │
   │  write_key ──► content addressing ─┐  │
   │  usage     ──► alloc  addressing  ─┤  │
   │                                    ▼  │
   │              w_write ──► erase + write│
   │                                       │
   │  read_keys ──► content addressing     │
   │                        ──► w_read     │
   │                            ──► r_t   │
   └────────────────────┬──────────────────┘
                        │  r_t  (R × W read vector)
                        │
               ┌────────▼───────────────┐
               │  h_t + read_proj(r_t)  │
               └────────┬───────────────┘
                        │
                   ┌────▼─────┐
                   │  LM Head │
                   └──────────┘
```

### Composite Loss

```
L = L_LM  +  λ_r · L_routing  +  λ_e · L_entropy
```

| Term | Purpose |
|:---|:---|
| `L_LM` | Cross-entropy next-token prediction |
| `L_routing` | Rewards high write gate when memory diverges from the no-memory baseline (KL-weighted gate loss) |
| `L_entropy` | Penalises diffuse write distributions — encourages sparse, focused writes |

---

## Installation

```bash
git clone https://github.com/your-username/dnc-gpt2.git
cd dnc-gpt2

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets tqdm matplotlib
```

> **Note:** PyTorch ≥ 2.0 is required for optional `torch.compile` support. CUDA is strongly recommended.

---

## Quickstart

```bash
python dnc_gpt2.py
```

This will:
1. Download WikiText-2 automatically via 🤗 Datasets
2. Train GPT-2 Baseline → then GPT-2 + DNC for 1 epoch each
3. Save checkpoints to `./checkpoints/`
4. Save a training curve plot to `./plots/training_curves.png`
5. Print a final comparison table and per-token memory write inspection

---

## Configuration

All hyperparameters are defined in the `Config` class at the top of `dnc_gpt2.py`.

### Model

| Parameter | Default | Description |
|:---|:---:|:---|
| `hidden_size` | `768` | GPT-2 embedding / hidden dimension |
| `num_layers` | `6` | Number of transformer layers |
| `num_heads` | `8` | Number of attention heads |
| `mem_slots` | `64` | Memory rows — N in the N × W matrix |
| `mem_width` | `128` | Memory column width — W |
| `num_reads` | `4` | Number of DNC read heads — R |

### Training

| Parameter | Default | Description |
|:---|:---:|:---|
| `epochs` | `1` | Training epochs (used when `max_steps=0`) |
| `max_steps` | `0` | If > 0, overrides epoch-based stopping |
| `batch_size` | `4` | Sequences per batch |
| `seq_len` | `128` | Token sequence length |
| `lr` | `3e-4` | AdamW learning rate |
| `grad_clip` | `1.0` | Gradient norm clipping threshold |
| `lambda_routing` | `0.1` | Routing loss weight λ_r |
| `lambda_entropy` | `0.05` | Entropy loss weight λ_e |

### Logging & Checkpointing

| Parameter | Default | Description |
|:---|:---:|:---|
| `log_every_steps` | `50` | Steps between train metric recordings |
| `save_every_steps` | `3000` | Steps between validation + checkpoint saves |
| `save_dir` | `checkpoints/` | Checkpoint output directory |
| `plot_dir` | `plots/` | Plot output directory |

### Hardware

| Parameter | Default | Description |
|:---|:---:|:---|
| `use_compile` | `False` | Enable `torch.compile` — disable with DataParallel |
| `compile_mode` | `reduce-overhead` | Mode passed to `torch.compile` |
| `num_gpus` | auto | Uses up to 2 GPUs via `nn.DataParallel` |

> ⚠️ `torch.compile` and `nn.DataParallel` are mutually exclusive. Set `num_gpus=1` to enable compilation.

---

## Outputs

### Checkpoints

```
checkpoints/
├── GPT2_Baseline_step3000.pt
├── GPT2_+_DNC_step3000.pt
├── GPT2_Baseline_FINAL_step3621.pt
└── GPT2_+_DNC_FINAL_step3621.pt
```

Each `.pt` file contains `model_state_dict`, `optimizer_state_dict`, `global_step`, `epoch`, and `val_ppl`.

### Training Curves

A dark-themed side-by-side plot of CE loss and top-1 accuracy vs. step, overlaying both models with train (thin line) and validation checkpoint (dashed + markers) series.

```
plots/
└── training_curves.png
```

### Comparison Table

Printed to stdout after training completes:

```
══════════════════════════════════════════════════════════════════
  FINAL COMPARISON — Baseline GPT2 vs GPT2 + DNC
══════════════════════════════════════════════════════════════════
  Metric                   GPT2 Baseline       GPT2 + DNC   Δ
  ──────────────────────────────────────────────────────────────
  Perplexity ↓                    45.21            41.88   ▼7.4% ✓
  Top-1 Accuracy ↑               0.3412           0.3601   ▲5.5% ✓
  Avg Write Gate                     —             0.512
  Write Sparsity ↑                   —             0.731
  Memory KL ↑                        —             0.284
```

### Memory Write Inspection

Per-token write gate values printed as inline bar charts reveal which tokens the model commits to external memory:

```
  Token                  Gate  bar
  ──────────────────────────────────────────────────────────
  Albert                 0.821  ████████████████████████
  Einstein               0.143  ████
  was                    0.031
  born                   0.612  ██████████████████
  in                     0.028
  1879                   0.897  ██████████████████████████
```

---

## Metrics Reference

| Metric | DNC only | Description |
|:---|:---:|:---|
| `loss` | | Cross-entropy loss |
| `ppl` | | Perplexity |
| `bpt` | | Bits per token |
| `top1_acc` | | Top-1 next-token accuracy |
| `top5_acc` | | Top-5 next-token accuracy |
| `confidence` | | Mean probability assigned to the correct token |
| `mean_rank` | | Mean rank of the correct token in logit distribution |
| `pred_entropy` | | Entropy of the predicted distribution |
| `avg_gate` | ✓ | Mean write gate value across all tokens |
| `gate_std` | ✓ | Std of write gate — higher = more polarised |
| `write_rate` | ✓ | Fraction of tokens with gate > 0.7 |
| `write_sparsity` | ✓ | Normalised write attention entropy — higher = more focused |
| `mem_kl` | ✓ | KL divergence between memory-augmented and baseline logits |

---

## Project Structure

```
dnc-gpt2/
├── dnc_gpt2.py          # Full implementation — single file
├── checkpoints/         # Saved model checkpoints (auto-created)
├── plots/               # Training curve plots (auto-created)
└── README.md
```

---

## References

```bibtex
@article{graves2016hybrid,
  title   = {Hybrid computing using a neural network with dynamic external memory},
  author  = {Graves, Alex and Wayne, Greg and Reynolds, Malcolm and others},
  journal = {Nature},
  volume  = {538},
  pages   = {471--476},
  year    = {2016}
}

@article{radford2019language,
  title  = {Language Models are Unsupervised Multitask Learners},
  author = {Radford, Alec and Wu, Jeffrey and Child, Rewon and others},
  year   = {2019}
}

@article{merity2016pointer,
  title  = {Pointer Sentinel Mixture Models},
  author = {Merity, Stephen and Xiong, Caiming and Bradbury, James and Socher, Richard},
  year   = {2016}
}
```

---

<div align="center">
<sub>Built with ❤️ using PyTorch · HuggingFace Transformers · WikiText-2</sub>
</div>
