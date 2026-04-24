"""
Differentiable Neural Computer (DNC) + GPT2 Training
WITH torch.compile optimization - FIXED for DataParallel
"""

import math
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Config, GPT2Model

# Check for torch.compile availability
TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile')
if TORCH_COMPILE_AVAILABLE:
    print(f"✅ torch.compile available (PyTorch {torch.__version__})")
else:
    print(f"⚠️ torch.compile not available (need PyTorch 2.0+)")

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

class Config:
    # ── model ─────────────────────────────────────────────────────
    model_name  = "gpt2"
    hidden_size = 768
    num_layers  = 6
    num_heads   = 8

    # ── DNC memory ────────────────────────────────────────────────
    mem_slots   = 64
    mem_width   = 128
    num_reads   = 4

    # ── training mode ─────────────────────────────────────────────
    epochs      = 1
    max_steps   = 0

    # ── training hyperparams ──────────────────────────────────────
    batch_size  = 4
    seq_len     = 128
    lr          = 3e-4
    grad_clip   = 1.0

    # ── loss weights ──────────────────────────────────────────────
    lambda_routing = 0.1
    lambda_entropy = 0.05

    # ── checkpointing ─────────────────────────────────────────────
    save_dir         = "checkpoints"
    save_every_steps = 2000

    # ── compilation ───────────────────────────────────────────────
    use_compile = False          # DISABLED by default for DataParallel compatibility
    compile_mode = "reduce-overhead"
    
    # ── hardware ──────────────────────────────────────────────────
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = min(2, torch.cuda.device_count())


cfg = Config()


# ══════════════════════════════════════════════════════════════════
# DNC MEMORY MODULE
# ══════════════════════════════════════════════════════════════════

class DNCMemory(nn.Module):
    def __init__(self, mem_slots, mem_width, num_reads, controller_size):
        super().__init__()
        self.N = mem_slots
        self.W = mem_width
        self.R = num_reads

        self.write_key_proj  = nn.Linear(controller_size, mem_width)
        self.write_vec_proj  = nn.Linear(controller_size, mem_width)
        self.erase_vec_proj  = nn.Linear(controller_size, mem_width)
        self.write_gate_proj = nn.Linear(controller_size, 1)
        self.read_key_proj   = nn.Linear(controller_size, mem_width * num_reads)
        self.temp            = nn.Parameter(torch.ones(1) * 2.0)

    def get_write_weighting(self, write_key, memory, usage):
        key_norm  = F.normalize(write_key.unsqueeze(1), dim=-1)
        mem_norm  = F.normalize(memory, dim=-1)
        sim       = (key_norm * mem_norm).sum(-1)
        w_content = F.softmax(sim * self.temp, dim=-1)
        w_alloc   = F.softmax((1.0 - usage) * self.temp, dim=-1)
        return 0.5 * w_content + 0.5 * w_alloc

    def get_read_weighting(self, read_keys, memory):
        key_norm = F.normalize(read_keys, dim=-1)
        mem_norm = F.normalize(memory, dim=-1)
        sim      = torch.bmm(key_norm, mem_norm.transpose(1, 2))
        return F.softmax(sim * self.temp, dim=-1)

    def write(self, memory, w_write, write_vec, erase_vec, write_gate):
        w          = w_write.unsqueeze(-1)
        e          = erase_vec.unsqueeze(1)
        v          = write_vec.unsqueeze(1)
        g          = write_gate.unsqueeze(-1)
        memory_new = memory * (1 - w * e) + w * v
        return memory + g * (memory_new - memory)

    def read(self, memory, w_read):
        r = torch.bmm(w_read, memory)
        return r.view(r.size(0), -1)

    def forward(self, hidden, memory, usage):
        write_key  = self.write_key_proj(hidden)
        write_vec  = self.write_vec_proj(hidden)
        erase_vec  = torch.sigmoid(self.erase_vec_proj(hidden))
        write_gate = torch.sigmoid(self.write_gate_proj(hidden))
        read_keys  = self.read_key_proj(hidden).view(-1, self.R, self.W)

        w_write    = self.get_write_weighting(write_key, memory, usage)
        w_read     = self.get_read_weighting(read_keys, memory)
        memory_new = self.write(memory, w_write, write_vec, erase_vec, write_gate)
        usage_new  = usage + (1 - usage) * w_write.detach()
        read_vec   = self.read(memory_new, w_read)

        return read_vec, memory_new, usage_new, write_gate, w_write


# ══════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════

def _gpt2_cfg(cfg, vocab_size):
    return GPT2Config(
        vocab_size  = vocab_size,
        n_embd      = cfg.hidden_size,
        n_layer     = cfg.num_layers,
        n_head      = cfg.num_heads,
        n_positions = cfg.seq_len,
        resid_pdrop = 0.1,
        attn_pdrop  = 0.1,
    )


class BaselineGPT2(nn.Module):
    """Vanilla GPT2, no memory."""
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.transformer = GPT2Model(_gpt2_cfg(cfg, vocab_size))
        self.lm_head     = nn.Linear(cfg.hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, **kwargs):
        h = self.transformer(input_ids).last_hidden_state
        return self.lm_head(h)


class DNCLLM(nn.Module):
    """GPT2 controller + DNC external memory."""
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg         = cfg
        self.transformer = GPT2Model(_gpt2_cfg(cfg, vocab_size))
        self.memory      = DNCMemory(
            cfg.mem_slots, cfg.mem_width, cfg.num_reads, cfg.hidden_size
        )
        self.read_proj   = nn.Linear(cfg.num_reads * cfg.mem_width, cfg.hidden_size)
        self.lm_head     = nn.Linear(cfg.hidden_size, vocab_size, bias=False)

    def init_memory(self, batch_size, device):
        memory = torch.zeros(batch_size, self.cfg.mem_slots, self.cfg.mem_width, device=device)
        usage  = torch.zeros(batch_size, self.cfg.mem_slots, device=device)
        return memory, usage

    def forward(self, input_ids, memory, usage):
        hidden_states = self.transformer(input_ids).last_hidden_state
        all_logits, all_gates, all_ww = [], [], []

        for t in range(input_ids.size(1)):
            h_t = hidden_states[:, t, :]
            read_vec, memory, usage, write_gate, w_write = self.memory(h_t, memory, usage)
            h_out = h_t + self.read_proj(read_vec)
            all_logits.append(self.lm_head(h_out))
            all_gates.append(write_gate)
            all_ww.append(w_write)

        logits      = torch.stack(all_logits, dim=1)
        write_gates = torch.stack(all_gates,  dim=1)
        w_writes    = torch.stack(all_ww,     dim=1)
        return logits, memory, usage, write_gates, w_writes

    def forward_no_memory(self, input_ids):
        h = self.transformer(input_ids).last_hidden_state
        return self.lm_head(h)


# ══════════════════════════════════════════════════════════════════
# COMPILATION WRAPPER - FIXED for DataParallel
# ══════════════════════════════════════════════════════════════════

def maybe_compile(model, model_name):
    """Apply torch.compile if available and configured.
    IMPORTANT: Only compile if NOT using DataParallel, or compile each replica separately."""
    if TORCH_COMPILE_AVAILABLE and cfg.use_compile:
        # Don't compile if we're going to wrap with DataParallel
        if cfg.num_gpus > 1:
            print(f"  ⚠️ torch.compile disabled for {model_name} when using DataParallel")
            print(f"     (compile + DataParallel has compatibility issues)")
            return model
        
        print(f"  🔧 Compiling {model_name} with mode='{cfg.compile_mode}'...")
        try:
            # Compile the full model
            compiled_model = torch.compile(model, mode=cfg.compile_mode, fullgraph=False)
            print(f"  ✅ {model_name} compiled successfully")
            return compiled_model
        except Exception as e:
            print(f"  ⚠️ Compilation failed: {e}, using uncompiled model")
            return model
    return model


# ══════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def lm_loss(logits, targets):
    shift_logits  = logits[:, :-1].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        ignore_index=-100,
    )


def routing_loss(logits_mem, logits_no_mem, write_gates):
    B, T, V  = logits_mem.shape
    p_mem    = logits_mem[:, :-1].reshape(-1, V)
    p_no_mem = logits_no_mem[:, :-1].reshape(-1, V)
    gates    = write_gates[:, :-1].reshape(-1)
    kl       = F.kl_div(
        F.log_softmax(p_no_mem, dim=-1),
        F.softmax(p_mem, dim=-1),
        reduction='none',
    ).sum(-1).detach()
    return -(gates * kl).mean()


def entropy_loss(w_writes):
    eps = 1e-8
    return -(w_writes * (w_writes + eps).log()).sum(-1).mean()


# ══════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════

class MetricTracker:
    def __init__(self):
        self.data = defaultdict(list)

    def update(self, d):
        for k, v in d.items():
            self.data[k].append(v)

    def avg(self, k):
        v = self.data[k]
        return sum(v) / len(v) if v else 0.0

    def all_avgs(self):
        return {k: self.avg(k) for k in self.data}


def compute_metrics(logits, input_ids,
                    write_gates=None, w_writes=None, logits_no_mem=None):
    B, T, V      = logits.shape
    metrics      = {}
    flat_pred    = logits[:, :-1].reshape(-1, V)
    flat_targets = input_ids[:, 1:].reshape(-1)
    probs        = F.softmax(flat_pred, dim=-1)

    ce                  = F.cross_entropy(flat_pred, flat_targets)
    metrics["loss"]     = ce.item()
    metrics["ppl"]      = math.exp(ce.item())
    metrics["bpt"]      = ce.item() / math.log(2)
    metrics["top1_acc"] = (flat_pred.argmax(-1) == flat_targets).float().mean().item()
    top5                = flat_pred.topk(5, dim=-1).indices
    metrics["top5_acc"] = (top5 == flat_targets.unsqueeze(1)).any(1).float().mean().item()

    correct_probs         = probs[torch.arange(len(flat_targets)), flat_targets]
    metrics["confidence"] = correct_probs.mean().item()
    correct_logit         = flat_pred[torch.arange(len(flat_targets)), flat_targets]
    ranks                 = (flat_pred > correct_logit.unsqueeze(1)).sum(1).float()
    metrics["mean_rank"]  = ranks.mean().item()
    metrics["pred_entropy"] = -(probs * (probs + 1e-8).log()).sum(-1).mean().item()

    if write_gates is not None:
        gates                 = write_gates[:, :-1].reshape(-1)
        metrics["avg_gate"]   = gates.mean().item()
        metrics["gate_std"]   = gates.std().item()
        metrics["write_rate"] = (gates > 0.7).float().mean().item()

    if w_writes is not None:
        w   = w_writes[:, :-1]
        eps = 1e-8
        H   = -(w * (w + eps).log()).sum(-1).mean()
        metrics["write_sparsity"] = (1.0 - (H / math.log(w.size(-1))).clamp(0, 1)).item()

    if logits_no_mem is not None:
        p_m  = F.softmax(logits[:, :-1].reshape(-1, V), dim=-1)
        p_nm = F.log_softmax(logits_no_mem[:, :-1].reshape(-1, V), dim=-1)
        metrics["mem_kl"] = F.kl_div(p_nm, p_m, reduction='batchmean').item()

    return metrics


# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════

def get_dataloader(cfg, tokenizer, split="train"):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation     = True,
            max_length     = cfg.seq_len,
            padding        = "max_length",
            return_tensors = None,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids"])
    return DataLoader(tokenized, batch_size=cfg.batch_size,
                      shuffle=(split == "train"), drop_last=True)


# ══════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════

def save_checkpoint(raw_model, optimizer, global_step, epoch, val_ppl, tag, cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)
    path = os.path.join(cfg.save_dir, f"{tag}_step{global_step}.pt")
    torch.save({
        "global_step": global_step,
        "epoch"      : epoch,
        "val_ppl"    : val_ppl,
        "model_state": raw_model.state_dict(),
        "opt_state"  : optimizer.state_dict(),
    }, path)
    tqdm.write(f"  💾  Saved → {path}  (val_ppl={val_ppl:.2f})")
    return path


# ══════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, raw_model, val_loader, cfg, is_dnc, desc="val"):
    model.eval()
    tracker = MetricTracker()
    bar     = tqdm(val_loader, desc=f"  [{desc}]", dynamic_ncols=True, leave=False)

    for batch in bar:
        input_ids = batch["input_ids"].to(cfg.device)
        B         = input_ids.size(0)

        if is_dnc:
            memory, usage = raw_model.init_memory(B, cfg.device)
            logits, _, _, write_gates, w_writes = model(input_ids, memory, usage)
            logits_no_mem = raw_model.forward_no_memory(input_ids)
            m = compute_metrics(logits, input_ids,
                                write_gates=write_gates,
                                w_writes=w_writes,
                                logits_no_mem=logits_no_mem)
        else:
            logits = model(input_ids)
            m      = compute_metrics(logits, input_ids)

        tracker.update(m)
        bar.set_postfix(ppl=f"{m['ppl']:.2f}", top1=f"{m['top1_acc']:.3f}")

    return tracker.all_avgs()


# ══════════════════════════════════════════════════════════════════
# INFINITE EPOCH GENERATOR
# ══════════════════════════════════════════════════════════════════

def _infinite_epochs():
    i = 0
    while True:
        yield i
        i += 1


# ══════════════════════════════════════════════════════════════════
# CORE TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def train_model(model, raw_model, train_loader, val_loader, cfg, is_dnc, tag):
    optimizer   = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    use_steps   = cfg.max_steps > 0
    max_steps   = cfg.max_steps if use_steps else cfg.epochs * len(train_loader)
    global_step = 0
    epoch       = 0
    val_metrics = {}
    last_saved_step = 0

    mode_str = f"max_steps={cfg.max_steps}" if use_steps else f"epochs={cfg.epochs}"
    tqdm.write(f"\n  Mode: {mode_str}  |  steps/epoch: {len(train_loader)}"
               f"  |  save every {cfg.save_every_steps} steps")

    epoch_iter = range(cfg.epochs) if not use_steps else _infinite_epochs()

    for epoch in epoch_iter:
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        train_bar = tqdm(
            train_loader,
            desc          = f"[{tag}] epoch {epoch+1}",
            dynamic_ncols = True,
            leave         = True,
        )

        for batch in train_bar:
            if global_step >= max_steps:
                train_bar.close()
                break

            input_ids = batch["input_ids"].to(cfg.device)
            B         = input_ids.size(0)

            if is_dnc:
                memory, usage = raw_model.init_memory(B, cfg.device)
                logits, _, _, write_gates, w_writes = model(input_ids, memory, usage)
                with torch.no_grad():
                    logits_no_mem = raw_model.forward_no_memory(input_ids)
                L_lm  = lm_loss(logits, input_ids)
                L_rt  = routing_loss(logits, logits_no_mem, write_gates)
                L_ent = entropy_loss(w_writes)
                loss  = L_lm + cfg.lambda_routing * L_rt + cfg.lambda_entropy * L_ent
                train_bar.set_postfix(
                    step = global_step,
                    loss = f"{loss.item():.3f}",
                    lm   = f"{L_lm.item():.3f}",
                    rt   = f"{L_rt.item():.4f}",
                    ent  = f"{L_ent.item():.3f}",
                    gate = f"{write_gates.mean().item():.3f}",
                )
            else:
                logits = model(input_ids)
                loss   = lm_loss(logits, input_ids)
                train_bar.set_postfix(step=global_step, loss=f"{loss.item():.3f}")

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_loss  += loss.item()
            n_batches   += 1
            global_step += 1

            if global_step % cfg.save_every_steps == 0 and global_step != last_saved_step:
                last_saved_step = global_step
                val_metrics = validate(model, raw_model, val_loader, cfg, is_dnc,
                                       f"{tag} step {global_step}")
                tqdm.write(
                    f"\n  [{tag}] step {global_step} | "
                    f"val_ppl={val_metrics['ppl']:.2f} | "
                    f"top1={val_metrics['top1_acc']:.4f} | "
                    f"top5={val_metrics['top5_acc']:.4f}"
                )
                save_checkpoint(raw_model, optimizer, global_step, epoch,
                                val_metrics["ppl"], tag, cfg)
                model.train()

        if n_batches > 0:
            tqdm.write(
                f"\n  [{tag}] epoch {epoch+1} done | "
                f"avg_loss={epoch_loss/n_batches:.4f} | "
                f"train_ppl={math.exp(epoch_loss/n_batches):.2f} | "
                f"global_step={global_step}"
            )

        if global_step >= max_steps:
            break

    if last_saved_step != global_step:
        tqdm.write(f"\n  [{tag}] Final validation...")
        val_metrics = validate(model, raw_model, val_loader, cfg, is_dnc, f"{tag} final")
        tqdm.write(
            f"  [{tag}] FINAL | "
            f"ppl={val_metrics['ppl']:.2f} | "
            f"top1={val_metrics['top1_acc']:.4f} | "
            f"top5={val_metrics['top5_acc']:.4f} | "
            f"bpt={val_metrics['bpt']:.4f}"
        )
        save_checkpoint(raw_model, optimizer, global_step, epoch,
                        val_metrics["ppl"], f"{tag}_FINAL", cfg)

    val_metrics["params"] = sum(p.numel() for p in raw_model.parameters())
    return val_metrics


# ══════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════

def print_comparison(all_results):
    names      = list(all_results.keys())
    col_w      = 20
    name_w     = 24
    W          = name_w + col_w * len(names) + col_w + 4

    metrics_cfg = [
        ("ppl",            "Perplexity ↓",        ".2f",  False),
        ("bpt",            "Bits/Token ↓",         ".4f",  False),
        ("loss",           "CE Loss ↓",            ".4f",  False),
        ("top1_acc",       "Top-1 Accuracy ↑",     ".4f",  True ),
        ("top5_acc",       "Top-5 Accuracy ↑",     ".4f",  True ),
        ("confidence",     "Confidence ↑",         ".4f",  True ),
        ("mean_rank",      "Mean Rank ↓",          ".1f",  False),
        ("pred_entropy",   "Pred Entropy",         ".4f",  None ),
        ("avg_gate",       "Avg Write Gate",       ".4f",  None ),
        ("gate_std",       "Gate Polarization ↑",  ".4f",  True ),
        ("write_rate",     "Write Rate >0.7",      ".4f",  None ),
        ("write_sparsity", "Write Sparsity ↑",     ".4f",  True ),
        ("mem_kl",         "Memory KL ↑",          ".4f",  True ),
    ]

    print("\n" + "═" * W)
    print("  FINAL COMPARISON — Baseline GPT2 vs GPT2 + DNC")
    print("═" * W)
    print(f"  {'Metric':<{name_w}}", end="")
    for n in names:
        print(f"{n:>{col_w}}", end="")
    print(f"  {'Δ (DNC vs Base)':>{col_w}}")
    print("  " + "─" * (W - 2))

    for key, label, fmt, hib in metrics_cfg:
        vals = {n: all_results[n].get(key, None) for n in names}
        if all(v is None for v in vals.values()):
            continue
        print(f"  {label:<{name_w}}", end="")
        for n in names:
            v = vals[n]
            s = format(v, fmt) if v is not None else "—"
            print(f"{s:>{col_w}}", end="")

        b = vals[names[0]]
        l = vals[names[-1]]
        if b is not None and l is not None and b != 0:
            delta     = l - b
            delta_pct = 100 * delta / abs(b)
            arrow     = "▲" if delta > 0 else "▼"
            if   hib is True:  good = delta > 0
            elif hib is False: good = delta < 0
            else:              good = None
            badge = " ✓" if good else (" ✗" if good is False else "")
            print(f"  {arrow}{abs(delta_pct):.1f}%{badge}", end="")
        print()

    print("  " + "─" * (W - 2))
    print(f"  {'Parameters':<{name_w}}", end="")
    for n in names:
        p = all_results[n].get("params", 0)
        print(f"{f'{p/1e6:.1f}M':>{col_w}}", end="")
    print()
    print("═" * W + "\n")


# ══════════════════════════════════════════════════════════════════
# INSPECT MEMORY WRITES
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def inspect_writes(model, tokenizer, text, cfg):
    model.eval()
    input_ids     = tokenizer(text, return_tensors="pt")["input_ids"].to(cfg.device)
    memory, usage = model.init_memory(1, cfg.device)
    _, _, _, write_gates, _ = model(input_ids, memory, usage)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    gates  = write_gates[0, :, 0].cpu().tolist()

    print(f"\n  {'Token':<22} {'Gate':>6}  bar")
    print("  " + "─" * 60)
    for tok, gate in zip(tokens, gates):
        bar = "█" * int(gate * 30)
        print(f"  {tok:<22} {gate:>6.3f}  {bar}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run():
    mode_str = f"max_steps={cfg.max_steps}" if cfg.max_steps > 0 else f"epochs={cfg.epochs}"
    compile_str = f" | torch.compile: {cfg.compile_mode}" if (TORCH_COMPILE_AVAILABLE and cfg.use_compile and cfg.num_gpus <= 1) else ""
    
    print(f"\n{'═'*55}")
    print(f"  Device       : {cfg.device}")
    print(f"  GPUs         : {torch.cuda.device_count()} available, using {cfg.num_gpus}")
    print(f"  Training mode: {mode_str}")
    print(f"  Compilation  : {compile_str if compile_str else 'disabled'}")
    print(f"  Checkpoint   : every {cfg.save_every_steps} steps → ./{cfg.save_dir}/")
    print(f"{'═'*55}\n")

    tokenizer  = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    train_loader = get_dataloader(cfg, tokenizer, "train")
    val_loader   = get_dataloader(cfg, tokenizer, "validation")

    all_results = {}

    for label, is_dnc in [("GPT2 Baseline", False), ("GPT2 + DNC", True)]:
        print(f"\n{'═'*55}")
        print(f"  TRAINING: {label}")
        print(f"{'═'*55}")

        model = (DNCLLM(cfg, vocab_size) if is_dnc else BaselineGPT2(cfg, vocab_size))
        model = model.to(cfg.device)
        
        # Apply compilation BEFORE DataParallel
        model = maybe_compile(model, label)

        if cfg.num_gpus > 1:
            model = nn.DataParallel(model, device_ids=list(range(cfg.num_gpus)))
            print(f"  DataParallel on GPUs: {list(range(cfg.num_gpus))}")

        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        n_params  = sum(p.numel() for p in raw_model.parameters())
        print(f"  Parameters: {n_params/1e6:.1f}M\n")

        results              = train_model(model, raw_model, train_loader, val_loader,
                                           cfg, is_dnc, label.replace(" ", "_"))
        all_results[label]   = results

    print_comparison(all_results)

    print("  Memory write inspection (DNC):\n")
    dnc_ckpt_path = None
    if os.path.exists(cfg.save_dir):
        for f in sorted(os.listdir(cfg.save_dir)):
            if "GPT2_+_DNC_FINAL" in f:
                dnc_ckpt_path = os.path.join(cfg.save_dir, f)
                break

    if dnc_ckpt_path:
        dnc_inspect = DNCLLM(cfg, vocab_size).to(cfg.device)
        ckpt        = torch.load(dnc_ckpt_path, map_location=cfg.device)
        dnc_inspect.load_state_dict(ckpt["model_state"])
        # Don't compile for inspection to avoid issues
        inspect_writes(
            dnc_inspect, tokenizer,
            "Albert Einstein was born in 1879 in Ulm. "
            "He developed the theory of relativity.",
            cfg,
        )
    else:
        print("  No DNC checkpoint found for inspection.")

    return all_results


if __name__ == "__main__":
    # Set use_compile=False for DataParallel compatibility
    cfg.use_compile = False  # ← Change to True only if num_gpus=1
    results = run()
