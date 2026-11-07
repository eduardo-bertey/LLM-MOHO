#!/usr/bin/env python3
"""Destilar FF capa 15 Qwen2.5-0.5B -> MoE, BLOQUE A BLOQUE, Wikipedia ES sin repetir.

Uso en Colab:
    !git clone --depth 1 -b fix/ff-eval-real-activations https://github.com/eduardo-bertey/LLM-MOHO.git
    %cd LLM-MOHO/qwen-ff-local
    !python3 train_15_ff.py

Regla: 1 bloque >= 500 k tokens (489 seq x 1024 = 500 736 tokens). El script:
    1) carga el bloque de Wikipedia ES en streaming (sin repetir articulo),
    2) entrena la MoE sobre ese bloque,
    3) LIBERA la RAM/GPU del bloque anterior antes de cargar el siguiente,
    4) imprime cos/R2/NMSE en eval fijo y guarda (sobrescribe) moe_layer15.pt.
"""
import json
import math
import os
import subprocess
import sys
import time

LAYER = int(os.environ.get("LAYER", "15"))
SEQ = int(os.environ.get("SEQ", "1024"))
BS_FWD = int(os.environ.get("BS_FWD", "8"))         # secuencias por forward del modelo
LR = float(os.environ.get("LR", "3e-3"))
SEED = int(os.environ.get("SEED", "123"))
BLOCK_SEQS = int(os.environ.get("BLOCK_SEQS", "489"))  # >= 500 k tokens por bloque (489 x 1024 = 500 736)
BATCH = int(os.environ.get("BATCH", "64"))
W_LB = float(os.environ.get("W_LB", "0.01"))
N_EXPERTS, TOP_K = 4, 2
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_DIR = os.environ.get("MODEL_DIR", "./modelo-qwen2.5-0.5b")
CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_layer15.pt")

LOG = open("train_15_ff.log", "a", buffering=1)


def log(m):
    print(m, flush=True)
    LOG.write(m + "\n")


log("=" * 60)
log(f"MoE FF{LAYER} Qwen2.5-0.5B | 4 expertos top-2 | BLOCK_SEQS {BLOCK_SEQS} | lr {LR} batch {BATCH}")

import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"torch {torch.__version__} | device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
torch.manual_seed(SEED)

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import datasets
except ImportError:
    log("instalando datasets...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    import datasets

if os.path.isdir(MODEL_DIR) and os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
    log(f"Modelo ya en {MODEL_DIR}")
    src = MODEL_DIR
else:
    log(f"Descargando {MODEL_ID}...")
    src = MODEL_ID

tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32).to(device)
model.eval()
cfg = model.config
HIDDEN, INTER, N_LAYERS = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
INTER_E = INTER // N_EXPERTS
log(f"hidden {HIDDEN} | inter {INTER} | MoE {N_EXPERTS}x{INTER_E} | params {N_EXPERTS*3*INTER_E*HIDDEN:,}")
m_hook = model.model.layers[LAYER].mlp
for p in m_hook.parameters():
    p.requires_grad_(False)


def stream_wikipedia(BLOCK_SEQS, SEQ):
    """Devuelve bloques de BLOCK_SEQS secuencias (cada una SEQ tokens).
    Itera wiki ES dos snapshots sin repetir articulo."""
    total = 0
    bloque = 0
    for config in ["20231101.es", "20240501.es"]:
        ds = datasets.load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
        blk = []
        for art in ds:
            txt = (art.get("text") or "").strip()
            if len(txt) < 2000:
                continue
            ids = tok(txt, add_special_tokens=False)["input_ids"]
            i = 0
            while i + SEQ <= len(ids):
                blk.append(ids[i:i + SEQ])
                total += 1
                i += SEQ
                if len(blk) >= BLOCK_SEQS:
                    yield blk, bloque
                    blk = []
                    bloque += 1
        if blk:
            yield blk, bloque
            bloque += 1
    log(f"stream completo: bloques {bloque} | secuencias {total}")


def init_origen(shape, fan_in):
    w = torch.empty(shape)
    torch.nn.init.normal_(w, mean=0.0, std=1.0 / math.sqrt(fan_in))
    return torch.nn.Parameter(w)


class MoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.ModuleList()
        for _ in range(N_EXPERTS):
            e = torch.nn.Module()
            e.gate = init_origen((INTER_E, HIDDEN), HIDDEN)
            e.up = init_origen((INTER_E, HIDDEN), HIDDEN)
            e.down = init_origen((HIDDEN, INTER_E), INTER_E)
            self.experts.append(e)
        self.router_w = torch.nn.Parameter(torch.randn(N_EXPERTS, HIDDEN) * 0.01)
        self.router_b = torch.nn.Parameter(torch.zeros(N_EXPERTS))

    def forward(self, x):
        orig = x.shape
        x = x.reshape(-1, x.shape[-1])
        logits = x @ self.router_w.T + self.router_b
        probs = torch.softmax(logits, dim=-1)
        wt, idx = torch.topk(probs, TOP_K, dim=-1)
        wt = wt / wt.sum(dim=-1, keepdim=True)
        y = torch.zeros_like(x)
        for e in range(N_EXPERTS):
            rows = (idx == e).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            xe = x[rows]
            h = F.silu(xe @ self.experts[e].gate.T) * (xe @ self.experts[e].up.T)
            cols = (idx[rows] == e).nonzero(as_tuple=True)[1]
            y[rows] += (h @ self.experts[e].down.T) * wt[rows, cols][:, None]
        f = torch.bincount(idx.reshape(-1), minlength=N_EXPERTS).float() / idx.numel()
        lb = N_EXPERTS * (f * probs.mean(0)).sum()
        return y.reshape(orig), lb, f


moe = MoE().to(device)
opt = torch.optim.AdamW(moe.parameters(), lr=LR, weight_decay=0.01)

X_eval = None
Y_eval = None
bloque_idx = 0
step = 0
vec_total = 0
t0 = time.time()

for bloque, bloque_idx in stream_wikipedia(BLOCK_SEQS, SEQ):
    # forward en sub-batches, NO ids_t entero (OOM en T4)
    cap = []

    def hook(module, inp, out):
        cap.append((inp[0].detach().reshape(-1, HIDDEN).cpu().half(),
                    out.detach().reshape(-1, HIDDEN).cpu().half()))

    handle = m_hook.register_forward_hook(hook)
    t_start = time.time()
    for j in range(0, len(bloque), BS_FWD):
        ids_t = torch.tensor(bloque[j:j + BS_FWD], device=device)
        with torch.no_grad():
            model(input_ids=ids_t)
        v = (j // BS_FWD + 1) * BS_FWD * SEQ
        print(f"[capture] bloque {bloque_idx} sub {j//BS_FWD + 1}/{(len(bloque)+BS_FWD-1)//BS_FWD} vec {v}", flush=True)
        del ids_t
    handle.remove()
    n_cap = sum(c[0].shape[0] for c in cap)
    print(f"[capture] bloque {bloque_idx} terminado en {time.time() - t_start:.1f}s, vectores {n_cap}", flush=True)


    X = torch.cat([h[0].float() for h in cap], dim=0)
    Y = torch.cat([o[0].float() for o in cap], dim=0)
    del cap
    n = X.shape[0]
    vec_total += n

    # eval fijo (primer bloque solo): 512 vectores que nunca se entrenan
    if X_eval is None:
        X_eval, Y_eval = X[:512].clone(), Y[:512].clone()

    # entrenar sobre ese bloque sin retener memoria (batches barajados)
    perm = torch.randperm(n)                     # cpu
    for j in range(0, n, BATCH):
        idx_cpu = perm[j:j + BATCH]             # indexar en CPU (X vive en cpu)
        if len(idx_cpu) == 0:
            break
        xb = X[idx_cpu].to(device).contiguous()
        yb = Y[idx_cpu].to(device).contiguous()
        m_out, lb, f = moe(xb)
        cos_b = F.cosine_similarity(m_out, yb, dim=-1).mean()
        loss = (m_out - yb).pow(2).mean() / (yb ** 2).mean() + 0.1 * (1 - cos_b) + W_LB * lb
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        del xb, yb, idx_cpu
    # liberar RAM del bloque
    del X, Y
    if device == "cuda":
        torch.cuda.empty_cache()

    cos = r2 = nmse = 0.0
    if X_eval is not None:
        with torch.no_grad():
            ye, _ = moe(X_eval.to(device))
            yt = Y_eval.to(device)
            cos = F.cosine_similarity(ye, yt, dim=-1).mean().item()
            r2 = 1 - ((ye - yt) ** 2).sum().item() / max(((yt - yt.mean(0)) ** 2).sum().item(), 1e-12)
            nmse = (((ye - yt) ** 2).mean() / (yt ** 2).mean()).item()
    # guardar SOBRESCRIBIENDO
    sd = {k: v.detach().cpu() for k, v in moe.state_dict().items()}
    sd.update({"layer": LAYER, "n_experts": N_EXPERTS, "top_k": TOP_K, "inter_e": INTER_E,
               "hidden": HIDDEN, "steps_seen": step, "lr": LR, "blocks": bloque_idx,
               "vec_total": vec_total, "cos": cos, "r2": r2})
    with open(CHECKPOINT, "wb") as f:
        torch.save(sd, f)
    log(f"[bloque {bloque_idx:3d}] steps {step} | vec {vec_total} | cos {cos:.4f} R2 {r2:.4f} | [save] {CHECKPOINT}")

del model, tok, m_hook
log(f"[time] {time.time() - t0:.1f}s | bloques {bloque_idx + 1} | pasos {step} | vectores {vec_total}")
log("OK: moe_layer15.pt sobrescrito.")
