#!/usr/bin/env python3
"""Destilar FF capa 15 Qwen2.5-0.5B -> MoE, BLOQUE A BLOQUE, Wikipedia ES sin repetir.

Mismo flujo que colab_destilar_moe.py (captura real, eval disjoint, BLD, micro-batches),
pero bloque a bloque:
    1 bloque = BLOCK_SEQS secuencias de 1024 tokens (>= 500k tokens).
    1) carga el bloque en streaming (sin repetir articulo),
    2) captura SOLO la entrada h de la FF (la salida t se recalcula con swiglu),
    3) entrena la MoE sobre ese bloque (shuffle sin reemplazo),
    4) LIBERA la RAM/GPU del bloque antes de cargar el siguiente,
    5) eval cos/R2/NMSE en set disjoint y guarda (sobrescribe) moe_layer15.pt.

Uso en Colab:
    !git clone --depth 1 -b fix/ff-eval-real-activations https://github.com/eduardo-bertey/LLM-MOHO.git
    %cd LLM-MOHO/qwen-ff-local
    !python3 train_15_ff.py
"""
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
BLOCK_SEQS = int(os.environ.get("BLOCK_SEQS", "489"))  # 489 x 1024 = 500 736 tokens por bloque
BATCH = int(os.environ.get("BATCH", "64"))             # vectores por paso
MICRO = int(os.environ.get("MICRO", "1024"))           # micro-batches dentro del paso
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "20"))   # eval cada N pasos
EPOCHS = int(os.environ.get("EPOCHS", "8"))            # pasadas sobre cada bloque
W_LB = float(os.environ.get("W_LB", "0.01"))
N_EXPERTS, TOP_K = 4, 2
TARGET_COS, TARGET_R2 = 0.95, 0.90
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_DIR = os.environ.get("MODEL_DIR", "./modelo-qwen2.5-0.5b")
CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_layer15.pt")

LOG = open("train_15_ff.log", "a", buffering=1)


def log(m):
    print(m, flush=True)
    LOG.write(m + "\n")


def fail(m):
    log(f"[ERROR] {m}")
    sys.exit(1)


log("=" * 60)
log(f"MoE FF{LAYER} Qwen2.5-0.5B | 4 expertos top-{TOP_K} | BLOCK_SEQS {BLOCK_SEQS} "
    f"({BLOCK_SEQS * SEQ} tokens) | epochs {EPOCHS} | lr {LR} batch {BATCH} micro {MICRO}")

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
log(f"hidden {HIDDEN} | inter {INTER} | capas {N_LAYERS} | MoE {N_EXPERTS}x{INTER_E} "
    f"| params {N_EXPERTS * 3 * INTER_E * HIDDEN:,} (profesora {3 * INTER * HIDDEN:,})")

teacher = model.model.layers[LAYER].mlp
for p in teacher.parameters():
    p.requires_grad_(False)


def stream_wikipedia(BLOCK_SEQS, SEQ):
    """Bloques de BLOCK_SEQS secuencias (cada una SEQ tokens), wiki ES dos snapshots
    sin repetir articulo."""
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


def swiglu(x, gate, up, down):
    return F.silu(x @ gate.T) * (x @ up.T) @ down.T


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
log(f"MoE params: {sum(p.numel() for p in moe.parameters()):,}")

g_w = teacher.gate_proj.weight.float()
u_w = teacher.up_proj.weight.float()
d_w = teacher.down_proj.weight.float()


def eval_moe():
    """cos/R2/NMSE en el set disjoint (targets recalculados con la profesora)."""
    with torch.no_grad():
        y_t_chunks, y_chunks = [], []
        for i in range(0, X_eval.shape[0], MICRO):
            xb = X_eval[i:i + MICRO].to(device)
            y_t_chunks.append(swiglu(xb, g_w, u_w, d_w))
            ye, _lb_e, _f_e = moe(xb)
            y_chunks.append(ye)
        yt = torch.cat(y_t_chunks)
        y = torch.cat(y_chunks)
        cos = F.cosine_similarity(y, yt, dim=-1).mean().item()
        r2 = 1 - ((y - yt) ** 2).sum().item() / max(((yt - yt.mean(0)) ** 2).sum().item(), 1e-12)
        nmse = (((y - yt) ** 2).mean() / (yt ** 2).mean()).item()
        probs = torch.softmax(y @ moe.router_w.T + moe.router_b, dim=-1)
        idx = torch.topk(probs, TOP_K, dim=-1).indices
        f = torch.bincount(idx.reshape(-1), minlength=N_EXPERTS).float() / idx.numel()
        lb = (N_EXPERTS * (f * probs.mean(0)).sum()).item()
    return cos, r2, nmse, lb


X_eval = None
bloque_idx = 0
step = 0
vec_total = 0
t0 = time.time()

for bloque, bloque_idx in stream_wikipedia(BLOCK_SEQS, SEQ):
    # --- captura: SOLO la entrada h de la FF (sub-batches, no 489 de golpe: OOM T4) ---
    cap = []

    def hook(module, inp, out):
        cap.append(inp[0].detach().reshape(-1, HIDDEN).cpu().half())

    handle = teacher.register_forward_hook(hook)
    t_start = time.time()
    n_sub = (len(bloque) + BS_FWD - 1) // BS_FWD
    for j in range(0, len(bloque), BS_FWD):
        ids_t = torch.tensor(bloque[j:j + BS_FWD], device=device)
        with torch.no_grad():
            model(input_ids=ids_t)
        sub = j // BS_FWD
        if sub % 10 == 0:
            log(f"[capture] bloque {bloque_idx} sub {sub + 1}/{n_sub} vec {(sub + 1) * BS_FWD * SEQ}")
        del ids_t
    handle.remove()
    X = torch.cat(cap, dim=0)
    del cap
    n = X.shape[0]
    vec_total += n
    log(f"[capture] bloque {bloque_idx} terminado en {time.time() - t_start:.1f}s, vectores {n}")

    # --- eval fijo disjoint: del PRIMER bloque, jamas se entrena ---
    if X_eval is None:
        n_eval = max(512, n // 5)
        X_eval = X[:n_eval].float().clone()
        X_train = X[n_eval:]
        log(f"eval disjoint {X_eval.shape[0]} vectores (nunca se entrena) | "
            f"train bloque 0: {X_train.shape[0]} vectores")
        cos0, r20, nmse0, lb0 = eval_moe()
        log(f"[eval inicial] cos {cos0:.4f} R2 {r20:.4f} NMSE {nmse0:.4f} lb {lb0:.4f}")
    else:
        X_train = X
    del X

    # --- entrenar ese bloque: EPOCHS pasadas, shuffle sin reemplazo por epoca ---
    n_train = X_train.shape[0]
    t_block = time.time()
    for ep in range(EPOCHS):
        perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(SEED + step + ep))
        for j in range(0, n_train, BATCH):
            idx_cpu = perm[j:j + BATCH]              # X_train vive en CPU
            if len(idx_cpu) == 0:
                break
            xb = X_train[idx_cpu].to(device).float()
            with torch.no_grad():
                yb = swiglu(xb, g_w, u_w, d_w)
            opt.zero_grad(set_to_none=True)
            loss_train = 0.0
            for i in range(0, xb.shape[0], MICRO):
                x_m, y_m = xb[i:i + MICRO], yb[i:i + MICRO]
                y_s, lb, _ = moe(x_m)
                nmse = ((y_s - y_m) ** 2).mean() / (y_m ** 2).mean()
                cos_b = F.cosine_similarity(y_s, y_m, dim=-1).mean()
                loss = (nmse + 0.1 * (1 - cos_b) + W_LB * lb) / max(1, (xb.shape[0] + MICRO - 1) // MICRO)
                loss.backward()
                loss_train += loss.item()
            opt.step()
            step += 1
            del xb, yb, idx_cpu
            if step % EVAL_EVERY == 0:
                cos, r2, nmse, lb = eval_moe()
                log(f"paso {step} ep {ep + 1}/{EPOCHS} loss {loss_train:.4f} | EVAL cos {cos:.4f} "
                    f"R2 {r2:.4f} NMSE {nmse:.4f} lb {lb:.4f} | bloque {bloque_idx} "
                    f"{time.time() - t_block:.0f}s")
        log(f"[bloque {bloque_idx} ep {ep + 1}/{EPOCHS}] steps {step} | {time.time() - t_block:.0f}s")

    # --- LIBERAR la RAM del bloque antes del siguiente ---
    del X_train
    if device == "cuda":
        torch.cuda.empty_cache()

    cos, r2, nmse, lb = eval_moe()
    sd = {k: v.detach().cpu() for k, v in moe.state_dict().items()}
    sd.update({"layer": LAYER, "n_experts": N_EXPERTS, "top_k": TOP_K, "inter_e": INTER_E,
               "hidden": HIDDEN, "steps_seen": step, "lr": LR, "blocks": bloque_idx,
               "vec_total": vec_total, "cos": cos, "r2": r2, "nmse": nmse})
    with open(CHECKPOINT, "wb") as f:
        torch.save(sd, f)
    log(f"[bloque {bloque_idx:3d}] steps {step} | vec {vec_total} | cos {cos:.4f} R2 {r2:.4f} "
        f"NMSE {nmse:.4f} | [save] {CHECKPOINT}")

cos, r2, nmse, lb = eval_moe()
ok = cos >= TARGET_COS and r2 > TARGET_R2
log(f"[eval final] cos {cos:.4f} R2 {r2:.4f} NMSE {nmse:.4f} lb {lb:.4f} | "
    f"meta cos>={TARGET_COS} y R2>{TARGET_R2}: {'SI' if ok else 'NO'}")
log(f"[time] {time.time() - t0:.1f}s | bloques {bloque_idx + 1} | pasos {step} | vectores {vec_total}")
log("OK: moe_layer15.pt sobrescrito.")
