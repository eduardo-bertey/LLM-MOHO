#!/usr/bin/env python3
"""MoE D2DMoE en 3 etapas, capa 4, Qwen2.5-0.5B, solo run en Colab.

Defaults sin argumento: LAYER=4, CLAMP=-4, 5 expertos, top-3, 100+100 steps.

Paper: Szatkowski et al., D2DMoE, arXiv:2310.04361,
codigo https://github.com/bartwojcik/D2DMoE

Las 3 etapas, cada una con su eval contra la FF maestra original:
  ETAPA 1 - SPARSITY en -10: se entrena la FF sola (copia de la profesora)
    con perdida BLD NMSE+0.1*(1-cos) para no romper la salida,
    + W_HOYER * Hoyer sobre la pre-activacion gate clampeada en -10
    (dsti_clamp_displacement = -10 en scripts/d2dmoe_gpt2_combined.py:224).
    Objetivo: que mas neuronas queden apagadas, sin perder cos/R2.
  ETAPA 2 - PARTE: KMeans sobre gate_proj de la FF ya escasa,
    4 expertos x 1216 neuronas, copiando los pesos de la FF escasa.
    Router por regresion, top-3 duro.
    Eval sin entrenar, para ver cuanto se pierde solo por partir.
  ETAPA 3 - FINE: se entrena la MoE completa (expertos + router)
    en top-3 duro con straight-through, perdida BLD + 0.01*lb,
    sobre entrada/salida de la FF maestra original.
    Eval final top-3 contra la profesora original.

Todo con vectores reales: se pasan tokens de Wikipedia ES por Qwen base
y se capturan X (entrada FF) e Y (salida profesora original).

Uso en Colab (una celda):
    !python3 colab_moe_d2dmoe.py

Ajustes (variables de entorno):
  Sin argumento: LAYER=4  CLAMP=-4  N_EXPERTS=5  TOPK=3  STEPS1=100  STEPS3=100
  BATCH=56  SEQ=1024  LR=5e-4  N_TOKENS_CAPTURA=500000  W_HOYER=0.1  SEED=123
"""
import json
import math
import os
import subprocess
import sys
import time

LAYER = int(os.environ.get("LAYER", "4"))
BATCH = int(os.environ.get("BATCH", "56"))
SEQ = int(os.environ.get("SEQ", "1024"))
STEPS1 = int(os.environ.get("STEPS1", "100"))
STEPS3 = int(os.environ.get("STEPS3", "100"))
LR = float(os.environ.get("LR", "5e-4"))
SEED = int(os.environ.get("SEED", "123"))
N_TOKENS_CAPTURA = int(os.environ.get("N_TOKENS_CAPTURA", "500000"))
CLAMP = float(os.environ.get("CLAMP", "-4"))
W_HOYER = float(os.environ.get("W_HOYER", "0.1"))
MICRO = int(os.environ.get("MICRO", "1024"))
W_LB = float(os.environ.get("W_LB", "0.01"))
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_DIR = os.environ.get("MODEL_DIR", "./modelo-qwen2.5-0.5b")
N_EXPERTS = int(os.environ.get("N_EXPERTS", "5"))
TOPK = int(os.environ.get("TOPK", "3"))
TARGET_COS, TARGET_R2 = 0.95, 0.90

LOG = open("colab_moe_d2dmoe.log", "a", buffering=1)


def log(m):
    print(m, flush=True)
    LOG.write(m + "\n")


def fail(m):
    log(f"[ERROR] {m}")
    sys.exit(1)


log("=" * 60)
log(f"D2DMoE 3 etapas FF{LAYER} Qwen2.5-0.5B | clamp {CLAMP}, W_HOYER {W_HOYER}")
log(f"etapa1 sparsity {STEPS1} steps | etapa2 parte KMeans | etapa3 fine MoE {STEPS3} steps")
log(f"batch {BATCH} seq {SEQ} lr {LR} captura {N_TOKENS_CAPTURA} tokens wiki")

import torch
import numpy as np

log(f"torch {torch.__version__} | cuda: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")
torch.manual_seed(SEED)
np.random.seed(SEED)

from transformers import AutoModelForCausalLM, AutoTokenizer

if os.path.isdir(MODEL_DIR) and os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
    src = MODEL_DIR
else:
    log(f"Modelo NO esta en {MODEL_DIR}, se descarga de {MODEL_ID} (~1 GB)...")
    src = MODEL_ID

tok = AutoTokenizer.from_pretrained(src, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32, trust_remote_code=False).to(device)
model.eval()
cfg = model.config
HIDDEN, INTER, N_LAYERS = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
INTER_E = INTER // N_EXPERTS
EXPERT_SIZES = [INTER // N_EXPERTS + (1 if i < INTER % N_EXPERTS else 0) for i in range(N_EXPERTS)]  # ej 5 exp: [973,973,973,973,972]
log(f"Modelo: hidden {HIDDEN}, inter {INTER}, capas {N_LAYERS}")

# --- Wikipedia ES, 500k tokens ---
TEXTO_PPL_WIKI = None


def cargar_wikipedia():
    try:
        import datasets
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
        import datasets
    ds = datasets.load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)
    seqs, total, n_art = [], 0, 0
    for art in ds:
        txt = (art.get("text") or "").strip()
        if len(txt) < 2000:
            continue
        n_art += 1
        if n_art == 6:
            globals()["TEXTO_PPL_WIKI"] = txt[:4000]
            continue
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        for i in range(0, len(ids) - SEQ + 1, SEQ):
            seqs.append(ids[i:i + SEQ])
            total += SEQ
            if total >= N_TOKENS_CAPTURA:
                return seqs, n_art, total
    return seqs, n_art, total


try:
    SEQS_IDS, N_ART, N_TOK = cargar_wikipedia()
    log(f"Wikipedia ES: {N_ART} articulos, {len(SEQS_IDS)} secuencias x {SEQ} = {N_TOK} tokens")
except Exception as e:
    fail(f"no se pudo bajar Wikipedia: {e}")
if TEXTO_PPL_WIKI is None:
    TEXTO_PPL_WIKI = "Montevideo es la capital de Uruguay y esta sobre el Rio de la Plata."

# --- captura: X entrada FF, Y salida profesora ORIGINAL ---
log(f"Pasando tokens por Qwen base y capturando FF{LAYER}: {len(SEQS_IDS)} seq x {SEQ} tokens...")
capturados_x = []


def hook_fn(module, inp, out):
    capturados_x.append(inp[0].detach().float().cpu())


teacher = model.model.layers[LAYER].mlp
h1 = teacher.register_forward_hook(hook_fn)
with torch.no_grad():
    for j, ids in enumerate(SEQS_IDS):
        model(input_ids=torch.tensor([ids], device=device))
        if (j + 1) % 50 == 0:
            log(f"  captura {j + 1}/{len(SEQS_IDS)}")
h1.remove()
X = torch.cat([t.reshape(-1, HIDDEN) for t in capturados_x], dim=0)
del capturados_x
log(f"Activaciones reales: X {X.shape} vectores")
if X.shape[0] < 5000:
    fail(f"muy pocos vectores ({X.shape[0]})")
perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(SEED))
n_eval = max(512, X.shape[0] // 5)
X_eval, X_train = X[perm[:n_eval]], X[perm[n_eval:]]
log(f"train {X_train.shape[0]} vectores / eval {X_eval.shape[0]} vectores (eval nunca se entrena)")
for p in teacher.parameters():
    p.requires_grad_(False)

# salida profesora ORIGINAL, una sola vez, para todas las etapas
with torch.no_grad():
    Y_eval = teacher(X_eval.to(device)).float().cpu()
    log(f"Y profesora original: {Y_eval.shape}")


def swiglu_torch(x, gate, up, down):
    return torch.nn.functional.silu(x @ gate.T) * (x @ up.T) @ down.T


def hoyer_clamped(preact, clamp=CLAMP):
    c = torch.clamp(preact, max=clamp) - clamp
    return ((c.abs().sum(dim=-1) ** 2) / (c.pow(2).sum(dim=-1) + 1e-15)).mean()


def sparsity_gate(gate_w, x_sample):
    with torch.no_grad():
        g = x_sample.to(device) @ gate_w.T
        return {
            "lt_-10": (g < -10).float().mean().item(),
            "lt_-5": (g < -5).float().mean().item(),
            "lt_-2": (g < -2).float().mean().item(),
            "lt_-1": (g < -1).float().mean().item(),
            "lt_0": (g < 0).float().mean().item(),
            "silu_lt_0.01": (torch.nn.functional.silu(g).abs() < 0.01).float().mean().item(),
        }


def eval_ff(gate_w, up_w, down_w, tag):
    with torch.no_grad():
        y = swiglu_torch(X_eval.to(device), gate_w, up_w, down_w)
        cos = torch.nn.functional.cosine_similarity(y, Y_eval.to(device), dim=-1).mean().item()
        r2 = 1 - ((y - Y_eval.to(device)) ** 2).sum().item() / max(((Y_eval - Y_eval.mean(0)) ** 2).sum().item(), 1e-12)
        nmse = (((y - Y_eval.to(device)) ** 2).mean() / (Y_eval.to(device) ** 2).mean()).item()
    sp = sparsity_gate(gate_w, X_eval[:2048])
    log(f"[{tag}] cos {cos:.4f} R2 {r2:.4f} NMSE {nmse:.4f} | gate<-10 {sp['lt_-10']*100:.1f}% <-5 {sp['lt_-5']*100:.1f}% <-2 {sp['lt_-2']*100:.1f}% <-1 {sp['lt_-1']*100:.1f}% <0 {sp['lt_0']*100:.1f}% SiLU~0 {sp['silu_lt_0.01']*100:.1f}%")
    return cos, r2, nmse, sp


resultados = {}

# ================= ETAPA 1: SPARSITY en -10 =================
log("=" * 60)
log("ETAPA 1: SPARSITY en -10, entrenar la FF sola (copia profesora)")
log("Perdida: BLD NMSE+0.1*(1-cos) para no romper la salida + W_HOYER*Hoyer(clamp -10)")
gate1 = teacher.gate_proj.weight.float().clone().detach().requires_grad_(True)
up1 = teacher.up_proj.weight.float().clone().detach().requires_grad_(True)
down1 = teacher.down_proj.weight.float().clone().detach().requires_grad_(True)
eval_ff(gate1, up1, down1, "etapa1 inicial (copia profesora)")
opt1 = torch.optim.Adam([gate1, up1, down1], lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
vec_por_paso = BATCH * SEQ
t0 = time.time()
n_train = X_train.shape[0]
order = torch.randperm(n_train, generator=torch.Generator().manual_seed(SEED))
pos = 0
for step in range(1, STEPS1 + 1):
    if pos + vec_por_paso > n_train:
        order = torch.randperm(n_train, generator=torch.Generator().manual_seed(SEED + step))
        pos = 0
    xb = X_train[order[pos:pos + vec_por_paso]].to(device)
    pos += vec_por_paso
    with torch.no_grad():
        yb = teacher(xb)
    opt1.zero_grad()
    loss_train = 0.0
    n_xb = xb.shape[0]
    n_micro = (n_xb + MICRO - 1) // MICRO
    for i in range(0, n_xb, MICRO):
        x_m, y_m = xb[i:i + MICRO], yb[i:i + MICRO]
        g_m = x_m @ gate1.T
        y_s = torch.nn.functional.silu(g_m) * (x_m @ up1.T) @ down1.T
        nmse = ((y_s - y_m) ** 2).mean() / (y_m ** 2).mean()
        cos = torch.nn.functional.cosine_similarity(y_s, y_m, dim=-1).mean()
        h_loss = hoyer_clamped(g_m)
        loss = (nmse + 0.1 * (1 - cos) + W_HOYER * h_loss) / n_micro
        loss.backward()
        loss_train += loss.item() * n_micro
    opt1.step()
    if step % 20 == 0 or step == 1:
        cos, r2, nmse, sp = eval_ff(gate1, up1, down1, f"etapa1 paso {step}/{STEPS1} loss {loss_train:.4f}")
cos1, r21, nmse1, sp1 = eval_ff(gate1, up1, down1, "ETAPA 1 FINAL")
resultados["etapa1"] = {"cos": cos1, "r2": r21, "nmse": nmse1, "sparsity": sp1, "steps": STEPS1}
torch.save({"gate": gate1.detach().cpu(), "up": up1.detach().cpu(), "down": down1.detach().cpu(),
            "cos": cos1, "r2": r21, "nmse": nmse1},
           f"ff_sparse_d2dmoe_capa{LAYER}_etapa1.pt")
log(f"Etapa 1 guardada: ff_sparse_d2dmoe_capa{LAYER}_etapa1.pt | {time.time() - t0:.0f}s")

# ================= ETAPA 2: PARTE =================
log("=" * 60)
log(f"ETAPA 2: PARTE, KMeans sobre gate_proj de la FF escasa, {N_EXPERTS} expertos {EXPERT_SIZES}")
try:
    from sklearn.cluster import KMeans
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"])
    from sklearn.cluster import KMeans
gate_w = gate1.detach().cpu().numpy()
gate_norm = gate_w / (np.linalg.norm(gate_w, axis=1, keepdims=True) + 1e-12)
km = KMeans(n_clusters=N_EXPERTS, n_init=4, random_state=SEED).fit(gate_norm)
log(f"  tamanos de cluster KMeans: {np.bincount(km.labels_, minlength=N_EXPERTS).tolist()} (ideal {INTER_E})")
dists = km.transform(gate_norm)
order_neuronas = np.argsort(dists.min(axis=1))
labels_bal = np.full(INTER, -1, dtype=int)
ocup = np.zeros(N_EXPERTS, dtype=int)
for n_idx in order_neuronas:
    for e_idx in np.argsort(dists[n_idx]):
        if ocup[e_idx] < EXPERT_SIZES[e_idx]:
            labels_bal[n_idx] = e_idx
            ocup[e_idx] += 1
            break
log(f"  tamanos balanceados: {ocup.tolist()}")


class MoE(torch.nn.Module):
    def __init__(self, labels_bal, gate_src, up_src, down_src):
        super().__init__()
        self.experts = torch.nn.ModuleList()
        for e in range(N_EXPERTS):
            idx = torch.tensor(np.where(labels_bal == e)[0])
            m = torch.nn.Module()
            m.gate = torch.nn.Parameter(gate_src[idx].clone())
            m.up = torch.nn.Parameter(up_src[idx].clone())
            m.down = torch.nn.Parameter(down_src[:, idx].clone())
            self.experts.append(m)
        self.router = torch.nn.Linear(HIDDEN, N_EXPERTS)
        with torch.no_grad():
            self.router.weight.mul_(0.01)
            self.router.bias.zero_()

    def forward(self, x, topk=TOPK):
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        logits = self.router(x)
        probs = torch.softmax(logits, dim=-1)
        # top-3 duro, straight-through en train: forward usa top-3, backward usa softmax
        idx = torch.topk(probs, topk, dim=-1).indices
        w_hard = torch.zeros_like(probs).scatter_(1, idx, torch.gather(probs, 1, idx))
        w_hard = w_hard / (w_hard.sum(dim=-1, keepdim=True) + 1e-12)
        w = w_hard + (probs - probs.detach())  # STE
        y = torch.zeros_like(x)
        for e in range(N_EXPERTS):
            rows = w[:, e] > 0
            if rows.sum() == 0:
                continue
            ye = swiglu_torch(x[rows], self.experts[e].gate, self.experts[e].up, self.experts[e].down)
            y[rows] += ye * w[rows, e][:, None]
        f = (w > 0).float().mean(0)
        lb = N_EXPERTS * (f * probs.mean(0)).sum()
        return y.reshape(orig_shape), lb, f


moe = MoE(labels_bal, gate1.detach(), up1.detach(), down1.detach()).to(device)
log(f"MoE params totales: {sum(p.numel() for p in moe.parameters()):,}")


def eval_moe(tag):
    with torch.no_grad():
        y, lb, f = moe(X_eval.to(device), topk=TOPK)
        cos = torch.nn.functional.cosine_similarity(y, Y_eval.to(device), dim=-1).mean().item()
        r2 = 1 - ((y - Y_eval.to(device)) ** 2).sum().item() / max(((Y_eval - Y_eval.mean(0)) ** 2).sum().item(), 1e-12)
        nmse = (((y - Y_eval.to(device)) ** 2).mean() / (Y_eval.to(device) ** 2).mean()).item()
    log(f"[{tag}] cos {cos:.4f} R2 {r2:.4f} NMSE {nmse:.4f} lb {lb.item():.4f} frac {['%.2f' % v for v in f.cpu().tolist()]}")
    return cos, r2, nmse, lb.item()


cos2, r22, nmse2, lb2 = eval_moe("ETAPA 2, partida sin entrenar, top-3")
resultados["etapa2"] = {"cos": cos2, "r2": r22, "nmse": nmse2, "lb": lb2}

# ================= ETAPA 3: FINE =================
log("=" * 60)
log("ETAPA 3: FINE, entrenar la MoE en top-3 duro sobre entrada/salida de la FF maestra ORIGINAL")
opt3 = torch.optim.Adam(moe.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
t0 = time.time()
order = torch.randperm(n_train, generator=torch.Generator().manual_seed(SEED + 1000))
pos = 0
for step in range(1, STEPS3 + 1):
    if pos + vec_por_paso > n_train:
        order = torch.randperm(n_train, generator=torch.Generator().manual_seed(SEED + step))
        pos = 0
    xb = X_train[order[pos:pos + vec_por_paso]].to(device)
    pos += vec_por_paso
    with torch.no_grad():
        yb = teacher(xb)
    opt3.zero_grad()
    loss_train = 0.0
    n_xb = xb.shape[0]
    n_micro = (n_xb + MICRO - 1) // MICRO
    for i in range(0, n_xb, MICRO):
        x_m, y_m = xb[i:i + MICRO], yb[i:i + MICRO]
        y_s, lb, _ = moe(x_m, topk=TOPK)
        nmse = ((y_s - y_m) ** 2).mean() / (y_m ** 2).mean()
        cos = torch.nn.functional.cosine_similarity(y_s, y_m, dim=-1).mean()
        loss = (nmse + 0.1 * (1 - cos) + W_LB * lb) / n_micro
        loss.backward()
        loss_train += loss.item() * n_micro
    opt3.step()
    if step % 20 == 0 or step == 1:
        cos, r2, nmse, lb = eval_moe(f"etapa3 paso {step}/{STEPS3} loss {loss_train:.4f}")
cos3, r23, nmse3, lb3 = eval_moe("ETAPA 3 FINAL")
ok = cos3 >= TARGET_COS and r23 > TARGET_R2
resultados["etapa3"] = {"cos": cos3, "r2": r23, "nmse": nmse3, "lb": lb3, "steps": STEPS3}
log(f"Etapa 3 meta cos>={TARGET_COS} y R2>{TARGET_R2}: {'SI' if ok else 'NO'} | {time.time() - t0:.0f}s")

# --- inferencia: Qwen normal vs Qwen con MoE D2DMoE en la capa ---
log("=" * 60)
log("Inferencia: Qwen normal vs Qwen con MoE D2DMoE en la capa %d (mismas KV, mismas lineales)" % LAYER)
PROMPTS = [
    "La capital de Uruguay es",
    "El agua hierve a una temperatura de",
    "La fotosintesis es el proceso por el cual",
    "Montevideo es la capital de",
    "El idioma oficial de Espana es",
    "La Revolucion de Mayo ocurrio en",
]


def generar(modelo, prompt):
    ids = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = modelo.generate(**ids, max_new_tokens=30, do_sample=False,
                              use_cache=True, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


class MoEWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        y, _, _ = self.m(x, topk=TOPK)
        return y


mlp_original = model.model.layers[LAYER].mlp
comparaciones = []
for PROMPT_GEN in PROMPTS:
    with torch.no_grad():
        texto_normal = generar(model, PROMPT_GEN)
    model.model.layers[LAYER].mlp = MoEWrapper(moe).to(device)
    with torch.no_grad():
        texto_moe = generar(model, PROMPT_GEN)
    model.model.layers[LAYER].mlp = mlp_original
    comparaciones.append((PROMPT_GEN, texto_normal, texto_moe))
    log(f"--- prompt: {PROMPT_GEN!r}")
    log(f"[Qwen normal] {texto_normal!r}")
    log(f"[Qwen con MoE D2DMoE] {texto_moe!r}")
    log(f"Textos iguales: {'SI' if texto_normal == texto_moe else 'NO'}")


def ppl(modelo):
    ids = tok(TEXTO_PPL_WIKI, return_tensors="pt").to(device)
    with torch.no_grad():
        out = modelo(**ids, labels=ids["input_ids"])
    return math.exp(out.loss.item())


with torch.no_grad():
    ppl_normal = ppl(model)
    model.model.layers[LAYER].mlp = MoEWrapper(moe).to(device)
    ppl_moe = ppl(model)
    model.model.layers[LAYER].mlp = mlp_original
log(f"PPL Wikipedia (no vista) normal {ppl_normal:.2f} | PPL con MoE D2DMoE {ppl_moe:.2f} | ratio {ppl_moe / ppl_normal:.3f}")
torch.save({"state_dict": moe.state_dict(), "layer": LAYER, "inter_expert": INTER_E,
            "n_experts": N_EXPERTS, "clamp": CLAMP, "cos": cos3, "r2": r23, "nmse": nmse3,
            "etapa1": resultados["etapa1"], "etapa2": resultados["etapa2"],
            "batch": BATCH, "seq": SEQ, "steps3": STEPS3, "lr": LR, "seed": SEED},
           f"moe_d2dmoe_qwen25_capa{LAYER}.pt")
with open("resultados-colab-moe-d2dmoe.md", "w") as fo:
    fo.write(f"# MoE D2DMoE 3 etapas FF{LAYER} Qwen2.5-0.5B en Colab\n\n"
             f"- Datos: Wikipedia ES, {N_ART} articulos, {N_TOK} tokens\n"
             f"- Etapa 1 sparsity clamp {CLAMP}, W_HOYER {W_HOYER}, {STEPS1} steps: "
             f"cos {cos1:.4f}, R2 {r21:.4f}, NMSE {nmse1:.4f}, gate<-10 {sp1['lt_-10']*100:.1f}%\n"
             f"- Etapa 2 parte KMeans, sin entrenar, top-3: cos {cos2:.4f}, R2 {r22:.4f}, NMSE {nmse2:.4f}\n"
             f"- Etapa 3 fine MoE top-3, {STEPS3} steps: cos {cos3:.4f}, R2 {r23:.4f}, NMSE {nmse3:.4f}, lb {lb3:.4f}\n"
             f"- Meta cos>={TARGET_COS} y R2>{TARGET_R2}: {'SI' if ok else 'NO'}\n\n"
             f"## Inferencia\n\n")
    for pr, tn, tm in comparaciones:
        fo.write(f"### Prompt: {pr!r}\n\n- Qwen normal: {tn!r}\n- Qwen con MoE D2DMoE: {tm!r}\n\n")
    fo.write(f"- PPL Wikipedia (no vista) normal {ppl_normal:.2f} | PPL con MoE D2DMoE {ppl_moe:.2f} | ratio {ppl_moe / ppl_normal:.3f}\n")
log("Guardado: moe_d2dmoe_qwen25_capa0.pt, ff_sparse_d2dmoe_capa0_etapa1.pt, resultados-colab-moe-d2dmoe.md, colab_moe_d2dmoe.log")
log("RESUMEN 3 ETAPAS: "
    f"1 sparsity cos {cos1:.4f}/R2 {r21:.4f}/NMSE {nmse1:.4f} | "
    f"2 parte cos {cos2:.4f}/R2 {r22:.4f}/NMSE {nmse2:.4f} | "
    f"3 fine cos {cos3:.4f}/R2 {r23:.4f}/NMSE {nmse3:.4f} | meta {'SI' if ok else 'NO'}")
