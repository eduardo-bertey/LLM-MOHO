#!/usr/bin/env python3
"""Continuar la MoE de la capa 0 desde el checkpoint de 900 steps, con fine-tuning.

Idea de Juls 2026-11-07: agarrar la MoE que ya entreno 900 steps y darle
900 steps mas, pero en fine-tuning: LR a la mitad (2.5e-4, en vez de 5e-4),
para que no se desarme lo aprendido y solo afine.

Uso en Colab (una celda, en la misma carpeta donde quedo el checkpoint):
    !python3 colab_moe_continuar.py

Checkpoint que busca (en orden):
    moe_destilada_qwen25_capa0_paso900.pt
    moe_destilada_qwen25_capa0.pt
Si no esta, falla y te dice que corras primero colab_destilar_moe.py.

Datos: Wikipedia ES, 500k tokens reales (igual que antes), captura nueva.
Escalera: prueba completa cada 300 pasos (1200, 1500, 1800 totales),
con checkpoint por escalon: ..._paso1200.pt, ..._paso1500.pt, ..._paso1800.pt

Ajustes (variables de entorno):
  LAYER=0  BATCH=56  SEQ=1024  STEPS=900  LR=2.5e-4  SEED=123
  N_TOKENS_CAPTURA=500000  CKPT=moe_destilada_qwen25_capa0_paso900.pt
"""
import json
import math
import os
import subprocess
import sys
import time

LAYER = int(os.environ.get("LAYER", "0"))
BATCH = int(os.environ.get("BATCH", "56"))
SEQ = int(os.environ.get("SEQ", "1024"))
STEPS = int(os.environ.get("STEPS", "900"))
LR = float(os.environ.get("LR", "2.5e-4"))
SEED = int(os.environ.get("SEED", "123"))
N_TOKENS_CAPTURA = int(os.environ.get("N_TOKENS_CAPTURA", "500000"))
CKPT = os.environ.get("CKPT", f"moe_destilada_qwen25_capa{LAYER}_paso900.pt")
MICRO = int(os.environ.get("MICRO", "1024"))
W_LB = float(os.environ.get("W_LB", "0.01"))
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_DIR = os.environ.get("MODEL_DIR", "./modelo-qwen2.5-0.5b")
N_EXPERTS, TOP_K = 4, 2
TARGET_COS, TARGET_R2 = 0.95, 0.90

LOG = open("colab_moe_continuar.log", "a", buffering=1)


def log(m):
    print(m, flush=True)
    LOG.write(m + "\n")


def fail(m):
    log(f"[ERROR] {m}")
    sys.exit(1)


log("=" * 60)
log(f"Continuar MoE FF{LAYER} desde {CKPT} | +{STEPS} steps, LR fine-tuning {LR}")
log(f"batch {BATCH} seq {SEQ} steps {STEPS} captura {N_TOKENS_CAPTURA} tokens wiki")

import torch
import numpy as np

log(f"torch {torch.__version__} | cuda: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    device = "cpu"
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
log(f"Modelo: hidden {HIDDEN}, inter {INTER}, capas {N_LAYERS}")

# --- Wikipedia ES, 500k tokens ---
N_SEQ = N_TOKENS_CAPTURA // SEQ
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

# --- captura real de esta capa ---
log(f"Capturando activaciones REALES: {len(SEQS_IDS)} seq x {SEQ} tokens...")
capturados = []


def hook_fn(module, inp, out):
    capturados.append(inp[0].detach().float().cpu())


handle = model.model.layers[LAYER].mlp.register_forward_hook(hook_fn)
with torch.no_grad():
    for j, ids in enumerate(SEQS_IDS):
        model(input_ids=torch.tensor([ids], device=device))
        if (j + 1) % 50 == 0:
            log(f"  captura {j + 1}/{len(SEQS_IDS)}")
handle.remove()
X = torch.cat([t.reshape(-1, HIDDEN) for t in capturados], dim=0)
del capturados
log(f"Activaciones reales: {X.shape} vectores")
if X.shape[0] < 5000:
    fail(f"muy pocos vectores ({X.shape[0]})")
perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(SEED))
n_eval = max(512, X.shape[0] // 5)
X_eval, X_train = X[perm[:n_eval]], X[perm[n_eval:]]
log(f"train {X_train.shape[0]} vectores / eval {X_eval.shape[0]} vectores (eval nunca se entrena)")

teacher = model.model.layers[LAYER].mlp
for p in teacher.parameters():
    p.requires_grad_(False)


def swiglu(x, gate, up, down):
    return torch.nn.functional.silu(x @ gate.T) * (x @ up.T) @ down.T


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
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        logits = x @ self.router_w.T + self.router_b
        probs = torch.softmax(logits, dim=-1)
        w_top, idx = torch.topk(probs, TOP_K, dim=-1)
        w_top = w_top / w_top.sum(dim=-1, keepdim=True)
        y = torch.zeros_like(x)
        for e in range(N_EXPERTS):
            rows, cols = torch.where(idx == e)
            if rows.numel() == 0:
                continue
            ye = swiglu(x[rows], self.experts[e].gate, self.experts[e].up, self.experts[e].down)
            y[rows] += ye * w_top[rows, cols][:, None]
        f = torch.bincount(idx.reshape(-1), minlength=N_EXPERTS).float() / idx.numel()
        lb = N_EXPERTS * (f * probs.mean(0)).sum()
        return y.reshape(orig_shape), lb, f


moe = MoE().to(device)
# --- cargar checkpoint de 900 steps (pesos, no init de origen) ---
ckpt_path = CKPT
if not os.path.isfile(ckpt_path):
    ckpt_path = f"moe_destilada_qwen25_capa{LAYER}.pt"
if not os.path.isfile(ckpt_path):
    fail(f"no encontre el checkpoint {CKPT} ni moe_destilada_qwen25_capa{LAYER}.pt. "
         "Corre primero colab_destilar_moe.py (900 steps).")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
moe.load_state_dict(ckpt["state_dict"])
log(f"Checkpoint cargado: {ckpt_path} | cos {ckpt.get('cos')} R2 {ckpt.get('r2')} "
    f"NMSE {ckpt.get('nmse')} paso {ckpt.get('paso', '?')}")
PASO_INICIAL = int(ckpt.get("paso", 900))


def eval_moe():
    with torch.no_grad():
        y_t = swiglu(X_eval.to(device), teacher.gate_proj.weight.float(),
                     teacher.up_proj.weight.float(), teacher.down_proj.weight.float())
        y, lb, f = moe(X_eval.to(device))
        cos = torch.nn.functional.cosine_similarity(y, y_t, dim=-1).mean().item()
        r2 = 1 - ((y - y_t) ** 2).sum().item() / max(((y_t - y_t.mean(0)) ** 2).sum().item(), 1e-12)
        nmse = (((y - y_t) ** 2).mean() / (y_t ** 2).mean()).item()
    return cos, r2, nmse, lb.item(), f.cpu().tolist()


cos0, r20, nmse0, lb0, f0 = eval_moe()
log(f"[eval al cargar] cos {cos0:.4f} R2 {r20:.4f} NMSE {nmse0:.4f} lb {lb0:.4f} frac {['%.2f' % v for v in f0]}")


class MoEWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None

    def forward(self, x):
        y, _, _ = self.m(x)
        return y


def generar_prueba(modelo, prompt):
    ids = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = modelo.generate(**ids, max_new_tokens=30, do_sample=False,
                              use_cache=True, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


def ppl_prueba(modelo):
    ids = tok(TEXTO_PPL_WIKI, return_tensors="pt").to(device)
    with torch.no_grad():
        out = modelo(**ids, labels=ids["input_ids"])
    return math.exp(out.loss.item())


ETAPAS = []


def prueba_escalon(paso_total, cos_e, r2_e, nmse_e):
    moe.eval()
    mlp_orig = model.model.layers[LAYER].mlp
    model.model.layers[LAYER].mlp = MoEWrapper(moe).to(device)
    with torch.no_grad():
        ppl_m = ppl_prueba(model)
        txt_m = generar_prueba(model, "La capital de Uruguay es")
    model.model.layers[LAYER].mlp = mlp_orig
    moe.train()
    log(f"[ESCALON paso {paso_total}] cos {cos_e:.4f} R2 {r2_e:.4f} NMSE {nmse_e:.4f} | "
        f"PPL MoE {ppl_m:.2f} | texto: {txt_m!r}")
    torch.save({"state_dict": moe.state_dict(), "layer": LAYER, "inter_expert": INTER_E,
                "n_experts": N_EXPERTS, "top_k": TOP_K, "cos": cos_e, "r2": r2_e, "nmse": nmse_e,
                "ppl_moe": ppl_m, "paso": paso_total, "batch": BATCH, "seq": SEQ, "lr": LR, "seed": SEED},
               f"moe_destilada_qwen25_capa{LAYER}_paso{paso_total}.pt")
    ETAPAS.append({"paso": paso_total, "cos": cos_e, "r2": r2_e, "nmse": nmse_e, "ppl_moe": ppl_m,
                   "texto": txt_m})


# --- fine-tuning: LR a la mitad, Adam nuevo (no se cargan momentos viejos) ---
opt = torch.optim.Adam(moe.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
vec_por_paso = BATCH * SEQ
t0 = time.time()
n_train = X_train.shape[0]
order = torch.randperm(n_train, generator=torch.Generator().manual_seed(SEED + 999))
pos = 0
for step in range(1, STEPS + 1):
    paso_total = PASO_INICIAL + step
    if pos + vec_por_paso > n_train:
        order = torch.randperm(n_train, generator=torch.Generator().manual_seed(SEED + step))
        pos = 0
    xb = X_train[order[pos:pos + vec_por_paso]].to(device)
    pos += vec_por_paso
    with torch.no_grad():
        yb = swiglu(xb, teacher.gate_proj.weight.float(), teacher.up_proj.weight.float(),
                    teacher.down_proj.weight.float())
    opt.zero_grad()
    loss_train = 0.0
    n_xb = xb.shape[0]
    n_micro = (n_xb + MICRO - 1) // MICRO
    for i in range(0, n_xb, MICRO):
        x_m, y_m = xb[i:i + MICRO], yb[i:i + MICRO]
        y_s, lb, _ = moe(x_m)
        nmse = ((y_s - y_m) ** 2).mean() / (y_m ** 2).mean()
        cos = torch.nn.functional.cosine_similarity(y_s, y_m, dim=-1).mean()
        loss = (nmse + 0.1 * (1 - cos) + W_LB * lb) / n_micro
        loss.backward()
        loss_train += loss.item() * n_micro
    opt.step()
    if step % 20 == 0 or step == 1:
        cos, r2, nmse, lb, f = eval_moe()
        log(f"paso {paso_total} (+{step}/{STEPS}) loss_train {loss_train:.4f} | EVAL cos {cos:.4f} R2 {r2:.4f} NMSE {nmse:.4f} "
            f"lb {lb:.4f} frac {['%.2f' % v for v in f]} | {time.time() - t0:.0f}s")
    if step % 300 == 0:
        c_e, r_e, n_e, _, _ = eval_moe()
        prueba_escalon(paso_total, c_e, r_e, n_e)

cos, r2, nmse, lb, f = eval_moe()
ok = cos >= TARGET_COS and r2 > TARGET_R2
log(f"[eval final paso {PASO_INICIAL + STEPS}] cos {cos:.4f} R2 {r2:.4f} NMSE {nmse:.4f} lb {lb:.4f} | meta: {'SI' if ok else 'NO'}")
log(f"tiempo total: {time.time() - t0:.0f}s")

# --- inferencia final: 6 prompts, normal vs MoE, PPL wiki no vista ---
log("=" * 60)
log("Inferencia: Qwen normal vs Qwen con MoE en la capa %d (mismas KV, mismas lineales)" % LAYER)
PROMPTS = [
    "La capital de Uruguay es",
    "El agua hierve a una temperatura de",
    "La fotosintesis es el proceso por el cual",
    "Montevideo es la capital de",
    "El idioma oficial de Espana es",
    "La Revolucion de Mayo ocurrio en",
]
GEN_TOKENS = 30


def generar(modelo, prompt):
    ids = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = modelo.generate(**ids, max_new_tokens=GEN_TOKENS, do_sample=False,
                              use_cache=True, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


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
    log(f"[Qwen con MoE] {texto_moe!r}")
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
log(f"PPL Wikipedia (no vista) normal {ppl_normal:.2f} | PPL con MoE {ppl_moe:.2f} | ratio {ppl_moe / ppl_normal:.3f}")
torch.save({"state_dict": moe.state_dict(), "layer": LAYER, "inter_expert": INTER_E,
            "n_experts": N_EXPERTS, "top_k": TOP_K, "cos": cos, "r2": r2, "nmse": nmse,
            "ppl_base": ppl_normal, "ppl_moe": ppl_moe, "paso": PASO_INICIAL + STEPS,
            "batch": BATCH, "seq": SEQ, "steps": STEPS, "lr": LR, "seed": SEED},
           f"moe_destilada_qwen25_capa{LAYER}_paso{PASO_INICIAL + STEPS}.pt")
with open("resultados-colab-moe-continuar.md", "w") as fo:
    fo.write(f"# MoE FF{LAYER} Qwen2.5-0.5B - continuar desde paso {PASO_INICIAL} (+{STEPS} steps, LR {LR})\n\n"
             f"- Checkpoint inicial: {ckpt_path}\n"
             f"- Datos: Wikipedia ES, {N_ART} articulos, {N_TOK} tokens\n"
             f"- Batch {BATCH} seq {SEQ}, lr fine-tuning {LR}, train {n_train} vectores / eval {X_eval.shape[0]}\n"
             f"- Eval final: cos {cos:.4f}, R2 {r2:.4f}, NMSE {nmse:.4f}, lb {lb:.4f}\n"
             f"- Meta cos>={TARGET_COS} y R2>{TARGET_R2}: {'SI' if ok else 'NO'}\n\n"
             f"## Escalera\n\n")
    for e in ETAPAS:
        fo.write(f"- Paso {e['paso']}: cos {e['cos']:.4f} R2 {e['r2']:.4f} NMSE {e['nmse']:.4f} | "
                 f"PPL MoE {e['ppl_moe']:.2f} | texto: {e['texto']!r}\n")
    fo.write("\n## Inferencia\n\n")
    for pr, tn, tm in comparaciones:
        fo.write(f"### Prompt: {pr!r}\n\n- Qwen normal: {tn!r}\n- Qwen con MoE: {tm!r}\n\n")
    fo.write(f"- PPL Wikipedia (no vista) normal {ppl_normal:.2f} | PPL con MoE {ppl_moe:.2f} | ratio {ppl_moe / ppl_normal:.3f}\n")
log(f"Guardado: moe_destilada_qwen25_capa{LAYER}_paso{PASO_INICIAL + STEPS}.pt, resultados-colab-moe-continuar.md, colab_moe_continuar.log")
