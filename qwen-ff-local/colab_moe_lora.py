#!/usr/bin/env python3
"""MoE + LoRA, con datos de Wikipedia NUEVOS (no entrenados antes).

Idea de Juls 2026-11-07:
  - Agarrar la MoE que ya entreno 900 + 900 steps en fine-tuning
    (checkpoint de colab_moe_continuar.py, paso 1800).
  - Seguir entrenando con una parte de Wikipedia que NUNCA se uso antes
    (se saltean los primeros 200 articulos, que son los que ya se usaron),
    500k tokens nuevos.
  - Agregar un adaptador LoRA sobre la salida de la MoE:
        y = MoE(x) + B @ A @ x
    A: (r, hidden), init random chico; B: (hidden, r), init en CERO,
    asi al principio el LoRA no cambia nada (y = MoE(x)).
    r = 16 por defecto. La MoE sigue entrenando en fine-tuning (LR baja)
    y el LoRA entrena con LR mas alta, para corregir el error residual.
  - Escalera: prueba completa cada 300 pasos, 3 veces (2100, 2400, 2700
    totales), con checkpoint en cada escalon, guardando MoE + LoRA.

Uso en Colab (misma carpeta donde quedo el checkpoint de continuar):
    !python3 colab_moe_lora.py

Checkpoint que busca (en orden):
    moe_destilada_qwen25_capa0_paso1800.pt
    moe_destilada_qwen25_capa0_paso1500.pt
    moe_destilada_qwen25_capa0_paso1200.pt
    moe_destilada_qwen25_capa0_paso900.pt
Si no esta, falla y te dice que corras primero colab_moe_continuar.py.

Ajustes (variables de entorno):
  LAYER=0  BATCH=56  SEQ=1024  STEPS=900  LR_MOE=1e-4  LR_LORA=1e-3
  R_LORA=16  SEED=123  N_TOKENS_CAPTURA=500000  SALTAR_ARTICULOS=200
  CKPT=moe_destilada_qwen25_capa0_paso1800.pt
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
LR_MOE = float(os.environ.get("LR_MOE", "1e-4"))
LR_LORA = float(os.environ.get("LR_LORA", "1e-3"))
R_LORA = int(os.environ.get("R_LORA", "16"))
SEED = int(os.environ.get("SEED", "123"))
N_TOKENS_CAPTURA = int(os.environ.get("N_TOKENS_CAPTURA", "500000"))
SALTAR_ARTICULOS = int(os.environ.get("SALTAR_ARTICULOS", "200"))
CKPT = os.environ.get("CKPT", f"moe_destilada_qwen25_capa{LAYER}_paso1800.pt")
MICRO = int(os.environ.get("MICRO", "1024"))
W_LB = float(os.environ.get("W_LB", "0.01"))
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_DIR = os.environ.get("MODEL_DIR", "./modelo-qwen2.5-0.5b")
N_EXPERTS, TOP_K = 4, 2
TARGET_COS, TARGET_R2 = 0.95, 0.90

LOG = open("colab_moe_lora.log", "a", buffering=1)


def log(m):
    print(m, flush=True)
    LOG.write(m + "\n")


def fail(m):
    log(f"[ERROR] {m}")
    sys.exit(1)


log("=" * 60)
log(f"MoE + LoRA FF{LAYER} desde {CKPT} | +{STEPS} steps")
log(f"LoRA r={R_LORA} | LR MoE {LR_MOE} | LR LoRA {LR_LORA} | wiki NUEVA, salta {SALTAR_ARTICULOS} articulos")

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
log(f"Modelo: hidden {HIDDEN}, inter {INTER}, capas {N_LAYERS}")

# --- Wikipedia NUEVA: saltear los articulos ya usados antes ---
TEXTO_PPL_WIKI = None


def cargar_wikipedia_nueva():
    try:
        import datasets
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
        import datasets
    ds = datasets.load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)
    seqs, total, n_art, n_saltados = [], 0, 0, 0
    for art in ds:
        txt = (art.get("text") or "").strip()
        if len(txt) < 2000:
            continue
        n_art += 1
        if n_art <= SALTAR_ARTICULOS:
            n_saltados += 1
            continue
        if n_art == SALTAR_ARTICULOS + 6:  # articulo aparte para PPL, nunca en train
            globals()["TEXTO_PPL_WIKI"] = txt[:4000]
            continue
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        for i in range(0, len(ids) - SEQ + 1, SEQ):
            seqs.append(ids[i:i + SEQ])
            total += SEQ
            if total >= N_TOKENS_CAPTURA:
                return seqs, n_art, total, n_saltados
    return seqs, n_art, total, n_saltados


try:
    SEQS_IDS, N_ART, N_TOK, N_SALT = cargar_wikipedia_nueva()
    log(f"Wikipedia ES NUEVA: salteados {N_SALT} articulos, usados {N_ART - N_SALT}, "
        f"{len(SEQS_IDS)} secuencias x {SEQ} = {N_TOK} tokens")
except Exception as e:
    fail(f"no se pudo bajar Wikipedia: {e}")
if TEXTO_PPL_WIKI is None:
    TEXTO_PPL_WIKI = "Montevideo es la capital de Uruguay y esta sobre el Rio de la Plata."

# --- captura real de esta capa, con los datos NUEVOS ---
log(f"Capturando activaciones REALES (datos nuevos): {len(SEQS_IDS)} seq x {SEQ} tokens...")
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
log(f"Activaciones reales (nuevas): {X.shape} vectores")
if X.shape[0] < 5000:
    fail(f"muy pocos vectores ({X.shape[0]})")
perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(SEED + 7))
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


class MoELoRA(torch.nn.Module):
    """MoE + adaptador LoRA en la salida: y = MoE(x) + B @ A @ x.
    B arranca en cero, asi al cargar el checkpoint la salida no cambia."""

    def __init__(self, moe):
        super().__init__()
        self.moe = moe
        self.lora_A = torch.nn.Parameter(torch.randn(R_LORA, HIDDEN) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(HIDDEN, R_LORA))

    def forward(self, x):
        y, lb, f = self.moe(x)
        # LoRA sobre la misma entrada x (2D o 3D, se aplana igual que la MoE)
        orig_shape = x.shape
        x2 = x.reshape(-1, x.shape[-1])
        y2 = y.reshape(-1, y.shape[-1])
        y2 = y2 + (x2 @ self.lora_A.T) @ self.lora_B.T
        return y2.reshape(orig_shape), lb, f


moe = MoE().to(device)
# --- cargar checkpoint de continuar (paso 1800, 1500, 1200 o 900) ---
ckpt_path = CKPT
if not os.path.isfile(ckpt_path):
    for p in [f"moe_destilada_qwen25_capa{LAYER}_paso1800.pt",
              f"moe_destilada_qwen25_capa{LAYER}_paso1500.pt",
              f"moe_destilada_qwen25_capa{LAYER}_paso1200.pt",
              f"moe_destilada_qwen25_capa{LAYER}_paso900.pt",
              f"moe_destilada_qwen25_capa{LAYER}.pt"]:
        if os.path.isfile(p):
            ckpt_path = p
            break
if not os.path.isfile(ckpt_path):
    fail(f"no encontre ningun checkpoint de continuar. Corre primero colab_moe_continuar.py.")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
moe.load_state_dict(ckpt["state_dict"])
log(f"Checkpoint cargado: {ckpt_path} | cos {ckpt.get('cos')} R2 {ckpt.get('r2')} "
    f"NMSE {ckpt.get('nmse')} paso {ckpt.get('paso', '?')}")
PASO_INICIAL = int(ckpt.get("paso", 1800))
moe_lora = MoELoRA(moe).to(device)
log(f"LoRA r={R_LORA} agregado: params LoRA {R_LORA * HIDDEN * 2:,} | "
    f"MoE {sum(p.numel() for p in moe.parameters()):,}")


def eval_moe_lora():
    with torch.no_grad():
        y_t = swiglu(X_eval.to(device), teacher.gate_proj.weight.float(),
                     teacher.up_proj.weight.float(), teacher.down_proj.weight.float())
        y, lb, f = moe_lora(X_eval.to(device))
        cos = torch.nn.functional.cosine_similarity(y, y_t, dim=-1).mean().item()
        r2 = 1 - ((y - y_t) ** 2).sum().item() / max(((y_t - y_t.mean(0)) ** 2).sum().item(), 1e-12)
        nmse = (((y - y_t) ** 2).mean() / (y_t ** 2).mean()).item()
    return cos, r2, nmse, lb.item(), f.cpu().tolist()


cos0, r20, nmse0, lb0, f0 = eval_moe_lora()
log(f"[eval al cargar + LoRA en cero] cos {cos0:.4f} R2 {r20:.4f} NMSE {nmse0:.4f} lb {lb0:.4f}")


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
    moe_lora.eval()
    mlp_orig = model.model.layers[LAYER].mlp
    model.model.layers[LAYER].mlp = MoEWrapper(moe_lora).to(device)
    with torch.no_grad():
        ppl_m = ppl_prueba(model)
        txt_m = generar_prueba(model, "La capital de Uruguay es")
    model.model.layers[LAYER].mlp = mlp_orig
    moe_lora.train()
    log(f"[ESCALON paso {paso_total}] cos {cos_e:.4f} R2 {r2_e:.4f} NMSE {nmse_e:.4f} | "
        f"PPL MoE+LoRA {ppl_m:.2f} | texto: {txt_m!r}")
    torch.save({"state_dict": moe.state_dict(), "lora_A": moe_lora.lora_A.detach().cpu(),
                "lora_B": moe_lora.lora_B.detach().cpu(), "r_lora": R_LORA,
                "layer": LAYER, "inter_expert": INTER_E, "n_experts": N_EXPERTS, "top_k": TOP_K,
                "cos": cos_e, "r2": r2_e, "nmse": nmse_e, "ppl_moe": ppl_m, "paso": paso_total,
                "batch": BATCH, "seq": SEQ, "lr_moe": LR_MOE, "lr_lora": LR_LORA, "seed": SEED},
               f"moe_lora_qwen25_capa{LAYER}_paso{paso_total}.pt")
    ETAPAS.append({"paso": paso_total, "cos": cos_e, "r2": r2_e, "nmse": nmse_e, "ppl_moe": ppl_m,
                   "texto": txt_m})


# --- entrenar MoE (LR baja) + LoRA (LR alta), Adam por grupo ---
opt = torch.optim.Adam([
    {"params": moe.parameters(), "lr": LR_MOE},
    {"params": [moe_lora.lora_A, moe_lora.lora_B], "lr": LR_LORA},
], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
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
        y_s, lb, _ = moe_lora(x_m)
        nmse = ((y_s - y_m) ** 2).mean() / (y_m ** 2).mean()
        cos = torch.nn.functional.cosine_similarity(y_s, y_m, dim=-1).mean()
        loss = (nmse + 0.1 * (1 - cos) + W_LB * lb) / n_micro
        loss.backward()
        loss_train += loss.item() * n_micro
    opt.step()
    if step % 20 == 0 or step == 1:
        cos, r2, nmse, lb, f = eval_moe_lora()
        log(f"paso {paso_total} (+{step}/{STEPS}) loss_train {loss_train:.4f} | EVAL cos {cos:.4f} R2 {r2:.4f} NMSE {nmse:.4f} "
            f"lb {lb:.4f} frac {['%.2f' % v for v in f]} | {time.time() - t0:.0f}s")
    if step % 300 == 0:  # escalera x3: 2100, 2400, 2700
        c_e, r_e, n_e, _, _ = eval_moe_lora()
        prueba_escalon(paso_total, c_e, r_e, n_e)

cos, r2, nmse, lb, f = eval_moe_lora()
ok = cos >= TARGET_COS and r2 > TARGET_R2
log(f"[eval final paso {PASO_INICIAL + STEPS}] cos {cos:.4f} R2 {r2:.4f} NMSE {nmse:.4f} lb {lb:.4f} | meta: {'SI' if ok else 'NO'}")
log(f"tiempo total: {time.time() - t0:.0f}s")

# --- inferencia final: 6 prompts, normal vs MoE+LoRA, PPL wiki nueva no vista ---
log("=" * 60)
log("Inferencia: Qwen normal vs Qwen con MoE+LoRA en la capa %d (mismas KV, mismas lineales)" % LAYER)
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


mlp_original = model.model.layers[LAYER].mlp
comparaciones = []
for PROMPT_GEN in PROMPTS:
    with torch.no_grad():
        texto_normal = generar(model, PROMPT_GEN)
    model.model.layers[LAYER].mlp = MoEWrapper(moe_lora).to(device)
    with torch.no_grad():
        texto_moe = generar(model, PROMPT_GEN)
    model.model.layers[LAYER].mlp = mlp_original
    comparaciones.append((PROMPT_GEN, texto_normal, texto_moe))
    log(f"--- prompt: {PROMPT_GEN!r}")
    log(f"[Qwen normal] {texto_normal!r}")
    log(f"[Qwen con MoE+LoRA] {texto_moe!r}")
    log(f"Textos iguales: {'SI' if texto_normal == texto_moe else 'NO'}")


def ppl(modelo):
    ids = tok(TEXTO_PPL_WIKI, return_tensors="pt").to(device)
    with torch.no_grad():
        out = modelo(**ids, labels=ids["input_ids"])
    return math.exp(out.loss.item())


with torch.no_grad():
    ppl_normal = ppl(model)
    model.model.layers[LAYER].mlp = MoEWrapper(moe_lora).to(device)
    ppl_moe = ppl(model)
    model.model.layers[LAYER].mlp = mlp_original
log(f"PPL Wikipedia nueva (no vista) normal {ppl_normal:.2f} | PPL con MoE+LoRA {ppl_moe:.2f} | ratio {ppl_moe / ppl_normal:.3f}")
with open("resultados-colab-moe-lora.md", "w") as fo:
    fo.write(f"# MoE+LoRA FF{LAYER} Qwen2.5-0.5B - desde paso {PASO_INICIAL} (+{STEPS} steps)\n\n"
             f"- Checkpoint inicial: {ckpt_path}\n"
             f"- Datos: Wikipedia ES NUEVA, salteados {N_SALT} articulos, {N_TOK} tokens\n"
             f"- LoRA r={R_LORA}, LR MoE {LR_MOE}, LR LoRA {LR_LORA}\n"
             f"- Batch {BATCH} seq {SEQ}, train {n_train} vectores / eval {X_eval.shape[0]}\n"
             f"- Eval final: cos {cos:.4f}, R2 {r2:.4f}, NMSE {nmse:.4f}, lb {lb:.4f}\n"
             f"- Meta cos>={TARGET_COS} y R2>{TARGET_R2}: {'SI' if ok else 'NO'}\n\n"
             f"## Escalera\n\n")
    for e in ETAPAS:
        fo.write(f"- Paso {e['paso']}: cos {e['cos']:.4f} R2 {e['r2']:.4f} NMSE {e['nmse']:.4f} | "
                 f"PPL MoE+LoRA {e['ppl_moe']:.2f} | texto: {e['texto']!r}\n")
    fo.write("\n## Inferencia\n\n")
    for pr, tn, tm in comparaciones:
        fo.write(f"### Prompt: {pr!r}\n\n- Qwen normal: {tn!r}\n- Qwen con MoE+LoRA: {tm!r}\n\n")
    fo.write(f"- PPL Wikipedia nueva (no vista) normal {ppl_normal:.2f} | PPL con MoE+LoRA {ppl_moe:.2f} | ratio {ppl_moe / ppl_normal:.3f}\n")
log(f"Guardado: moe_lora_qwen25_capa{LAYER}_paso{PASO_INICIAL + STEPS}.pt, resultados-colab-moe-lora.md, colab_moe_lora.log")
