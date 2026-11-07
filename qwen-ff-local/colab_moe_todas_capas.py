#!/usr/bin/env python3
"""MoE capa por capa, de la 0 a la 23, Qwen2.5-0.5B, solo run en Colab.

Idea de Juls 2026-11-07: ver que capas son mas dificiles de sustituir por una MoE.
  - Datos: Wikipedia ES, 500k tokens reales (una sola captura por capa).
  - Por cada capa L (0..23):
      1. Capturar las activaciones REALES que entran a la MLP de esa capa.
      2. Destilar una MoE de 4 expertos inter 1216, presupuesto = profesora
         (13.074.432 params, top-k=2, activos/token 6.537.216).
      3. Medir cos / R2 / NMSE en eval (nunca entrenado).
      4. Reinsertar SOLO esa capa en Qwen original (mismas KV, mismas lineales,
         solo cambia esa MLP) y medir PPL en un articulo de Wikipedia no visto
         + generar con un prompt.
  - Al final: tabla comparativa, de la capa mas facil a la mas dificil.

Uso en Colab (una celda):
    !git clone --depth 1 -b fix/ff-eval-real-activations https://github.com/eduardo-bertey/LLM-MOHO.git
    %cd LLM-MOHO/qwen-ff-local
    !python3 colab_moe_todas_capas.py

Ajustes (variables de entorno):
  BATCH=56  SEQ=1024  STEPS=100  LR=5e-4  SEED=123  N_TOKENS_CAPTURA=500000
  CAPA_INICIO=0  CAPA_FIN=23  (para retomar: CAPA_INICIO=12 python3 colab_moe_todas_capas.py)
  Con 500k tokens y 100 steps por capa, ~3-4 min por capa, ~1.5 h total en T4.
"""
import json
import math
import os
import subprocess
import sys
import time

BATCH = int(os.environ.get("BATCH", "56"))
SEQ = int(os.environ.get("SEQ", "1024"))
STEPS = int(os.environ.get("STEPS", "100"))
LR = float(os.environ.get("LR", "5e-4"))
SEED = int(os.environ.get("SEED", "123"))
N_TOKENS_CAPTURA = int(os.environ.get("N_TOKENS_CAPTURA", "500000"))
CAPA_INICIO = int(os.environ.get("CAPA_INICIO", "0"))
CAPA_FIN = int(os.environ.get("CAPA_FIN", "23"))
MICRO = int(os.environ.get("MICRO", "1024"))
W_LB = float(os.environ.get("W_LB", "0.01"))
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_DIR = os.environ.get("MODEL_DIR", "./modelo-qwen2.5-0.5b")
N_EXPERTS, TOP_K = 4, 2
TARGET_COS, TARGET_R2 = 0.95, 0.90

LOG = open("colab_moe_todas_capas.log", "a", buffering=1)


def log(m):
    print(m, flush=True)
    LOG.write(m + "\n")


def fail(m):
    log(f"[ERROR] {m}")
    sys.exit(1)


log("=" * 60)
log(f"MoE capa por capa Qwen2.5-0.5B | capas {CAPA_INICIO}..{CAPA_FIN}")
log(f"batch {BATCH} seq {SEQ} steps/capa {STEPS} lr {LR} captura {N_TOKENS_CAPTURA} tokens wiki")

import torch
import numpy as np

log(f"torch {torch.__version__} | cuda: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    log("AVISO: sin GPU, muy lento. Colab: Entorno > GPU (T4).")
    device = "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

from transformers import AutoModelForCausalLM, AutoTokenizer

if os.path.isdir(MODEL_DIR) and os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
    log(f"Modelo YA ESTA en {MODEL_DIR}, no se descarga.")
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
log(f"MoE por capa: {N_EXPERTS} expertos x inter {INTER_E} = {N_EXPERTS * 3 * INTER_E * HIDDEN:,} params = profesora")

# --- Wikipedia ES, 500k tokens, una sola vez (se reusan para todas las capas) ---
N_SEQ = N_TOKENS_CAPTURA // SEQ
TEXTO_PPL_WIKI = None


def cargar_wikipedia():
    try:
        import datasets
    except ImportError:
        log("instalando datasets para bajar Wikipedia...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
        import datasets
    ds = datasets.load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)
    seqs, total, n_art = [], 0, 0
    for art in ds:
        txt = (art.get("text") or "").strip()
        if len(txt) < 2000:
            continue
        n_art += 1
        if n_art == 6:  # articulo aparte para PPL, nunca entra en train
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

# PPL base (Qwen original, sin tocar nada) en el articulo no visto
def ppl_con_mlp(mlp_nueva):
    ids = tok(TEXTO_PPL_WIKI, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**ids, labels=ids["input_ids"])
    return math.exp(out.loss.item())


with torch.no_grad():
    PPL_BASE = ppl_con_mlp(None)
log(f"PPL base Qwen original (wiki no vista): {PPL_BASE:.2f}")

PROMPT_GEN = "La capital de Uruguay es"


def generar(modelo, prompt):
    ids = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = modelo.generate(**ids, max_new_tokens=30, do_sample=False,
                              use_cache=True, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


with torch.no_grad():
    TEXTO_BASE_GEN = generar(model, PROMPT_GEN)
log(f"[Qwen original] {TEXTO_BASE_GEN!r}")


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


resultados = []
t_total = time.time()
for LAYER in range(CAPA_INICIO, CAPA_FIN + 1):
    t0 = time.time()
    log("=" * 60)
    log(f"### CAPA {LAYER}/{N_LAYERS - 1}")
    # 1. captura real de ESTA capa
    capturados = []

    def hook_fn(module, inp, out):
        capturados.append(inp[0].detach().float().cpu())

    handle = model.model.layers[LAYER].mlp.register_forward_hook(hook_fn)
    with torch.no_grad():
        for j, ids in enumerate(SEQS_IDS):
            model(input_ids=torch.tensor([ids], device=device))
            if (j + 1) % 100 == 0:
                log(f"  captura capa {LAYER}: {j + 1}/{len(SEQS_IDS)}")
    handle.remove()
    X = torch.cat([t.reshape(-1, HIDDEN) for t in capturados], dim=0)
    del capturados
    log(f"  Activaciones capa {LAYER}: {X.shape} vectores")
    if X.shape[0] < 5000:
        log(f"  [AVISO] pocos vectores en capa {LAYER}, se salta")
        continue
    perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(SEED))
    n_eval = max(512, X.shape[0] // 5)
    X_eval, X_train = X[perm[:n_eval]], X[perm[n_eval:]]
    n_train = X_train.shape[0]

    teacher = model.model.layers[LAYER].mlp
    for p in teacher.parameters():
        p.requires_grad_(False)

    moe = MoE().to(device)

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
    log(f"  [eval inicial capa {LAYER}] cos {cos0:.4f} R2 {r20:.4f} NMSE {nmse0:.4f}")
    opt = torch.optim.Adam(moe.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    vec_por_paso = BATCH * SEQ
    order = torch.randperm(n_train, generator=torch.Generator().manual_seed(SEED))
    pos = 0
    for step in range(1, STEPS + 1):
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
            c, r, n, lbv, fv = eval_moe()
            log(f"  capa {LAYER} paso {step}/{STEPS} loss {loss_train:.4f} | EVAL cos {c:.4f} R2 {r:.4f} NMSE {n:.4f} lb {lbv:.4f}")
    cos, r2, nmse, lb, f = eval_moe()
    ok = cos >= TARGET_COS and r2 > TARGET_R2
    log(f"  [eval final capa {LAYER}] cos {cos:.4f} R2 {r2:.4f} NMSE {nmse:.4f} lb {lb:.4f} | meta: {'SI' if ok else 'NO'}")

    # 4. reinsertar SOLO esta capa y medir PPL + generacion
    mlp_original = model.model.layers[LAYER].mlp
    model.model.layers[LAYER].mlp = MoEWrapper(moe).to(device)
    with torch.no_grad():
        ppl_moe = ppl_con_mlp(None)
        texto_moe = generar(model, PROMPT_GEN)
    model.model.layers[LAYER].mlp = mlp_original
    ratio = ppl_moe / PPL_BASE
    log(f"  [reinsersion capa {LAYER}] PPL normal {PPL_BASE:.2f} | PPL MoE {ppl_moe:.2f} | ratio {ratio:.3f}")
    log(f"  [Qwen con MoE capa {LAYER}] {texto_moe!r}")
    torch.save({"state_dict": moe.state_dict(), "layer": LAYER, "inter_expert": INTER_E,
                "n_experts": N_EXPERTS, "top_k": TOP_K, "cos": cos, "r2": r2, "nmse": nmse,
                "ppl_base": PPL_BASE, "ppl_moe": ppl_moe, "batch": BATCH, "seq": SEQ,
                "steps": STEPS, "lr": LR, "seed": SEED, "n_tokens_wiki": N_TOK},
               f"moe_destilada_qwen25_capa{LAYER}.pt")
    resultados.append({"capa": LAYER, "cos": cos, "r2": r2, "nmse": nmse, "lb": lb,
                       "ppl_base": PPL_BASE, "ppl_moe": ppl_moe, "ratio_ppl": ratio,
                       "meta": bool(ok), "texto_moe": texto_moe, "tiempo_s": time.time() - t0,
                       "train": int(n_train), "eval": int(X_eval.shape[0])})
    with open("resultados-todas-capas.json", "w") as fo:
        json.dump(resultados, fo, indent=2, ensure_ascii=False)
    del X, X_train, X_eval, moe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- tabla final: de la mas facil a la mas dificil (por cos, y por ratio PPL) ---
log("=" * 60)
log("TABLA FINAL (ordenada por cos, de mas facil a mas dificil de sustituir)")
for r in sorted(resultados, key=lambda d: -d["cos"]):
    log(f"  capa {r['capa']:2d} | cos {r['cos']:.4f} R2 {r['r2']:.4f} NMSE {r['nmse']:.4f} | "
        f"PPL {r['ppl_moe']:.2f} ratio {r['ratio_ppl']:.3f} | meta {'SI' if r['meta'] else 'NO'} | {r['tiempo_s']:.0f}s")
log("TABLA POR PPL (ratio mas alto = mas dificil de sustituir)")
for r in sorted(resultados, key=lambda d: -d["ratio_ppl"]):
    log(f"  capa {r['capa']:2d} | ratio PPL {r['ratio_ppl']:.3f} | cos {r['cos']:.4f} R2 {r['r2']:.4f}")
with open("resultados-todas-capas.md", "w") as fo:
    fo.write("# MoE capa por capa Qwen2.5-0.5B (Wikipedia ES, 500k tokens)\n\n"
             f"- PPL base Qwen original (wiki no vista): {PPL_BASE:.2f}\n"
             f"- Prompt: {PROMPT_GEN!r} | Qwen original: {TEXTO_BASE_GEN!r}\n\n"
             "| capa | cos | R2 | NMSE | PPL MoE | ratio PPL | meta |\n"
             "|---:|---:|---:|---:|---:|---:|:---:|\n")
    for r in sorted(resultados, key=lambda d: d["capa"]):
        fo.write(f"| {r['capa']} | {r['cos']:.4f} | {r['r2']:.4f} | {r['nmse']:.4f} | "
                 f"{r['ppl_moe']:.2f} | {r['ratio_ppl']:.3f} | {'SI' if r['meta'] else 'NO'} |\n")
    fo.write("\n## Generacion por capa\n\n")
    for r in sorted(resultados, key=lambda d: d["capa"]):
        fo.write(f"- Capa {r['capa']}: {r['texto_moe']!r}\n")
log(f"Guardado: resultados-todas-capas.md/.json, moe_destilada_qwen25_capa{{L}}.pt, tiempo total {time.time() - t_total:.0f}s")
