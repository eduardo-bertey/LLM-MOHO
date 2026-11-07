#!/usr/bin/env python3
"""Destilar la FF de Qwen2.5-0.5B a MoE, solo run en Colab.

Uso en Colab (una celda):
    !git clone --depth 1 -b fix/ff-eval-real-activations https://github.com/eduardo-bertey/LLM-MOHO.git
    %cd LLM-MOHO/qwen-ff-local
    !python3 colab_destilar_moe.py

POR QUE EL TRAIN ANTERIOR ESTABA MAL (ver captura de Juls 2026-11-07):
  - Solo se capturaron 125 vectores reales (train 61 / eval 64).
  - Cada paso muestreaba 16384 vectores CON REEMPLAZO de esos 61:
    cada vector se repetia ~268 veces por paso, 300 pasos = memorizacion.
  - Resultado: loss de train -> 0.0000, pero eval clavada en
    cos 0.4174 / R2 -0.4804 / NMSE 1.1597 desde el paso 60.
    El loss que bajaba era el de train memorizado, no generalizacion.
  ARREGLO: capturar MUCHOS vectores reales (>= 20000), pasar cada vector
  ~1 vez por epoca (shuffle sin reemplazo), y medir eval en un set aparte
  que el train no ve.

MoE con PRESUPUESTO IGUAL A LA PROFESORA:
  - Profesora FF0: inter 4864, 3 * 4864 * 896 = 13.074.432 params.
  - MoE: 4 expertos, cada uno inter 1216 (= 4864 / 4).
    Total expertos: 4 * 3 * 1216 * 896 = 13.074.432 params, IGUAL que la profesora.
    Router: 4 * 896 + 4 = 3.588 params extra (despreciable).
  - Top-k=2: activos por token 2 * 3 * 1216 * 896 = 6.537.216 params (la mitad
    de la profesora por token, el resto es capacidad total).
  - Cada experto se destila desde la profesora con la misma perdida BLD
    (NMSE + 0.1*(1-cos)) + perdida de balanceo del router (peso 0.01).

Ajustes (variables de entorno):
  LAYER=0  BATCH=56  SEQ=1024  STEPS=900  LR=5e-4  SEED=123
  N_TOKENS_CAPTURA=500000 (Wikipedia en espanol, ~500 seq x 1024 tokens)
  N_SEQ_CAPTURA = secuencias de SEQ tokens a pasar por Qwen para capturar
  activaciones (500 x 1024 = 512000 vectores). Si Colab va lento, bajar a 200.
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
LR = float(os.environ.get("LR", "5e-4"))
SEED = int(os.environ.get("SEED", "123"))
N_TOKENS_CAPTURA = int(os.environ.get("N_TOKENS_CAPTURA", "500000"))
N_SEQ_CAPTURA = int(os.environ.get("N_SEQ_CAPTURA", str(N_TOKENS_CAPTURA // SEQ)))
MICRO = int(os.environ.get("MICRO", "1024"))
W_LB = float(os.environ.get("W_LB", "0.01"))
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_DIR = os.environ.get("MODEL_DIR", "./modelo-qwen2.5-0.5b")
N_EXPERTS, TOP_K = 4, 2
TARGET_COS, TARGET_R2 = 0.95, 0.90

LOG = open("colab_destilar_moe.log", "a", buffering=1)


def log(m):
    print(m, flush=True)
    LOG.write(m + "\n")


def fail(m):
    log(f"[ERROR] {m}")
    sys.exit(1)


log("=" * 60)
log(f"MoE FF{LAYER} Qwen2.5-0.5B | 4 expertos inter 1216, presupuesto total = profesora")
log(f"batch {BATCH} seq {SEQ} steps {STEPS} lr {LR} captura {N_SEQ_CAPTURA} seq x {SEQ} tokens")

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
INTER_E = INTER // N_EXPERTS  # 1216
log(f"Modelo: hidden {HIDDEN}, inter {INTER}, capas {N_LAYERS}")
log(f"MoE: {N_EXPERTS} expertos x inter {INTER_E} = {N_EXPERTS * 3 * INTER_E * HIDDEN:,} params "
    f"(profesora {3 * INTER * HIDDEN:,}) | activos/token top-{TOP_K}: {TOP_K * 3 * INTER_E * HIDDEN:,}")

# --- datos: Wikipedia en espanol, ~500k tokens reales (no sinteticos) ---
# Se baja por streaming con `datasets` (wikimedia/wikipedia 20231101.es),
# se tokeniza cada articulo y se corta en secuencias de SEQ tokens.
# Un articulo aparte queda como texto de PPL que el train nunca ve.
TEXTOS_BASE = [
    "La destilacion de una red neuronal consiste en que un modelo alumno aprenda a imitar a un modelo profesor, copiando sus salidas en datos reales y no solo las etiquetas correctas. En el caso de una red feedforward, el alumno aprende a producir la misma salida que la profesora ante la misma entrada, aunque tenga menos parametros o una estructura distinta, como una mezcla de expertos.",
    "Montevideo es la capital de Uruguay y esta sobre el Rio de la Plata. Fue fundada en el siglo dieciocho y hoy es el centro politico, economico y cultural del pais. Su puerto es uno de los mas importantes de la region y su rambla recorre toda la costa de la ciudad.",
    "Un transformador es una arquitectura de red neuronal que usa el mecanismo de atencion para relacionar las palabras de una frase entre si. Cada capa tiene atencion multi-cabeza y una red feedforward, y se apilan muchas capas para formar modelos de lenguaje grandes.",
    "La fotosintesis es el proceso por el cual las plantas convierten la luz del sol, el agua y el dioxido de carbono en oxigeno y glucosa. Este proceso ocurre en los cloroplastos y es la base de casi toda la vida en la Tierra, porque produce el oxigeno que respiramos.",
]
TEXTO_PPL_WIKI = None  # articulo de Wikipedia no visto en train, para la PPL


def cargar_wikipedia():
    """Devuelve (secuencias_ids, n_articulos, n_tokens): listas de ids de SEQ tokens."""
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
        # articulo 6 queda aparte para PPL: no entra en train, nunca se ve
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
    log(f"[AVISO] no se pudo bajar Wikipedia ({e}), uso los 4 textos base (fallback)")
    SEQS_IDS = []
    i = 0
    while len(SEQS_IDS) < N_SEQ_CAPTURA:
        ids = tok(TEXTOS_BASE[i % len(TEXTOS_BASE)], add_special_tokens=False)["input_ids"][:SEQ]
        SEQS_IDS.append(ids)
        i += 1
    N_ART, N_TOK = 0, len(SEQS_IDS) * SEQ
if TEXTO_PPL_WIKI is None:
    TEXTO_PPL_WIKI = TEXTOS_BASE[1]

# --- captura real: pasar cada secuencia UNA vez, sin reemplazo, seq 1024 ---
log(f"Capturando activaciones REALES: {len(SEQS_IDS)} seq x {SEQ} tokens...")
capturados = []


def hook_fn(module, inp, out):
    capturados.append(inp[0].detach().float().cpu())


handle = model.model.layers[LAYER].mlp.register_forward_hook(hook_fn)
with torch.no_grad():
    for j, ids in enumerate(SEQS_IDS):
        ids_t = torch.tensor([ids], device=device)
        model(input_ids=ids_t)
        if (j + 1) % 50 == 0:
            log(f"  captura {j + 1}/{len(SEQS_IDS)}")
handle.remove()
X = torch.cat([t.reshape(-1, HIDDEN) for t in capturados], dim=0)
log(f"Activaciones reales: {X.shape} vectores")
if X.shape[0] < 5000:
    fail(f"muy pocos vectores ({X.shape[0]}), el train va a memorizar. Subi N_SEQ_CAPTURA.")
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
        # Al generar, Qwen pasa hidden_states 3D (batch, seq, hidden);
        # en train es 2D (N, hidden). Aplanar a 2D y volver a la forma original.
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
log(f"MoE params totales: {sum(p.numel() for p in moe.parameters()):,} "
    f"(expertos {N_EXPERTS * 3 * INTER_E * HIDDEN:,} + router {moe.router_w.numel() + moe.router_b.numel():,})")


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
log(f"[eval inicial] cos {cos0:.4f} R2 {r20:.4f} NMSE {nmse0:.4f} lb {lb0:.4f} frac {['%.2f' % v for v in f0]}")

opt = torch.optim.Adam(moe.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
vec_por_paso = BATCH * SEQ
t0 = time.time()
# epocas reales: shuffle sin reemplazo, cada vector ~1 vez por epoca
n_train = X_train.shape[0]
order = torch.randperm(n_train, generator=torch.Generator().manual_seed(SEED))
pos = 0
for step in range(1, STEPS + 1):
    if pos + vec_por_paso > n_train:  # nueva epoca, re-shuffle
        order = torch.randperm(n_train, generator=torch.Generator().manual_seed(SEED + step))
        pos = 0
    xb = X_train[order[pos:pos + vec_por_paso]].to(device)
    pos += vec_por_paso
    with torch.no_grad():
        yb = swiglu(xb, teacher.gate_proj.weight.float(), teacher.up_proj.weight.float(),
                    teacher.down_proj.weight.float())
    opt.zero_grad()
    loss_train = 0.0
    # ARREGLO nan: xb puede ser mas chico que vec_por_paso (train 8112 < 65536),
    # antes el loop llegaba a micro-batches VACIOS y mean() de vacio = nan.
    # Ahora se itera solo sobre los vectores reales de xb.
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
        log(f"paso {step}/{STEPS} loss_train {loss_train:.4f} | EVAL cos {cos:.4f} R2 {r2:.4f} NMSE {nmse:.4f} "
            f"lb {lb:.4f} frac {['%.2f' % v for v in f]} | {time.time() - t0:.0f}s")

cos, r2, nmse, lb, f = eval_moe()
ok = cos >= TARGET_COS and r2 > TARGET_R2
log(f"[eval final] cos {cos:.4f} R2 {r2:.4f} NMSE {nmse:.4f} lb {lb:.4f} | meta cos>={TARGET_COS} y R2>{TARGET_R2}: {'SI' if ok else 'NO'}")
log(f"tiempo total: {time.time() - t0:.0f}s")

# ------------------------------------------------- 8. inferencia: Qwen normal vs Qwen con MoE
# Misma dim (hidden 896), mismas KV y mismas lineales al generar:
# solo se reemplaza la MLP de la capa LAYER por la MoE (entrada 896 -> salida 896).
# Atencion, KV cache, embeddings, lm_head y el resto de capas NO se tocan.
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


# wrapper drop-in: misma firma que la MLP original (x -> y), sin tocar nada mas
moe.eval()


class MoEWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        # nombres que el modelo espera en una MLP, para que nada falle al generar
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None

    def forward(self, x):
        y, _, _ = self.m(x)
        return y


mlp_original = model.model.layers[LAYER].mlp
comparaciones = []
for PROMPT_GEN in PROMPTS:
    with torch.no_grad():
        texto_normal = generar(model, PROMPT_GEN)
    model.model.layers[LAYER].mlp = MoEWrapper(moe).to(device)
    with torch.no_grad():
        texto_moe = generar(model, PROMPT_GEN)
    model.model.layers[LAYER].mlp = mlp_original  # restaurar
    comparaciones.append((PROMPT_GEN, texto_normal, texto_moe))
    log(f"--- prompt: {PROMPT_GEN!r}")
    log(f"[Qwen normal] {texto_normal!r}")
    log(f"[Qwen con MoE] {texto_moe!r}")
    log(f"Textos iguales: {'SI' if texto_normal == texto_moe else 'NO'}")
# PPL en un articulo de Wikipedia NO visto en train (mismo texto, mismo modelo,
# solo cambia la MLP de la capa)
TEXTO_PPL = TEXTO_PPL_WIKI


def ppl(modelo):
    ids = tok(TEXTO_PPL, return_tensors="pt").to(device)
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
            "batch": BATCH, "seq": SEQ, "steps": STEPS, "lr": LR, "seed": SEED,
            "n_tokens_wiki": N_TOK, "n_articulos_wiki": N_ART},
           f"moe_destilada_qwen25_capa{LAYER}.pt")
with open("resultados-colab-moe.md", "w") as fo:
    fo.write(f"# MoE FF{LAYER} Qwen2.5-0.5B en Colab\n\n"
             f"- Datos: Wikipedia ES, {N_ART} articulos, {N_TOK} tokens, {len(SEQS_IDS)} seq x {SEQ}\n"
             f"- 4 expertos inter {INTER_E}, total {N_EXPERTS * 3 * INTER_E * HIDDEN:,} params = profesora\n"
             f"- Batch {BATCH} seq {SEQ}, steps {STEPS}, lr {LR}, train {n_train} vectores / eval {X_eval.shape[0]}\n"
             f"- Eval real: cos {cos:.4f}, R2 {r2:.4f}, NMSE {nmse:.4f}, lb {lb:.4f}\n"
             f"- Meta cos>={TARGET_COS} y R2>{TARGET_R2}: {'SI' if ok else 'NO'}\n\n"
             f"## Inferencia ({GEN_TOKENS} tokens, greedy, misma KV)\n\n")
    for pr, tn, tm in comparaciones:
        fo.write(f"### Prompt: {pr!r}\n\n- Qwen normal: {tn!r}\n- Qwen con MoE: {tm!r}\n"
                 f"- Textos iguales: {'SI' if tn == tm else 'NO'}\n\n")
    fo.write(f"- PPL Wikipedia (no vista) normal {ppl_normal:.2f} | PPL con MoE {ppl_moe:.2f} | ratio {ppl_moe / ppl_normal:.3f}\n")
log("Guardado: moe_destilada_qwen25_capa%d.pt, resultados-colab-moe.md, colab_destilar_moe.log" % LAYER)
