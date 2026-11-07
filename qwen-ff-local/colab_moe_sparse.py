#!/usr/bin/env python3
"""MoE SPARSE estilo D2DMoE, capa 0 primero, Qwen2.5-0.5B, solo run en Colab.

Paper: Szatkowski et al., "Exploiting Activation Sparsity with Dense to
Dynamic-k MoE Conversion" (D2DMoE), arXiv:2310.04361, codigo
https://github.com/bartwojcik/D2DMoE (bajado a ~/workspace/d2dmoe).

Que hace D2DMoE, adaptado a Qwen (SwiGLU, no GELU):
  1. ESCASEZ: en el repo, `dsti_clamp_displacement = -10` en
     scripts/d2dmoe_gpt2_combined.py:224. Las pre-activaciones de GELU se
     clampean en -10 y se les aplica perdida Hoyer para forzar escasez.
     Aca: pre-activacion = salida de gate_proj (antes de SiLU). SiLU(x) ~ 0
     cuando x es muy negativo, asi que clamplear en -10 y Hoyer empuja a
     que muchas neuronas queden apagadas.
  2. EXPERTOS POR CLUSTERING (no init de origen): en
     architectures/moe/moefication.py:219,230, para Gemma (gate/up/down,
     como Qwen) agrupan por `gate_proj` "because they are actually sparse",
     con KMeansConstrained, copiando los pesos de la profesora.
     Aca: KMeans sobre las filas de gate_proj normalizadas, 4 expertos
     x 1216 neuronas, copiando gate/up/down de la profesora segun el cluster.
  3. ROUTER POR REGRESION + DYNAMIC-K: el router no es top-k fijo, predice
     la contribucion (norma) de cada experto. En inferencia se ordenan los
     expertos por contribucion y se activan hasta acumular tau (0.9).
  4. FINE-TUNING sobre entrada/salida de la FF maestra, como antes:
     con los vectores reales X (entrada a la FF0, post-RMSNorm) e Y
     (salida de la profesora), se entrena la MoE + router con
     perdida BLD NMSE + 0.1*(1-cos) + 0.01*lb + 0.1*Hoyer_clamped(-10).

Si, hay que pasar tokens por el modelo base: es lo que ya hacemos.
Se pasan 500k tokens de Wikipedia ES por Qwen, se capturan X (entrada)
y G (pre-activacion gate), y con eso se hace todo lo demas. Sin tokens
no hay vectores reales, no hay clustering ni router.

Uso en Colab (una celda):
    !git clone --depth 1 -b fix/ff-eval-real-activations https://github.com/eduardo-bertey/LLM-MOHO.git
    %cd LLM-MOHO/qwen-ff-local
    !python3 colab_moe_sparse.py

Ajustes (variables de entorno):
  LAYER=0  BATCH=56  SEQ=1024  STEPS=300  LR=5e-4  SEED=123
  N_TOKENS_CAPTURA=500000  CLAMP=-10  TAU=0.9  W_HOYER=0.1
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
STEPS = int(os.environ.get("STEPS", "300"))
LR = float(os.environ.get("LR", "5e-4"))
SEED = int(os.environ.get("SEED", "123"))
N_TOKENS_CAPTURA = int(os.environ.get("N_TOKENS_CAPTURA", "500000"))
CLAMP = float(os.environ.get("CLAMP", "-10"))
TAU = float(os.environ.get("TAU", "0.9"))
W_HOYER = float(os.environ.get("W_HOYER", "0.1"))
MICRO = int(os.environ.get("MICRO", "1024"))
W_LB = float(os.environ.get("W_LB", "0.01"))
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_DIR = os.environ.get("MODEL_DIR", "./modelo-qwen2.5-0.5b")
N_EXPERTS = 4
TARGET_COS, TARGET_R2 = 0.95, 0.90

LOG = open("colab_moe_sparse.log", "a", buffering=1)


def log(m):
    print(m, flush=True)
    LOG.write(m + "\n")


def fail(m):
    log(f"[ERROR] {m}")
    sys.exit(1)


log("=" * 60)
log(f"MoE SPARSE (D2DMoE) FF{LAYER} Qwen2.5-0.5B | clamp {CLAMP}, tau dyn-k {TAU}")
log(f"batch {BATCH} seq {SEQ} steps {STEPS} lr {LR} captura {N_TOKENS_CAPTURA} tokens wiki")

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

# --- captura: X = entrada a la FF, G = pre-activacion gate (para escasez) ---
# Si, hay que pasar los tokens por el modelo base: sin eso no hay vectores.
log(f"Pasando tokens por Qwen base y capturando FF{LAYER}: {len(SEQS_IDS)} seq x {SEQ} tokens...")
capturados_x, capturados_g = [], []


def hook_fn(module, inp, out):
    capturados_x.append(inp[0].detach().float().cpu())


# hook en gate_proj para la pre-activacion (salida de gate, antes de SiLU)
def hook_gate(module, inp, out):
    capturados_g.append(out.detach().float().cpu())


teacher = model.model.layers[LAYER].mlp
h1 = teacher.register_forward_hook(hook_fn)
h2 = teacher.gate_proj.register_forward_hook(hook_gate)
with torch.no_grad():
    for j, ids in enumerate(SEQS_IDS):
        model(input_ids=torch.tensor([ids], device=device))
        if (j + 1) % 50 == 0:
            log(f"  captura {j + 1}/{len(SEQS_IDS)}")
h1.remove()
h2.remove()
X = torch.cat([t.reshape(-1, HIDDEN) for t in capturados_x], dim=0)
G = torch.cat([t.reshape(-1, INTER) for t in capturados_g], dim=0)
del capturados_x, capturados_g
log(f"Activaciones reales: X {X.shape} vectores, G pre-activacion {G.shape}")
# escasez real de la capa: % de neuronas gate con pre-activacion < CLAMP (apagadas)
frac_apagadas = (G < CLAMP).float().mean().item()
frac_silu_casi0 = (torch.nn.functional.silu(G).abs() < 0.01).float().mean().item()
log(f"Escasez capa {LAYER}: gate < {CLAMP}: {frac_apagadas * 100:.1f}% | SiLU(gate) ~0: {frac_silu_casi0 * 100:.1f}%")
if X.shape[0] < 5000:
    fail(f"muy pocos vectores ({X.shape[0]})")
perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(SEED))
n_eval = max(512, X.shape[0] // 5)
X_eval, X_train = X[perm[:n_eval]], X[perm[n_eval:]]
G_eval = G[perm[:n_eval]]
del G
log(f"train {X_train.shape[0]} vectores / eval {X_eval.shape[0]} vectores (eval nunca se entrena)")
for p in teacher.parameters():
    p.requires_grad_(False)


def swiglu(x, gate, up, down):
    return torch.nn.functional.silu(x @ gate.T) * (x @ up.T) @ down.T


def hoyer_clamped(preact, clamp=CLAMP):
    # D2DMoE: clamped = min(preact, clamp) - clamp ; Hoyer = (sum|x|)^2 / sum(x^2)
    c = torch.clamp(preact, max=clamp) - clamp
    return ((c.abs().sum(dim=-1) ** 2) / (c.pow(2).sum(dim=-1) + 1e-15)).mean()


# --- 2. expertos por KMeans sobre gate_proj (copiando pesos de la profesora) ---
log("Clustering gate_proj con KMeans en 4 expertos (copiando pesos de la profesora)...")
try:
    from sklearn.cluster import KMeans
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"])
    from sklearn.cluster import KMeans
gate_w = teacher.gate_proj.weight.float().detach().cpu().numpy()  # (4864, 896)
gate_norm = gate_w / (np.linalg.norm(gate_w, axis=1, keepdims=True) + 1e-12)
# KMeans balanceado a mano: clusters de 1216 neuronas cada uno
km = KMeans(n_clusters=N_EXPERTS, n_init=4, random_state=SEED).fit(gate_norm)
labels = km.labels_
# balancear: si un cluster quedo desparejo, reasignar las neuronas mas lejanas
counts = np.bincount(labels, minlength=N_EXPERTS)
log(f"  tamanos de cluster KMeans: {counts.tolist()} (ideal {INTER_E} cada uno)")
# reasignacion simple por distancia al centroide, con tope de 1216 por experto
dists = km.transform(gate_norm)  # (4864, 4)
order_neuronas = np.argsort(dists.min(axis=1))  # las mas seguras primero
labels_bal = np.full(INTER * 1, -1, dtype=int)
ocup = np.zeros(N_EXPERTS, dtype=int)
for n_idx in order_neuronas:
    # experto mas cercano con lugar libre
    for e_idx in np.argsort(dists[n_idx]):
        if ocup[e_idx] < INTER_E:
            labels_bal[n_idx] = e_idx
            ocup[e_idx] += 1
            break
log(f"  tamanos balanceados: {ocup.tolist()}")


class MoE(torch.nn.Module):
    def __init__(self, labels_bal):
        super().__init__()
        self.experts = torch.nn.ModuleList()
        for e in range(N_EXPERTS):
            idx = torch.tensor(np.where(labels_bal == e)[0])
            m = torch.nn.Module()
            # copiar pesos de la profesora segun el cluster (D2DMoE)
            m.gate = torch.nn.Parameter(teacher.gate_proj.weight.float()[idx].clone())
            m.up = torch.nn.Parameter(teacher.up_proj.weight.float()[idx].clone())
            m.down = torch.nn.Parameter(teacher.down_proj.weight.float()[:, idx].clone())
            self.experts.append(m)
        # router por regresion: predice la norma de contribucion de cada experto
        self.router = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN, 128), torch.nn.Tanh(),
            torch.nn.Linear(128, N_EXPERTS))
        # init router chico para no romper al principio
        with torch.no_grad():
            self.router[2].weight.mul_(0.01)
            self.router[2].bias.zero_()

    def expert_norms(self, x):
        # norma de salida de cada experto (para entrenar el router por regresion)
        norms = []
        for e in range(N_EXPERTS):
            ye = swiglu(x, self.experts[e].gate, self.experts[e].up, self.experts[e].down)
            norms.append(ye.norm(dim=-1))
        return torch.stack(norms, dim=-1)  # (N, 4)

    def forward(self, x, tau=TAU):
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        pred = self.router(x)  # (N, 4) contribucion predicha (puede ser negativa, se usa softplus)
        pred_pos = torch.nn.functional.softplus(pred)
        shares = pred_pos / (pred_pos.sum(dim=-1, keepdim=True) + 1e-12)
        # dynamic-k: ordenar por share, activar hasta acumular tau
        sorted_shares, sorted_idx = torch.sort(shares, dim=-1, descending=True)
        cum = torch.cumsum(sorted_shares, dim=-1)
        mask_sorted = torch.ones_like(cum, dtype=torch.bool)
        mask_sorted[:, 1:] = cum[:, :-1] < tau
        mask = torch.zeros_like(shares, dtype=torch.bool)
        mask.scatter_(1, sorted_idx, mask_sorted)
        # pesos: shares renormalizados sobre los activos
        w = torch.where(mask, shares, torch.zeros_like(shares))
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-12)
        y = torch.zeros_like(x)
        for e in range(N_EXPERTS):
            rows = mask[:, e]
            if rows.sum() == 0:
                continue
            ye = swiglu(x[rows], self.experts[e].gate, self.experts[e].up, self.experts[e].down)
            y[rows] += ye * w[rows, e][:, None]
        # balanceo: fraccion de tokens por experto (para log, no para forzar top-k)
        f = mask.float().mean(0)
        lb = N_EXPERTS * (f * shares.mean(0)).sum()
        return y.reshape(orig_shape), lb, f, mask.float().mean().item()


moe = MoE(labels_bal).to(device)
log(f"MoE SPARSE params totales: {sum(p.numel() for p in moe.parameters()):,} "
    f"(expertos {N_EXPERTS * 3 * INTER_E * HIDDEN:,} + router {sum(p.numel() for p in moe.router.parameters()):,})")


def eval_moe():
    with torch.no_grad():
        y_t = swiglu(X_eval.to(device), teacher.gate_proj.weight.float(),
                     teacher.up_proj.weight.float(), teacher.down_proj.weight.float())
        y, lb, f, k_medio = moe(X_eval.to(device))
        cos = torch.nn.functional.cosine_similarity(y, y_t, dim=-1).mean().item()
        r2 = 1 - ((y - y_t) ** 2).sum().item() / max(((y_t - y_t.mean(0)) ** 2).sum().item(), 1e-12)
        nmse = (((y - y_t) ** 2).mean() / (y_t ** 2).mean()).item()
    return cos, r2, nmse, lb.item(), f.cpu().tolist(), k_medio


cos0, r20, nmse0, lb0, f0, k0 = eval_moe()
log(f"[eval inicial, expertos clusterizados sin fine-tuning] cos {cos0:.4f} R2 {r20:.4f} NMSE {nmse0:.4f} "
    f"k_medio {k0:.2f} expertos/token frac {['%.2f' % v for v in f0]}")

# --- 4. fine-tuning sobre entrada/salida de la FF maestra, como antes ---
opt = torch.optim.Adam(moe.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
vec_por_paso = BATCH * SEQ
t0 = time.time()
n_train = X_train.shape[0]
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
        # contribucion real de cada experto (para entrenar el router por regresion)
        with torch.no_grad():
            pass
    opt.zero_grad()
    loss_train = 0.0
    n_xb = xb.shape[0]
    n_micro = (n_xb + MICRO - 1) // MICRO
    for i in range(0, n_xb, MICRO):
        x_m, y_m = xb[i:i + MICRO], yb[i:i + MICRO]
        y_s, lb, _, _ = moe(x_m)
        nmse = ((y_s - y_m) ** 2).mean() / (y_m ** 2).mean()
        cos = torch.nn.functional.cosine_similarity(y_s, y_m, dim=-1).mean()
        # Hoyer con clamp -10 sobre la pre-activacion gate de la MoE (primer experto, muestra)
        gate_pre = x_m @ moe.experts[0].gate.T
        h_loss = hoyer_clamped(gate_pre)
        # router por regresion: predecir la norma real de cada experto
        with torch.no_grad():
            target_norms = moe.expert_norms(x_m)
            target_shares = target_norms / (target_norms.sum(dim=-1, keepdim=True) + 1e-12)
        pred = torch.nn.functional.softplus(moe.router(x_m))
        pred_shares = pred / (pred.sum(dim=-1, keepdim=True) + 1e-12)
        router_loss = ((pred_shares - target_shares) ** 2).mean()
        loss = (nmse + 0.1 * (1 - cos) + W_LB * lb + W_HOYER * h_loss + router_loss) / n_micro
        loss.backward()
        loss_train += loss.item() * n_micro
    opt.step()
    if step % 20 == 0 or step == 1:
        cos, r2, nmse, lb, f, k_medio = eval_moe()
        log(f"paso {step}/{STEPS} loss_train {loss_train:.4f} | EVAL cos {cos:.4f} R2 {r2:.4f} NMSE {nmse:.4f} "
            f"lb {lb:.4f} k_medio {k_medio:.2f} frac {['%.2f' % v for v in f]} | {time.time() - t0:.0f}s")

cos, r2, nmse, lb, f, k_medio = eval_moe()
ok = cos >= TARGET_COS and r2 > TARGET_R2
log(f"[eval final] cos {cos:.4f} R2 {r2:.4f} NMSE {nmse:.4f} lb {lb:.4f} k_medio {k_medio:.2f} | meta: {'SI' if ok else 'NO'}")
log(f"tiempo total: {time.time() - t0:.0f}s")

# --- inferencia: Qwen normal vs Qwen con MoE SPARSE en la capa ---
log("=" * 60)
log("Inferencia: Qwen normal vs Qwen con MoE SPARSE en la capa %d (mismas KV, mismas lineales)" % LAYER)
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
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None

    def forward(self, x):
        y, _, _, _ = self.m(x)
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
    log(f"[Qwen con MoE SPARSE] {texto_moe!r}")
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
log(f"PPL Wikipedia (no vista) normal {ppl_normal:.2f} | PPL con MoE SPARSE {ppl_moe:.2f} | ratio {ppl_moe / ppl_normal:.3f}")
torch.save({"state_dict": moe.state_dict(), "layer": LAYER, "inter_expert": INTER_E,
            "n_experts": N_EXPERTS, "tau": TAU, "clamp": CLAMP, "cos": cos, "r2": r2, "nmse": nmse,
            "k_medio": k_medio, "frac_apagadas": frac_apagadas, "batch": BATCH, "seq": SEQ,
            "steps": STEPS, "lr": LR, "seed": SEED},
           f"moe_sparse_qwen25_capa{LAYER}.pt")
with open("resultados-colab-moe-sparse.md", "w") as fo:
    fo.write(f"# MoE SPARSE (D2DMoE) FF{LAYER} Qwen2.5-0.5B en Colab\n\n"
             f"- Datos: Wikipedia ES, {N_ART} articulos, {N_TOK} tokens\n"
             f"- Escasez gate < {CLAMP}: {frac_apagadas * 100:.1f}% | SiLU ~0: {frac_silu_casi0 * 100:.1f}%\n"
             f"- 4 expertos inter {INTER_E} por KMeans sobre gate_proj, copiando pesos profesora\n"
             f"- Router por regresion + dynamic-k tau {TAU}, k_medio {k_medio:.2f} expertos/token\n"
             f"- Batch {BATCH} seq {SEQ}, steps {STEPS}, lr {LR}, train {n_train} vectores / eval {X_eval.shape[0]}\n"
             f"- Eval real: cos {cos:.4f}, R2 {r2:.4f}, NMSE {nmse:.4f}, lb {lb:.4f}\n"
             f"- Meta cos>={TARGET_COS} y R2>{TARGET_R2}: {'SI' if ok else 'NO'}\n\n"
             f"## Inferencia\n\n")
    for pr, tn, tm in comparaciones:
        fo.write(f"### Prompt: {pr!r}\n\n- Qwen normal: {tn!r}\n- Qwen con MoE SPARSE: {tm!r}\n\n")
    fo.write(f"- PPL Wikipedia (no vista) normal {ppl_normal:.2f} | PPL con MoE SPARSE {ppl_moe:.2f} | ratio {ppl_moe / ppl_normal:.3f}\n")
log(f"Guardado: moe_sparse_qwen25_capa{LAYER}.pt, resultados-colab-moe-sparse.md, colab_moe_sparse.log")
