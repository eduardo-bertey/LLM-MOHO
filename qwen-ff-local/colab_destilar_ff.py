#!/usr/bin/env python3
"""Destilar la FF de Qwen2.5-0.5B, solo run en Colab.

Uso en Colab (una celda, nada mas):
    !git clone --depth 1 -b fix/ff-eval-real-activations https://github.com/eduardo-bertey/LLM-MOHO.git
    %cd LLM-MOHO/qwen-ff-local
    !python3 colab_destilar_ff.py

Que hace, en orden:
  1. Verifica entorno: GPU, torch, transformers, RAM.
  2. Verifica si el modelo Qwen2.5-0.5B ya esta en ./modelo-qwen2.5-0.5b
     (o en HF cache); si no esta, lo descarga de Qwen/Qwen2.5-0.5B.
  3. Selecciona la FF a destilar (LAYER, por defecto 0) y captura
     activaciones REALES: pasa textos en espanol por Qwen y guarda la
     entrada real a la MLP (post-RMSNorm de la capa), nunca ruido gaussiano.
  4. Ajusta la destilacion: alumna SwiGLU al 75% (inter 4864 -> 3648),
     init de origen (randn * 1/sqrt(fan_in), seed 123), SIN copiar pesos
     de la profesora. Perdida BLD: NMSE + 0.1 * (1 - cos).
  5. Entrena con batch 32, seq 512 para empezar (16384 vectores/paso,
     en micro-batches de 1024 para no reventar la GPU).
  6. Evalua en real: cos / R2 / NMSE. Meta: cos >= 0.95 y R2 > 0.90.
  7. Guarda checkpoint, log y resultados. No toca main, no sube nada a git.

Ajustes rapidos (variables de entorno en Colab):
  LAYER=0  BATCH=32  SEQ=512  STEPS=300  LR=5e-4  REDUCCION=0.75  SEED=123
  Ej: !LAYER=12 STEPS=200 python3 colab_destilar_ff.py
"""
import json
import math
import os
import sys
import time

# ---------------------------------------------------------------- ajustes
LAYER = int(os.environ.get("LAYER", "0"))
BATCH = int(os.environ.get("BATCH", "64"))      # secuencias por paso
SEQ = int(os.environ.get("SEQ", "1024"))         # tokens por secuencia
STEPS = int(os.environ.get("STEPS", "300"))
LR = float(os.environ.get("LR", "5e-4"))
REDUCCION = float(os.environ.get("REDUCCION", "0.75"))  # 0.75 = inter 3648
SEED = int(os.environ.get("SEED", "123"))
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_DIR = os.environ.get("MODEL_DIR", "./modelo-qwen2.5-0.5b")
MICRO = int(os.environ.get("MICRO", "1024"))   # vectores por micro-batch
TARGET_COS = 0.95
TARGET_R2 = 0.90

TEXTOS = [
    "El agua hierve a cien grados y el sol sale por el este cada manana.",
    "La destilacion de una red neuronal consiste en que un modelo alumno aprenda a imitar a un modelo profesor.",
    "Montevideo es la capital de Uruguay y esta sobre el Rio de la Plata.",
    "Un transformador usa atencion para relacionar las palabras de una frase entre si.",
    "La fotosintesis permite a las plantas convertir la luz del sol en energia quimica.",
    "En invierno el frio hace que la gente se abrigue mas y tome bebidas calientes.",
]

LOG = open("colab_destilar_ff.log", "a", buffering=1)


def log(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n")


def fail(msg):
    log(f"[ERROR] {msg}")
    sys.exit(1)


# ---------------------------------------------------------------- 1. entorno
log("=" * 60)
log(f"Destilar FF{LAYER} Qwen2.5-0.5B | batch {BATCH} seq {SEQ} steps {STEPS} lr {LR} reduccion {REDUCCION}")
try:
    import torch
    import numpy as np
except ImportError as e:
    fail(f"falta torch/numpy en Colab: {e}. En Colab veni instalado; si no, !pip install torch transformers datasets")

log(f"torch {torch.__version__} | cuda disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    device = "cuda"
else:
    log("AVISO: sin GPU, va a ser muy lento. En Colab: Entorno > Cambiar tipo de entorno > GPU (T4).")
    device = "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------- 2. modelo: verifica / descarga
from transformers import AutoModelForCausalLM, AutoTokenizer

if os.path.isdir(MODEL_DIR) and os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
    log(f"Modelo YA ESTA en {MODEL_DIR}, no se descarga.")
    src = MODEL_DIR
else:
    log(f"Modelo NO esta en {MODEL_DIR}, se descarga de {MODEL_ID} (~1 GB)...")
    src = MODEL_ID

tok = AutoTokenizer.from_pretrained(src, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    src, torch_dtype=torch.float32, trust_remote_code=False).to(device)
model.eval()
cfg = model.config
HIDDEN, INTER, N_LAYERS = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
log(f"Modelo cargado: hidden {HIDDEN}, inter {INTER}, capas {N_LAYERS}, vocab {cfg.vocab_size}")
if not (0 <= LAYER < N_LAYERS):
    fail(f"LAYER={LAYER} fuera de rango (0..{N_LAYERS - 1})")
INTER_STUDENT = int(INTER * REDUCCION)
log(f"FF a destilar: capa {LAYER} | profesora inter {INTER} -> alumna inter {INTER_STUDENT} ({REDUCCION * 100:.0f}%)")

# ------------------------------------------- 3. activaciones reales (no gauss)
log("Capturando activaciones REALES de la capa %d..." % LAYER)
capturados = []


def hook_fn(module, inp, out):
    # entrada real a la MLP: post-RMSNorm, antes de gate/up
    capturados.append(inp[0].detach().float().cpu())


handle = model.model.layers[LAYER].mlp.register_forward_hook(hook_fn)
with torch.no_grad():
    for txt in TEXTOS:
        ids = tok(txt, return_tensors="pt").to(device)
        model(**ids)
handle.remove()
X = torch.cat([t.reshape(-1, HIDDEN) for t in capturados], dim=0)
log(f"Activaciones reales capturadas: {X.shape} vectores (solo entradas, sin ruido)")
# split fijo train/eval 80/20
perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(SEED))
n_eval = max(64, X.shape[0] // 5)
X_eval, X_train = X[perm[:n_eval]], X[perm[n_eval:]]
log(f"train {X_train.shape[0]} vectores / eval {X_eval.shape[0]} vectores")

# ------------------------------------------------- 4. profesora y alumna
teacher = model.model.layers[LAYER].mlp
for p in teacher.parameters():
    p.requires_grad_(False)


def swiglu(x, gate, up, down):
    return torch.nn.functional.silu(x @ gate.T) * (x @ up.T) @ down.T


class Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # init de origen: randn * 1/sqrt(fan_in), SIN copiar pesos de la profesora
        def init(shape, fan_in):
            w = torch.empty(shape)
            torch.nn.init.normal_(w, mean=0.0, std=1.0 / math.sqrt(fan_in))
            return torch.nn.Parameter(w)
        self.gate = init((INTER_STUDENT, HIDDEN), HIDDEN)
        self.up = init((INTER_STUDENT, HIDDEN), HIDDEN)
        self.down = init((HIDDEN, INTER_STUDENT), INTER_STUDENT)

    def forward(self, x):
        return swiglu(x, self.gate, self.up, self.down)


student = Student().to(device)
n_params = sum(p.numel() for p in student.parameters())
log(f"Alumna: {n_params:,} params (profesora: {sum(p.numel() for p in teacher.parameters()):,})")


def metrics(x, y_teacher):
    with torch.no_grad():
        y = student(x)
        cos = torch.nn.functional.cosine_similarity(y, y_teacher, dim=-1).mean().item()
        ss_res = ((y - y_teacher) ** 2).sum().item()
        ss_tot = ((y_teacher - y_teacher.mean(0)) ** 2).sum().item()
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        nmse = (((y - y_teacher) ** 2).mean() / (y_teacher ** 2).mean()).item()
    return cos, r2, nmse


with torch.no_grad():
    Y_eval = swiglu(X_eval.to(device),
                    teacher.gate_proj.weight.float(), teacher.up_proj.weight.float(),
                    teacher.down_proj.weight.float())
cos0, r20, nmse0 = metrics(X_eval.to(device), Y_eval)
log(f"[eval inicial] cos {cos0:.4f} R2 {r20:.4f} NMSE {nmse0:.4f} (init de origen, sin entrenar)")

# ------------------------------------------------------------- 5. entrenar
opt = torch.optim.Adam(student.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
vec_por_paso = BATCH * SEQ
log(f"Entrenando: {STEPS} pasos x {vec_por_paso} vectores/paso (micro-batch {MICRO})")
t0 = time.time()
for step in range(1, STEPS + 1):
    # batch 32 x seq 512 = 16384 vectores, muestreados con reemplazo de train
    idx = torch.randint(0, X_train.shape[0], (vec_por_paso,), generator=torch.Generator().manual_seed(SEED + step))
    xb = X_train[idx].to(device)
    with torch.no_grad():
        yb = swiglu(xb, teacher.gate_proj.weight.float(), teacher.up_proj.weight.float(),
                    teacher.down_proj.weight.float())
    opt.zero_grad()
    loss_total = 0.0
    for i in range(0, vec_por_paso, MICRO):
        x_m, y_m = xb[i:i + MICRO], yb[i:i + MICRO]
        y_s = student(x_m)
        nmse = ((y_s - y_m) ** 2).mean() / (y_m ** 2).mean()
        cos = torch.nn.functional.cosine_similarity(y_s, y_m, dim=-1).mean()
        loss = nmse + 0.1 * (1 - cos)
        (loss / (vec_por_paso // MICRO)).backward()
        loss_total += loss.item()
    opt.step()
    if step % 20 == 0 or step == 1:
        cos, r2, nmse = metrics(X_eval.to(device), Y_eval)
        log(f"paso {step}/{STEPS} loss {loss_total:.4f} | eval cos {cos:.4f} R2 {r2:.4f} NMSE {nmse:.4f} | {time.time() - t0:.0f}s")

# ---------------------------------------------------------------- 6. eval final
cos, r2, nmse = metrics(X_eval.to(device), Y_eval)
ok = cos >= TARGET_COS and r2 > TARGET_R2
log(f"[eval final] cos {cos:.4f} R2 {r2:.4f} NMSE {nmse:.4f} | meta cos>={TARGET_COS} y R2>{TARGET_R2}: {'SI' if ok else 'NO'}")
log(f"tiempo total: {time.time() - t0:.0f}s")

# ---------------------------------------------------------------- 7. guardar
torch.save({"state_dict": student.state_dict(), "layer": LAYER, "inter": INTER_STUDENT,
            "hidden": HIDDEN, "cos": cos, "r2": r2, "nmse": nmse,
            "batch": BATCH, "seq": SEQ, "steps": STEPS, "lr": LR, "seed": SEED},
           f"ff_destilada_qwen25_capa{LAYER}.pt")
with open("resultados-colab-ff.md", "w") as f:
    f.write(f"# Destilacion FF{LAYER} Qwen2.5-0.5B en Colab\n\n"
            f"- Alumna inter {INTER} -> {INTER_STUDENT} ({REDUCCION * 100:.0f}%), init origen, sin copia de profesora\n"
            f"- Batch {BATCH} seq {SEQ} ({vec_por_paso} vectores/paso), steps {STEPS}, lr {LR}, seed {SEED}\n"
            f"- Eval real ({X_eval.shape[0]} vectores): cos {cos:.4f}, R2 {r2:.4f}, NMSE {nmse:.4f}\n"
            f"- Meta cos>={TARGET_COS} y R2>{TARGET_R2}: {'SI' if ok else 'NO'}\n")
log("Guardado: ff_destilada_qwen25_capa%d.pt, resultados-colab-ff.md, colab_destilar_ff.log" % LAYER)
log("Listo. Bajate los archivos desde Colab (Files > Download) o guardalos en Drive.")
