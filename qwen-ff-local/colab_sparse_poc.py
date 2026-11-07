#!/usr/bin/env python3
"""Prueba de concepto SPARSE, capa 0, Qwen2.5-0.5B, Colab.
Ocupa muy poco: solo 10 secuencias x 512 tokens = 5120 tokens,
sin entrenar nada, solo medir.

Que mide:
  - Escasez de la pre-activacion gate (antes de SiLU) a diferentes
    escalas / umbrales: -10, -5, -2, -1, 0.
    % de neuronas apagadas en cada umbral, por token y por secuencia.
  - Diversidad: ¿las mismas neuronas se apagan siempre, o cambia
    segun el texto? Se mide por secuencia: que neuronas estan activas
    en cada texto, y el solapamiento entre textos (Jaccard).
    Si siempre son las mismas, no hay diversidad para hacer expertos.
    Si cambia mucho, si hay diversidad y vale la pena clusterizar.
  - Clustering rapido: KMeans sobre gate_proj (4 expertos) y que tan
    separados quedan los clusters (silhouette aproximada por distancia
    intra vs inter cluster).

Uso en Colab (una celda):
    !python3 colab_sparse_poc.py

Ajustes (variables de entorno):
  LAYER=0  N_SEQ=10  SEQ=512  SEED=123
"""
import math
import os
import sys
import time

LAYER = int(os.environ.get("LAYER", "0"))
N_SEQ = int(os.environ.get("N_SEQ", "10"))
SEQ = int(os.environ.get("SEQ", "512"))
SEED = int(os.environ.get("SEED", "123"))
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_DIR = os.environ.get("MODEL_DIR", "./modelo-qwen2.5-0.5b")

print("=" * 60)
print(f"SPARSE PoC FF{LAYER} Qwen2.5-0.5B | {N_SEQ} seq x {SEQ} tokens = {N_SEQ * SEQ} tokens")
print("Sin entrenar, solo medir escasez y diversidad. Ocupa muy poco.")

import torch
import numpy as np

print(f"torch {torch.__version__} | cuda: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

from transformers import AutoModelForCausalLM, AutoTokenizer

if os.path.isdir(MODEL_DIR) and os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
    src = MODEL_DIR
else:
    print(f"Modelo NO esta en {MODEL_DIR}, se descarga de {MODEL_ID} (~1 GB)...")
    src = MODEL_ID

tok = AutoTokenizer.from_pretrained(src, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32, trust_remote_code=False).to(device)
model.eval()
cfg = model.config
HIDDEN, INTER = cfg.hidden_size, cfg.intermediate_size
print(f"Modelo: hidden {HIDDEN}, inter {INTER}, capas {cfg.num_hidden_layers}")

TEXTOS = [
    "Montevideo es la capital de Uruguay y esta sobre el Rio de la Plata. Fue fundada en el siglo dieciocho y hoy es el centro politico, economico y cultural del pais.",
    "La fotosintesis es el proceso por el cual las plantas convierten la luz del sol, el agua y el dioxido de carbono en oxigeno y glucosa.",
    "Un transformador es una arquitectura de red neuronal que usa el mecanismo de atencion para relacionar las palabras de una frase entre si.",
    "El agua hierve a una temperatura de 100 grados Celsius a nivel del mar, y se usa para cocinar, limpiar y generar vapor en la industria.",
    "La Revolucion de Mayo ocurrio en 1810 en Buenos Aires, cuando se formo la Primera Junta de gobierno.",
    "La destilacion de una red neuronal consiste en que un modelo alumno aprenda a imitar a un modelo profesor, copiando sus salidas en datos reales.",
    "El idioma oficial de Espana es el espanol, que se habla tambien en casi toda America Latina y en partes de Estados Unidos.",
    "La economia de un pais depende de su produccion, su comercio exterior, su moneda y sus politicas fiscales y monetarias.",
    "Los dinosaurios dominaron la Tierra durante mas de 150 millones de anos, hasta su extincion hace unos 66 millones de anos.",
    "La programacion en Python es popular por su sintaxis simple, sus librerias y su comunidad, y se usa en ciencia de datos y web.",
]
TEXTOS = TEXTOS[:N_SEQ]

# captura gate pre-activacion por secuencia (para medir diversidad por texto)
capturados_g = []
capturados_x = []


def hook_gate(module, inp, out):
    capturados_g.append(out.detach().float().cpu())


def hook_x(module, inp, out):
    capturados_x.append(inp[0].detach().float().cpu())


teacher = model.model.layers[LAYER].mlp
h1 = teacher.gate_proj.register_forward_hook(hook_gate)
h2 = teacher.register_forward_hook(hook_x)
t0 = time.time()
with torch.no_grad():
    for txt in TEXTOS:
        ids = tok(txt, return_tensors="pt", truncation=True, max_length=SEQ).to(device)
        model(**ids)
h1.remove()
h2.remove()
print(f"Captura: {time.time() - t0:.1f}s, {sum(t.shape[0] for t in capturados_g)} tokens en total")

# --- 1. escasez a diferentes escalas / umbrales ---
print("\n--- ESCASEZ: % de neuronas gate apagadas por umbral ---")
print("Umbral = pre-activacion gate < umbral  (SiLU ~0 cuando es muy negativo)")
for umbral in [-10, -5, -2, -1, 0, 1]:
    fracs = []
    for g in capturados_g:
        gg = g.reshape(-1, INTER)
        fracs.append((gg < umbral).float().mean().item())
    print(f"  umbral {umbral:>3}: {np.mean(fracs) * 100:5.1f}% apagadas "
          f"(por secuencia: min {np.min(fracs) * 100:.1f}% max {np.max(fracs) * 100:.1f}%)")
# tambien SiLU(gate) casi cero
for eps in [0.01, 0.1]:
    fracs = []
    for g in capturados_g:
        gg = torch.nn.functional.silu(g.reshape(-1, INTER))
        fracs.append((gg.abs() < eps).float().mean().item())
    print(f"  |SiLU(gate)| < {eps}: {np.mean(fracs) * 100:5.1f}% apagadas")

# --- 2. diversidad: ¿las mismas neuronas activas en cada texto? ---
print("\n--- DIVERSIDAD: neuronas activas por texto (gate > 0) ---")
activos_por_texto = []
for g in capturados_g:
    gg = g.reshape(-1, INTER)
    # neurona activa si en promedio supera 0 en esa secuencia
    activo = (gg.mean(dim=0) > 0).numpy()
    activos_por_texto.append(activo)
    print(f"  texto {len(activos_por_texto)}: {activo.sum()} / {INTER} neuronas activas "
          f"({activo.mean() * 100:.1f}%)")
# Jaccard entre textos: 1 = siempre las mismas, 0 = totalmente distintas
print("\n  Solapamiento Jaccard entre textos (1 = mismas neuronas, 0 = distintas):")
jac = []
for i in range(len(activos_por_texto)):
    for j in range(i + 1, len(activos_por_texto)):
        a, b = activos_por_texto[i], activos_por_texto[j]
        inter = (a & b).sum()
        union = (a | b).sum()
        jac.append(inter / max(union, 1))
print(f"  Jaccard medio: {np.mean(jac):.3f} (min {np.min(jac):.3f}, max {np.max(jac):.3f})")
if np.mean(jac) > 0.9:
    print("  -> POCA diversidad: casi siempre las mismas neuronas, clusterizar no ayuda mucho.")
elif np.mean(jac) < 0.5:
    print("  -> MUCHA diversidad: cada texto usa neuronas distintas, hay para clusterizar.")
else:
    print("  -> Diversidad MEDIA: hay solapamiento pero tambien especializacion por texto.")

# --- 3. clustering rapido sobre gate_proj ---
print("\n--- CLUSTERING KMeans gate_proj (4 expertos) ---")
try:
    from sklearn.cluster import KMeans
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"])
    from sklearn.cluster import KMeans
gate_w = teacher.gate_proj.weight.float().detach().cpu().numpy()
gate_norm = gate_w / (np.linalg.norm(gate_w, axis=1, keepdims=True) + 1e-12)
km = KMeans(n_clusters=4, n_init=4, random_state=SEED).fit(gate_norm)
counts = np.bincount(km.labels_, minlength=4)
print(f"  tamanos de cluster: {counts.tolist()} (ideal {INTER // 4} cada uno)")
# distancia intra vs inter (silhouette aproximada)
dists = km.transform(gate_norm)
intra = dists[np.arange(len(gate_w)), km.labels_].mean()
# distancia al centroide mas cercano que no es el propio
dists_sorted = np.sort(dists, axis=1)
inter = dists_sorted[:, 1].mean()
print(f"  distancia media intra-cluster: {intra:.3f}")
print(f"  distancia media al otro cluster mas cercano: {inter:.3f}")
print(f"  separacion inter/intra: {inter / max(intra, 1e-9):.2f}x "
      f"(<1.5 = clusters poco separados, >2 = bien separados)")
print(f"\nTiempo total PoC: {time.time() - t0:.1f}s")
print("Guardado: nada en disco, solo este log. Para el MoE SPARSE completo, correr colab_moe_sparse.py")
