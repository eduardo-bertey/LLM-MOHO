#!/usr/bin/env python3
"""Captura v7: 10 articulos NUEVOS mas (tercer grupo de temas).

Motivo: e16/e16c llegaron a eval cos 0,9507 (umbral de cos
alcanzado) pero R2 0,8973 < 0,90 por 0,0027; los ultimos 80 pasos
sobre los mismos datos sumaron apenas +0,0007 de cos. El dato
diverso es la palanca que viene funcionando (v5: +0,0073,
v6: +0,0053). Aca se agregan 10 articulos NUEVOS de Wikipedia
en espanol (Arte, Cine, Historia, Tecnologia, Internet,
Politica, Psicologia, Educacion, Deporte, Religion), tokens
0-900 de cada uno (~140 ventanas, ~8960 vectores nuevos), y se
suman al train de la v6. El eval NO cambia (2361 vectores)
para que las comparaciones sigan siendo honestas.
Salida: activaciones-reales-qwen25-v7.npz.
Textos descargados con descargar_wiki_v7.py (API es.wikipedia.org).
"""
import re
import sys
import time

import numpy as np

sys.path.insert(0, "/home/hatch/workspace/qwen-ff-local")
sys.path.insert(0, "/home/hatch/workspace/qwen-ff-local/pylibs")
import capturar_activaciones_reales as cap
from capturar_activaciones_v2 import forward_chunk

BASE = "/home/hatch/workspace/qwen-ff-local"
MODEL_DIR = BASE + "/modelo-qwen2.5-0.5b"
SAFETENSORS = MODEL_DIR + "/model.safetensors"
OUT = BASE + "/activaciones-reales-qwen25-v7.npz"
WIKI = BASE + "/wiki_es"  # en workspace: /tmp se borra entre sesiones
CHUNK = 64
TOK_END = 900

NEW_FILES = ["Arte", "Cine", "Historia", "Tecnología", "Internet",
             "Política", "Psicología", "Educación", "Deporte", "Religión"]


def main():
    t0 = time.time()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(MODEL_DIR + "/tokenizer.json")
    windows = []
    for fn in NEW_FILES:
        text = re.sub(r"\s+", " ", open(f"{WIKI}/{fn}.txt").read()).strip()
        ids = tok.encode(text).ids[:TOK_END]
        n0 = len(windows)
        for i in range(0, len(ids) - CHUNK + 1, CHUNK):
            windows.append(ids[i:i + CHUNK])
        print(f"train nuevo {fn}: {len(ids)} tok -> {len(windows) - n0} ventanas",
              flush=True)
    print(f"ventanas nuevas: {len(windows)}", flush=True)

    st = cap.SafeTensors(SAFETENSORS)
    uniq = sorted({t for ids in windows for t in ids})
    emb_rows = st.read_rows("model.embed_tokens.weight", uniq)
    row_of = {t: i for i, t in enumerate(uniq)}
    w = {
        "in": st.read("model.layers.0.input_layernorm.weight"),
        "q": st.read("model.layers.0.self_attn.q_proj.weight"),
        "bq": st.read("model.layers.0.self_attn.q_proj.bias"),
        "k": st.read("model.layers.0.self_attn.k_proj.weight"),
        "bk": st.read("model.layers.0.self_attn.k_proj.bias"),
        "v": st.read("model.layers.0.self_attn.v_proj.weight"),
        "bv": st.read("model.layers.0.self_attn.v_proj.bias"),
        "o": st.read("model.layers.0.self_attn.o_proj.weight"),
        "post": st.read("model.layers.0.post_attention_layernorm.weight"),
        "g": st.read("model.layers.0.mlp.gate_proj.weight"),
        "u": st.read("model.layers.0.mlp.up_proj.weight"),
        "d": st.read("model.layers.0.mlp.down_proj.weight"),
    }
    xs, ys = [], []
    for n, ids in enumerate(windows):
        x_ff, y = forward_chunk(ids, st, emb_rows, row_of, w)
        xs.append(x_ff)
        ys.append(y)
        if (n + 1) % 25 == 0:
            print(f"{n + 1}/{len(windows)} ventanas ({time.time() - t0:.0f}s)",
                  flush=True)
    x_new = np.concatenate(xs).astype(np.float32)
    y_new = np.concatenate(ys).astype(np.float32)
    print(f"vectores nuevos: {x_new.shape[0]}", flush=True)

    v6 = np.load(BASE + "/activaciones-reales-qwen25-v6.npz")
    x_tr = np.concatenate([v6["x_train"], x_new])
    y_tr = np.concatenate([v6["y_train"], y_new])
    print(f"total v7: train {x_tr.shape[0]} vectores, "
          f"eval {v6['x_eval'].shape[0]} vectores (sin cambios)", flush=True)
    np.savez(OUT, x_train=x_tr, y_train=y_tr,
             x_eval=v6["x_eval"], y_eval=v6["y_eval"],
             tok_train=v6["tok_train"], tok_eval=v6["tok_eval"])
    print(f"guardado: {OUT}", flush=True)
    print(f"tiempo total: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
