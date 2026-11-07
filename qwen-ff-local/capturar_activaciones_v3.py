#!/usr/bin/env python3
"""Captura v3: MAS datos reales de FF0 (resto de los articulos de la v2).

Motivo: e11 llego a eval cos 0,8836 (desde 0,8606 en paso 140) con
train 9539 vectores; el gap train/eval (0,98 vs 0,88) sigue pidiendo
mas datos. Aca se toman los tokens 900-1900 de los 10 articulos de
train de la v2 (~9600 vectores nuevos) y se suman al train de la v2.
El eval NO cambia (2361 vectores) para que las comparaciones sigan
siendo honestas. Salida: activaciones-reales-qwen25-v3.npz.
"""
import re
import sys
import time

import numpy as np

sys.path.insert(0, "/home/hatch/workspace/qwen-ff-local")
sys.path.insert(0, "/home/hatch/workspace/qwen-ff-local/pylibs")
import capturar_activaciones_reales as cap
from capturar_activaciones_v2 import forward_chunk, TRAIN_FILES

BASE = "/home/hatch/workspace/qwen-ff-local"
MODEL_DIR = BASE + "/modelo-qwen2.5-0.5b"
SAFETENSORS = MODEL_DIR + "/model.safetensors"
OUT = BASE + "/activaciones-reales-qwen25-v3.npz"
WIKI = BASE + "/wiki_es"  # en workspace: /tmp se borra entre sesiones
CHUNK = 64
TOK_START, TOK_END = 900, 1900


def main():
    t0 = time.time()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(MODEL_DIR + "/tokenizer.json")
    windows = []
    for fn in TRAIN_FILES:
        text = re.sub(r"\s+", " ", open(f"{WIKI}/{fn}.txt").read()).strip()
        ids = tok.encode(text).ids[TOK_START:TOK_END]
        for i in range(0, len(ids) - CHUNK + 1, CHUNK):
            windows.append(ids[i:i + CHUNK])
        print(f"train {fn}: {len(ids)} tok -> {len(windows)} ventanas acumuladas",
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

    v2 = np.load(BASE + "/activaciones-reales-qwen25-v2.npz")
    x_tr = np.concatenate([v2["x_train"], x_new])
    y_tr = np.concatenate([v2["y_train"], y_new])
    print(f"total v3: train {x_tr.shape[0]} vectores, "
          f"eval {v2['x_eval'].shape[0]} vectores (sin cambios)", flush=True)
    np.savez(OUT, x_train=x_tr, y_train=y_tr,
             x_eval=v2["x_eval"], y_eval=v2["y_eval"],
             tok_train=v2["tok_train"], tok_eval=v2["tok_eval"])
    print(f"guardado: {OUT}", flush=True)
    print(f"tiempo total: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
