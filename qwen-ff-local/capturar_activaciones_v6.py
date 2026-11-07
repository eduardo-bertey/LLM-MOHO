#!/usr/bin/env python3
"""Captura v6: MAS datos de los articulos NUEVOS de la v5 (segundo tramo).

Motivo: e15 llego a eval cos 0,9454 (desde 0,9381 en e14) con
train 37699 vectores (v5: 10 articulos nuevos). El dato diverso
ayudo (+0,0073). Faltan 0,0046 para 0,95. Aca se toman los tokens
900-2900 de los mismos 10 articulos nuevos de la v5 (2000 tok
c/u, ~31 ventanas c/u, ~19840 vectores nuevos) y se suman al
train de la v5. El eval NO cambia (2361 vectores) para que las
comparaciones sigan siendo honestas.
Salida: activaciones-reales-qwen25-v6.npz.
"""
import re
import sys
import time

import numpy as np

sys.path.insert(0, "/home/hatch/workspace/qwen-ff-local")
sys.path.insert(0, "/home/hatch/workspace/qwen-ff-local/pylibs")
import capturar_activaciones_reales as cap
from capturar_activaciones_v2 import forward_chunk
from capturar_activaciones_v5 import NEW_FILES

BASE = "/home/hatch/workspace/qwen-ff-local"
MODEL_DIR = BASE + "/modelo-qwen2.5-0.5b"
SAFETENSORS = MODEL_DIR + "/model.safetensors"
OUT = BASE + "/activaciones-reales-qwen25-v6.npz"
WIKI = BASE + "/wiki_es"  # en workspace: /tmp se borra entre sesiones
CHUNK = 64
TOK_START, TOK_END = 900, 2900


def main():
    t0 = time.time()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(MODEL_DIR + "/tokenizer.json")
    windows = []
    for fn in NEW_FILES:
        text = re.sub(r"\s+", " ", open(f"{WIKI}/{fn}.txt").read()).strip()
        ids = tok.encode(text).ids[TOK_START:TOK_END]
        n0 = len(windows)
        for i in range(0, len(ids) - CHUNK + 1, CHUNK):
            windows.append(ids[i:i + CHUNK])
        print(f"train {fn}: {len(ids)} tok -> {len(windows) - n0} ventanas",
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

    v5 = np.load(BASE + "/activaciones-reales-qwen25-v5.npz")
    x_tr = np.concatenate([v5["x_train"], x_new])
    y_tr = np.concatenate([v5["y_train"], y_new])
    print(f"total v6: train {x_tr.shape[0]} vectores, "
          f"eval {v5['x_eval'].shape[0]} vectores (sin cambios)", flush=True)
    np.savez(OUT, x_train=x_tr, y_train=y_tr,
             x_eval=v5["x_eval"], y_eval=v5["y_eval"],
             tok_train=v5["tok_train"], tok_eval=v5["tok_eval"])
    print(f"guardado: {OUT}", flush=True)
    print(f"tiempo total: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
