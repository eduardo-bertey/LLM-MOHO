#!/usr/bin/env python3
"""Captura v2: MAS activaciones reales de FF0 de Qwen2.5-0.5B, solo numpy.

Motivo: e8r75 memorizo (train cos 0,9908, eval 0,6566) porque el train
tenia solo 579 vectores unicos. Aca se capturan miles de vectores de
prosa real en espanol (articulos de Wikipedia, dominio publico de
hecho via API), en ventanas de 64 tokens, mas los prompts originales.

Train: 10 articulos + 24 prompts originales.
Eval: 2 articulos distintos (Cafe, Electricidad) + 8 prompts originales.
Salida: activaciones-reales-qwen25-v2.npz (la v1 queda intacta).
"""
import json
import re
import sys
import time

import numpy as np

sys.path.insert(0, "/home/hatch/workspace/qwen-ff-local")
sys.path.insert(0, "/home/hatch/workspace/qwen-ff-local/pylibs")
import capturar_activaciones_reales as cap

BASE = "/home/hatch/workspace/qwen-ff-local"
MODEL_DIR = BASE + "/modelo-qwen2.5-0.5b"
SAFETENSORS = MODEL_DIR + "/model.safetensors"
OUT = BASE + "/activaciones-reales-qwen25-v2.npz"
WIKI = BASE + "/wiki_es"  # en workspace: /tmp se borra entre sesiones

HIDDEN = 896
N_HEADS = 14
N_KV = 2
HEAD_DIM = 64
CHUNK = 64
TRAIN_FILES = ["Fotosíntesis", "Buenos_Aires", "Miguel_de_Cervantes",
               "Segunda_Guerra_Mundial", "Sistema_solar", "Revolución_francesa",
               "Fútbol", "Computadora", "Océano_Atlántico", "Historia_de_Roma"]
EVAL_FILES = ["Café", "Electricidad"]
TOKENS_PER_TRAIN_FILE = 900   # ~14 ventanas de 64
TOKENS_PER_EVAL_FILE = 1100   # ~17 ventanas de 64


def forward_chunk(ids, st, emb_rows, row_of, w):
    """Forward manual capa 0 sobre una ventana de tokens. Devuelve (x_ff, y)."""
    seq = len(ids)
    x = np.stack([emb_rows[row_of[t]] for t in ids])
    h = cap.rmsnorm(x, w["in"])
    q = (h @ w["q"].T + w["bq"]).reshape(seq, N_HEADS, HEAD_DIM)
    k = (h @ w["k"].T + w["bk"]).reshape(seq, N_KV, HEAD_DIM)
    v = (h @ w["v"].T + w["bv"]).reshape(seq, N_KV, HEAD_DIM)
    q, k = cap.apply_rope(q, k)
    k_exp = np.repeat(k, N_HEADS // N_KV, axis=1)
    v_exp = np.repeat(v, N_HEADS // N_KV, axis=1)
    att = np.einsum("ihd,jhd->hij", q, k_exp) / np.sqrt(HEAD_DIM)
    mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
    att = np.where(mask[None, :, :], -1e9, att)
    att = att - att.max(axis=-1, keepdims=True)
    p = np.exp(att)
    p = p / p.sum(axis=-1, keepdims=True)
    o = np.einsum("hij,jhd->ihd", p, v_exp).reshape(seq, HIDDEN) @ w["o"].T
    x = x + o
    x_ff = cap.rmsnorm(x, w["post"])
    y = cap.ff(x_ff, w["g"], w["u"], w["d"])
    return x_ff, y


def main():
    t0 = time.time()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(MODEL_DIR + "/tokenizer.json")
    print("tokenizer cargado", flush=True)

    # Tokenizar articulos y armar ventanas de 64 tokens.
    windows = {"train": [], "eval": []}
    for split, files, n_tok in (("train", TRAIN_FILES, TOKENS_PER_TRAIN_FILE),
                                ("eval", EVAL_FILES, TOKENS_PER_EVAL_FILE)):
        for fn in files:
            text = open(f"{WIKI}/{fn}.txt").read()
            text = re.sub(r"\s+", " ", text).strip()
            ids = tok.encode(text).ids[:n_tok]
            for i in range(0, len(ids) - CHUNK + 1, CHUNK):
                windows[split].append(ids[i:i + CHUNK])
            print(f"{split} {fn}: {len(ids)} tok -> "
                  f"{len(windows[split])} ventanas acumuladas", flush=True)
    print(f"ventanas: train {len(windows['train'])}, eval {len(windows['eval'])}",
          flush=True)

    all_ids = [w for ws in windows.values() for w in ws]
    st = cap.SafeTensors(SAFETENSORS)
    uniq = sorted({t for ids in all_ids for t in ids})
    print(f"tokens unicos: {len(uniq)}", flush=True)
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
    print("pesos capa 0 cargados", flush=True)

    out = {}
    for split in ("train", "eval"):
        xs, ys, tids = [], [], []
        for n, ids in enumerate(windows[split]):
            x_ff, y = forward_chunk(ids, st, emb_rows, row_of, w)
            xs.append(x_ff)
            ys.append(y)
            tids.append(np.array(ids, dtype=np.int32))
            if (n + 1) % 25 == 0:
                print(f"{split}: {n + 1}/{len(windows[split])} ventanas "
                      f"({time.time() - t0:.0f}s)", flush=True)
        out[split] = (np.concatenate(xs).astype(np.float32),
                      np.concatenate(ys).astype(np.float32),
                      np.concatenate(tids))
        print(f"{split}: {out[split][0].shape[0]} vectores", flush=True)

    # Sumar los prompts originales de la v1 (mismo espacio, mismos keys).
    v1 = np.load(BASE + "/activaciones-reales-qwen25.npz")
    x_tr = np.concatenate([out["train"][0], v1["x_train"]])
    y_tr = np.concatenate([out["train"][1], v1["y_train"]])
    x_ev = np.concatenate([out["eval"][0], v1["x_eval"]])
    y_ev = np.concatenate([out["eval"][1], v1["y_eval"]])
    print(f"total v2: train {x_tr.shape[0]} vectores, eval {x_ev.shape[0]} vectores",
          flush=True)
    print(f"X_ff train: media={x_tr.mean():.4f} std={x_tr.std():.4f}", flush=True)
    np.savez(OUT, x_train=x_tr, y_train=y_tr, x_eval=x_ev, y_eval=y_ev,
             tok_train=out["train"][2], tok_eval=out["eval"][2])
    print(f"guardado: {OUT}", flush=True)
    print(f"tiempo total: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
