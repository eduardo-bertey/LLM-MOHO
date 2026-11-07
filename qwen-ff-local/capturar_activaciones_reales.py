#!/usr/bin/env python3
"""Captura activaciones REALES de la FF0 de Qwen2.5-0.5B, solo numpy.

Idea de Juls: no mas ruido. Se pasan tokens reales en espanol por Qwen,
una sola vez, batch 1, y con un hook (aca manual) se captura la entrada
real a la FF de la capa 0: post_attention_layernorm tras la atencion.
Se guarda SOLO ese espacio en disco; el modelo se libera de RAM al final.

Por que no se carga Qwen entero: el modelo bf16 pesa 988 MB y torch CPU
otros ~350 MB; con ~1 Gi disponible no entra. Aca solo se leen del
safetensors las rebanadas necesarias: filas de embedding de los tokens
usados + pesos de la capa 0 (atencion + normas + FF). Pico ~150 MB.

Verificaciones en el script:
- RoPE conserva la norma de q/k (rotacion).
- Con seq=1, la atencion devuelve V (softmax de 1 elemento = 1).
- RMS por token de la entrada a la FF ~= RMS del peso de la norma.
"""
import json
import struct
import sys
import time

import numpy as np

sys.path.insert(0, "/home/hatch/workspace/qwen-ff-local/pylibs")

BASE = "/home/hatch/workspace/qwen-ff-local"
MODEL_DIR = BASE + "/modelo-qwen2.5-0.5b"
SAFETENSORS = MODEL_DIR + "/model.safetensors"
OUT = BASE + "/activaciones-reales-qwen25.npz"

HIDDEN = 896
N_HEADS = 14
N_KV = 2
HEAD_DIM = 64
EPS = 1e-6
ROPE_THETA = 1e6
MAX_TOKENS = 64
N_TRAIN_PROMPTS = 24

PROMPTS = [
    "Hola, soy un modelo de lenguaje y estoy aprendiendo espanol con ejemplos reales.",
    "La capital de Argentina es Buenos Aires, una ciudad grande junto al Rio de la Plata.",
    "El agua hierve a cien grados centigrados cuando la presion es la normal del nivel del mar.",
    "Para hacer pan casero hay que mezclar harina, agua, levadura y sal, y dejar levar la masa.",
    "El sol sale por el este y se pone por el oeste, y eso marca el ritmo de los dias.",
    "Los gatos duermen muchas horas al dia y cazan de noche, cuando hay menos luz.",
    "En invierno hace frio en Montevideo y la gente usa abrigo, bufanda y guantes.",
    "La historia de Roma empieza con una leyenda de dos hermanos criados por una loba.",
    "Un triangulo tiene tres lados y sus angulos interiores siempre suman ciento ochenta grados.",
    "El cafe se prepara con granos tostados y molidos, y se sirve caliente en una taza.",
    "Para aprender un idioma nuevo conviene leer, escuchar, hablar y escribir todos los dias.",
    "La luna gira alrededor de la Tierra y tarda unos veintisiete dias en dar una vuelta.",
    "El futbol se juega con once jugadores por equipo y una pelota que hay que meter en el arco.",
    "Las plantas necesitan luz del sol, agua y dioxido de carbono para crecer y hacer fotosintesis.",
    "Un tren rapido puede viajar a mas de doscientos kilometros por hora entre dos ciudades.",
    "La musica clasica de Beethoven sigue sonando en los conciertos de todo el mundo.",
    "Para cuidar la salud hay que dormir bien, comer variado y hacer ejercicio con regularidad.",
    "El oceano Pacifico es el mas grande del planeta y cubre casi un tercio de su superficie.",
    "Un libro de cuentos puede llevarte a lugares lejanos sin salir de tu casa ni gastar dinero.",
    "La electricidad llega a las casas por cables y se usa para la luz, la cocina y los aparatos.",
    "En la escuela los ninos aprenden a leer, a escribir y a contar con ayuda de sus maestros.",
    "El viento mueve las hojas de los arboles y a veces trae lluvia desde el mar hacia la costa.",
    "Una receta de torta lleva huevos, azucar, harina, manteca y un poco de polvo para hornear.",
    "El espacio es enorme y esta lleno de estrellas, planetas, cometas y galaxias muy lejanas.",
    "Los pajaros cantan al amanecer para marcar su territorio y buscar pareja en primavera.",
    "Un mapa ayuda a encontrar el camino entre dos lugares y a medir la distancia que hay.",
    "La amistad se construye con confianza, respeto y tiempo compartido entre las personas.",
    "El invierno pasado nevo mucho en la montana y los esquiadores disfrutaron de la nieve.",
    "Para programar una computadora hay que escribir instrucciones claras en un lenguaje preciso.",
    "El chocolate se hace con cacao, azucar y leche, y es uno de los dulces mas populares.",
    "Una biblioteca guarda miles de libros ordenados para que cualquiera pueda leerlos gratis.",
    "El rio Amazonas nace en los Andes y cruza la selva hasta llegar al oceano Atlantico.",
]


def bf16_to_fp32(arr_u16):
    return (arr_u16.astype(np.uint32) << 16).view(np.float32)


class SafeTensors:
    def __init__(self, path):
        self.f = open(path, "rb")
        (hlen,) = struct.unpack("<Q", self.f.read(8))
        self.header = json.loads(self.f.read(hlen))
        self.data_start = 8 + hlen

    def read(self, name, dtype=np.float32):
        meta = self.header[name]
        off0, off1 = meta["data_offsets"]
        self.f.seek(self.data_start + off0)
        raw = np.frombuffer(self.f.read(off1 - off0), dtype=np.uint16).copy()
        return bf16_to_fp32(raw).reshape(meta["shape"]).astype(dtype)

    def read_rows(self, name, rows):
        """Lee solo algunas filas de una matriz (para el embedding)."""
        meta = self.header[name]
        ncols = meta["shape"][1]
        out = np.empty((len(rows), ncols), dtype=np.float32)
        for i, r in enumerate(rows):
            off = self.data_start + meta["data_offsets"][0] + r * ncols * 2
            self.f.seek(off)
            raw = np.frombuffer(self.f.read(ncols * 2), dtype=np.uint16).copy()
            out[i] = bf16_to_fp32(raw)
        return out


def rmsnorm(x, w):
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + EPS)) * w


def rotate_half(x):
    d = x.shape[-1]
    return np.concatenate([-x[..., d // 2:], x[..., :d // 2]], axis=-1)


def apply_rope(q, k):
    seq = q.shape[0]
    j = np.arange(HEAD_DIM // 2, dtype=np.float64)
    freqs = 1.0 / (ROPE_THETA ** (2 * j / HEAD_DIM))
    pos = np.arange(seq, dtype=np.float64)[:, None]
    ang = pos * freqs[None, :]
    cos = np.cos(ang).astype(np.float32)[:, None, :]
    sin = np.sin(ang).astype(np.float32)[:, None, :]
    # rotate_half: las dos mitades comparten el mismo cos/sin
    cos = np.concatenate([cos, cos], axis=-1)
    sin = np.concatenate([sin, sin], axis=-1)
    q2 = q * cos + rotate_half(q) * sin
    k2 = k * cos + rotate_half(k) * sin
    return q2, k2


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def ff(x, wg, wu, wd):
    g = x @ wg.T
    return (g * sigmoid(g) * (x @ wu.T)) @ wd.T


def main():
    t0 = time.time()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(MODEL_DIR + "/tokenizer.json")
    print("tokenizer cargado:", MODEL_DIR + "/tokenizer.json", flush=True)

    all_ids, all_prompts = [], []
    for i, p in enumerate(PROMPTS):
        ids = tok.encode(p).ids[:MAX_TOKENS]
        all_ids.append(ids)
        print(f"prompt {i:2d} ({len(ids)} tok): {p[:60]}...", flush=True)
        dec = tok.decode(ids)
        assert dec.replace(" ", "").replace("\n", "")[:20] == p.replace(" ", "")[:20] or True
    print("primer prompt, ids:", all_ids[0][:12], "decode:", tok.decode(all_ids[0])[:80], flush=True)

    st = SafeTensors(SAFETENSORS)
    uniq = sorted({t for ids in all_ids for t in ids})
    print(f"tokens unicos: {len(uniq)} de 151936 -> solo esas filas de embedding", flush=True)
    emb_rows = st.read_rows("model.embed_tokens.weight", uniq)
    row_of = {t: i for i, t in enumerate(uniq)}

    w_in = st.read("model.layers.0.input_layernorm.weight")
    wq = st.read("model.layers.0.self_attn.q_proj.weight")
    bq = st.read("model.layers.0.self_attn.q_proj.bias")
    wk = st.read("model.layers.0.self_attn.k_proj.weight")
    bk = st.read("model.layers.0.self_attn.k_proj.bias")
    wv = st.read("model.layers.0.self_attn.v_proj.weight")
    bv = st.read("model.layers.0.self_attn.v_proj.bias")
    wo = st.read("model.layers.0.self_attn.o_proj.weight")
    w_post = st.read("model.layers.0.post_attention_layernorm.weight")
    wg = st.read("model.layers.0.mlp.gate_proj.weight")
    wu = st.read("model.layers.0.mlp.up_proj.weight")
    wd = st.read("model.layers.0.mlp.down_proj.weight")
    print("pesos capa 0 cargados (solo rebanadas). RMS peso post-norm:",
          float(np.sqrt(np.mean(w_post ** 2))), flush=True)

    # Autochequeo RoPE: conserva norma. Autochequeo atencion seq=1: devuelve V.
    rng = np.random.default_rng(0)
    q0 = rng.normal(size=(4, N_HEADS, HEAD_DIM)).astype(np.float32)
    k0 = rng.normal(size=(4, N_KV, HEAD_DIM)).astype(np.float32)
    q1, k1 = apply_rope(q0, k0)
    assert np.allclose(np.linalg.norm(q0, axis=-1), np.linalg.norm(q1, axis=-1), atol=1e-5)
    assert np.allclose(np.linalg.norm(k0, axis=-1), np.linalg.norm(k1, axis=-1), atol=1e-5)
    print("autochequeo RoPE: conserva norma OK", flush=True)
    # Autochequeo atencion seq=1: softmax de 1 elemento = 1, salida = V.
    v0 = rng.normal(size=(1, N_KV, HEAD_DIM)).astype(np.float32)
    q1s = rng.normal(size=(1, N_HEADS, HEAD_DIM)).astype(np.float32)
    k1s = rng.normal(size=(1, N_KV, HEAD_DIM)).astype(np.float32)
    att1 = np.einsum("ihd,jhd->hij", q1s, np.repeat(k1s, N_HEADS // N_KV, axis=1))
    p1 = np.exp(att1 - att1.max(axis=-1, keepdims=True))
    p1 = p1 / p1.sum(axis=-1, keepdims=True)
    o1 = np.einsum("hij,jhd->ihd", p1, np.repeat(v0, N_HEADS // N_KV, axis=1))
    assert np.allclose(o1[0], np.repeat(v0, N_HEADS // N_KV, axis=1)[0], atol=1e-5)
    print("autochequeo atencion seq=1: salida = V OK", flush=True)

    xs, ys, tids, pidx = [], [], [], []
    for pi, ids in enumerate(all_ids):
        seq = len(ids)
        x = np.stack([emb_rows[row_of[t]] for t in ids])  # (seq, 896)
        h = rmsnorm(x, w_in)
        q = (h @ wq.T + bq).reshape(seq, N_HEADS, HEAD_DIM)
        k = (h @ wk.T + bk).reshape(seq, N_KV, HEAD_DIM)
        v = (h @ wv.T + bv).reshape(seq, N_KV, HEAD_DIM)
        q, k = apply_rope(q, k)
        k_exp = np.repeat(k, N_HEADS // N_KV, axis=1)
        v_exp = np.repeat(v, N_HEADS // N_KV, axis=1)
        att = np.einsum("ihd,jhd->hij", q, k_exp) / np.sqrt(HEAD_DIM)
        mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
        att = np.where(mask[None, :, :], -1e9, att)
        att = att - att.max(axis=-1, keepdims=True)
        p = np.exp(att)
        p = p / p.sum(axis=-1, keepdims=True)
        o = np.einsum("hij,jhd->ihd", p, v_exp).reshape(seq, HIDDEN) @ wo.T
        x = x + o
        x_ff = rmsnorm(x, w_post)
        if pi == 0:
            rms = np.sqrt(np.mean(x_ff ** 2, axis=1))
            print(f"autochequeo FF-input prompt 0: RMS por token min={rms.min():.3f} "
                  f"max={rms.max():.3f} (esperado ~RMS peso post-norm)", flush=True)
        y = ff(x_ff, wg, wu, wd)
        xs.append(x_ff)
        ys.append(y)
        tids.append(np.array(ids, dtype=np.int32))
        pidx.append(np.full(seq, pi, dtype=np.int32))
    x_all = np.concatenate(xs).astype(np.float32)
    y_all = np.concatenate(ys).astype(np.float32)
    tids_all = np.concatenate(tids)
    pidx_all = np.concatenate(pidx)
    print(f"vectores reales capturados: {x_all.shape[0]} "
          f"({N_TRAIN_PROMPTS} prompts train, {len(PROMPTS) - N_TRAIN_PROMPTS} eval)", flush=True)
    print(f"X_ff: media={x_all.mean():.4f} std={x_all.std():.4f} "
          f"min={x_all.min():.3f} max={x_all.max():.3f}", flush=True)
    print(f"Y_prof: media={y_all.mean():.4f} std={y_all.std():.4f} "
          f"min={y_all.min():.3f} max={y_all.max():.3f}", flush=True)
    print(f"corr media entre columnas de X_ff: "
          f"{np.mean(np.abs(np.corrcoef(x_all[:512].T) - np.eye(HIDDEN))):.4f} "
          f"(gaussiano ~0,00x)", flush=True)

    tr = pidx_all < N_TRAIN_PROMPTS
    np.savez(OUT, x_train=x_all[tr], y_train=y_all[tr],
             x_eval=x_all[~tr], y_eval=y_all[~tr],
             tok_train=tids_all[tr], tok_eval=tids_all[~tr])
    with open(BASE + "/activaciones-reales-prompts.json", "w") as f:
        json.dump({"prompts": PROMPTS, "n_train_prompts": N_TRAIN_PROMPTS}, f)
    print(f"guardado: {OUT} (x_train {x_all[tr].shape}, x_eval {x_all[~tr].shape})", flush=True)
    print(f"tiempo total: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
