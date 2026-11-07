#!/usr/bin/env python3
"""FF aislada Qwen2.5-0.5B, capa 0, solo numpy. v6: BLD (Puzzle, NVIDIA),
destilacion HONESTA al 75% (intermediate 4864 -> 3648, ~9,8M params).

La alumna NO copia pesos de la profesora: init de origen del repo
(randn * 1/sqrt(fan_in), seed 123), y aprende solo por destilacion
con perdida NMSE Puzzle + 0,1*(1-cos) sobre activaciones reales
post-RMSNorm de tokens por Qwen. Profesora congelada, gradientes
aislados, entradas siempre del padre. El gaussiano es solo smoke test.
Uso: python3 ff_aislada_qwen25.py --inter 3648 --batch 4 --seq 64 --steps 50 --lr 1e-3 --tag e8r75
"""
import argparse
import json
import os
import struct
import time

import numpy as np

BASE = "/home/hatch/workspace/qwen-ff-local"
MODEL_DIR = BASE + "/modelo-qwen2.5-0.5b"
SAFETENSORS = MODEL_DIR + "/model.safetensors"
POOL_NPZ = BASE + "/checkpoints/pool_qwen25.npz"
EVAL_NPZ = BASE + "/checkpoints/eval_qwen25.npz"

HIDDEN = 896
INTER = 4864
MICRO = 256


def mem_available_mib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def bf16_to_fp32(arr_u16):
    return (arr_u16.astype(np.uint32) << 16).view(np.float32)


def load_ff0(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        out = {}
        for name in ("model.layers.0.mlp.gate_proj.weight",
                     "model.layers.0.mlp.up_proj.weight",
                     "model.layers.0.mlp.down_proj.weight"):
            meta = header[name]
            off0, off1 = meta["data_offsets"]
            f.seek(8 + hlen + off0)
            raw = np.frombuffer(f.read(off1 - off0), dtype=np.uint16).copy()
            out[name] = bf16_to_fp32(raw).reshape(meta["shape"])
        return (out["model.layers.0.mlp.gate_proj.weight"],
                out["model.layers.0.mlp.up_proj.weight"],
                out["model.layers.0.mlp.down_proj.weight"])


def fp32_to_bf16_to_fp32(arr):
    """fp32 -> bf16 (round-to-nearest-even, como torch .to(bfloat16)) -> fp32."""
    u = arr.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)).astype(np.uint32)
    return (u & 0xFFFF0000).view(np.float32)


def init_origen(shape, scale, r):
    """Init de origen del repo (ff_train_seq.py / ff_compare.py, seed 123):
    randn * (1/sqrt(fan_in)), con paso por bf16 y vuelta a fp32."""
    return fp32_to_bf16_to_fp32(
        r.normal(0.0, 1.0, size=shape).astype(np.float32) * np.float32(scale))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def silu(z):
    return z * sigmoid(z)


def silu_prime(z):
    s = sigmoid(z)
    return s * (1.0 + z * (1.0 - s))


def ff_forward(x, wg, wu, wd):
    g = x @ wg.T
    u = x @ wu.T
    h = silu(g) * u
    return h @ wd.T, (g, u, h)


def loss_and_grad_y(y, yt, kind):
    """Perdida y gradiente respecto a la salida y (micro-lote).
    - nmse_cos: NMSE Puzzle = sum((y-yt)^2)/sum(yt^2)  +  0.1 * media(1-cos_i)
    - mse: media((y-yt)^2)  (solo para comparar)
    Devuelve (loss, dy). Gradiente verificado contra diferencias finitas."""
    if kind == "mse":
        diff = y - yt
        return float(np.mean(diff ** 2)), 2.0 * diff / diff.size
    diff = y - yt
    # NMSE Puzzle (sin centrar): sum((y-yt)^2) / sum(yt^2)
    s = float(np.sum(yt ** 2)) + 1e-12
    nmse = float(np.sum(diff ** 2)) / s
    dy = 2.0 * diff / s
    # Termino coseno: media por vector de (1 - cos_i)
    yn = np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
    ytn = np.linalg.norm(yt, axis=1, keepdims=True) + 1e-12
    dot = np.sum(y * yt, axis=1, keepdims=True)
    cos_i = dot / (yn * ytn)
    loss_cos = 0.1 * float(np.mean(1.0 - cos_i))
    dy += 0.1 * (-yt / (yn * ytn) + y * dot / (yn ** 3 * ytn)) / y.shape[0]
    return nmse + loss_cos, dy


def metrics(y, yt):
    num = np.sum(y * yt, axis=1)
    den = np.linalg.norm(y, axis=1) * np.linalg.norm(yt, axis=1) + 1e-12
    cos = float(np.mean(num / den))
    mag = float(np.linalg.norm(y) / (np.linalg.norm(yt) + 1e-12))
    rel = float(np.linalg.norm(y - yt) / (np.linalg.norm(yt) + 1e-12))
    ss_res = float(np.sum((y - yt) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    nmse = float(np.sum((y - yt) ** 2) / (np.sum(yt ** 2) + 1e-12))
    return cos, mag, rel, float(1.0 - ss_res / ss_tot), nmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--pool-size", type=int, default=4096)
    ap.add_argument("--init", default="origen", choices=["origen", "teacher", "teacher_noise"])
    ap.add_argument("--inter", type=int, default=4864)
    ap.add_argument("--noise-frac", type=float, default=0.1)
    ap.add_argument("--data", default="real", choices=["real", "gauss"])
    ap.add_argument("--data-file", default="activaciones-reales-qwen25.npz")
    ap.add_argument("--resume", default="")
    ap.add_argument("--loss", default="nmse_cos", choices=["nmse_cos", "mse"])
    args = ap.parse_args()

    t0 = time.time()
    global INTER
    INTER = args.inter
    vec_step = args.batch * args.seq
    rng = np.random.default_rng(args.seed)
    print(f"=== {args.tag}: inter={INTER} batch={args.batch} seq={args.seq} "
          f"vectores/paso={vec_step} steps={args.steps} lr={args.lr} "
          f"init={args.init} seed={args.seed} data={args.data} loss={args.loss} ===", flush=True)
    print(f"[mem] disponible antes: {mem_available_mib():.0f} MiB", flush=True)
    if args.data == "real":
        print(f"entradas: ACTIVACIONES REALES ({args.data_file})", flush=True)
    else:
        print("entradas: gaussianas N(0,1) [SMOKE, NO REAL]", flush=True)

    wg_t, wu_t, wd_t = load_ff0(SAFETENSORS)
    print(f"FF0 profesora: 13,074,432 params, "
          f"{(wg_t.nbytes + wu_t.nbytes + wd_t.nbytes) / 1e6:.1f} MB fp32", flush=True)
    n_params_s = 2 * INTER * HIDDEN + HIDDEN * INTER
    print(f"FF alumna: inter={INTER}, {n_params_s:,} params "
          f"({100.0 * INTER / 4864:.0f}% de la profesora)", flush=True)

    if args.data == "real":
        d = np.load(BASE + "/" + args.data_file)
        x_pool, y_pool = d["x_train"], d["y_train"]
        x_ev, y_ev = d["x_eval"], d["y_eval"]
        print(f"datos reales: train {x_pool.shape[0]} vectores, "
              f"eval {x_ev.shape[0]} vectores (prompts distintos)", flush=True)
    else:
        # Pool fijo gaussiano: entradas + salida profesora, calculado una vez.
        if os.path.exists(POOL_NPZ):
            d = np.load(POOL_NPZ)
            x_pool, y_pool = d["x"], d["y"]
            print(f"pool reusado de disco: {x_pool.shape[0]} vectores", flush=True)
        else:
            rp = np.random.default_rng(1234)
            x_pool = rp.normal(0, 1, size=(args.pool_size, HIDDEN)).astype(np.float32)
            y_pool = np.empty((args.pool_size, HIDDEN), dtype=np.float32)
            for i in range(0, args.pool_size, MICRO):
                y_pool[i:i + MICRO], _ = ff_forward(x_pool[i:i + MICRO], wg_t, wu_t, wd_t)
            np.savez(POOL_NPZ, x=x_pool, y=y_pool)
            print(f"pool creado: {args.pool_size} vectores, salida profesora en disco", flush=True)

        if os.path.exists(EVAL_NPZ):
            d = np.load(EVAL_NPZ)
            x_ev, y_ev = d["x"], d["y"]
        else:
            re_ = np.random.default_rng(999)
            x_ev = re_.normal(0, 1, size=(1024, HIDDEN)).astype(np.float32)
            y_ev, _ = ff_forward(x_ev, wg_t, wu_t, wd_t)
            np.savez(EVAL_NPZ, x=x_ev, y=y_ev)
    print(f"[mem] disponible tras cargar FF0+datos: {mem_available_mib():.0f} MiB", flush=True)

    if args.init == "origen":
        # Init del repo original: randn * 1/sqrt(fan_in), seed 123, paso bf16.
        # Orden de sorteo igual que en el repo: gate, up, down, mismo generador.
        wg_s = init_origen((INTER, HIDDEN), 1.0 / np.sqrt(HIDDEN), rng)
        wu_s = init_origen((INTER, HIDDEN), 1.0 / np.sqrt(HIDDEN), rng)
        wd_s = init_origen((HIDDEN, INTER), 1.0 / np.sqrt(INTER), rng)
        print(f"init origen: gate/up ({INTER}x{HIDDEN}) std={wg_s.std():.4f} (1/sqrt(896)=0,0334), "
              f"down std={wd_s.std():.4f} (1/sqrt({INTER})={1.0 / np.sqrt(INTER):.4f}), con paso bf16", flush=True)
    elif args.init == "teacher":
        # Copia exacta de la profesora (BLD: init desde el padre).
        wg_s, wu_s, wd_s = wg_t.copy(), wu_t.copy(), wd_t.copy()
        print("init teacher: copia exacta de la profesora", flush=True)
    else:  # teacher_noise: profesora + ruido chico (fracion del std de cada matriz)
        def _noisy(w):
            return (w + rng.normal(0, args.noise_frac * float(w.std()),
                                   size=w.shape)).astype(np.float32)
        wg_s, wu_s, wd_s = _noisy(wg_t), _noisy(wu_t), _noisy(wd_t)
        print(f"init teacher_noise: profesora + ruido {args.noise_frac:.2f}*std", flush=True)
    params = [wg_s, wu_s, wd_s]
    if args.resume:
        # Reanudar desde un checkpoint (pesos alumna guardados, mismo inter).
        rc = np.load(args.resume)
        params = [rc["gate_proj"].astype(np.float32),
                  rc["up_proj"].astype(np.float32),
                  rc["down_proj"].astype(np.float32)]
        wg_s, wu_s, wd_s = params
        print(f"[resume] cargado checkpoint {args.resume} "
              f"(Adam m/v reinician en cero)", flush=True)
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8

    y0, _ = ff_forward(x_ev, wg_s, wu_s, wd_s)
    cos, mag, rel, r2, nmse = metrics(y0, y_ev)
    print(f"[eval inicial] cos={cos:.4f} mag={mag:.4f} rel_err={rel:.4f} R2={r2:.4f} NMSE={nmse:.4f}", flush=True)

    n_pool = x_pool.shape[0]
    for step in range(1, args.steps + 1):
        ts = time.time()
        idx = rng.integers(0, n_pool, size=vec_step)
        grads = [np.zeros_like(p) for p in params]
        loss_acc = 0.0
        y_last = yt_last = None
        for i in range(0, vec_step, MICRO):
            xb = x_pool[idx[i:i + MICRO]]
            ytb = y_pool[idx[i:i + MICRO]]
            y, (g, u, h) = ff_forward(xb, wg_s, wu_s, wd_s)
            l_mb, dy = loss_and_grad_y(y, ytb, args.loss)
            loss_acc += l_mb * len(xb)
            dwd = dy.T @ h
            dh = dy @ wd_s
            ds = dh * u
            du = dh * silu(g)
            dg = ds * silu_prime(g)
            grads[0] += dg.T @ xb
            grads[1] += du.T @ xb
            grads[2] += dwd
            y_last, yt_last = y, ytb
        n_mb = (vec_step + MICRO - 1) // MICRO
        for i in range(3):
            grads[i] /= n_mb
            m[i] = b1 * m[i] + (1 - b1) * grads[i]
            v[i] = b2 * v[i] + (1 - b2) * (grads[i] ** 2)
            mh = m[i] / (1 - b1 ** step)
            vh = v[i] / (1 - b2 ** step)
            params[i] -= args.lr * mh / (np.sqrt(vh) + eps)
        if step % 5 == 0 or step == 1 or step == args.steps:
            cos, mag, rel, r2, nmse = metrics(y_last, yt_last)
            print(f"paso {step:3d}/{args.steps} loss={loss_acc / vec_step:.6f} "
                  f"cos={cos:.4f} mag={mag:.4f} rel_err={rel:.4f} R2={r2:.4f} NMSE={nmse:.4f} "
                  f"({time.time() - ts:.1f}s, mem_disp={mem_available_mib():.0f} MiB)", flush=True)
        if step % 20 == 0:
            ck = f"{BASE}/checkpoints/ff_destilada_qwen25_{args.tag}_step{step}.npz"
            np.savez(ck, gate_proj=params[0], up_proj=params[1], down_proj=params[2])
            print(f"[checkpoint] paso {step} -> {ck}", flush=True)

    yf, _ = ff_forward(x_ev, wg_s, wu_s, wd_s)
    cos, mag, rel, r2, nmse = metrics(yf, y_ev)
    ok = cos >= 0.95 and r2 > 0.90
    print(f"[eval final {args.tag}] cos={cos:.4f} mag={mag:.4f} rel_err={rel:.4f} R2={r2:.4f} NMSE={nmse:.4f}", flush=True)
    print(f"[eval final {args.tag}] criterio cos>=0.95 y R2>0.90: {'SI' if ok else 'NO'}", flush=True)
    out = f"{BASE}/checkpoints/ff_destilada_qwen25_{args.tag}.npz"
    np.savez(out, gate_proj=params[0], up_proj=params[1], down_proj=params[2])
    print(f"pesos alumna: {out}", flush=True)
    print(f"RESULT tag={args.tag} inter={INTER} batch={args.batch} seq={args.seq} steps={args.steps} "
          f"lr={args.lr} init={args.init} data={args.data} loss={args.loss} "
          f"cos={cos:.4f} mag={mag:.4f} "
          f"rel_err={rel:.4f} R2={r2:.4f} NMSE={nmse:.4f} tiempo={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
