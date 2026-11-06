"""Un batch, un paso: entrada -> FF K2 -> salida -> FF MOHO (cero, sin entrenar).

K2   = down(silu(gate(x)) * up(x))  con pesos layer0_*.weight de HF.
MOHO = misma arquitectura SiLU-gated, pesos creados de cero (randn) y guardados
       como moho_layer0_*.weight en esta misma carpeta. Sin entrenar.

Uso: python ff-in/k2/ff_compare.py   (desde LLM-MOHO)
Memoria: pesos ~94 MB (47 K2 + 47 MOHO) + 1 entrada + 2 salidas.
"""
import json
import os
import time

import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))   # ff-in/k2
MO = os.path.join(BASE, "moho_layer0_gate_proj.weight")
MU = os.path.join(BASE, "moho_layer0_up_proj.weight")
MD = os.path.join(BASE, "moho_layer0_down_proj.weight")
GATE_F = os.path.join(BASE, "layer0_gate_proj.weight")
UP_F = os.path.join(BASE, "layer0_up_proj.weight")
DOWN_F = os.path.join(BASE, "layer0_down_proj.weight")


def load_raw(p, shape):
    data = open(p, "rb").read()
    return torch.frombuffer(bytearray(data), dtype=torch.bfloat16).view(*shape)


def save_raw(t, p):
    open(p, "wb").write(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                         if False else t.detach().cpu().contiguous().numpy().tobytes()
                         if t.dtype != torch.bfloat16 else
                         t.detach().cpu().view(torch.uint8).numpy().tobytes())
    return p


def ff(x, gate, up, down):
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


def stats(y):
    return (f"mean={y.float().mean():.6f} std={y.float().std():.4f} "
            f"min={y.float().min():.4f} max={y.float().max():.4f} l2={y.float().norm():.4f}")


def main():
    t0 = time.time()
    meta = json.load(open(os.path.join(BASE, "meta.json")))
    print(f"[meta] " + ", ".join(f"{k.split('.')[-2]}={v['shape']}" for k, v in meta.items() if k != "__source__"))

    gate = load_raw(GATE_F, (5120, 1536))
    up = load_raw(UP_F, (5120, 1536))
    down = load_raw(DOWN_F, (1536, 5120))
    print(f"[k2] pesos cargados: gate/up/down bf16")

    # MOHO de cero (sin entrenar): si ya existen se cargan, si no se crean y guardan
    if os.path.exists(MO) and os.path.exists(MU) and os.path.exists(MD):
        mg = load_raw(MO, (5120, 1536))
        mu = load_raw(MU, (5120, 1536))
        md = load_raw(MD, (1536, 5120))
        print("[moho] pesos de cero encontrados, reutilizo")
    else:
        g = torch.Generator().manual_seed(123)
        scale = 1.0 / (1536 ** 0.5)
        mg = torch.randn((5120, 1536), generator=g, dtype=torch.float32) * scale
        mu = torch.randn((5120, 1536), generator=g, dtype=torch.float32) * scale
        md = torch.randn((1536, 5120), generator=g, dtype=torch.float32) * (1.0 / (5120 ** 0.5))
        mg = mg.to(torch.bfloat16)
        mu = mu.to(torch.bfloat16)
        md = md.to(torch.bfloat16)
        for t, p in ((mg, MO), (mu, MU), (md, MD)):
            open(p, "wb").write(t.contiguous().view(torch.uint8).numpy().tobytes())
        print(f"[moho] pesos de cero creados y guardados: moho_layer0_*")

    torch.manual_seed(0)
    x = torch.randn(1, 1536, dtype=torch.bfloat16)
    print(f"[in] x shape={tuple(x.shape)} dtype={x.dtype}")

    with torch.no_grad():
        yk = ff(x, gate, up, down)
        ym = ff(x, mg, mu, md)

    print(f"[k2]   out shape={tuple(yk.shape)} {stats(yk)}")
    print(f"[moho] out shape={tuple(ym.shape)} {stats(ym)}")
    print(f"[time] {time.time() - t0:.2f}s")
    try:
        import psutil
        print(f"[mem] RSS python ~{psutil.Process().memory_info().rss / 1e6:.1f} MB")
    except ImportError:
        print("[mem] pesos ~94MB + salidas despreciables")
    print("OK: 1 batch, 1 paso. K2 (pesos HF) y MOHO (cero, sin entrenar) con la misma entrada.")


if __name__ == "__main__":
    main()
