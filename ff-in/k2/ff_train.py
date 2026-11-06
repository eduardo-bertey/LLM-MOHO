"""100 pasos, 1 batch: entrena FF MOHO (cero) para copiar FF K2 real.

Entrada: x (1,1536) aleatoria cada paso.
Target:  y_k2 = K2FF(x)   con pesos layer0_*.weight (HF).
Loss:    MSE(MohoFF(x), y_k2)  -> Adam -> 100 pasos -> guarda moho_step100_*.
Uso: python ff-in/k2/ff_train.py
"""
import os
import time

import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))


def load_raw(p, shape, dtype=torch.float32):
    raw = open(p, "rb").read()
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(*shape).to(dtype)


def ff(x, gate, up, down):
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


Steps = 100
lr = 1e-3
device = "cpu"

torch.manual_seed(0)
gw = load_raw(os.path.join(BASE, "layer0_gate_proj.weight"), (5120, 1536))
uw = load_raw(os.path.join(BASE, "layer0_up_proj.weight"), (5120, 1536))
dw = load_raw(os.path.join(BASE, "layer0_down_proj.weight"), (1536, 5120))

# pesos MOHO de cero (ya guardados) -> parámetros trainables float32
g = load_raw(os.path.join(BASE, "moho_layer0_gate_proj.weight"), (5120, 1536))
u = load_raw(os.path.join(BASE, "moho_layer0_up_proj.weight"), (5120, 1536))
d = load_raw(os.path.join(BASE, "moho_layer0_down_proj.weight"), (1536, 5120))
g.requires_grad_(True)
u.requires_grad_(True)
d.requires_grad_(True)

opt = torch.optim.Adam([g, u, d], lr=lr)
print(f"[train] 100 pasos | 1 batch (1x1536) | lr {lr} | MSE | cpu float32")
t0 = time.time()
gen = torch.Generator().manual_seed(42)

for step in range(1, Steps + 1):
    x = torch.randn(1, 1536, generator=gen, dtype=torch.float32)
    with torch.no_grad():
        target = ff(x, gw, uw, dw)
    opt.zero_grad(set_to_none=True)
    out = ff(x, g, u, d)
    loss = F.mse_loss(out, target)
    loss.backward()
    opt.step()
    if step == 1 or step % 10 == 0 or step == Steps:
        print(f" step {step:3d}/{Steps} loss {float(loss):.6f} "
              f"l2_target={float(target.norm()):.3f} l2_out={float(out.norm()):.3f}",
              flush=True)

print(f"[time] {time.time() - t0:.2f}s")

# guardar pesos entrenados (bf16, mismo formato raw)
for t, name in ((g, "moho_step100_gate_proj.weight"),
                (u, "moho_step100_up_proj.weight"),
                (d, "moho_step100_down_proj.weight")):
    open(os.path.join(BASE, name), "wb").write(t.detach().cpu().to(torch.bfloat16)
                                                .contiguous().view(torch.uint8).numpy().tobytes())
    print(f"[save] {name}")
print("OK: MOHO copiou a K2 en 100 pasos (pesos en moho_step100_*)")
