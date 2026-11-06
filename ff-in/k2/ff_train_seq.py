"""
Un solo run en Colab: descarga el rango FF de K2 y entrena FF MOHO de cero.
Cada 500 pasos para el destil, carga K2, cambia FF0 por el destil,
genera 10 tokens con original y con destil, compara, libera y sigue.
NO toca k2/infer.py.

Colab: !python ff-in/k2/ff_train_seq.py
"""
import json
import os
import struct
import time

import torch
import torch.nn.functional as F

try:
    import requests
except ImportError:
    requests = None

BASE = os.path.dirname(os.path.abspath(__file__))
URL = "https://huggingface.co/IFM/K2-Horizon-0.9B/resolve/main/model-00000-of-00001.safetensors"
SHAPES = {
    "layer0_gate_proj.weight": (5120, 1536),
    "layer0_up_proj.weight": (5120, 1536),
    "layer0_down_proj.weight": (1536, 5120),
}
KEYS = {
    "layer0_gate_proj.weight": "model.layers.0.mlp.gate_proj.weight",
    "layer0_up_proj.weight": "model.layers.0.mlp.up_proj.weight",
    "layer0_down_proj.weight": "model.layers.0.mlp.down_proj.weight",
}

BATCH = 64
SEQ = 1024
STEPS = 10000
LR = 1e-3
LOG_EVERY = 10
EVAL_EVERY = 500
EVAL_TOKENS = 10
EVAL_PROMPT = "Hola, soy K2 y"


def eval_10_tokens(g, u, d, gw, uw, dw, step):
    """Para el destil: congela gradientes, carga K2, compara 10 tokens."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # detener destil y liberar gradientes
    for p in (g, u, d):
        p.requires_grad_(False)
    print(f"[eval] paso {step}: cargando K2...", flush=True)
    tok = AutoTokenizer.from_pretrained("IFM/K2-Horizon-0.9B", trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        "IFM/K2-Horizon-0.9B", dtype=torch.bfloat16,
        device_map="cuda:0" if device == "cuda" else "cpu",
        trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval()

    def gen10():
        inp = tok(EVAL_PROMPT, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=EVAL_TOKENS,
                                 do_sample=False, use_cache=True)
        ids = out[0][inp.input_ids.shape[1]:].tolist()
        txt = tok.decode(ids, skip_special_tokens=True)
        return ids, txt

    # 1) original: FF K2 de HF
    ids_orig, txt_orig = gen10()

    # 2) destil: copiar pesos MOHO (en CPU float32 -> bf16 -> GPU del modelo)
    mlp = model.model.layers[0].mlp
    # el modelo K2 con trust_remote_code puede omitir prefijo model.
    if mlp is None:
        mlp = model.layers[0].mlp
    pairs = ((mlp.gate_proj.weight, g), (mlp.up_proj.weight, u),
             (mlp.down_proj.weight, d))
    for dst, src in pairs:
        dst.data.copy_(src.detach().cpu().to(dst.dtype))
    ids_dist, txt_dist = gen10()
    match = sum(a == b for a, b in zip(ids_orig, ids_dist))
    print(f"[eval] paso {step}: match {match}/{EVAL_TOKENS}")
    print(f"[eval] orig : {ids_orig} -> {txt_orig!r}")
    print(f"[eval] dist : {ids_dist} -> {txt_dist!r}")

    # liberar modelo y reactivar destil
    del model, tok
    if device == "cuda":
        torch.cuda.empty_cache()
    for p in (g, u, d):
        p.requires_grad_(True)
    print("[eval] destil reanudado", flush=True)
    return match


def _get_range(url, s, e):
    r = requests.get(url, headers={"Range": f"bytes={s}-{e}"}, timeout=120)
    assert r.status_code == 206, f"Range falló: {r.status_code}"
    return r.content


def ensure_k2_weights():
    """Si no existen, baja solo el header + los 3 tensores por Range."""
    if all(os.path.exists(os.path.join(BASE, k)) for k in SHAPES):
        print("[k2] pesos ya en disco")
        return
    if requests is None:
        raise ImportError("pip install requests")
    h = requests.get(URL, headers={"Range": "bytes=0-7"}, timeout=60)
    hlen = struct.unpack("<Q", h.content)[0]
    hdr = json.loads(_get_range(URL, 8, 7 + hlen))
    base = 8 + hlen
    for fname, hfkey in KEYS.items():
        out = os.path.join(BASE, fname)
        if os.path.exists(out):
            continue
        s, eoff = hdr[hfkey]["data_offsets"]
        data = _get_range(URL, base + s, base + eoff - 1)
        open(out, "wb").write(data)
        print(f"[k2] {fname} {len(data)} bytes ({hfkey})")
    json.dump({k: hdr[k] for k in hdr if k != "__metadata__"},
              open(os.path.join(BASE, "meta.json"), "w"), indent=1)


def load_raw(name, shape, dtype=torch.float32):
    raw = open(os.path.join(BASE, name), "rb").read()
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(*shape).to(dtype)


def ff(x, gate, up, down):
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


def main():
    ensure_k2_weights()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    torch.manual_seed(0)

    gw = load_raw("layer0_gate_proj.weight", (5120, 1536)).to(device)
    uw = load_raw("layer0_up_proj.weight", (5120, 1536)).to(device)
    dw = load_raw("layer0_down_proj.weight", (1536, 5120)).to(device)

    # FF MOHO de cero: si hay pesos de cero se reutilizan, si no se crean
    names = [("moho_layer0_gate_proj.weight", (5120, 1536)),
             ("moho_layer0_up_proj.weight", (5120, 1536)),
             ("moho_layer0_down_proj.weight", (1536, 5120))]
    if all(os.path.exists(os.path.join(BASE, n)) for n, _ in names):
        g = load_raw(names[0][0], names[0][1]).to(device)
        u = load_raw(names[1][0], names[1][1]).to(device)
        d = load_raw(names[2][0], names[2][1]).to(device)
        print("[moho] pesos de cero en disco")
    else:
        gen = torch.Generator().manual_seed(123)
        g = (torch.randn(5120, 1536, generator=gen, dtype=torch.float32) * (1 / 1536 ** 0.5)
             ).to(torch.bfloat16).to(torch.float32).to(device)
        u = (torch.randn(5120, 1536, generator=gen, dtype=torch.float32) * (1 / 1536 ** 0.5)
             ).to(torch.bfloat16).to(torch.float32).to(device)
        d = (torch.randn(1536, 5120, generator=gen, dtype=torch.float32) * (1 / 5120 ** 0.5)
             ).to(torch.bfloat16).to(torch.float32).to(device)
        for t, (n, _) in zip((g, u, d), names):
            open(os.path.join(BASE, n), "wb").write(t.cpu().to(torch.bfloat16)
                                                    .contiguous().view(torch.uint8).numpy().tobytes())
        print("[moho] pesos de cero creados y guardados")
        g = g.to(device)
        u = u.to(device)
        d = d.to(device)
    for p in (g, u, d):
        p.requires_grad_(True)

    opt = torch.optim.AdamW([g, u, d], lr=LR, weight_decay=0.01)
    print(f"[train] {STEPS} pasos | batch {BATCH} seq {SEQ} "
          f"({BATCH * SEQ} tokens) | lr {LR} | AdamW | MSE")
    gen = torch.Generator(device=device).manual_seed(42)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x = torch.randn(BATCH * SEQ, 1536, generator=gen,
                        dtype=torch.float32, device=device)
        with torch.no_grad():
            target = ff(x, gw, uw, dw)
        opt.zero_grad(set_to_none=True)
        out = ff(x, g, u, d)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt.step()
        if step % LOG_EVERY == 0 or step == 1:
            cos = F.cosine_similarity(target.detach().ravel(), out.detach().ravel(), dim=0).item()
            mag_ratio = out.norm().item() / max(target.norm().item(), 1e-8)
            rel_err = (out - target).norm().item() / max(target.norm().item(), 1e-8)
            r2 = 1 - ((out - target).pow(2).sum() / target.pow(2).sum()).item()
            print(f" step {step:5d}/{STEPS} loss {loss.detach().item():.6f} cos {cos:.4f} "
                  f"mag_ratio {mag_ratio:.4f} rel_err {rel_err:.4f} R2 {r2:.4f} "
                  f"l2_target={target.norm().item():.1f} l2_out={out.norm().item():.1f}",
                  flush=True)

        if step % EVAL_EVERY == 0:
            try:
                eval_10_tokens(g, u, d, gw, uw, dw, step)
            except Exception as e:
                print(f"[eval] ERROR {e}; sigo con destil", flush=True)
                for p in (g, u, d):
                    p.requires_grad_(True)
                if device == "cuda":
                    torch.cuda.empty_cache()
    print(f"[time] {time.time() - t0:.1f}s")

    for t, name in ((g, "moho_step10k_gate_proj.weight"),
                    (u, "moho_step10k_up_proj.weight"),
                    (d, "moho_step10k_down_proj.weight")):
        open(os.path.join(BASE, name), "wb").write(t.detach().cpu().to(torch.bfloat16)
                                                    .contiguous().view(torch.uint8).numpy().tobytes())
        print(f"[save] {name}")
    print("OK: entrenado en Colab (batch 64 seq 1024, 10000 pasos)")


if __name__ == "__main__":
    main()
