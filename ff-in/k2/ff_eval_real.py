"""
Evalúa la FF0 destilada (MOHO) contra la FF0 real de K2 con activaciones
reales, no solo con ruido gaussiano.

Por qué: en ff_train.py y ff_train_seq.py el destilado se entrena y se
mide con x ~ N(0, 1) iid. Eso es un smoke test útil y común para ver si
la función se aprende en un caso simple, pero las activaciones reales
de un transformer no son gaussianas independientes: están correlacionadas
por el texto, tienen otra media, otra varianza por dimensión y otra
estructura. Un buen coseno con ruido no garantiza que copie bien con
texto real. Por eso este script mide las dos cosas por separado y la
métrica que manda es la de activaciones reales.

Qué hace:
  1. Carga K2 una sola vez (no en cada evaluación) y registra con un hook
     la entrada real a la MLP de la capa 0 al pasar prompts en español.
  2. Compara FF_K2 vs FF_MOHO sobre esas activaciones reales:
     coseno, mag_ratio, rel_err y R2.
  3. Repite la misma comparación sobre ruido gaussiano, etiquetada
     explícitamente como smoke test, para no confundirla con la real.
  4. Compara 10 tokens generados con la FF original y con la FF
     destilada, reutilizando el mismo modelo cargado.

Uso:
  python ff-in/k2/ff_eval_real.py [--gaussian-only] [--eval-tokens 10]
  python ff-in/k2/ff_eval_real.py --prompts-file prompts.txt

Requiere: torch, transformers, requests (para bajar los pesos FF por
Range si no están en disco, igual que ff_train_seq.py).
Los pesos *.weight están en .gitignore, no se suben al repo.
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "IFM/K2-Horizon-0.9B"

DEFAULT_PROMPTS = [
    "Hola, soy K2 y",
    "La capital de Argentina es",
    "El agua hierve a",
    "Don Quijote fue escrito por",
    "La fotosíntesis ocurre en",
    "El río más largo del mundo es",
    "Buenos Aires es la capital de",
    "La velocidad de la luz es de",
]

# Reutiliza la descarga por Range y load_raw de ff_train_seq.py si está
# disponible, para no duplicar esa lógica. Si no, define versiones locales.
try:
    from ff_train_seq import ensure_k2_weights, load_raw, ff
except Exception:  # pragma: no cover - fallback standalone
    import struct

    try:
        import requests
    except ImportError:
        requests = None

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

    def _get_range(url, s, e):
        r = requests.get(url, headers={"Range": f"bytes={s}-{e}"}, timeout=120)
        assert r.status_code == 206, f"Range falló: {r.status_code}"
        return r.content

    def ensure_k2_weights():
        if all(os.path.exists(os.path.join(BASE, k)) for k in SHAPES):
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
            open(out, "wb").write(_get_range(URL, base + s, base + eoff - 1))

    def load_raw(name, shape, dtype=torch.float32):
        raw = open(os.path.join(BASE, name), "rb").read()
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(*shape).to(dtype)

    def ff(x, gate, up, down):
        return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


def metrics(target, out):
    """Métricas comparables entre la evaluación gaussiana y la real."""
    target = target.detach().float()
    out = out.detach().float()
    cos = F.cosine_similarity(target.ravel(), out.ravel(), dim=0).item()
    mag_ratio = out.norm().item() / max(target.norm().item(), 1e-8)
    rel_err = (out - target).norm().item() / max(target.norm().item(), 1e-8)
    r2 = 1 - ((out - target).pow(2).sum() / target.pow(2).sum()).item()
    return {"cos": cos, "mag_ratio": mag_ratio, "rel_err": rel_err, "r2": r2}


def evaluate_on_activations(x, gw, uw, dw, g, u, d):
    """Compara FF_K2 vs FF_MOHO sobre un lote de activaciones dadas."""
    with torch.no_grad():
        target = ff(x, gw, uw, dw)
        out = ff(x, g, u, d)
    return metrics(target, out)


def load_moho_weights(device, trained_name="moho_step10k"):
    """Carga pesos MOHO entrenados si existen, si no los de cero / random."""
    names = [
        (f"{trained_name}_gate_proj.weight", (5120, 1536)),
        (f"{trained_name}_up_proj.weight", (5120, 1536)),
        (f"{trained_name}_down_proj.weight", (1536, 5120)),
    ]
    if all(os.path.exists(os.path.join(BASE, n)) for n, _ in names):
        print(f"[moho] pesos entrenados en disco: {trained_name}_*")
        return tuple(load_raw(n, s).to(device) for n, s in names)
    zero_names = [
        ("moho_layer0_gate_proj.weight", (5120, 1536)),
        ("moho_layer0_up_proj.weight", (5120, 1536)),
        ("moho_layer0_down_proj.weight", (1536, 5120)),
    ]
    if all(os.path.exists(os.path.join(BASE, n)) for n, _ in zero_names):
        print("[moho] pesos de cero en disco (sin entrenar)")
        return tuple(load_raw(n, s).to(device) for n, s in zero_names)
    print("[moho] sin pesos en disco: baseline random (sin entrenar)")
    gen = torch.Generator().manual_seed(123)
    g = torch.randn(5120, 1536, generator=gen) * (1 / 1536 ** 0.5)
    u = torch.randn(5120, 1536, generator=gen) * (1 / 1536 ** 0.5)
    d = torch.randn(1536, 5120, generator=gen) * (1 / 5120 ** 0.5)
    return g.to(device), u.to(device), d.to(device)


def get_layer0_mlp(model):
    """Devuelve la MLP de la capa 0, con o sin prefijo 'model.'."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError("No se encontró model.layers[0] en el modelo cargado")
    return layers[0].mlp


def collect_real_activations(model, tok, prompts, max_activations=4096):
    """Pasa prompts reales por K2 y captura la entrada real a la MLP0."""
    mlp = get_layer0_mlp(model)
    captured = []

    def hook(_module, inputs, _output):
        # inputs[0]: (batch, seq, 1536), entrada real a la FF0
        captured.append(inputs[0].detach().float().cpu())

    handle = mlp.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for prompt in prompts:
                inp = tok(prompt, return_tensors="pt").to(model.device)
                # model(**inp) sin generate: solo forward para capturar
                model(**inp)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("No se capturó ninguna activación real")
    x = torch.cat([t.reshape(-1, t.shape[-1]) for t in captured], dim=0)
    print(f"[real] activaciones capturadas: {x.shape[0]} x {x.shape[1]}")
    print(f"[real] media={x.mean().item():.4f} std={x.std().item():.4f} "
          f"min={x.min().item():.4f} max={x.max().item():.4f}")
    if x.shape[0] > max_activations:
        idx = torch.randperm(x.shape[0])[:max_activations]
        x = x[idx]
        print(f"[real] subsample a {max_activations} activaciones")
    return x


def generate_with_ff(model, tok, prompt, n_tokens, ff_weights=None):
    """Genera n_tokens; si ff_weights está dado, reemplaza la FF0 antes."""
    mlp = get_layer0_mlp(model)
    original = None
    if ff_weights is not None:
        g, u, d = ff_weights
        original = tuple(p.detach().clone() for p in
                         (mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight))
        for dst, src in zip((mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight),
                            (g, u, d)):
            dst.data.copy_(src.detach().to(dst.dtype))
    try:
        inp = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=n_tokens,
                                 do_sample=False, use_cache=True)
        ids = out[0][inp["input_ids"].shape[1]:].tolist()
        return ids, tok.decode(ids, skip_special_tokens=True)
    finally:
        if original is not None:
            for dst, src in zip((mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight),
                                original):
                dst.data.copy_(src)


def main():
    ap = argparse.ArgumentParser(description="Eval FF0 MOHO con activaciones reales")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--prompts-file", default=None, help="Un prompt por línea")
    ap.add_argument("--max-activations", type=int, default=4096)
    ap.add_argument("--eval-tokens", type=int, default=10)
    ap.add_argument("--eval-prompt", default="Hola, soy K2 y")
    ap.add_argument("--gaussian-only", action="store_true",
                    help="Solo smoke test gaussiano, sin cargar el modelo completo")
    ap.add_argument("--moho", default="moho_step10k",
                    help="Prefijo de pesos MOHO entrenados (sin _gate_proj.weight)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    torch.manual_seed(0)

    ensure_k2_weights()
    gw = load_raw("layer0_gate_proj.weight", (5120, 1536)).to(device)
    uw = load_raw("layer0_up_proj.weight", (5120, 1536)).to(device)
    dw = load_raw("layer0_down_proj.weight", (1536, 5120)).to(device)
    g, u, d = load_moho_weights(device, trained_name=args.moho)

    # 1) Smoke test gaussiano: útil, pero NO es la métrica principal.
    gen = torch.Generator(device=device).manual_seed(42)
    x_gauss = torch.randn(4096, 1536, generator=gen, dtype=torch.float32, device=device)
    m_gauss = evaluate_on_activations(x_gauss, gw, uw, dw, g, u, d)
    print(f"[gauss SMOKE] cos {m_gauss['cos']:.4f} mag {m_gauss['mag_ratio']:.4f} "
          f"rel_err {m_gauss['rel_err']:.4f} R2 {m_gauss['r2']:.4f} "
          f"(solo smoke test, no evalúa texto real)")

    if args.gaussian_only:
        print("OK: smoke test gaussiano hecho. Usá sin --gaussian-only para la eval real.")
        return

    # 2) Evaluación real.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        device_map="cuda:0" if device == "cuda" else "cpu",
        trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval()

    t0 = time.time()
    x_real = collect_real_activations(model, tok, prompts,
                                      max_activations=args.max_activations)
    m_real = evaluate_on_activations(x_real.to(device), gw, uw, dw, g, u, d)
    print(f"[REAL] cos {m_real['cos']:.4f} mag {m_real['mag_ratio']:.4f} "
          f"rel_err {m_real['rel_err']:.4f} R2 {m_real['r2']:.4f} "
          f"({x_real.shape[0]} activaciones reales)")
    print(f"[REAL] gauss cos {m_gauss['cos']:.4f} vs real cos {m_real['cos']:.4f} "
          f"-> si difieren mucho, el destilado no generaliza del ruido al texto")

    # 3) Generación: mismo modelo, FF original vs FF destilada.
    ids_orig, txt_orig = generate_with_ff(model, tok, args.eval_prompt, args.eval_tokens)
    ids_dist, txt_dist = generate_with_ff(model, tok, args.eval_prompt, args.eval_tokens,
                                          ff_weights=(g, u, d))
    match = sum(a == b for a, b in zip(ids_orig, ids_dist))
    print(f"[gen] match {match}/{args.eval_tokens}")
    print(f"[gen] orig : {ids_orig} -> {txt_orig!r}")
    print(f"[gen] dist : {ids_dist} -> {txt_dist!r}")
    print(f"[time] {time.time() - t0:.1f}s")
    print("OK: eval real hecha. La métrica que manda es [REAL], no [gauss SMOKE].")


if __name__ == "__main__":
    main()
