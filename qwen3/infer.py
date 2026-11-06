"""Chat Qwen3-0.8B (Qwen). Loop interactivo en GPU.

Uso Colab: !python qwen3/infer.py [--max-new 2048 --temp 0.6 --top-p 0.95 --top-k 20]
Escribí tu mensaje y Enter. 'salir' para terminar.
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-0.8B"


def main():
    ap = argparse.ArgumentParser(description="Chat Qwen3-0.8B")
    ap.add_argument("--model", default=BASE, help="ID en HF o ruta local")
    ap.add_argument("--max-new", type=int, default=2048, help="max tokens nuevos")
    ap.add_argument("--temp", type=float, default=0.6, help="temperatura (0-2)")
    ap.add_argument("--top-p", type=float, default=0.95, help="nucleus sampling")
    ap.add_argument("--top-k", type=int, default=20, help="top-k por token")
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--compile", action="store_true", help="torch.compile")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"],
                    help="implementación de atención")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "sin CUDA: este script exige GPU"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation=args.attn,
        )
        print(f"attn: {args.attn}")
    except Exception as e:
        print(f"attn {args.attn} no soportado ({e}), eager")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    model.eval()
    model.config.use_cache = True

    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile: on")
        except Exception as e:
            print(f"torch.compile no ({e}), sigo sin compilar")

    print(f"Modelo en: {model.device} | VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print("Chat listo (salir = terminar).")

    history = []
    while True:
        try:
            msg = input("\nvos> ").strip()
        except EOFError:
            break
        if msg.lower() in ("salir", "exit", "quit"):
            break
        if not msg:
            continue

        history.append({"role": "user", "content": msg})
        text = tok.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inp = tok(text, return_tensors="pt").to(model.device)
        inp.pop("token_type_ids", None)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=args.max_new,
                                 temperature=args.temp, top_p=args.top_p,
                                 top_k=args.top_k, do_sample=True,
                                 repetition_penalty=args.repetition_penalty,
                                 use_cache=True)
        dt = time.time() - t0
        new_toks = out[0].shape[0] - inp["input_ids"].shape[1]
        gen = out[0][inp["input_ids"].shape[1]:]
        ans = tok.decode(gen, skip_special_tokens=False)
        print(f"\nqwen3> {ans}")
        print(f"[{new_toks} toks en {dt:.1f}s = {new_toks / max(dt, 1e-3):.1f} t/s | "
              f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB]")
        history.append({"role": "assistant", "content": ans[-1500:]})


if __name__ == "__main__":
    main()
