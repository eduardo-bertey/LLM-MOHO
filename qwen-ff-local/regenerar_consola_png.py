#!/usr/bin/env python3
"""Regenera consola-ff-0.5b.png desde consola-ff-0.5b.log (ultimas lineas).

Fondo oscuro, texto monoespaciado, como una captura de terminal.
Uso: python3 regenerar_consola_png.py [n_lineas]
"""
import sys

from PIL import Image, ImageDraw, ImageFont

BASE = "/home/hatch/workspace/qwen-ff-local"
LOG = BASE + "/consola-ff-0.5b.log"
PNG = BASE + "/consola-ff-0.5b.png"

N = int(sys.argv[1]) if len(sys.argv) > 1 else 55
lines = open(LOG).read().splitlines()[-N:]
lines = [ln[:150] for ln in lines]

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
    font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 13)
except OSError:
    font = ImageFont.load_default()
    font_b = font

W, H = 1180, 20 + 19 * len(lines) + 10
img = Image.new("RGB", (W, H), (12, 14, 18))
d = ImageDraw.Draw(img)
d.text((10, 4), "consola-ff-0.5b.log  —  FF0 Qwen2.5-0.5B, destilación honesta al 75%",
       fill=(120, 200, 120), font=font_b)
for i, ln in enumerate(lines):
    y = 22 + 19 * i
    if ln.startswith("RESULT") or "[eval final" in ln:
        fill = (120, 220, 130)
        f = font_b
    elif "criterio" in ln:
        fill = (255, 210, 90)
        f = font_b
    elif ln.startswith("[aborto") or "ABORT" in ln:
        fill = (255, 120, 110)
        f = font_b
    elif ln.startswith("===") or ln.startswith("[eval inicial"):
        fill = (130, 180, 255)
        f = font_b
    else:
        fill = (200, 205, 210)
        f = font
    d.text((10, y), ln, fill=fill, font=f)
img.save(PNG)
print(f"guardado: {PNG} ({W}x{H}, {len(lines)} lineas)")
