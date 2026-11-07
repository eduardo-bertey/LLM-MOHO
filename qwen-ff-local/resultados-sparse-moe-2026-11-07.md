# Resultados sparse / MoE, capa 0, Qwen2.5-0.5B — 2026-11-07

Todos los tests usan los mismos vectores reales v7: 66.499 train / 2.361 eval,
activaciones post-RMSNorm de tokens reales por Qwen. Meta: cos >= 0.95 y R2 > 0.90
en eval real (nunca entrenada). NMSE = error cuadratico medio / potencia media
de la profesora. Los numeros son del log, no hay PPL de reinsercion en estos
tests locales (eso es Colab, con GPU).

## Resumen

| Test | cos | R2 | NMSE | Meta |
|---|---|---|---|---|
| FF sparse sola, 10% podada | 0.9916 | 0.9808 | 0.0192 | SI |
| MoE v1, train suave, top-2 | 0.9065 | 0.7236 | 0.2764 | NO |
| MoE v2, train top-2 duro, 4 exp. | 0.9532 | 0.9009 | 0.0991 | SI |
| D2DMoE clamp -3, 3 etapas, top-2, 4 exp. | 0.9531 | 0.9001 | 0.0999 | SI |
| D2DMoE clamp -4, 5 exp., top-3 | sin eval final * | — | — | — |
| ToMoE 1 capa, hiperred, p=0.5 | 0.8843 | 0.7577 | 0.2423 | NO |
| ToMoE 1 capa, MoE por co-activacion, top-2 | 0.8467 | 0.5528 | 0.4472 | NO |

\* D2DMoE -4 se corto por reinicio de la VM en etapa 3, paso 80/100.
Train en ese punto: cos 0.9648, R2 0.9238, NMSE 0.0762. No hay eval final.

## 1. PoC sparsity, capa 0

10.000 vectores reales, sin entrenar.
- gate < -10: 0.0% | < -5: 0.0% | < -2: 1.3% | < -1: 9.5% | < 0: 56.7%
- |SiLU(gate)| < 0.01: 3.3%
- Jaccard entre trozos: 0.942 (poca diversidad, casi siempre las mismas neuronas)
- Aproximacion sin entrenar (500 vectores):
  - apagar gate < -1: cos 0.9759, R2 0.9430, NMSE 0.0477
  - apagar gate < 0: cos 0.9184, R2 0.8156, NMSE 0.1544
  - top-1216 por token (25%): cos 0.9128, R2 0.7940, NMSE 0.1724

Conclusion del PoC: el clamp -10 de D2DMoE no apaga nada en Qwen capa 0,
porque la pre-activacion gate va de -7 a 4,6.

## 2. FF sparse sola, 10% podada

Script: `ff_sparse_train.py`, checkpoint `ff_sparse_qwen25_e1sparse10.npz`.
- Podadas las 487 neuronas menos activas, quedan 4377 / 4864 (11.765.376 params).
- Init: copia de la profesora, solo las neuronas que quedan.
- 100 steps, batch 4, seq 64, lr 5e-4, 686 s.
- Eval inicial, podada sin entrenar: cos 0.9885, R2 0.9746, NMSE 0.0254
- **Eval final: cos 0.9916, R2 0.9808, NMSE 0.0192, meta SI**

Esto NO es una MoE, es una FF podada sola.

## 3. MoE v1, train suave

Script: `moe_local_train.py`, checkpoint `moe_qwen25_e1moe.npz`.
- 4 expertos x 1216 por KMeans sobre gate_proj, copia profesora, router 896->4.
- Train con gating suave (los 4 expertos), eval con top-2 duro.
- Eval inicial top-2: cos 0.7977, R2 0.3976, NMSE 0.6024
- Eval final suave: cos 0.9866, R2 0.9682, NMSE 0.0318
- **Eval final top-2: cos 0.9065, R2 0.7236, NMSE 0.2764, meta NO**
- Brecha soft vs top-2: el train no era top-2, al cortar se pierde magnitud.

## 4. MoE v2, train top-2 duro

Script: `moe_local_train_v2.py`, checkpoint `moe_qwen25_e2moe_top2.npz`.
- Mismo setup que v1, pero train ya en top-2 duro con straight-through
  (forward top-2, backward softmax).
- Eval inicial top-2: cos 0.7977, R2 0.3976, NMSE 0.6024
- Paso 50: cos 0.9538, R2 0.9027
- **Eval final top-2: cos 0.9532, R2 0.9009, NMSE 0.0991, lb 1.1063, meta SI**
  (justo: +0.0032 cos, +0.0009 R2)
- 100 steps, 749 s. Usa las 4864 neuronas completas, no la FF podada.

## 5. D2DMoE, 3 etapas

### 5a. Clamp -3, 4 expertos, top-2

Script: `d2dmoe_local.py`, checkpoint `moe_d2dmoe_e2d2d_m3.npz`, 1698 s.
Colab: `colab_moe_d2dmoe.py` (commit 2be4886e).
- Etapa 1 sparsity -3, FF sola: cos 0.9908, R2 0.9788, NMSE 0.0212,
  Hoyer 4.72 -> 0.65, gate < -1 9.4% -> 5.7%
- Etapa 2 parte, KMeans, sin entrenar, top-2: cos 0.8047, R2 0.3982, NMSE 0.6018
- **Etapa 3 fine top-2: cos 0.9531, R2 0.9001, NMSE 0.0999, meta SI**
- Casi identico a MoE v2 (0.9532 / 0.9009 / 0.0991): la sparsity no ayudo al final.
- Clamp -10, cortado en paso 60: Hoyer 0.0000, NMSE 0.0134, cos 0.9943.

### 5b. Clamp -4, 5 expertos, top-3

Script: `d2dmoe_local_v2.py`, checkpoint etapa 1 `ff_d2dmoe_e3d2d_m4k3e5_etapa1.npz`.
- Etapa 1 sparsity -4: **cos 0.9928, R2 0.9831, NMSE 0.0169**,
  Hoyer 0.86 -> 0.28, gate < -1 9.4% -> 8.5%
- Etapa 2 parte, 5 expertos (973, 973, 973, 973, 972), sin entrenar, top-3:
  cos 0.8481, R2 0.3267, NMSE 0.6733, lb 1.1349
- Etapa 3 fine top-3: **cortada por reinicio de la VM (~20:11 -03) en paso 80/100.**
  Train paso 80: cos 0.9648, R2 0.9238, NMSE 0.0762, lb 1.0605.
  No hay eval final ni checkpoint de la MoE. La etapa 1 si quedo guardada.

## 6. ToMoE, 1 capa

Repos clonados: `gaosh/ToMoE` (oficial, arXiv:2501.15316, Phi-2 / LLaMA-2/3 /
Qwen-2.5, sin Gemma) y `loktar00/ToMoE` (fork, agrega Qwen3.5-2B/4B/9B).
Script local: `tomoe_1capa.py` / `tomoe_etapaB.py`,
checkpoint `tomoe_1capa_e1tomoe.npz`.
- Hiperred 896->128->4864, p=0.5, FF congelada, no se tocan los pesos.
- Etapa A inicial: cos 0.8313, R2 0.6412, NMSE 0.3588
- **Etapa A final: cos 0.8843, R2 0.7577, NMSE 0.2423, meta NO**
  (estancada en 0.90 desde el paso 20)
- Etapa B, MoE por co-activacion, router top-2, solo router entrenado:
  inicial cos 0.8224, NMSE 0.5745
- **Etapa B final: cos 0.8467, R2 0.5528, NMSE 0.4472, meta NO**

## 7. NMSE y PPL

Objetivo seguro por capa (24 capas, total NMSE < 0.10, PPL ratio < 1.10):
cos > 0.9979, NMSE < 0.0042, R2 > 0.9958.
Para PPL 4-8% peor (ratio 1.04-1.08): NMSE por capa 0.0017-0.0033.
Ningun test de MoE llega a eso: FF sparse 0.0192 (6-11x arriba),
MoE v2 / D2DMoE ~0.099 (30-58x arriba), ToMoE 0.4472.

## 8. VM

RAM 7.7 Gi total, disponible ~650 Mi, globo del host 6.44 Gi, sin GPU, sin torch.
El reinicio de la VM a las ~20:11 -03 mato el test D2DMoE -4 en etapa 3, paso 80.
Los archivos en `~/workspace` sobreviven al reinicio, los procesos no.
