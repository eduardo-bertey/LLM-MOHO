# Resultados: FF aislada Qwen2.5-0.5B, capa 0 (BLD)

Fecha: 2026-11-07 (UTC). Trabajo: `~/workspace/qwen-ff-local/`
Modelo: Qwen2.5-0.5B (Qwen3-0.8B devuelve 401 en HF desde esta máquina).
Rama: `fix/ff-eval-real-activations`. Meta: **cos ≥ 0,95 y R² > 0,90 en eval**,
mismos parámetros (intermediate 4864), con activaciones reales.

## Correcciones de Juls (2026-11-07)

- **No hay fase de reducción.** Nadie dijo reducir. Se queda en los mismos
  parámetros para las otras FF también, y se ve qué pasa.
- **Xavier no lo nombró Juls.** Fue una elección del primer agente, no una
  instrucción de Juls. Los escalones e1-e3 usaron Xavier por decisión del
  agente; quedan como baseline de control, nada más.
- **Init no tiene por qué ser random, ni los datos.** Técnica: init desde la
  profesora (copia o copia + ruido chico), datos reales de tokens por Qwen.
- **Técnica de referencia:** Blockwise Local Distillation (BLD) de Puzzle
  (NVIDIA): sub-bloque aislado imita al padre, entradas siempre del padre,
  gradientes aislados, profesora congelada. Pérdida NMSE Puzzle + 0,1·(1−cos).
- **Plan aprobado por Juls:** límite 12 horas desde 2026-11-06 23:54 -03.
  Se puede cambiar init, pérdida, lr, batch, pasos y datos con criterio
  propio; los papers y el repo son referencia, no regla.

## Config real de Qwen2.5-0.5B (config.json verificado)

hidden 896, intermediate 4864, 24 capas, 14 cabezas de atención,
2 cabezas KV, head_dim 64, vocab 151936, tie_word_embeddings, bf16,
RoPE theta 1e6. FF0: gate/up 4864×896, down 896×4864,
**13.074.432 parámetros**, 52,3 MB en fp32 (26 MB en bf16).

## Cómo se obtuvieron las activaciones reales

Qwen entero en RAM no entra (modelo bf16 988 MB + torch ~350 MB,
disponible ~0,5-1 Gi). En su lugar, con solo numpy:

- Tokenizer real (`tokenizers` 0.23.3) sobre 32 prompts en español.
- Del safetensors en disco solo se leyeron las rebanadas necesarias:
  415 filas de embedding + pesos de la capa 0 (atención, normas, FF).
- Forward manual capa 0 en numpy: embedding → RMSNorm → atención GQA
  con RoPE (rotate_half) y máscara causal → residual → post_attention_layernorm.
  Esa salida es la entrada real a la FF (el espacio que se copia).
- Autochequeos OK: RoPE conserva norma; atención seq=1 devuelve V;
  RMS por token de la entrada a la FF 0,59-0,71 (esperado ~0,69,
  RMS del peso de la post-norm); decode de los tokens = prompt original.
- **764 vectores reales**: 579 train (24 prompts) + 185 eval (8 prompts
  distintos). Guardados en `activaciones-reales-qwen25.npz` (5,3 MB,
  solo en disco local, NO va a git).
- Estadística: X_ff media 0,0011, std 0,6555, min −9,18, max 12,04;
  corr media entre columnas 0,0716 (el gaussiano da ~0,005).
  Y_prof std 0,2063. Colas pesadas, correlacionadas: no es ruido.

## Destilación honesta al 75% (2026-11-07, Juls)

- **«Es trampa» (Juls, 2026-11-07):** e6r/e7r partían de cos 0,9944
  (profesora+ruido 10%) y llegar a 0,9965 es solo recuperar la
  perturbación, no destilar. Esa meta queda anulada como prueba.
- **Corrección de Juls:** FF reducida al **75%** (intermediate
  4864 → 3648, **9.805.824 params**), **destilar, NO copiar pesos
  del original**: nada de copia de profesora, ni selección de
  neuronas (Channel Contribution anulado), ni profesora+ruido.
  Init de origen del repo (randn·1/√fan_in, seed 123), pérdida
  NMSE + 0,1·(1−cos) sobre activaciones reales, BLD.
- Meta honesta: eval real cos ≥ 0,95 y R² > 0,90 partiendo de un
  init sin copia, con cos inicial bajo.
- Script v6: `--inter` (default 4864) para la alumna reducida.
  Límite 12 h desde 2026-11-07 00:24 -03, reporte cada 30 min.

| Intento | Init | Datos | Batch×seq | Pasos | LR | cos inicial | cos eval | R² eval | NMSE | mag | Resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke75 | origen | reales | 1×64 | 2 | 1e-3 | −0,0027 | 0,2760 | −0,5057 | 1,5057 | 0,97 | Smoke OK: cos inicial bajo, aprende |
| e8r75 | origen | reales | 4×64 | 50 | 1e-3 | −0,0027 | **0,6566** | 0,3723 | 0,6277 | 0,89 | **NO**: train cos 0,9908, eval 0,6566 → memoriza |
| e9r75v2 | origen | reales v2 | 4×64 | 140/200* | 1e-3 | −0,0004 | 0,8606† | 0,7187† | 0,2813† | 0,92† | **NO** (proceso muerto en paso 145, ver nota) |
| e10r75v2c | resume p140 | reales v2 | 8×64 | 5/100 | 3e-3 | 0,8606 | — | — | — | — | **ABORTADO**: lr 3e-3 + Adam reiniciado → loss 0,09→0,75 |
| e11r75v2c | resume p140 | reales v2 | 8×64 | 100 | 1e-3 | 0,8606 | **0,8836** | 0,7592 | 0,2408 | 0,94 | **NO**: mejora +0,023, gap train 0,98 / eval 0,88 |
| e12r75v3c | resume e11 | reales v3 | 4×64 | 300 | 1e-3 | 0,8836 | **0,9173** | 0,8278 | 0,1722 | 0,96 | **NO**: mejora +0,034, gap train 0,976 / eval 0,917; train cos estancado ~0,97 desde paso 140 → cuello de generalización, no de optimización |
| e13r75v4c | resume e12 p300 | reales v4 | 4×64 | 280+20* | 1e-3 | 0,9173 | 0,9255† | 0,8454† | 0,1546† | 0,97† | **NO**: mejor checkpoint paso 280: **cos 0,9298, R² 0,8552**; los últimos 20 pasos con Adam reiniciado a lr 1e-3 lo empeoraron (0,9298→0,9255) |
| e14r75v4c | resume e13 p280 | reales v4 | 4×64 | 200 | 5e-4 | 0,9298 | **0,9381** | 0,8714 | 0,1286 | 0,97 | **NO**: mejora +0,0083, nuevo mejor; train cos 0,9836, gap train 0,984 / eval 0,938; eval casi plana al final (p120 0,9374, p160 0,9377, p200 0,9381) → cuello generalización, toca dato diverso (v5) |
| e15r75v5c | resume e14 final | reales v5 | 4×64 | 300 | 5e-4 | 0,9381 | **0,9454** | 0,8862 | 0,1138 | 0,97 | **NO**: mejora +0,0073, nuevo mejor; el dato diverso v5 ayudó; train cos 0,9829, gap 0,037; faltan 0,0046 para 0,95 y R² 0,8862 < 0,90 |
| e16r75v6c | resume e15 final | reales v6 | 4×64 | 220+80* | 5e-4 | 0,9454 | **0,9507**† | 0,8973† | 0,1027† | 0,97 | **NO (casi)**: cos ≥0,95 SÍ (0,9507) pero R² 0,8973 < 0,90 por 0,0027; el checkpoint paso 220 ya daba cos 0,9500 / R² 0,8958; los 80 pasos de la continuación sumaron apenas +0,0007 cos |
| e17r75v7c | resume e16c final | reales v7 | 4×64 | 300 | 5e-4 | 0,9507 | **0,9540** | 0,9038 | 0,0962 | 0,98 | **SÍ — META HONESTA ALCANZADA**: cos ≥0,95 y R² >0,90; train cos 0,9792; el dato diverso v7 empujó R² de 0,8973 a 0,9038 |
| e17br75v7c | resume e16c final | reales v7 | 4×64 | 300 | 5e-4 | 0,9507 | **0,9542** | 0,9041 | 0,0959 | 0,98 | **SÍ — CONFIRMA la meta**: segunda semilla (133), mismo punto de partida y datos que e17; resultado casi idéntico (0,9540→0,9542 cos, 0,9038→0,9041 R²): no fue suerte de semilla |

\* e9r75v2 murió en el paso ~145 (sin mensaje de error, sin OOM en
dmesg; la sesión del proceso se perdió). Checkpoint paso 140 intacto.
† eval del checkpoint paso 140 medida con corrida `--steps 0`
(`eval140`): cos 0,8606, R² 0,7187. El gap se achicó mucho vs e8
(train 0,99 / eval 0,66 → train 0,955 / eval 0,86 con datos v2).
Script v7: `--resume` para continuar desde un checkpoint (Adam m/v
reinician en cero — ojo: subir el lr justo después de un resume
desestabiliza, como pasó en e10).

## Diagnóstico e8r75: sobreajuste por falta de datos

Train cos 0,9908 (R² 0,9837) pero eval cos 0,6566 (R² 0,3723).
50 pasos × 256 vectores sacados de solo 579 vectores únicos de train
≈ 22 épocas: la alumna memoriza el train y no generaliza a los 8
prompts nuevos de eval. Más pasos o más lr no arreglan esto.
**La salida es más datos reales**: capturar miles de vectores de
prosa real en español (artículos de Wikipedia), no más épocas sobre
los mismos 579. Plan: v2 con ~8000 train / ~2000 eval, y re-entrenar
desde origen, 200 pasos, lr 1e-3.
Captura v2 hecha (2026-11-07 ~03:35 UTC, 184 s): 12 artículos de
Wikipedia en español en ventanas de 64 tokens, más los prompts
originales. **Train 9539 vectores, eval 2361 vectores**
(10 artículos train + 24 prompts; 2 artículos eval distintos —
Café, Electricidad — + 8 prompts). Archivo
`activaciones-reales-qwen25-v2.npz` (solo local, NO va a git);
script `capturar_activaciones_v2.py`; el entrenamiento acepta
`--data-file` (default v1).

\* e13: el proceso principal murió silencioso en el paso ~290 (2ª muerte
silenciosa, igual que e9; sin error ni OOM, memoria disponible ~2 Gi al
relanzar). Se continuó desde el checkpoint paso 280 con 20 pasos más.
† eval final de la continuación (20 pasos, seed 127). El mejor peso de
e13 es el **checkpoint paso 280: cos 0,9298, R² 0,8552, NMSE 0,1448**,
medido como `[eval inicial]` de la continuación. Lección: tras un resume
(Adam reiniciado) el lr 1e-3 mete ruido y puede empeorar el eval; el
próximo intento (e14) usa lr 5e-4 desde ese checkpoint paso 280.
Captura v4: tokens 1900-2900 de los 10 artículos train, +9600 vectores
→ train 28739, eval 2361 sin cambios.
e14 (2026-11-07 ~05:35-06:18 UTC, 1597 s): resume del checkpoint
paso 280 de e13, lr 5e-4 (más suave que 1e-3, sin divergencia),
200 pasos, seed 128, datos v4. Train cos: paso 1 0,9737 → paso 120
0,9844 → paso 200 0,9836. Eval final **cos 0,9381, R² 0,8714,
NMSE 0,1286 — NO, nuevo mejor (+0,0083)**. Curva de eval medida
post-hoc sobre checkpoints (`--steps 0`): paso 120 → 0,9374,
paso 160 → 0,9377, paso 200 → 0,9381: el eval sube monótono pero
casi plano (+0,0007 en los últimos 80 pasos). Más pasos sobre los
mismos datos no llegan a 0,95 a este ritmo.
Captura v5 (2026-11-07 ~06:23 UTC): 10 artículos NUEVOS de
Wikipedia en español (Matemáticas, Biología, Química, Música,
Filosofía, Economía, Medicina, Astronomía, Literatura, Geografía),
tokens 0-900 c/u, ~8960 vectores nuevos que se suman al train v4
→ train ~37700, eval 2361 sin cambios. Scripts
`descargar_wiki_v5.py` + `capturar_activaciones_v5.py`. Próximo
intento (e15): resume del final de e14, datos v5, 300 pasos,
lr 5e-4, seed 129.
e15 (2026-11-07 ~06:29-07:08 UTC, 2365 s): resume del final de
e14, lr 5e-4, 300 pasos, seed 129, datos v5. Train cos: paso 240
0,9778 → paso 280 0,9825 → paso 300 0,9829 (el dato nuevo es más
difícil al principio, después se recupera). Eval final
**cos 0,9454, R² 0,8862, NMSE 0,1138 — NO, nuevo mejor (+0,0073)**.
El dato diverso v5 ayudó casi tanto como el cambio de lr de e14.
Faltan 0,0046 para cos 0,95 y R² 0,8862 < 0,90.
Push a GitHub 2026-11-07 ~06:30-07:00 UTC: `github push-files`
(App eldeeon20) llegó a pedir aprobación de escritura pero la
tarjeta expiró sin que nadie la aprobara (approval_expired,
30 min de espera; no quedó ninguna aprobación pendiente).
El push NO se hizo. Queda pendiente: reintentar con aprobación
del usuario o con el token eduardo-bertey por stdin como antes.
El PNG no lo acepta push-files (no es UTF-8): va por API aparte.
Captura v6: tokens 900-2900 de los 10 artículos nuevos de la v5,
~19840 vectores nuevos → train ~57500, eval 2361 sin cambios.
Script `capturar_activaciones_v6.py`. Próximo intento (e16):
resume del final de e15, datos v6, 300 pasos, lr 5e-4, seed 130.
e16 (2026-11-07 ~07:18-07:47 UTC): resume del final de e15,
lr 5e-4, 300 pasos, seed 130, datos v6. **El proceso principal
murió silencioso en el paso ~225 (3ª muerte, igual que e9 y e13;
sin OOM en dmesg, 2,1 Gi disponibles; coincidió con un reinicio
del servicio)**. Checkpoint paso 220 intacto. Su `[eval inicial]`
al reanudar ya era **cos 0,9500, R² 0,8958, NMSE 0,1042**: el
umbral de cos se cruzó en el paso 220. Continuación e16c
(80 pasos, seed 131, lr 5e-4): train cos 0,975-0,981, eval final
**cos 0,9507, R² 0,8973, NMSE 0,1027 — NO por R² (falta 0,0027)**.
Los últimos 80 pasos sumaron apenas +0,0007 de cos: más pasos
sobre los mismos datos ya no rinden. Próximo intento (e17):
10 artículos NUEVOS (v7) + 300 pasos desde el final de e16c.
Captura v7 (2026-11-07 ~07:56 UTC, 153 s): 10 artículos NUEVOS
(Arte, Cine, Historia, Tecnología, Internet, Política,
Psicología, Educación, Deporte, Religión), tokens 0-900 c/u,
+8960 vectores → train 66499, eval 2361 sin cambios. Scripts
`descargar_wiki_v7.py` + `capturar_activaciones_v7.py`.
e17 (2026-11-07 ~08:00-08:35 UTC, 1488 s): resume del final de
e16c, lr 5e-4, 300 pasos, seed 132, datos v7. Train cos:
paso 150 0,9818 → paso 300 0,9792. Eval final **cos 0,9540,
R² 0,9038, NMSE 0,0962 — META HONESTA ALCANZADA (cos ≥0,95 y
R² >0,90)**. Progresión honesta completa desde init origen
(cos inicial −0,0027): e8 0,6566 → e9 0,8606 → e11 0,8836 →
e12 0,9173 → e13 0,9298 → e14 0,9381 → e15 0,9454 → e16 0,9507
→ e17 0,9540. Los márgenes son justos (cos +0,0040, R² +0,0038
sobre el umbral), así que se corre una confirmación con segunda
semilla (e17b, seed 133, mismo punto de partida e16c, datos v7).
e17b (2026-11-07 ~08:37-09:02 UTC, 1491 s): confirmación,
seed 133. Train cos: paso 270 0,9831 → paso 300 0,9798. Eval
final **cos 0,9542, R² 0,9041, NMSE 0,0959 — SÍ**. Diferencia
con e17: +0,0002 cos, +0,0003 R². **Meta honesta ALCANZADA y
CONFIRMADA con dos semillas.** Cos inicial del proyecto (init
origen, sin copia): −0,0027. Cos final: 0,9540 / 0,9542.
**Qué falta:** (1) reinserción de la FF destilada en el modelo
completo y test PPL/KL — la similitud de salidas de la FF no
garantiza por sí sola que el modelo genere igual; (2) el eval
es chico (2361 vectores, 2 artículos + 8 prompts); (3) el push
a GitHub sigue pendiente (la aprobación de `push-files` expiró;
reintentar con aprobación del usuario o con el token
eduardo-bertey por stdin). Los pesos finales están en
`checkpoints/ff_destilada_qwen25_e17r75v7c.npz` (seed 132) y
`checkpoints/ff_destilada_qwen25_e17br75v7c.npz` (seed 133),
solo locales, no van a git.

## Tabla de intentos (eval en datos retenidos, lo que manda)

| Intento | Init | Datos | Batch×seq | Pasos | LR | cos eval | R² eval | NMSE | mag | Resultado |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 (matado) | origen* | gauss fresco | 4×64 | 10/100 | 1e-3 | — | — | — | — | Parado: ~15 s/paso, no terminaba |
| e1 | Xavier | gauss pool 4096 | 1×64 | 20 | 1e-3 | 0,0430 | −0,4505 | — | 0,72 | No generaliza |
| e2 | Xavier | gauss pool 4096 | 1×64 | 20 | 1e-2 | 0,0047 | −263,0 | — | 16,2 | **Diverge** |
| e3 | Xavier | gauss pool 4096 | 4×64 | 50 | 3e-3 | 0,0586 | −0,7061 | — | 0,90 | No generaliza |
| e4r | origen | reales | 1×64 | 20 | 1e-3 | 0,6023 | 0,2546 | — | 0,91 | Gran salto vs gauss |
| e5copy | teacher (copia) | reales | — | 0 | — | **1,0000** | 1,0000 | 0,0000 | 1,00 | Sanity: pipeline OK |
| e5r | teacher+ruido 10% | reales | 1×64 | 20 | 1e-3 | 0,9515 | 0,8872 | 0,1128 | 0,91 | cos pasa, R² no |
| e6r | teacher+ruido 10% | reales | 4×64 | 50 | 3e-4 | **0,9965** | **0,9918** | **0,0082** | 0,98 | **META: SI** |
| e7r | teacher+ruido 10% | reales | 4×64 | 50 | 3e-4 | **0,9966** | **0,9921** | **0,0079** | 0,98 | **META: SI** (seed 456, confirmación) |

\* v1 usaba el init de origen del repo (randn·1/√fan_in), no Xavier.
Xavier fue la elección del primer agente en e1-e3, no de Juls.

## Hallazgos

1. **Con gaussiano la alumna memoriza, no generaliza.** e3 llegó a cos
   0,61 en el lote de entrenamiento pero solo 0,059 en eval (vectores
   frescos). Diagnóstico: pool visto cos 0,217 vs fresco 0,043.
   El cos de entrenamiento con gaussiano no sirve para nada.
2. **Con datos reales el salto es enorme.** e4r (origen, 20 pasos):
   eval cos 0,6023, R² 0,2546. El mismo presupuesto con gaussiano
   daba 0,043. El espacio real es lo que hay que copiar.
3. **lr 1e-2 diverge** (loss 0,14 → 160, mag 43). lr 1e-3 es el techo
   práctico con batch chico; con batch 4, lr 3e-3 entrena estable.
4. **Con init profesora+ruido 10% se parte de cos 0,9944** y el
   entrenamiento con lr 1e-3 lo empeora (0,9515 en eval, R² 0,8872):
   pasos demasiado grandes para un buen init. Hay que suavizar:
   batch más grande + lr más bajo (e6r: batch 4, lr 3e-4).
5. **Velocidad:** BLAS de numpy 1,8 GFLOPS (1 hilo). Un paso de
   256 vectores tarda ~7 s; uno de 64 vectores, ~1,7 s.
   Batch 32 × seq 1024 (32768 vectores/paso) no es viable de una:
   se procesa en micro-lotes de 256 con acumulación de gradientes.
6. **Memoria:** todo el experimento corre con 280-960 MiB disponibles;
   el proceso usa ~560 MB RSS. Qwen queda en disco (988 MB),
   la RAM es solo para la FF.

## Archivos

- `ff_aislada_qwen25.py` — script v5 (BLD, NMSE Puzzle + 0,1·(1−cos),
  init origen/teacher/teacher_noise, datos reales o gauss, checkpoint
  cada 20 pasos). Gradientes verificados contra diferencias finitas.
- `capturar_activaciones_reales.py` — captura FF0 con tokens reales.
- `activaciones-reales-qwen25.npz` — solo local, no va a git.
- `consola-ff-0.5b.log` — consola completa de todos los intentos.
- `consola-ff-0.5b.png` — captura de la consola.
- `modelo-qwen2.5-0.5b/` — modelo en disco local, no va a git.

## Estado

**Meta ALCANZADA en e6r (2026-11-07 ~03:05 UTC) y CONFIRMADA en e7r
(seed 456, ~03:10 UTC):** eval cos **0,9965 / 0,9966**,
R² **0,9918 / 0,9921**, NMSE **0,0082 / 0,0079**, mag 0,98,
en ~326 s cada una. Config: init profesora+ruido 10%, batch 4, seq 64,
50 pasos, lr 3e-4, pérdida NMSE Puzzle + 0,1·(1−cos),
datos reales (579 train / 185 eval).
Supera también el margen sugerido por la investigación
(cos ≥0,97 y NMSE ≤0,06). La reproducibilidad con otra semilla
confirma que no fue suerte.
Próximo paso natural: mismos parámetros para las otras FF
(capas 1, 12, 23, …), sin reducir, y ver qué pasa.
