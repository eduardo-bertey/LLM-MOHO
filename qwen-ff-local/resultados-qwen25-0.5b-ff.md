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
