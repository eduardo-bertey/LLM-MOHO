# Especialización de expertos por contribución — MoE capa 0 D2DMoE

Fecha: 2026-11-08 15:40 -03 (18:40 UTC). Script: `expert_specialization.py`, log: `expert_specialization.log` (32 s, solo numpy, lotes de 256).

Checkpoint analizado: `checkpoints/moe_d2dmoe_e3d2d_m4k3e5_resume.npz` — D2DMoE local, clamp -4, 5 expertos, top-3.
Datos: eval v7 (`activaciones-reales-qwen25-v7.npz`, 2361 vectores, nunca entrenados).

Motivo: el paper "SAE Interprets MoE" (arXiv:2611.08089) que Juls pidió explicar hoy dice que el gating no mide contribución. Acá se mide la contribución exacta, sin entrenar un SAE: `contrib_e = w_e * FF_e(x)`, la salida MoE es la suma. Un SAE daría la especialización fina por latente; esto responde la pregunta gruesa: ¿los 5 expertos se reparten el trabajo o son copias?

## Reproducción verificada
Eval final reproducida exacta: **cos 0.9598 / R² 0.9120 / NMSE 0.0880** (cos = media por vector, igual que en `d2dmoe_resume_e3.py`).

## Resultado por experto
| exp | seleccionado | mayor contribuyente | cuota norma | cuota energía |
|-----|--------------|---------------------|-------------|---------------|
| 0 | 76.3% | 18.4% | 25.1% | 24.7% |
| 1 | 63.4% | 22.4% | 20.0% | 19.8% |
| 2 | 58.7% | 19.1% | 20.7% | 21.0% |
| 3 | 64.4% | 33.7% | 24.2% | 24.8% |
| 4 | 37.2% | 6.4% | 10.1% | 9.7% |

## Por token
- Número efectivo de expertos: media **2.78 de 3** (mediana 2.83) — los 3 seleccionados aportan, no hay un ganador que se lo lleve todo.
- Cuota del mayor contribuyente: media **44%**; en **0%** de los tokens un solo experto pasa del 80%; en 86.7% queda por debajo del 50%.
- Par que co-domina más: **expertos 2-3, 28.2%** de los tokens.

## ¿Hacen lo mismo?
- Coseno entre vectores medios de contribución: **0.86–0.93** (alto). Matiz: todos tienen que sumar a la misma salida de la profesora, así que la dirección media común infla este número; no basta para decir "copias".
- Coseno medio entre salidas por token, pares seleccionados juntos: **0.49–0.59** para la mayoría de los pares (0-1, 0-2, 0-3, 1-2, 1-3, 2-3); **0.76–0.77** para los pares con el experto 4 (0-4, 1-4). A nivel de token sí empujan en direcciones algo distintas, salvo el 4, el más redundante.

## Lectura honesta
Ni 5 copias ni 5 especialistas limpios. El trabajo se reparte (dominancia repartida, 2.78 expertos efectivos), con diversidad moderada a nivel de token, y un experto débil: **el 4 es el menos seleccionado (37.2%), el que menos domina (6.4%) y el más redundante con el 0 y el 1**. Si hay que podar o reentrenar un experto, es ese.

## Límite
Esto es la MoE **local D2DMoE** (5 expertos), no la MoE de Colab de Wikipedia (4 expertos, paso 1800, cos 0.9807): ese checkpoint vive en Colab, no en la VM. El mismo análisis sirve ahí si Juls baja el `.pt`. El paso siguiente para especialización fina sería el SAE sobre la salida, como en el paper.
