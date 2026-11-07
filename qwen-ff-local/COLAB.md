# Correr la destilación de la FF en Colab

## 1. Abrí Colab con GPU
- https://colab.research.google.com
- Entorno > Cambiar tipo de entorno > Acelerador por hardware: **GPU (T4)**

## 2. Una sola celda, pegar y correr
```python
!git clone --depth 1 -b fix/ff-eval-real-activations https://github.com/eduardo-bertey/LLM-MOHO.git
%cd LLM-MOHO/qwen-ff-local
!python3 colab_destilar_ff.py
```

Eso solo hace todo: verifica GPU/torch, verifica si el modelo ya está
en `./modelo-qwen2.5-0.5b` y si no lo descarga de `Qwen/Qwen2.5-0.5B`,
elige la FF de la capa 0, captura activaciones reales, destila y evalúa.

## 3. Qué entrena
- FF capa 0, alumna al 75%: intermediate 4864 -> 3648, init de origen,
  sin copiar pesos de la profesora. Pérdida BLD: NMSE + 0.1*(1-cos).
- **Batch 32, seq 512** para empezar (16384 vectores/paso, en micro-batches
  de 1024), 300 pasos, lr 5e-4, seed 123.
- Meta: cos >= 0.95 y R² > 0.90 en eval real.

## 4. Cambiar la capa o el tamaño
```python
!LAYER=12 STEPS=200 python3 colab_destilar_ff.py
!LAYER=23 BATCH=16 SEQ=256 python3 colab_destilar_ff.py
```

## 5. Archivos que quedan
- `ff_destilada_qwen25_capa0.pt` (checkpoint, solo local, NO subir pesos a git)
- `resultados-colab-ff.md`
- `colab_destilar_ff.log`

Bajalos desde Files > Download, o guardalos en Drive antes de que se cierre la sesión.
