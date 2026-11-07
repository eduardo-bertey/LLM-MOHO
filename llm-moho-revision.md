# Revisión de eduardo-bertey/LLM-MOHO

Fecha de la revisión: 2026-11-06
Repo: `eduardo-bertey/LLM-MOHO`
Alcance revisado: 17 archivos Python, 2.166 líneas, clonado en `/tmp/LLM-MOHO`. Todos los archivos pasaron `py_compile`.

## Veredicto

Compila sin errores, pero es un proyecto experimental, no está listo para usar ni para compartir.

## Qué es

Un laboratorio para jugar con modelos chicos en español, sobre todo `IFM/K2-Horizon-0.9B` y `Qwen/Qwen3-0.8B`. Tiene tres partes:

- `k2/train.py` + `train_data.py`: fine-tune con LoRA y un optimizador propio, LoFT, con datos en streaming de Wikipedia ES (3 MB por bloque), FineWeb2-HQ ES (2 MB) y tweets en español (1 MB).
- `k2/infer.py`, `chat.py`, `eval.py`: chat en GPU, evaluación con `lm-eval` o un mini-bench propio en español, y perplejidad.
- `ff-in/k2/`: destilación de la capa feed-forward 0 de K2 a una copia llamada MOHO, con descarga por rangos del safetensors para bajar solo 3 tensores.

## Lo que está bien

- La idea del optimizador LoFT está bastante completa: reescala gradientes, reproyecta momento y segundo momento, alterna la actualización de A y B, y tiene su propio clipping que intenta respetar esa proyección.
- El pipeline de datos tiene prefetch en hilo, detección de duplicados por MD5 y `skip_blocks` para retomar por bloque.
- `eval.py` tiene buen fallback: si no está `lm-eval`, usa un mini-bench en español.
- `ff_train_seq.py` mide más que la loss: coseno, error relativo y R², y cada 500 pasos compara 10 tokens generados con el modelo original vs. el destilado.
- `qwen3/infer.py` es el archivo más limpio, con `argparse` de verdad.

## Los problemas importantes

### 1. No hay documentación ni reproducibilidad
El README es solo el título, no hay `requirements.txt`, licencia, tests ni CI, y hay un solo commit llamado `o`. Tampoco se sabe qué versión de torch/transformers se usó.

### 2. El entrenamiento no guarda nada intermedio
En `k2/train.py` no hay checkpoints, ni validación, ni semilla, ni scheduler de LR. Si Colab se cae a los 400 pasos, perdés todo. Además pide repo, LR, steps y bloque por `input()`, sin validar, y cualquier valor inválido rompe con un crash.

### 3. Bug en el weight decay — el más fácil de arreglar
En `k2/loft_optim/optimizer.py`, tanto en `AdamW` (línea 88) como en `LoFTAdamW` (línea 251), se hace:

```python
p.data.add_(p.data, alpha=-group['weight_decay'])
```

sin multiplicar por el LR. Con `weight_decay=0` como en `train.py` no se nota, pero si alguien lo pone en 0,01, decae 1% por paso. Es incorrecto.

El arreglo correcto, como en el AdamW de PyTorch, es multiplicar por el LR:

```python
p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
```

El parche listo para aplicar está en `llm-moho-weight-decay.patch`, en esta misma carpeta. Fue verificado contra la copia clonada el 2026-11-06.

### 4. Acumulación de gradiente confusa
Se divide la loss por `GA=8`, pero el conteo de pasos y el cálculo de tokens/s son engañosos, y no hay shuffle ni packing. Los textos se concatenan sin EOS ni separador entre documentos, así que el modelo aprende transiciones entre documentos que no existen.

### 5. El destilado de FF está mal planteado
En `ff_train.py` y `ff_train_seq.py` entrenan con entradas `torch.randn`, gaussianas. Las activaciones reales de un transformer no son gaussianas independientes, así que copiar bien esa entrada no garantiza que funcione con texto real. Es el problema más conceptual del repo.

### 6. `ff_train.py` no corre como está clonado
Espera los archivos `layer0_*.weight` y `moho_*.weight` en disco, pero el `.gitignore` excluye `*.weight`. Solo `ff_train_seq.py` los descarga.

### 7. Seguridad
`trust_remote_code=True` en todos lados, el token de Hugging Face se pide con `input()` en texto plano y se hace `login()` global, y `create_repo(..., private=False)` sube el modelo mergeado como público por defecto.

### 8. Código muerto y duplicado
`wikipedia.py` tiene una clase `WikiDataset` de 227 líneas que `TrainData` no usa, `hf_callbacks.py` no se importa en ningún lado, y en `train.py` hay un `for g in model.parameters(): pass` sin sentido.

### 9. Chat con historial frágil
En `infer.py` y `chat.py` el historial crece sin límite, se trunca a 1.500 caracteres, y se decodifica con `skip_special_tokens=False`, así que el usuario puede ver tokens especiales y de razonamiento crudos.

### 10. Evaluación débil
El mini-bench tiene 24 preguntas y la respuesta correcta es siempre la opción 0, y `eval.py` carga el modelo dos veces seguidas sin liberar VRAM entre una y otra.

## Próximos pasos sugeridos, en orden

1. Aplicar el parche del weight decay.
2. Agregar checkpoints, semilla y EOS/separador entre documentos en `k2/train.py`.
3. Cambiar el destilado de FF para usar activaciones reales en vez de ruido gaussiano.
4. Después, README, `requirements.txt` y quitar `trust_remote_code=True` donde no haga falta.
