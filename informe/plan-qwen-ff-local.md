# Plan local seguro: Qwen3-0.8B 4-bit, FF original vs destilada

Fecha: 2026-11-06
Repo de trabajo: `eduardo-bertey/LLM-MOHO`, rama `fix/ff-eval-real-activations`
Alcance: solo Qwen3-0.8B, solo la FF de la capa 0. No se toca K2 en esta prueba.
Estado: PLAN, no ejecutado todavía. Ejecutar solo después de aprobar este MD.

## 1. Respuesta corta: ¿tengo 8 GB libres?

No. Medido en esta máquina el 2026-11-06:

- RAM total: 7,7 Gi
- RAM usada: 7,3 Gi
- RAM libre: 135 Mi
- RAM disponible: 503 Mi
- Swap: 1,0 Gi, usado 369 Mi
- CPU: 4 cores, AMD EPYC
- Disco en home: 149 GB libres
- GPU: ninguna (sin CUDA, verificado antes)
- Python: 3.12.3, numpy 1.26.4, sin torch ni transformers instalados

Conclusión: no hay 8 GB libres. Hay 7,7 GB en total y ahora mismo solo ~0,5 GB disponibles.
La prueba de ~3 GB que proponés es posible solo si se hace en pasos chicos,
con monitoreo y con criterio de parada. No es segura tal cual si se lanza
todo junto sin liberar memoria antes.

## 2. Objetivo

Probar en local, con Qwen3-0.8B en 4-bit:
1. Inferencia con la FF original.
2. Entrenar solo la FF destilada de la capa 0, con batch 4 y seq 64.
3. Comparar FF original vs FF destilada con activaciones reales,
   no solo con ruido gaussiano.

El gaussiano queda como smoke test, como en `ff-in/k2/ff_eval_real.py`.
La métrica que manda es la real.

## 3. Por qué Qwen3-0.8B y por qué solo la FF0

- Qwen3-0.8B completo en bf16 pesa ~1,6 GB. En 4-bit pesa ~0,5 GB.
  Ese es el número que diste vos y es correcto en orden de magnitud.
- Solo se entrena la FF de la capa 0, no el modelo entero.
  Esa FF es del orden de 10-20 M de parámetros (gate/up/down),
  según el intermediate exacto, que se verifica al cargar
  `config.json` en ejecución. En fp32 son <100 MB de pesos.
  Con gradientes y Adam, <300 MB extra para esa FF sola.
- K2-Horizon-0.9B queda fuera de esta prueba: hidden 1536,
  intermediate 5120, 28 capas, y su config usa código remoto.
  Qwen es más simple y su config es estándar.

Nota: el 2026-11-06 el `config.json` de `Qwen/Qwen3-0.8B` en
Hugging Face devolvió "Invalid username or password" desde esta
máquina por curl. Por eso las dimensiones exactas de Qwen3-0.8B
se verifican en el paso 1 al descargar, no se dan por cerradas acá.

## 4. Presupuesto de memoria (estimado, a verificar en ejecución)

| Cosa | Estimado |
|---|---|
| Python + torch CPU + transformers | 500 MB - 1.000 MB |
| Qwen3-0.8B en 4-bit (pesos) | ~500 MB |
| Activaciones, batch 4, seq 64, solo FF0 | ~100-300 MB |
| FF destilada fp32 + gradiente + Adam | ~200-300 MB |
| Tokenizer, prompts, logs | <100 MB |
| Total estimado en pico | ~1,8-2,8 GB |

Tu estimado de ~1,8 GB para el entrenamiento y ~3 GB en total
es razonable como orden de magnitud, pero no incluye el overhead
de torch ni la memoria ya usada por el sistema. Por eso el plan
empieza chico y sube solo si hay memoria disponible.

## 5. Pasos, en orden

### Paso 0 - Pre-chequeo, sin instalar nada pesado
- `free -h`, confirmar disponible >= 1 GB antes de seguir.
- Si disponible < 1 GB, parar. No ejecutar.
- Crear carpeta de trabajo: `~/workspace/qwen-ff-local/`
- No tocar `main` ni la rama de K2. Todo en local, sin push
  hasta que los resultados estén revisados.

### Paso 1 - Instalar solo lo necesario, CPU
- `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- `pip install transformers accelerate bitsandbytes`
- Verificar: `python -c "import torch; print(torch.__version__)"`
- Verificar bitsandbytes en CPU: la cuantización NF4 en CPU
  es limitada. Si bitsandbytes no funciona en CPU, alternativa
  segura: cargar en bf16 solo la FF0 y el resto en 4-bit, o usar
  `torch.float16` con `low_cpu_mem_usage=True` y batch 1.
  No forzar GPU, no hay.

### Paso 2 - Descargar Qwen3-0.8B y verificar config
- Descargar `config.json` y `tokenizer`, leer `hidden_size`,
  `intermediate_size`, `num_hidden_layers` y anotarlos en el log.
- Descargar pesos en 4-bit con `BitsAndBytesConfig(load_in_4bit=True,
  bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)`.
- Si la descarga falla por auth en HF, parar y revisar acceso.
  No usar tokens de GitHub para HF.

### Paso 3 - Smoke test de inferencia, batch 1, seq 32
- Cargar modelo, generar 10 tokens con el prompt
  "Hola, soy Qwen y", temperatura 0, `do_sample=False`.
- Registrar tiempo, tokens/s y memoria con `free -h` antes y después.
- Si OOM o swap > 80%, parar y bajar a seq 16.

### Paso 4 - Capturar activaciones reales de la FF0
- Con un hook en `model.layers[0].mlp`, pasar 8 prompts en español.
- Guardar las activaciones reales en disco, en fp16, máximo 4096.
- Medir media, std, min, max. Comparar con ruido gaussiano:
  si son muy distintas, queda registrado, no se ignora.

### Paso 5 - Entrenar solo la FF destilada
- Inicializar FF MOHO con los pesos de cero / random, como en
  `ff-in/k2/ff_compare.py`, semilla 123.
- Entrenar solo gate/up/down de la capa 0, resto congelado.
- Config segura inicial: batch 4, seq 64, lr 1e-3, AdamW, MSE,
  100 pasos primero, no 10.000.
- Log cada 10 pasos: loss, cos, mag_ratio, rel_err, R2.
- Guardar checkpoint cada 50 pasos en disco local, no en el repo.

### Paso 6 - Evaluar original vs destilada
- Con el script ya creado en la rama,
  `ff-in/k2/ff_eval_real.py`, adaptado a Qwen:
  - métricas reales con activaciones del paso 4,
  - métricas gaussianas solo como smoke test,
  - 10 tokens generados con FF original vs FF destilada.
- Criterio de éxito: cos real > 0,95 y R2 real > 0,90,
  y al menos 7/10 tokens iguales en generación.
  Si no se llega, no se declara éxito, se registra el resultado.

### Paso 7 - Guardar resultados y limpiar
- Guardar `resultados-qwen-ff-local.md` con métricas, memoria
  usada, tiempo y commits.
- Borrar pesos temporales grandes si no se necesitan.
- Solo después de revisar, subir el plan y los resultados
  a la rama, nunca a `main` directo.

## 6. Criterios de parada (abortar la prueba)

Parar inmediatamente si pasa cualquiera de estas:
- Disponible < 200 MB o swap > 80%.
- OOM, proceso matado por el sistema, o `free` muestra disponible 0.
- La descarga de Qwen falla por auth o por red.
- bitsandbytes 4-bit no funciona en CPU y no hay alternativa segura.
- La inferencia de 10 tokens tarda más de 5 minutos en CPU.

En ese caso se guarda el log hasta donde llegó y se revisa el plan,
no se reintenta a ciegas subiendo batch o seq.

## 7. Qué NO se hace en esta prueba

- No se entrena K2, no se entrena Qwen entero, no se hace fine-tune.
- No se sube nada a `main`.
- No se usan tokens de GitHub para Hugging Face.
- No se ejecuta con batch 64 ni seq 1024 como en `ff_train_seq.py`.
  Eso es para Colab con GPU, no para esta máquina.

## 8. Archivos que crea esta prueba

- `~/workspace/qwen-ff-local/plan-qwen-ff-local.md` (este archivo, copia)
- `~/workspace/qwen-ff-local/resultados-qwen-ff-local.md`
- `~/workspace/qwen-ff-local/ff_qwen_eval.py` (adaptación de `ff_eval_real.py` a Qwen)
- Checkpoints locales en `~/workspace/qwen-ff-local/checkpoints/`, no en git.

## 9. Aprobación para ejecutar

No ejecutar los pasos 1-7 hasta que Juls apruebe este plan.
Si aprueba, ejecutar en orden, con los pre-chequeos del paso 0,
y parar en el primer criterio de la sección 6.
