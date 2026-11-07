# Compresión de la KV cache en LLMs — Informe para LLM-MOHO

**Proyecto destinatario:** LLM-MOHO (repo `eduardo-bertey/LLM-MOHO` según la misión; **no verificado en la web** — no se encontró el repositorio en las búsquedas del 2026-11-07), modelos pequeños IFM/K2-Horizon-0.9B, Qwen2.5-0.5B y Qwen3-0.8B, en español, con pocos recursos (CPU / poca RAM, también Colab).
**Fecha de investigación:** 2026-11-07. **Idioma:** español.
**Método:** búsqueda en el índice web + lectura directa de abstracts de arXiv y de la documentación de Hugging Face y de la model card de Qwen3.5-0.8B. Las cifras con fuente se citan con su enlace; los cálculos de memoria por token son **cálculos propios** a partir de configuraciones verificadas (fórmula indicada) y van marcados como tales. Lo no verificado va marcado explícitamente como **no verificado**.

---

## Resumen ejecutivo

1. **La KV cache es el cuello de botella de memoria en inferencia, no los pesos**, cuando el contexto o el batch crecen: crece linealmente con nº de tokens × nº de capas × nº de cabezas KV × dimensión de cabeza × bytes por elemento. En los modelos de MOHO el coste por token varía enormemente según el modelo: **Qwen2.5-0.5B ≈ 12 KiB/token, Qwen3-0.6B ≈ 112 KiB/token, Qwen3.5-0.8B ≈ 12 KiB/token** (solo 6 de sus 24 capas son atención completa; el resto es atención lineal con estado fijo). Es decir, **el problema de KV no es igual en los tres modelos**: en Qwen2.5-0.5B la KV es pequeña y los pesos dominan; en Qwen3-0.6B la KV domina a partir de ~8–32K tokens.
2. **Corrección importante de nomenclatura:** no existe un «Qwen3-0.8B» oficial. El modelo de 0.8B es **Qwen3.5-0.8B** (marzo de 2026), una arquitectura **híbrida** (Gated DeltaNet + atención) con solo 6 capas de atención completa y contexto nativo de 262K tokens. Muchas técnicas de compresión de KV (evicción, SnapKV, MLA) asumen atención completa en todas las capas y **no se aplican tal cual** a sus 18 capas lineales. Ver §1.3 y §6.
3. **Recomendación (ver §5):** probar primero, en este orden: **(1) QuantizedCache de Hugging Face / KV q8_0 en llama.cpp** — un cambio de configuración, training-free, inmediato; **(2) evicción tipo StreamingLLM / SinkCache o SnapKV** con presupuesto explícito (p. ej. 512–1024 tokens) para contexto largo; **(3) solo como proyecto de investigación, MHA2MLA** (conversión a MLA con poco fine-tuning) sobre Qwen2.5-0.5B o Qwen3-0.6B, y **no** TransMLA (requiere ~6B tokens). **No** probar MLA desde cero, ni YOCO/CLA (requieren pre-entrenar), ni FlashMLA (solo GPU Hopper, inútil en CPU/Colab).
4. **Advertencia transversal:** casi todas las cifras de compresión de los papers se midieron en modelos de **7B–70B** (LLaMA, Mistral) con benchmarks en inglés (LongBench, WikiText-2, NIAH). **No hay evidencia verificada de que esos números se trasladen a modelos de 0.5–0.9B ni a español.** Cualquier técnica debe medirse en MOHO con perplejidad en español y con una tarea de recuperación (needle) antes de adoptarla.

---

## 1. ¿Qué es la KV cache, por qué crece y por qué comprimirla?

### 1.1 Definición

En un Transformer decoder autorregresivo, generar cada token nuevo exigiría recalcular la atención sobre todos los tokens anteriores. La **KV cache** guarda, para cada token ya procesado y en cada capa, los vectores **clave (K)** y **valor (V)** de la atención, de modo que en el paso de decodificación solo se calcula K/V del token nuevo y se reutiliza el resto. Es lo que documenta Hugging Face: «A KV *cache* stores these calculations so they can be reused without recomputing them» (https://huggingface.co/docs/transformers/main/kv_cache, leído 2026-11-07).

El proceso tiene dos fases (KIVI, arXiv:2402.02750, §2, leído en https://arxiv.org/html/2402.02750v2):

- **Prefill:** se procesa el prompt completo y se construye la KV de todos sus tokens.
- **Decoding:** token a token; en cada paso se añade un K/V nuevo y se lee **toda** la cache para calcular la atención.

### 1.2 Fórmula y coste por token y por capa

$$ \text{KV (bytes)} = 2 \times L \times n_{kv} \times d_{head} \times S \times B \times \text{bytes\_por\_elemento} $$

donde $L$ = nº de capas, $n_{kv}$ = nº de cabezas clave/valor, $d_{head}$ = dimensión por cabeza, $S$ = longitud de secuencia, $B$ = batch, y el factor 2 es K+V. Por token y por capa (BF16/FP16, 2 bytes): $2 \times n_{kv} \times d_{head} \times 2$ bytes.

Por qué crece: **linealmente con el contexto y con el batch**, mientras los pesos son fijos. El paper KIVI lo ilustra con PaLM-540B: con batch 512 y contexto 2048, la KV sola ocupa **3 TB, 3 veces los pesos** (KIVI, §1). Además el problema no es solo capacidad: en cada paso de decodificación la GPU debe **leer toda la KV desde memoria**, dejando los núcleos de cómputo ociosos — la decodificación está limitada por ancho de banda de memoria, no por FLOPs (KIVI §2; Pope et al. 2023, citado allí).

Comprimirla sirve para tres cosas: (a) caber en memoria (contextos más largos / más batch), (b) generar más rápido (menos bytes que mover por token), y (c) en el caso de MOHO, **que un modelo pequeño quepa en CPU/RAM limitada o en una GPU de Colab (T4 16 GB / L4 24 GB)**.

### 1.3 Coste en los modelos de MOHO (cálculos propios)

Fórmula aplicada a configuraciones verificadas (ver notas `notes/source-02-frameworks-modelos.md`). Son cálculos propios, no cifras de ningún paper.

| Modelo | Capas | Q heads / KV heads | head_dim | KV/token (BF16) | KV a 4K ctx | KV a 32K ctx | KV a 128K ctx |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 24 | 14 / 2 | 64 | **12 KiB** | 48 MiB | 384 MiB | 1,5 GiB |
| Qwen3-0.6B | 28 | 16 / 8 | 128 | **112 KiB** | 448 MiB | 3,5 GiB | 14 GiB |
| Qwen3.5-0.8B («Qwen3-0.8B» de la misión) | 24 (solo 6 con atención completa) | 8 / 2 (en capas gated-attention) | 256 | **12 KiB** | 48 MiB | 384 MiB | 1,5 GiB (a 262K: ~3,0 GiB) |
| K2-Horizon-0.9B | **no verificado** | **no verificado** | **no verificado** | **no calculable** | — | — | — |

Fuentes de las configs: Qwen2.5-0.5B (24 capas, hidden 896, 14/2 cabezas — tesis, https://lup.lub.lu.se/student-papers/record/9232626/file/9232631.pdf); Qwen3-0.6B (28 capas, 16/8, head_dim 128 — config.json en vivo, https://github.com/yashb98/BuildFromScratch/blob/HEAD/Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/architecture_plan.md y https://huggingface.co/Qwen/Qwen3-0.6B-FP8); Qwen3.5-0.8B (24 capas, layout 6×(3×Gated DeltaNet + 1×Gated Attention), 8 Q / 2 KV, head_dim 256, contexto 262144 — model card leída en vivo, https://huggingface.co/Qwen/Qwen3.5-0.8B).

**Lecturas clave para MOHO:**

- **Qwen2.5-0.5B ya trae GQA agresivo (2 cabezas KV): su KV es diminuta.** A 4K tokens son 48 MiB frente a ~1 GB de pesos BF16. Comprimir la KV aquí apenas se nota salvo con batch grande o contexto ≥32K. El cuello de botella real es el peso del modelo, no la KV.
- **Qwen3-0.6B es el caso opuesto:** 8 cabezas KV × head_dim 128 = 9,3× más KV por token que Qwen2.5-0.5B. A 32K ya son 3,5 GiB. Aquí la compresión de KV sí es prioritaria.
- **Qwen3.5-0.8B ya es eficiente por diseño:** solo 6 capas guardan KV; las 18 capas Gated DeltaNet mantienen un **estado recurrente de tamaño fijo** (no crece con el contexto). Su KV total a contexto completo (262K) es ~3 GiB. Las técnicas de evicción/compresión de KV solo afectan a esas 6 capas.
- **K2-Horizon-0.9B:** lanzado por IFM el 3-sep-2026, familia de 6 modelos 0.9B–375B, Apache 2.0, contexto 131072 para el 0.9B (https://ifm.ai/k2/press-release/; https://docs.sglang.io/cookbook/autoregressive/IFM/K2-Horizon). Su `config.json` (nº de cabezas KV, head_dim) **no se verificó** en esta investigación: hay una página secundaria sobre el derivado K2-Type-0.9B que indica 28 capas, hidden 1536, 32 Q / 8 KV heads (https://api.aiqre.com/models/ifm-k2-type-09b/), pero corresponde a un modelo derivado, no al base. **Antes de calcular su KV, leer `config.json` de `IFM/K2-Horizon-0.9B` en Hugging Face.**

**Referencia de escala (modelos grandes):** DeepSeek-V3 necesita **70 KB por token** de KV, frente a 516 KB de LLaMA-3.1 405B y 327 KB de Qwen2.5-72B (informe técnico DeepSeek-V3, arXiv:2412.19437, citado por https://temperature2.com/p/2026-10-02-did-you-know-kv-cache-economics/ y por el análisis ISCA'25). Verificación cruzada propia: (512+64) × 61 capas × 2 bytes = 70 272 B ≈ 70 KB. ✓

---

## 2. Técnicas principales

### 2.1 MLA — Multi-head Latent Attention (DeepSeek-V2/V3) y variantes

**Paper:** DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, arXiv:2405.04434, 7-may-2024. https://arxiv.org/abs/2405.04434
**Idea central:** en lugar de guardar K y V completos por cabeza, se proyecta el token a **un único vector latente comprimido** $c^{KV}$ de baja dimensión ($d_c$, 512 en DeepSeek-V3) más una clave RoPE desacoplada de 64 dim, y se reconstruyen K/V por cabeza en el momento de la atención (con «absorción» de pesos: las matrices de up-projection se pliegan en $W_Q$ y $W_O$, por lo que nunca se materializa la K completa). Es compresión **por rango**, no por nº de cabezas.
**Ratio:** **−93,3 % de KV cache** frente a DeepSeek 67B, con **5,76× más throughput máximo de generación** y −42,5 % de coste de entrenamiento, manteniendo o mejorando calidad (abstract del paper). En DeepSeek-V3 la cache es de 576 valores por token y capa frente a 32 768 en MHA equivalente.
**Pérdida de calidad:** según el propio paper, ninguna frente a su baseline (mejora en benchmarks). Un estudio de interpretabilidad con un modelo MLA de 114M entrenado desde cero reporta 81 % de reducción con su configuración (rango 128) y preservación de entidades del 98 % — pero es un modelo pequeño experimental, no DeepSeek (fuente secundaria, https://arxiv.org/html/2503.11486v1 — no verificado el paper completo).
**Benchmarks:** el paper evalúa DeepSeek-V2 (236B total / 21B activos, 128K contexto) en benchmarks generales; no hay LongBench/NIAH específico de MLA en el abstract.
**Requiere:** entrenar desde cero (o conversión, ver abajo). Es una **arquitectura**, no un truco de inferencia aplicable a un checkpoint GQA tal cual.

**DeepSeek-V3** (arXiv:2412.19437, dic-2024, https://arxiv.org/abs/2412.19437) reutiliza MLA sin cambios a escala 671B. **FlashMLA** es el kernel, no un paper: https://github.com/deepseek-ai/FlashMLA — kernel CUDA de decodificación MLA optimizado para GPU **Hopper (H100/H200/H800, SM90)** y Blackwell (SM100), BF16, KV paginada (bloque 64), hasta 3000 GB/s en memoria y 580 TFLOPS en cómputo en H800 según su README. **Incompatible con CPU, con Colab T4/L4 y con GPU de consumo**: en vLLM hay backend Triton MLA como fallback, 3–7× más lento que FlashMLA según documentación secundaria. vLLM 0.8.0 integró FlashMLA (https://github.com/vllm-project/vllm/releases/tag/v0.8.0). Repo del modelo: https://github.com/deepseek-ai/DeepSeek-V3.

**Variantes de conversión (clave para MOHO, porque evitan entrenar desde cero):**

| Variante | Paper | Idea | Resultado verificado | Coste |
|---|---|---|---|---|
| **MHA2MLA** | Ji et al., ACL 2025, arXiv:2502.14837 — https://arxiv.org/html/2502.14837v1 | Partial-RoPE (quitar RoPE de las dimensiones que menos aportan) + SVD conjunta de $W_K, W_V$ | Llama2-7B: **−92,19 % KV con −0,5 % en LongBench** (con cuantización Int4: −96,87 %, −3,2 %); solo con MLA −87,5 % y −3 %. Recuperación con **3–6 ‰ de los datos** de entrenamiento | Fine-tuning ligero. Repo: https://github.com/JT-Ushio/MHA2MLA |
| **TransMLA** | Meng et al., arXiv:2502.07864, NeurIPS 2025 Spotlight | Conversión GQA→MLA con compatibilidad total con el stack DeepSeek (vLLM, FlashMLA) | Llama-2-7B: −93 % KV, 10,6× speedup a 8K; training-free −68,75 % KV con −1,65 % en 6 benchmarks; **recuperar paridad exige ~6B tokens de fine-tuning** | Alto si se quiere paridad. Repo: https://github.com/fxmeng/TransMLA |
| **X-EcoMLA** | arXiv:2503.11132 (2025) | Destilación post-entrenamiento para «upcycling» a MLA | Llama3.2-1B: 6,4× compresión con la misma puntuación media usando 3,6B tokens / 70 h GPU AMD MI300; 10,6× con <0,1 % caída usando 7B tokens / 140 h | Medio-alto. Repo **no verificado** |

**Veredicto MLA para MOHO:** la idea es la más potente a largo plazo, pero **no es plug-and-play**: exige convertir el checkpoint y re-entrenar, aunque sea poco. De las tres variantes, **MHA2MLA es la única con un coste de datos compatible con un proyecto pequeño** (3–6 ‰ de los datos; en SmolLM-135M/360M/1B7, modelos de escala parecida a MOHO, también se evaluó). TransMLA es inviable para MOHO si se busca paridad (6B tokens). Además, MHA2MLA se probó en Llama2-7B y SmolLM, **no en Qwen** (que ya es GQA con RoPE en todas las dimensiones y con QK-norm en Qwen3): la partial-RoPE puede comportarse distinto. Ver §6.

### 2.2 GQA y MQA

- **MQA** — Shazeer (2019), *Fast Transformer Decoding: One Write-Head Is All You Need*, arXiv:1911.02150, https://arxiv.org/abs/1911.02150. Todas las cabezas de query comparten **una sola** cabeza K/V. KV = 1/H de MHA (p. ej. 32× menos). Rápido, pero degrada calidad y puede ser inestable en entrenamiento.
- **GQA** — Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023, arXiv:2305.13245, https://arxiv.org/abs/2305.13245. Las cabezas Q se agrupan en G grupos; cada grupo comparte una cabeza K/V. KV = G/H de MHA. **Uptraining:** convertir un checkpoint MHA por mean-pooling de las cabezas K/V de cada grupo y seguir pre-entrenando con solo el **5 % del cómputo original**. En T5-XXL: GQA-8 casi iguala a MHA en calidad (47,1 vs ~47,2 de media) a velocidad casi de MQA (0,28 s/muestra vs 0,24 s MQA y 1,51 s MHA).
- **Relevancia para MOHO: GQA ya está dentro de los modelos.** Qwen2.5-0.5B (14 Q / 2 KV), Qwen3-0.6B (16 Q / 8 KV) y Qwen3.5-0.8B (8 Q / 2 KV en capas de atención) ya son GQA; K2-Horizon-0.9B probablemente también (no verificado). **No hay nada que «probar» aquí salvo entender que la KV ya viene comprimida 7× (Qwen2.5-0.5B: 14→2) o 2× (Qwen3-0.6B: 16→8) respecto a MHA.** Una conversión adicional a MQA (2→1 o 8→1) requeriría fine-tuning y daría como mucho 2× u 8× más, con riesgo de calidad; es la vía de menor prioridad. Implementación: nativa en HF Transformers, vLLM y llama.cpp (es la arquitectura del modelo, no un plugin).

### 2.3 Cuantización de KV

La familia más madura y la más adecuada para MOHO: **training-free** (o casi), aplicable a cualquier checkpoint.

| Método | Paper | Idea central | Ratio / pérdida verificada |
|---|---|---|---|
| **KIVI** | Liu et al., ICML 2024, arXiv:2402.02750 — https://arxiv.org/abs/2402.02750 | K por canal, V por token (asimetría descubierta analizando outliers: las K tienen canales con magnitudes enormes; las V no). Ventana residual en FP16 para los tokens recientes; el resto a 2 bit por grupos de 32 | 2 bit; **2,6× menos memoria pico** (Llama-2-7B) con «poca o ninguna» caída de accuracy; batch 4×; throughput 2,35–3,47×. Repo: https://github.com/jy-yuan/KIVI |
| **KVQuant** | Hooper et al., NeurIPS 2024, arXiv:2401.18079 — https://arxiv.org/abs/2401.18079 | Per-channel K + **cuantizar antes de RoPE** (RoPE amplifica outliers) + niveles no uniformes por capa + dense-and-sparse por vector (aislar outliers) + atención a attention sinks | 3 bit con **<0,1 de aumento de perplejidad** (WikiText-2 y C4, LLaMA/Llama-2/Llama-3/Mistral); LLaMA-7B con **1M de contexto en 1× A100-80GB** y 10M en 8 GPU; ~1,7× speedup vs matmul FP16. Kernels CUDA propios. Repo: https://github.com/SqueezeAILab/KVQuant |
| **FlexGen** | Sheng et al., ICML 2023, arXiv:2303.06865 — https://arxiv.org/abs/2303.06865 | Sistema de offloading (GPU+CPU+disco, con programación por programación lineal) + cuantización por grupos a **4 bit de pesos y de KV**, sin reentrenar ni calibrar | OPT-175B en **una GPU de 16 GB**, 1 token/s, batch efectivo 144; «pérdida de accuracy despreciable» (según abstract). Pensado para **throughput por lotes**, no para latencia interactiva. Repo: https://github.com/FMInference/FlexGen |
| **GEAR** | Kang et al., 2024, arXiv:2403.05527 — https://arxiv.org/abs/2403.05527 | Cuantizar ~98 % a 4 bit uniforme y aproximar el **error de cuantización con una matriz low-rank + una matriz dispersa** de outliers (~2 % en FP16) | Casi sin pérdida a ~4 bit de media; hasta 2,38× throughput y 2,29× menos memoria pico; ratio **3,8×** en la tabla de MLSys 2025. Repo: https://github.com/opengear-project/GEAR |
| **ZipCache** | He et al., NeurIPS 2024, arXiv:2405.14256 — https://arxiv.org/abs/2405.14256 | Cuantización token-wise separable por canal + **bits mixtos por token según saliencia** (tokens importantes con más bits) | **4,98× compresión, 0,38 % de caída** (Mistral-7B, GSM8K); −37,3 % latencia prefill y −56,9 % decode (LLaMA-3-8B @4096) — cifras de survey secundario, no del abstract. Repo **no verificado** |
| **SKVQ / QAQ / MiKV** | Varios 2024 (ver tabla MLSys 2025, arXiv:2503.24000) | Variantes: clipping + reordenación de canales, bits adaptativos por calidad, precisión mixta por importancia | SKVQ 7,9×; QAQ 10×; MiKV 5× (ratios reclamados en la tabla comparativa de MLSys 2025; calidad no detallada en la fuente consultada) |

**Lo que MOHO puede usar hoy sin escribir un paper:** HF Transformers ya trae **QuantizedCache** (`cache_implementation="quantized"`, backends `quanto` int2/int4 por defecto, `hqq` int1–8, `fp8`) y llama.cpp trae `--cache-type-k/--cache-type-v q8_0|q4_0` (ver §5). KIVI/KVQuant/GEAR como papers son la referencia teórica, pero sus repos son implementaciones de investigación (PyTorch + kernels CUDA propios, pensados para LLaMA/Mistral 7B+ en GPU); **no hay evidencia de que funcionen tal cual en Qwen 0.5B en CPU**. La vía práctica es replicar la *idea* de KIVI (K por canal / V por token + ventana residual) sobre la QuantizedCache de HF, no portar el repo.

### 2.4 Evicción / poda de tokens

Training-free, pero **descarta información de forma permanente**: lo eviccionado no se puede recuperar si luego hace falta.

| Método | Paper | Idea | Ratio / calidad verificada |
|---|---|---|---|
| **StreamingLLM** | Xiao et al., ICLR 2024, arXiv:2309.17453 — https://arxiv.org/abs/2309.17453 | *Attention sinks*: los primeros ~4 tokens absorben mucha atención aunque no sean semánticamente importantes; se conservan **esos + una ventana deslizante** de tokens recientes. Memoria constante, longitud «infinita» | Estable hasta **4M tokens** (Llama-2, MPT, Falcon, Pythia); 22,2× speedup vs recomputación de ventana. Degrada el recuerdo de largo alcance (el medio se pierde). Repo: https://github.com/mit-han-lab/streaming-llm |
| **H2O** | Zhang et al., NeurIPS 2023, arXiv:2306.14048 — https://arxiv.org/abs/2306.14048 | *Heavy hitters*: pocos tokens acumulan la mayoría de la masa de atención; se conservan esos (por atención acumulada) + ventana reciente | Con 20 % de cache: **<0,5 de aumento de perplejidad** (LLaMA-2-7B); ratio 5× en tabla MLSys; hasta 29× throughput vs HF Accelerate (OPT-6.7B/30B). No reduce el prefill. Repo: https://github.com/FMInference/H2O |
| **SnapKV** | Li et al., NeurIPS 2024, arXiv:2404.14469 — https://arxiv.org/abs/2404.14469 | Al final del prefill, una **ventana de observación** (últimos tokens del prompt) vota qué tokens del prefijo son importantes; se hace pooling (agrupa vecinos) y se conserva el top-k por cabeza | **8,2× eficiencia de memoria** y 3,6× generación más rápida a 16K; 380K tokens en 1 A100; paridad en LongBench con presupuestos de 1–4K. Repo: https://github.com/FasterDecoding/SnapKV |
| **PyramidKV** | Cai et al., 2024, arXiv:2406.02069 — https://arxiv.org/abs/2406.02069 | El presupuesto de KV **no es uniforme por capa**: las capas bajas necesitan más cache y las altas menos (forma de pirámide) | **Iguala a full-cache en LongBench con solo 12 % de la KV**; con 0,7 % supera a otros métodos hasta +20,5 puntos absolutos (TREC); en NIAH, 128 entradas bastan para 100 % con LLaMA-3-70B. Implementación de referencia en https://github.com/Zefan-Cai/KVCache-Factory (hub que incluye H2O, SnapKV y PyramidKV; 1346 estrellas a 23-jun-2026 según su propio backlog) |
| **FastGen** | Ge et al., 2023 (ICLR 2024) | Política de evicción **adaptativa por cabeza**: cada cabeza usa una estrategia distinta (local, global, etc.) según su patrón de atención | Ratio 1,6× y 1,2× throughput en la tabla MLSys 2025. arXiv y repo **no verificados**. No confundir con **DeepSpeed-FastGen**, que es un sistema de serving distinto |
| **Scissorhands** | Liu et al., 2023 | Evicción por ventana con contador de importancia | 5× (tabla MLSys 2025). Repo no verificado |

**Crítica importante (2025–2026):** la evicción se ve muy bien en LongBench de un solo turno, pero falla en multi-turno y multi-instrucción. El benchmark **SCBench** (Microsoft) y el paper *The Pitfalls of KV Cache Compression* (Chen et al., arXiv:2510.00231, sep-2025) muestran que métodos tipo SnapKV/PyramidKV ignoran instrucciones completas en prompts multi-instrucción y degradan tareas agentic (SWE-bench, τ-bench) entre 8 y 25 puntos con presupuestos de 20 %/10 %/5 %, mientras LongBench apenas cae 1–2 puntos (según el survey de evicción consultado, https://github.com/satindergrewal/aviary-1m/blob/HEAD/docs/RESEARCH-KV-EVICTION-SPARSE-ATTN.md). Para MOHO, que trabaja en español y previsiblemente con chat multi-turno, esto es un riesgo de primer orden: **hay que evaluar con conversaciones multi-turno, no solo con LongBench.**

### 2.5 Fusión / merging, low-rank, SVD

| Método | Paper | Idea | Ratio / calidad |
|---|---|---|---|
| **Palu** | Chang et al., ICLR 2025, arXiv:2407.21118 — https://arxiv.org/abs/2407.21118 | Descomposición **SVD offline de las matrices de proyección K/V**; se cachea el latente low-rank y se reconstruye K/V al vuelo; asignación de rango por Fisher | **50 % de compresión** manteniendo accuracy; hasta 1,89× speedup en el módulo de atención con RoPE; combinado con cuantización 3-bit: **>91,25 % (11,4×)** con perplejidad 1,19 menor que KVQuant a tasa similar. Post-training (calibración; fine-tuning ligero opcional). Repo: https://github.com/shadowpa0327/Palu |
| **MiniCache** | Liu et al., NeurIPS 2024, arXiv:2405.14366 — https://arxiv.org/abs/2405.14366 | **Fusionar la KV de capas adyacentes** en capas medias-profundas: separar magnitud/dirección, interpolar direcciones con SLERP, no fusionar pares de tokens muy distintos | ~1,53× solo; **5,02× combinado con cuantización 4-bit**, ~5× throughput, −41 % memoria (LLaMA-2-7B, ShareGPT; cifras de survey). Training-free. Repo **no verificado** |
| **SparK** | Liao et al., AAAI 2026, arXiv:2508.15212 — https://arxiv.org/abs/2508.15212 | **Poda de canales** no estructurada y consciente de la query, con **recuperación** de los canales podados al calcular el score de atención | Training-free; **>30 % menos KV que los métodos de evicción** a igual longitud; con 80 % de canales podados, <5 % de degradación vs el método de evicción base (LongBench, LLaMA-3-8B). Repo anunciado: https://github.com/Xnhyacinth/SparK (el abstract dice «will be available» — estado no verificado) |
| **KVMerger / CaM / LESS / D2O** | Varios 2024–2025 | Fusionar tokens eviccionados en supervivientes (por similitud coseno) en vez de borrarlos; o recuperar con residuo low-rank disperso | CaM 3,3× (tabla MLSys 2025). Detalles y repos **no verificados** en esta investigación |
| **EigenAttention, xKV, ShadowKV, MatryoshkaKV, ThinK** | 2024–2025 | Low-rank / SVD sobre K pre-RoPE; compresión entre capas por SVD; poda de canales por importancia | xKV: hasta 8,03× en Llama-3.1-8B (RULER 88,5 %) según survey; resto no verificado en primario |

**Nota de 2026:** el paper *Quantization Dominates Rank Reduction for KV-Cache Compression* (arXiv:2604.11501, citado en un survey) sostiene que, a igualdad de calidad, la cuantización domina a la reducción de rango. Es un argumento a favor de priorizar cuantización frente a SVD en MOHO (cita secundaria, no leído el paper).

### 2.6 Compresión cruzada de capas: CLA, YOCO (y LCKV, FusedKV)

| Método | Paper | Idea | Ratio / calidad |
|---|---|---|---|
| **CLA** (Cross-Layer Attention) | Brandon et al., 2024, arXiv:2405.12981 — https://arxiv.org/abs/2405.12981 | Compartir las **activaciones K/V entre capas adyacentes** (factor de compartición 2: solo la mitad de las capas calculan K/V). Ortogonal a MQA/GQA: se combina | **2× menos KV que MQA** con +0,04–0,05 de perplejidad (escalas 1B y 3B, las más cercanas a MOHO). Requiere **pre-entrenar** con CLA. Repo **no localizado** |
| **YOCO** | Sun et al., NeurIPS 2024, arXiv:2405.05254 — https://arxiv.org/abs/2405.05254 | Arquitectura **decoder-decoder**: la mitad inferior (self-decoder, atención eficiente) construye **una única KV global** que reutilizan todas las capas superiores (cross-decoder). La KV deja de escalar con el nº de capas: O(N) en vez de O(N·L) | **~80× menos memoria KV en un modelo 65B**; 9,38× menos memoria total a 1M de contexto; prefill de 512K de 180 s a <6 s; 1M de contexto con needle casi perfecto. Requiere **entrenar desde cero**. Repo: https://github.com/microsoft/unilm/tree/master/YOCO |
| **LCKV / FusedKV** | Wu & Tu 2024; FusedKV arXiv:2512.03870 | Variantes de compartición entre capas; FusedKV reconstruye la KV de las capas altas fusionando las de capas bajas y medias | FusedKV: −50 % de cache con **menor perplejidad que el Transformer estándar** en modelos de 332M–4B (según su abstract) — rango de tamaño directamente relevante para MOHO, pero es una arquitectura nueva que exige entrenar. Repo **no verificado** |

**Veredicto:** CLA/YOCO son ideas arquitectónicas para **entrenar un modelo nuevo**, no para comprimir un checkpoint existente. Para MOHO solo tienen sentido si el proyecto contempla **pre-entrenar un modelo propio pequeño con CLA desde cero** (a escala 1B, CLA se probó y funciona) — un proyecto mayor, no un primer experimento. YOCO es descartable a corto plazo (cambia la arquitectura completa). La alternativa training-free en este eje es **MiniCache** (fusionar KV entre capas en inferencia, §2.5).

### 2.7 Otras recientes 2024–2026

- **KVCache-Factory / Awesome-LLM-KV-Cache** — hubs de implementación y bibliografía, útiles como punto de partida: https://github.com/Zefan-Cai/KVCache-Factory y https://github.com/Zefan-Cai/Awesome-LLM-KV-Cache.
- **Quest, InfLLM, ShadowKV, RetrievalAttention** — selección en tiempo de decodificación sin evicción permanente (mantienen la KV completa y solo atienden a un subconjunto): menos riesgo de pérdida que la evicción, pero más complejos de implementar; repos: `mit-han-lab/Quest`, `thunlp/InfLLM` (citados en survey; no verificados en vivo).
- **TurboQuant** (Google, arXiv:2504.?? — fecha abril 2025, según fuente secundaria): cuantización vectorial de KV casi óptima a ~3,5 bits. **No está en llama.cpp mainline** (petición cerrada como «not planned» en ago-2026). No es vía para MOHO hoy.
- **DeepSeek-V3.2/V4 (2025–2026):** MLA dispersa (sparse MLA) y atención híbrida comprimida; la V4.1-Flash llegaría a ~890 bytes/token según un análisis secundario — **no verificado en el informe técnico primario**; se cita solo como contexto de que la frontera se mueve hacia MLA + esparcidad, no como dato para decidir.
- **Rethinking KV Cache Compression** (Gao et al., MLSys 2025, arXiv:2503.24000, https://arxiv.org/abs/2503.24000): el survey de referencia para la tabla comparativa. Sus dos advertencias prácticas: (1) las implementaciones actuales (FlashAttention, PagedAttention) no están optimizadas para serving con KV comprimida, por lo que el throughput real puede ser peor de lo esperado; (2) comprimir la KV puede **alargar las respuestas generadas**, aumentando la latencia end-to-end. Repo: https://github.com/LLMkvsys/rethink-kv-compression.

---

## 3. Repositorios GitHub por técnica

> Estado de mantenimiento: **salvo indicación expresa, no se verificó en vivo la fecha del último commit ni el nº de estrellas** en esta investigación (las búsquedas devolvieron READMEs y citas, no la actividad del repo). «Usable en MOHO» es una valoración propia a partir de la documentación consultada, no una prueba ejecutada.

| Técnica | Repo (URL exacta) | Lenguaje / framework | Usable en PyTorch / HF / vLLM | Modelos pequeños (0.5–1B) |
|---|---|---|---|---|
| MLA (DeepSeek-V2) | https://github.com/deepseek-ai/DeepSeek-V2 | Python/PyTorch (inferencia propia) | Referencia; HF tiene `DeepseekV3` nativo; vLLM soporta MLA | No: es un checkpoint 236B, no una librería |
| MLA (DeepSeek-V3) | https://github.com/deepseek-ai/DeepSeek-V3 | Python/PyTorch | Sí (HF/vLLM) | No (671B) |
| FlashMLA | https://github.com/deepseek-ai/FlashMLA | CUDA C++/PyTorch | Kernel para vLLM/SGLang; **solo Hopper/Blackwell** | No en CPU/Colab |
| TransMLA | https://github.com/fxmeng/TransMLA | Python/PyTorch, HF + vLLM | Sí (convierte Qwen2.5/LLaMA/Mistral; registro vLLM incluido) | En principio sí (Qwen2.5-7B demo), pero exige fine-tuning de ~6B tokens |
| MHA2MLA | https://github.com/JT-Ushio/MHA2MLA | Python/PyTorch | Sí (HF); evaluado en SmolLM-135M/360M/1B7 | **Sí: es el más cercano a la escala MOHO** |
| KIVI | https://github.com/jy-yuan/KIVI | Python/PyTorch (+CUDA) | Investigación; no es un plugin HF | Probado en Llama/Falcon/Mistral 7B; 0.5B no verificado |
| KVQuant | https://github.com/SqueezeAILab/KVQuant | Python/PyTorch + kernels CUDA | Investigación; kernels CUDA propios | 7B–70B; 0.5B no verificado |
| GEAR | https://github.com/opengear-project/GEAR | Python/PyTorch | Investigación | 7–13B; no verificado en pequeños |
| FlexGen | https://github.com/FMInference/FlexGen | Python/PyTorch | Sistema propio (no HF/vLLM) | Pensado para modelos enormes en 1 GPU; útil como referencia de offloading, no como compresor |
| H2O | https://github.com/FMInference/H2O | Python/PyTorch | Monkeypatch HF | 6.7B–66B; adaptable a pequeños con trabajo propio |
| StreamingLLM | https://github.com/mit-han-lab/streaming-llm | Python/PyTorch | Sí (HF) | 7B–70B; la idea (sinks+ventana) es trivial de replicar en cualquier tamaño |
| SnapKV | https://github.com/FasterDecoding/SnapKV | Python/PyTorch | Monkeypatch HF | 7–35B; adaptable |
| PyramidKV / KVCache-Factory | https://github.com/Zefan-Cai/KVCache-Factory | Python (HF, nano-vLLM, mini-SGLang) | Sí: implementa H2O, SnapKV, PyramidKV y más | **Sí: es el hub más práctico para experimentar con evicción en HF** (1346★ a jun-2026) |
| Palu | https://github.com/shadowpa0327/Palu | Python/PyTorch | Conversión de checkpoint + grafo nuevo | 7–13B; conversión pesada para 0.5B |
| SparK | https://github.com/Xnhyacinth/SparK | (anunciado en el paper) | No verificado | No verificado |
| CLA | — (no localizado) | — | — | Escala 1B/3B en el paper, pero sin repo público localizado |
| YOCO | https://github.com/microsoft/unilm/tree/master/YOCO | Python/PyTorch (unilm) | Entrenamiento propio | 3B en el paper; no es aplicable a checkpoint existente |
| vLLM (infra) | https://github.com/vllm-project/vllm | Python/CUDA | — | CUDA-only; no CPU |
| llama.cpp (infra) | https://github.com/ggerganov/llama.cpp | C/C++ | GGUF; `--cache-type-k/v` | **Sí: la vía CPU/Colab para KV q8_0/q4_0** |
| Survey / benchmark | https://github.com/LLMkvsys/rethink-kv-compression | Python | Herramientas de evaluación MLSys 2025 | Útil para metodología de evaluación |

**Repositorio paraguas recomendado para explorar:** https://github.com/Zefan-Cai/Awesome-LLM-KV-Cache (lista curada de papers con código, clasificada por familia).

---

## 4. Tabla comparativa

Ratios = reducción de memoria KV reclamada en cada paper (modelos y contextos indicados; **no son comparables entre sí de forma directa** porque cada paper usa modelos, benchmarks y definiciones de «ratio» distintos). «Pérdida» = lo que reporta cada paper en su propio benchmark.

| Técnica | Tipo | Compresión típica | Impacto en calidad (benchmarks) | Coste de implementación | ¿Reentrenamiento / fine-tuning? |
|---|---|---|---|---|---|
| MLA (DeepSeek-V2) | Arquitectura (low-rank) | −93,3 % (15×) | Sin pérdida vs baseline (mejora) | Muy alto (kernel + arquitectura) | Sí: entrenar desde cero |
| MHA2MLA | Arquitectura (conversión) | −92,19 % (12,8×); −96,87 % con Int4 | −0,5 % LongBench (Llama2-7B) | Alto | Fine-tuning ligero (3–6 ‰ datos) |
| TransMLA | Arquitectura (conversión) | −93 % (14×); training-free −68,75 % | −1,65 % (6 benchmarks, training-free); paridad con ~6B tokens | Alto | Sí (6B tokens para paridad) |
| GQA | Arquitectura (cabezas) | H/G (2–8× vs MHA) | ~0 (GQA-8 ≈ MHA) | Nulo (ya viene en el modelo) | Solo si se convierte (5 % cómputo) |
| MQA | Arquitectura (cabezas) | H× (hasta 32×) | Caída notable | Nulo | Solo si se convierte |
| KIVI | Cuantización | 2 bit → 2,6× memoria pico (8× teórico) | ~0 (Llama/Mistral/Falcon) | Medio (kernels CUDA) | **No (training-free)** |
| KVQuant | Cuantización | 3 bit → ~5,3× teórico | <0,1 ppl (WikiText-2, C4) | Alto (kernels CUDA, calibración) | No (calibración de niveles) |
| GEAR | Cuantización + low-rank | ~4 bit → 3,8× | Casi sin pérdida | Medio-alto | No |
| ZipCache | Cuantización | 4,98× | −0,38 % (Mistral-7B GSM8K) | Medio | No |
| FlexGen | Cuantización + offloading | 4 bit (4×) + offload a CPU/disco | «Despreciable» (HELM) | Alto (sistema) | No |
| HF QuantizedCache / llama.cpp q8_0 | Cuantización (infra) | 2× (q8_0) / 4× (q4_0) | q8_0 ~0 en uso casual; q4_0 degrada contexto largo | **Muy bajo (un flag)** | **No** |
| StreamingLLM | Evicción | Memoria constante (fija por ventana) | Bien en streaming; mal en recall largo | Bajo | No |
| H2O | Evicción | 5× (20 % cache) | <0,5 ppl (LLaMA-2-7B) | Medio | No |
| SnapKV | Evicción | 8,2× memoria | Paridad LongBench (1–4K presupuesto) | Medio | No |
| PyramidKV | Evicción (por capas) | 8,3×; paridad con 12 % cache | Paridad LongBench con 12 %; +20,5 pts TREC con 0,7 % | Medio | No |
| FastGen | Evicción (por cabeza) | 1,6× | No verificado | Medio | No |
| Palu | Low-rank / SVD | 50 % solo; 11,4× con 3-bit | ~0 solo; ppl mejor que KVQuant | Alto (conversión + kernels) | Calibración; fine-tuning opcional |
| MiniCache | Merging entre capas | 1,53× solo; 5,02× con 4-bit | No detallado en fuentes consultadas | Medio-alto | No |
| SparK | Poda de canales | >30 % extra sobre evicción; 80 % poda de canales | <5 % vs evicción base (LongBench) | Medio | No |
| CLA | Cruzada de capas | 2× sobre MQA | +0,04–0,05 ppl (1B/3B) | Alto | Sí: pre-entrenar |
| YOCO | Cruzada de capas | ~80× (65B); 9,38× memoria total a 1M | Casi perfecto en NIAH a 1M | Muy alto | Sí: entrenar desde cero |

**Benchmarks usados en la literatura:** LongBench (el estándar para evicción), Needle-in-a-Haystack / NIAH (recuperación), RULER, WikiText-2 y C4 (perplejidad para cuantización), GSM8K/MMLU (tareas), PG-19. Ninguno de estos papers publica resultados en español ni en modelos de 0.5–0.9B (salvo SmolLM en MHA2MLA y 1B/3B en CLA).

---

## 5. Recomendación práctica para LLM-MOHO (0.5–0.9B, pocos recursos)

### 5.1 Orden recomendado

**Paso 0 — Medir antes de comprimir (1 día, obligatorio).**
Calcular la KV real de cada modelo con la fórmula del §1.3 y medir memoria y velocidad en el hardware real (CPU o Colab) con `use_cache=True` y con `use_cache=False` (línea base). En Qwen2.5-0.5B probablemente se descubrirá que **la KV no es el problema** a contextos cortos (48 MiB a 4K) y que lo que manda es cuantizar los **pesos** (GGUF Q4_K_M) antes que la KV. Fijar el protocolo de evaluación en español: perplejidad en un corpus español, una tarea de recuperación tipo needle-in-a-haystack en español, y una conversación multi-turno (la evicción falla justo ahí).

**Paso 1 — Cuantización de KV con lo que ya existe (primera técnica a probar).**
- **En Hugging Face Transformers:** `cache_implementation="quantized"` con `cache_config={"nbits": 4, "backend": "quanto"}` (o `nbits=8` / backend `fp8` para empezar con el menor riesgo). Documentación leída en vivo: https://huggingface.co/docs/transformers/main/kv_cache. Es training-free, funciona en CPU y en Colab, y no exige cambiar el modelo. Empezar con **8 bit (2×)** y solo bajar a 4 bit si la calidad en español aguanta. Añadir `cache_implementation="offloaded"` como alternativa/complemento si la RAM de GPU es el límite (mueve la KV a CPU capa a capa).
- **En llama.cpp (si MOHO usa GGUF):** `--cache-type-k q8_0 --cache-type-v q8_0` (2×, «esencialmente gratis» en calidad según la literatura de usuario; q4_0 solo si hace falta y sabiendo que degrada contexto largo y código). Repo: https://github.com/ggerganov/llama.cpp. Funciona en CPU pura, que es el escenario MOHO.
- **Por qué primero:** coste casi cero, reversible en un flag, compatible con los tres modelos (incluido el híbrido Qwen3.5-0.8B, donde solo afecta a sus 6 capas de atención), y con la mejor relación compresión/riesgo de toda la tabla. Si solo se hace una cosa, que sea esta.
- **Extensión opcional (trabajo de investigación, no de configuración):** implementar la idea de **KIVI** (K por canal, V por token + ventana residual FP16) como `Cache` personalizada de HF sobre la QuantizedCache. Es el paper de cuantización más citado y su asimetría es fácil de validar en Qwen con un script pequeño.

**Paso 2 — Evicción con presupuesto fijo (segunda técnica).**
- **StreamingLLM / SinkCache** primero: es la más simple (conservar los primeros 4 tokens + ventana de N tokens recientes) y HF ya trae `SinkCache` en su API de caches. Sirve para **generación larga / streaming** con memoria constante. Probar ventanas de 512, 1024 y 2048 tokens y medir la degradación en la tarea needle en español.
- **SnapKV después** (o PyramidKV vía KVCache-Factory, https://github.com/Zefan-Cai/KVCache-Factory): para **prompts largos** (RAG, documentos), donde la ventana de observación puede votar qué conservar. Es la evicción con mejor evidencia en LongBench. Presupuesto inicial: 1024 tokens.
- **Por qué segunda:** training-free y con repos usables, pero con **pérdida permanente de información** y riesgo demostrado en multi-turno/multi-instrucción (§2.4). No usar como valor por defecto en un chatbot; reservarla para el caso de uso de contexto largo. Combinable con el Paso 1 (cuantizar lo que queda tras eviccionar: p. ej. SnapKV 1024 tokens × q8_0 ≈ 16× menos que full FP16 en Qwen3-0.6B a 32K).

**Paso 3 — Conversión a MLA con MHA2MLA (proyecto de investigación, no experimento rápido).**
- Solo si los pasos 1–2 se quedan cortos y el proyecto dispone de capacidad de fine-tuning (aunque sea pequeña, en Colab). MHA2MLA (https://github.com/JT-Ushio/MHA2MLA) es la vía más barata: 3–6 ‰ de los datos, probada en SmolLM-1B (escala MOHO) y Llama2-7B, con −92 % de KV y −0,5 % en LongBench. Aplicarla a **Qwen2.5-0.5B o Qwen3-0.6B** (atención completa en todas las capas) con un corpus español pequeño, y comparar contra el Paso 1. **No** a Qwen3.5-0.8B en primera instancia (su arquitectura híbrida complica la conversión; no verificado que MHA2MLA la soporte).
- **No** TransMLA (6B tokens para recuperar paridad: fuera del presupuesto de MOHO), **no** entrenar MLA desde cero, **no** YOCO/CLA salvo que MOHO decida pre-entrenar un modelo propio (en ese caso CLA a escala 1B es la opción documentada).

### 5.2 Compatibilidad (resumen)

| | HF Transformers | vLLM | CPU / Colab |
|---|---|---|---|
| QuantizedCache / offloaded | ✅ Nativo (`cache_implementation=`) | vLLM tiene su propio `--kv-cache-dtype` (fp8) | ✅ CPU sí (HF); Colab T4/L4 sí |
| SinkCache / SlidingWindow | ✅ Nativo en la API de caches | ✅ (ventana deslizante según modelo) | ✅ |
| SnapKV / H2O / PyramidKV | ⚠️ Vía monkeypatch o KVCache-Factory | ⚠️ No nativo; hay que portar la política | ✅ En HF/CPU con trabajo propio |
| MLA (modelo convertido) | ✅ (modelo DeepSeek-like) | ✅ (backend MLA; FlashMLA solo Hopper) | ⚠️ Lento sin kernel; en CPU usar implementación PyTorch de referencia |
| FlashMLA | ❌ | ✅ Solo Hopper/Blackwell | ❌ No CPU, no T4/L4 |
| KIVI / KVQuant / Palu (repos) | ⚠️ Investigación, kernels CUDA | ❌ | ❌ Pensados para GPU |

**Nota sobre vLLM en MOHO:** vLLM es un servidor de producción CUDA (PagedAttention, prefix caching, offloading a CPU vía LMCache). En un proyecto experimental con CPU/poca RAM y Colab, **no es la herramienta principal**: el flujo de trabajo es HF Transformers (o llama.cpp) y vLLM solo tiene sentido si se dispone de una GPU con CUDA en Colab Pro o similar, donde entonces sí conviene activar prefix caching y `--kv-cache-dtype fp8`.

---

## 6. Limitaciones y riesgos

1. **Pérdida en contexto largo y en multi-turno.** La evicción borra tokens para siempre: si la pregunta final necesita un dato del medio del documento, ya no está. StreamingLLM pierde el medio por diseño; SnapKV/PyramidKV dependen de que la pregunta esté al final del prompt (ventana de observación). La evidencia 2025–2026 (SCBench; *The Pitfalls of KV Cache Compression*, arXiv:2510.00231) muestra degradación de 8–25 puntos en tareas agentic/multi-instrucción con presupuestos bajos, invisible en LongBench. **Mitigación:** evaluar siempre con multi-turno y needle en español; no fiarse solo de LongBench.
2. **Incompatibilidad con RoPE.** Es el problema central de MLA y de la cuantización de K: RoPE aplica una rotación dependiente de la posición a Q y K, lo que (a) impide absorber/compartir K entre cabezas o capas sin más, y (b) amplifica los outliers de K y empeora su cuantización. Por eso MHA2MLA/TransMLA introducen *partial-RoPE* (quitar RoPE de algunas dimensiones) y KVQuant cuantiza **antes** de RoPE. Qwen2.5/Qwen3 aplican RoPE en todas las dimensiones de Q/K y Qwen3 añade QK-norm: **cualquier conversión MLA o cuantización agresiva de K debe validarse específicamente en Qwen**, no asumir que los resultados de LLaMA se trasladan. Palu también paga un coste extra por «reconstruir y luego aplicar RoPE».
3. **GQA ya está «gastado» en los modelos pequeños.** Qwen2.5-0.5B ya tiene solo 2 cabezas KV: pasar a MQA (1 cabeza) solo da 2× y exige fine-tuning. En Qwen3-0.6B (8 KV) el margen es mayor (hasta 8×), pero con riesgo de calidad.
4. **Los kernels rápidos no corren en el hardware de MOHO.** FlashMLA (Hopper/Blackwell), los kernels CUDA de KVQuant/KIVI y vLLM (CUDA) no sirven en CPU. En CPU, comprimir la KV puede incluso **ralentizar** la generación si la descompresión/cuantización se hace en Python sin kernels (HF advierte que QuantizedCache puede dañar la latencia con contexto corto). Medir velocidad, no solo memoria.
5. **Qwen3.5-0.8B es híbrido.** Solo 6/24 capas tienen KV; las técnicas de evicción, CLA o MiniCache (que asumen KV en todas las capas) no se aplican directamente, y su estado Gated DeltaNet es otra memoria distinta (fija, no KV). Además requiere versiones recientes (main/nightly) de transformers/vLLM. **Verificar la compatibilidad de cada técnica con esta arquitectura antes de probarla.**
6. **K2-Horizon-0.9B es muy reciente (sep-2026) y poco documentado.** Su arquitectura incluye MoVA en los tamaños grandes; en el 0.9B (denso) la atención exacta y su soporte en HF/vLLM deben verificarse en su `config.json` y en su model card antes de aplicar cualquier técnica. Su informe técnico estaba «en progreso» a finales de sep-2026 (fuente: https://openlm.ai/k2-horizon/).
7. **La evidencia es en inglés y en modelos grandes.** Ningún paper de los citados evalúa en español ni (salvo SmolLM/CLA-1B) en 0.5–0.9B. Los ratios de compresión de la tabla son **techos medidos en 7B–70B**, no promesas para MOHO. La cuantización a 2 bit (KIVI) es la más arriesgada en modelos pequeños, que tienen menos redundancia.
8. **Compresión ≠ throughput.** El survey MLSys 2025 advierte que con KV comprimida el throughput real puede empeorar (implementaciones no optimizadas) y que las salidas pueden alargarse, aumentando la latencia total. Medir tokens/s y longitud de salida, no solo GB.
9. **Riesgo de «doble compresión» sin control.** Combinar evicción + cuantización + low-rank multiplica los errores (cada paper combina como mucho dos). Si se combinan, hacerlo de forma incremental y con evaluación en cada paso.

---

## 7. Qué no se pudo verificar

- **Repo LLM-MOHO** (`eduardo-bertey/LLM-MOHO`): no encontrado en búsquedas web (2026-11-07). No se verificó su existencia, contenido ni stack.
- **Config de IFM/K2-Horizon-0.9B** (cabezas KV, head_dim, tipo de atención): no verificada en el repo base de HF; solo hay datos de un derivado (K2-Type-0.9B) en una fuente secundaria.
- **Estado de mantenimiento de los repos** (último commit, issues abiertos, estrellas): no se consultó en vivo repo por repo; solo constan las estrellas de KVCache-Factory (1346, jun-2026), H2O (523) y SnapKV (321) citadas en el backlog de KVCache-Factory.
- **Repos de ZipCache, MiniCache, FastGen (evicción), CLA y X-EcoMLA**: no localizados/verificados en las fuentes consultadas.
- **Cifras de ZipCache, MiniCache, FastGen, SparK (combinado) y xKV**: proceden de surveys secundarios, no de la lectura del paper primario.
- **Cifra de 70 KB/token de DeepSeek-V3**: citada por dos fuentes secundarias a partir del informe técnico; verificada aritméticamente con la config (512+64, 61 capas), pero el abstract de arXiv:2412.19437 no se abrió directamente.
- **Resultados en español o en modelos 0.5–0.9B**: no existe (hasta donde llega esta investigación) ningún benchmark público de compresión de KV en español para estos tamaños; es un hueco que MOHO tendría que cubrir con su propia evaluación.
- **TurboQuant en llama.cpp**: la afirmación de que no está en mainline procede de una fuente secundaria (vucense.com); no se verificó en el repo de llama.cpp.

---

## Fuentes

### Papers (arXiv)

- DeepSeek-V2 / MLA — https://arxiv.org/abs/2405.04434
- DeepSeek-V3 — https://arxiv.org/abs/2412.19437
- GQA — https://arxiv.org/abs/2305.13245 · MQA — https://arxiv.org/abs/1911.02150
- KIVI — https://arxiv.org/abs/2402.02750 (leído: https://arxiv.org/html/2402.02750v2)
- KVQuant — https://arxiv.org/abs/2401.18079
- GEAR — https://arxiv.org/abs/2403.05527 · ZipCache — https://arxiv.org/abs/2405.14256
- FlexGen — https://arxiv.org/abs/2303.06865
- StreamingLLM — https://arxiv.org/abs/2309.17453 · H2O — https://arxiv.org/abs/2306.14048
- SnapKV — https://arxiv.org/abs/2404.14469 · PyramidKV — https://arxiv.org/abs/2406.02069
- Palu — https://arxiv.org/abs/2407.21118 · MiniCache — https://arxiv.org/abs/2405.14366
- SparK — https://arxiv.org/abs/2508.15212 (leído: https://arxiv.org/abs/2508.15212?context=cs.LG)
- CLA — https://arxiv.org/abs/2405.12981 · YOCO — https://arxiv.org/abs/2405.05254
- MHA2MLA — https://arxiv.org/abs/2502.14837 · TransMLA — https://arxiv.org/abs/2502.07864 · X-EcoMLA — https://arxiv.org/abs/2503.11132
- Rethinking KV Cache Compression (MLSys 2025) — https://arxiv.org/abs/2503.24000
- The Pitfalls of KV Cache Compression — https://arxiv.org/abs/2510.00231 (citado vía survey; no leído el primario)

### Documentación y modelos

- HF Transformers, Cache strategies (leído en vivo) — https://huggingface.co/docs/transformers/main/kv_cache
- Qwen3.5-0.8B, model card (leída en vivo) — https://huggingface.co/Qwen/Qwen3.5-0.8B
- Qwen3-0.6B-FP8 — https://huggingface.co/Qwen/Qwen3-0.6B-FP8
- IFM K2 Horizon, nota de prensa — https://ifm.ai/k2/press-release/
- SGLang, recetas K2 Horizon — https://docs.sglang.io/cookbook/autoregressive/IFM/K2-Horizon
- vLLM 0.8.0 (FlashMLA) — https://github.com/vllm-project/vllm/releases/tag/v0.8.0

### Repositorios

https://github.com/deepseek-ai/DeepSeek-V2 · https://github.com/deepseek-ai/DeepSeek-V3 · https://github.com/deepseek-ai/FlashMLA · https://github.com/fxmeng/TransMLA · https://github.com/JT-Ushio/MHA2MLA · https://github.com/jy-yuan/KIVI · https://github.com/SqueezeAILab/KVQuant · https://github.com/opengear-project/GEAR · https://github.com/FMInference/FlexGen · https://github.com/FMInference/H2O · https://github.com/mit-han-lab/streaming-llm · https://github.com/FasterDecoding/SnapKV · https://github.com/Zefan-Cai/KVCache-Factory · https://github.com/Zefan-Cai/Awesome-LLM-KV-Cache · https://github.com/shadowpa0327/Palu · https://github.com/Xnhyacinth/SparK · https://github.com/microsoft/unilm/tree/master/YOCO · https://github.com/vllm-project/vllm · https://github.com/ggerganov/llama.cpp · https://github.com/LLMkvsys/rethink-kv-compression

### Notas de trabajo

- `notes/source-01-papers-principales.md` — fichas de papers con cifras y URLs.
- `notes/source-02-frameworks-modelos.md` — frameworks, configs de los modelos MOHO y cálculos propios de KV/token.
