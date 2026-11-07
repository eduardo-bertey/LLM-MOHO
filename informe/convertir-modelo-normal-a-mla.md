# Convertir un modelo Transformer normal (MHA/GQA) a MLA — Investigación

**Fecha de investigación:** 7 de noviembre de 2026 (ventana de búsqueda: hasta oct. 2026 en arXiv/GitHub).
**Idioma:** títulos de papers en inglés, explicaciones en español.
**Distinción clave — leer primero:** hay dos mundos distintos y NO intercambiables:
- **MLA from-scratch (entrenar desde cero):** DeepSeek-V2/V3/V3.2, Kimi K2, GLM-5, etc. MLA se diseña antes del pre-entrenamiento (8,1 T tokens en V2; 14,8 T en V3). Es donde MLA rinde al 100%.
- **Conversión post-hoc / upcycling:** tomar un checkpoint ya entrenado con MHA/GQA (Llama, Qwen, Mistral…) y re-expresarlo como MLA con SVD / factorización de bajo rango + fine-tuning/distilación ligera. **Ningún método es gratis ni sin pérdida:** todos necesitan algún entrenamiento posterior (de 300 M a 11 B tokens según el método) y la pérdida sin entrenar es enorme en Llama (ver §6). La conversión solo tiene sentido si se dispone de presupuesto de fine-tuning y de un motor MLA (vLLM/SGLang/FlashMLA).

## Resumen ejecutivo

| Pregunta | Respuesta corta |
|---|---|
| ¿Qué es MLA? | Compresión conjunta de bajo rango de K y V en un único vector latente `c_KV` (dim. `d_c` ≪ `d_h·n_h`) que es lo único que se cachea; K/V se reconstruyen con up-projections, absorbibles en Q y O en inferencia. RoPE desacoplado en una clave pequeña aparte (64 dim. en DeepSeek). |
| ¿Cuánto ahorra? | DeepSeek-V2: **−93,3 % de KV-cache vs. MHA (DeepSeek 67B)**, throughput máx. de generación **5,76×**, −42,5 % de coste de entrenamiento. DeepSeek-V3: **70 KB/token** vs. 516 KB Llama-3.1-405B (GQA) y 327 KB Qwen2.5-72B. |
| Mejor conversión barata (pocos datos) | **MHA2MLA (ACL 2025)**: 0,3–0,6 % de los tokens de pre-entrenamiento (v1) / 0,6–1 % (v2); Llama2-7B −92,19 % KV con solo −0,5 % (v1) / −1 % (v2) en LongBench. |
| Mejor conversión sin entrenar | **TransMLA**: a −68,75 % KV, solo −1,65 % media en 6 benchmarks sin entrenar (vs. −21,85 % de MHA2MLA en la comparación del propio paper de TransMLA); a −93 % necesita 6 B tokens para recuperar (media 58,68 vs. 59,85 original). |
| Mejor compresión extrema (1B) | **X-EcoMLA**: 6,4× con la misma puntuación media (3,6 B tokens, 70 GPU-h MI300, teacher 8B); 10,6× con <0,1 % de caída (7 B tokens, 140 GPU-h). |
| Más moderno (2026, Qwen3/MoE) | **MLAfication (oct. 2026)**: primer método compatible con QKNorm y MoE; Qwen3-30B-A3B hasta 5,35× throughput en vLLM y −94,73 % KV con cuantización Int4. **CARE (mar. 2026, ICLR 2026)**: a igual presupuesto de KV, PPL one-shot hasta 215× menor que SVD uniforme. |
| ¿Hay un conversor "plug-and-play"? | **No.** Los repos existentes (TransMLA, MHA2MLA/MLAfication) son código de investigación: convierten solo familias Llama/Qwen/Mistral, requieren calibración (WikiText-2/C4/Alpaca) y fine-tuning posterior; sin él la PPL se dispara (Llama-2-7B: 5,47 → 41,61 en TransMLA). |

---

## 1. El MLA original — DeepSeek-V2 / DeepSeek-V3

### 1.1 Ficha

| Campo | Dato |
|---|---|
| Título | *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* |
| Autores / año | DeepSeek-AI et al. (156 autores) — enviado 7 may. 2024, v5 19 jun. 2024 |
| arXiv | https://arxiv.org/abs/2405.04434 |
| Código / checkpoints | https://github.com/deepseek-ai/DeepSeek-V2 |
| Modelo | MoE 236 B totales, 21 B activados/token, contexto 128 K, pre-entrenamiento 8,1 T tokens |

### 1.2 Qué es MLA (en español)

En MHA, cada token guarda por capa `2·n_h·d_h` elementos (K y V de cada cabeza). MLA lo sustituye por:

1. **Compresión conjunta de bajo rango de K y V:** `c_t^KV = W^DKV · h_t`, con `d_c ≪ d_h·n_h`. Solo `c_t^KV` se guarda en la KV-cache: `d_c·l` elementos por token (`l` = nº de capas).
2. **Reconstrucción bajo demanda:** `k^C = W^UK · c^KV`, `v^C = W^UV · c^KV`. En inferencia `W^UK` se **absorbe** en `W^Q` y `W^UV` en `W^O`, así que ni siquiera hay que materializar K/V completos (esto es la "absorción", clave para el speedup).
3. **RoPE desacoplado:** RoPE normal es incompatible con la compresión (la matriz de rotación quedaría entre `W^Q` y `W^UK` y no conmuta). DeepSeek separa una clave RoPE pequeña y compartida (`k^rope`, 64 dim. en V3) que sí se cachea aparte. Total cacheado en V3: `512 + 64 = 576` elementos/token/capa.
4. **También se comprime Q** (`c^Q`, `d_c'`) para ahorrar memoria de activación en entrenamiento, aunque eso no reduce la cache.

### 1.3 Números oficiales (vs. DeepSeek 67B, denso MHA)

- **KV-cache: −93,3 %; throughput máx. de generación: 5,76×; coste de entrenamiento: −42,5 %** — los tres en el abstract de https://arxiv.org/abs/2405.04434
- El paper afirma empíricamente que **MLA supera a MHA en calidad** (no solo empata), a diferencia de GQA/MQA que pierden calidad; la ablación MHA/GQA/MQA está en el Apéndice C.1 y la comparación MLA vs. MHA en C.2.
- Benchmarks V2 base (5-shot MMLU 78,5; HumanEval 48,0 pass@1) — a la par de Llama-3-70B (MMLU 78,9) con solo 21 B activados. GSM8K/MATH/HumanEval del chat (SFT/RL) mejoran mucho sobre la base, pero esos números mezclan SFT+RL, no aíslan MLA.

### 1.4 DeepSeek-V3 (continuación, from-scratch)

- *DeepSeek-V3 Technical Report*, arXiv:2412.19437 — https://arxiv.org/abs/2412.19437 — 671 B MoE, 37 B activados/token, MLA heredado de V2 + FP8 + Multi-Token Prediction.
- **70.272 bytes/token (≈70 KB)** en BF16: `576 × 2 bytes × 61 capas`. Comparación del propio informe: **Qwen2.5-72B (GQA) 327 KB/token; Llama-3.1-405B (GQA) 516 KB/token** → ~4,7× y ~7,4× menos que GQA, no solo que MHA.
- Precaución: V3.2 introduce DeepSeek Sparse Attention (DSA) y V4/V4.1 cambian de arquitectura; FlashMLA actual (sep. 2026) ya es para V4.1, no para V3 (ver §5).

---

## 2. Métodos de conversión / upcycling a MLA (post-hoc, no from-scratch)

Tabla resumen — todos requieren fine-tuning/distilación posterior; ninguno es conversión exacta sin pérdida a alta compresión.

| Método | Año / venue | arXiv | De → a | Idea central | Código |
|---|---|---|---|---|---|
| **MHA2MLA** | 2025, ACL 2025 | 2502.14837 | MHA/GQA → MLA | Partial-RoPE (quitar RoPE de dims. de baja contribución, criterio S₂-norm) + SVD conjunta de W_K/W_V | https://github.com/JT-Ushio/MHA2MLA → hoy https://github.com/JT-Ushio/MLAfication |
| **TransMLA** | 2025, NeurIPS 2025 Spotlight | 2502.07864 | GQA → MLA | Prueba de equivalencia GQA≡MLA a igual cache; RoRoPE (PCA sobre K) + FreqFold + BKV-PCA (balanceo K/V) | https://github.com/fxmeng/TransMLA |
| **X-EcoMLA** | 2025 (AMD) | 2503.11132 | MHA (GQA) → MLA híbrido | SVD estática/dinámica + destilación post-entrenamiento con teacher mayor; RoPE compartido | https://github.com/AMD-AGI/AMD-Hybrid-Models |
| **Zebra-Llama** | 2025, NeurIPS 2025 | 2505.17272 | Transformer → híbrido MLA+Mamba2 | Inicialización SVD de MLA + Intermediate Layer Distillation (ILD); capas MLA y Mamba2 alternadas | mismo repo AMD |
| **CARE** | 2026, ICLR 2026 | 2603.17946 | GQA/MHA → MLA (a igual KV) | Factorización *covariance-aware* (preserva activaciones, no solo pesos) + reparto de rango por capa (water-filling) + KV-parity mapping | https://github.com/FutureMLS-Lab/CARE ⚠️ no verificado en vivo |
| **MHA2MLA-VLM** | 2026, AAAI 2026 | 2601.11464 | VLM MHA/GQA → MLA | Partial-RoPE adaptativo por modalidad + SVD desacoplada visual/textual + PEFT; minimiza error de activación de salida | https://github.com/JT-Ushio/MHA2MLA-VLM (hoy consolidado en MLAfication) |
| **MLAfication** | 2026 (preprint) | 2610.17202 | MHA/GQA (+QKNorm, MoE) → MLA | Extensión de MHA2MLA: KNorm calibrada + destilación parameter-efficient + partial-RoPE + init. SVD activation-aware | https://github.com/JT-Ushio/MLAfication |
| **FunctionalMLA** | 2026 (preprint) | 2607.27269 | *Post-conversión* (sobre TransMLA/MHA2MLA) | Reconstrucción funcional E2E para modelos *draft* de speculative decoding: reproducir la salida post-W_O del bloque original | https://github.com/swyhahaha/FunctionalMLA ⚠️ no verificado en vivo |

### 2.1 MHA2MLA — *Towards Economical Inference: Enabling DeepSeek's MLA in Any Transformer-based LLMs*

- **Autores/año:** Tao Ji, Bin Guo, Yuanbin Wu, Qipeng Guo, Lixing Shen, Zhan Chen, Xipeng Qiu, Qi Zhang, Tao Gui (Fudan) — 20 feb. 2025, v2 3 oct. 2025; ACL 2025, pp. 33313–33328. https://arxiv.org/abs/2502.14837
- **Código:** https://github.com/JT-Ushio/MHA2MLA — **verificado en vivo 2026-11-07: el repo fue renombrado/consolidado a `JT-Ushio/MLAfication`** (216★), que agrupa MHA2MLA + MHA2MLA-VLM + MLAfication. Usar esa URL actual.
- **Método:** (1) *Partial-RoPE*: se elimina RoPE de las dimensiones/subespacios de frecuencia que menos contribuyen al attention score (estrategia S₂-norm: top-`r` subespacios por norma-2 media por cabeza); el resto pasa a NoPE. (2) *Low-rank*: SVD sobre `W_K`/`W_V` de los subespacios NoPE; **SVD conjunta** (`SVD_joint`, factoriza `[W_K; W_V]` concatenados) supera a la separada (`SVD_split`) en 0,74–0,92 %. Sin esto, RoPE bloquea la absorción y la transferencia cero-shot es impracticable (palabras del paper).
- **Coste:** solo **0,3–0,6 % de los datos de pre-entrenamiento (v1) / 0,6–1 % (v2)** — discrepancia entre versiones del abstract, ver §4. Modelos probados: SmolLM 135M/360M/1,7B, Llama-2-7B y 13B.
- **Números:** Llama2-7B **−92,19 % KV con −0,5 % (v1) / −1 % (v2) en LongBench**; con cuantización Int4 de la KV-cache llega a **−96,87 %** (v2; −3,2 % en LongBench a ese extremo). A −68,75 % (`d_kv`=32): caída media en commonsense de −0,25 % (135M), +0,03 % (360M y 1,7B) y −0,71 % (7B).

### 2.2 TransMLA — *TransMLA: Multi-Head Latent Attention Is All You Need* (título posterior: *…Migrating GQA Models to MLA with Full DeepSeek Compatibility and Speedup*)

- **Autores/año:** Fanxu Meng, Pingzhi Tang, Xiaojuan Tang, Zengwei Yao, Xing Sun, Muhan Zhang (Pekín + Xiaomi/Tencent) — 11 feb. 2025, v5 12 jun. 2025; NeurIPS 2025 Spotlight. https://arxiv.org/abs/2502.07864
- **Código:** https://github.com/fxmeng/TransMLA — **verificado en vivo** (README: converter `transmla/converter.py`, scripts `scripts/convert/qwen2.5-7B-Instruct.sh`, registro vLLM DeepSeek, solo modelos tipo Llama/Qwen/Mistral). Recomiendan `--qk-mqa-dim 64 --kv-lora-rank 512` para FlashMLA en H100.
- **Método:** teorema: **toda GQA se puede reescribir exactamente como MLA con la misma KV-cache; al revés no** (MLA es estrictamente más expresiva a igual cache). Pipeline: (1) fusionar cabezas KV de GQA; (2) **RoRoPE**: PCA sobre la salida de K para concentrar la información posicional en las dimensiones de la primera cabeza + **FreqFold** (plegado de frecuencias RoPE); (3) **BKV-PCA / Balanced KV**: balancear las normas de K y V antes de la PCA conjunta (sin balanceo, K domina y se pierde información de V). El resultado es un checkpoint en formato DeepSeek, compatible con vLLM/SGLang y sus kernels MLA/FP8/MTP.
- **Coste / números:** Llama-2-7B **−92,97 % KV, speedup 10,6× a 8K contexto en vLLM** (hasta 10,6× en GPU de 24 GB); **6 B tokens de fine-tuning** para recuperar rendimiento comparable (media 6 benchmarks 58,68 vs. 59,85 original). **Sin entrenar, a −68,75 %: solo −1,65 % media** (el paper afirma que MHA2MLA cae −21,85 % en esa misma comparación — cifra del paper de TransMLA, no independiente). A −93 % sin entrenar "mantiene respuestas con sentido", pero la PPL se dispara (ver §6).
- **Modelos:** SmolLM-1,7B, Llama-2-7B en experimentos principales; conversión de equivalencia demostrada también en Llama-3, Qwen-2.5 (7B/14B), Mistral/Mixtral, Gemma-2, Phi-4.

### 2.3 X-EcoMLA — *Upcycling Pre-Trained Attention into MLA for Efficient and Extreme KV Compression*

- **Autores/año:** Guihong Li, Mehdi Rezagholizadeh, Mingyu Yang, Vikram Appia, Emad Barsoum (AMD) — 14 mar. 2025, v4 8 sep. 2025. https://arxiv.org/abs/2503.11132
- **Código:** https://github.com/AMD-AGI/AMD-Hybrid-Models — **verificado en vivo** (14★; carpetas `X-EcoMLA` y `Zebra-Llama`). Ojo: alguna fuente secundaria cita `AMD-AIG-AIMA/AMD-Hybrid-Models`; la URL del paper y la verificada es `AMD-AGI`.
- **Método:** inicialización MLA por **SVD estructurada (estática o dinámica)** de los bloques de atención pre-entrenados + **destilación post-entrenamiento** del modelo original (o de un teacher mayor — "dark knowledge"). RoPE unificado con clave RoPE compartida entre cabezas (estilo DeepSeek). Variante MLA híbrida (no MLA puro de DeepSeek en todos los detalles).
- **Coste / números (Llama-3.2-1B-Instruct):** con teacher 8B, **6,4× compresión (= −84,4 %) con la misma puntuación media, 3,6 B tokens y 70 GPU-h en AMD MI300**; **10,6× con <0,1 % de caída media, 7 B tokens y 140 GPU-h**. Con teachers 1B/3B/8B a 3,6 B tokens: 1,9× / 3,6× / 6,4× respectivamente. **Limitación de evidencia: solo escala 1B demostrada en el paper**; el coste escala con el tamaño del modelo.

### 2.4 Zebra-Llama — *Towards Extremely Efficient Hybrid Models*

- **Autores/año:** Mingyu Yang, Mehdi Rezagholizadeh, Guihong Li, Vikram Appia, Emad Barsoum (AMD) — 22 may. 2025, v2 20 ene. 2026; NeurIPS 2025. https://arxiv.org/abs/2505.17272 — mismo repo AMD que X-EcoMLA.
- **Método:** **híbrido MLA + Mamba2** (SSM): parte de las capas se convierten a MLA (init. SVD refinada, `r_kv`=128 en 1B/3B, 160 en 8B) y parte a Mamba2, con **Intermediate Layer Distillation (ILD)** para alinear representaciones internas con el Transformer original. No es conversión pura a MLA: las capas Mamba2 eliminan la KV-cache por completo.
- **Coste / números:** familia 1B/3B/8B, **7–11 B tokens** de post-entrenamiento, teacher 8B. KV-cache al **3,9 % / 2 % / 2,73 % del original (25× / 50× / 36×)** preservando **100 % / 100 % / >97 %** de la media zero-shot en LM Harness (ARC, HellaSwag, MMLU, OpenBookQA, PIQA, RACE, WinoGrande). Zebra-8B supera a Minitron-8B en +7 % few-shot con 8× menos tokens y >12× menos KV. Throughput 2,6–3,8× vs. MambaInLlama hasta 32K (v1; la v2 dice 1,4–3,3× — discrepancia entre versiones).

### 2.5 CARE — *Covariance-Aware and Rank-Enhanced Decomposition for Enabling MLA*

- **Autores/año:** Zhongzhu Zhou, Fengxiang Bie, Ziyan Chen, Zhenyu Zhang, Yibo Yang, Junxiong Wang, Ben Athiwaratkun, Xiaoxia Wu, Shuaiwen Leon Song (Together AI) — 18 mar. 2026; ICLR 2026. https://arxiv.org/abs/2603.17946
- **Código:** https://github.com/FutureMLS-Lab/CARE — **⚠️ NO verificado en vivo en esta investigación** (solo citado en el paper); marcar como no confirmado.
- **Método:** conversión a **igual presupuesto de KV (KV-parity: no comprime más, da más expresividad por el mismo cache)**. Critica el SVD uniforme sobre pesos: ignora la covarianza de las activaciones y reparte el rango por igual entre capas. Tres pasos: (i) factorización que preserva activaciones (ponderada por covarianza de las entradas reales, con datos de calibración Alpaca/WikiText-2/PTB/C4); (ii) reparto de rango adaptativo por capa según su espectro singular (water-filling); (iii) KV-parity mapping a formato MLA.
- **Números:** Qwen3-4B / Qwen3-30B-A3B-Instruct-2507 y Llama-3.1-8B/70B-Instruct: **PPL one-shot hasta 215× menor y precisión media hasta 1,70× mayor que el baseline SVD uniforme** a igual KV; con un breve "healing" fine-tune tras el SVD, **recupera por completo la precisión del modelo original** (según el abstract). Es el método más reciente y el único que ataca explícitamente la inicialización, no solo el fine-tuning posterior.

### 2.6 MHA2MLA-VLM (visión-lenguaje)

- **Autores/año:** Xiaoran Fan, Zhichao Sun, Tao Ji, Lixing Shen, Tao Gui (Fudan) — 16 ene. 2026; AAAI 2026. https://arxiv.org/abs/2601.11464
- **Código:** https://github.com/JT-Ushio/MHA2MLA-VLM — hoy consolidado dentro de https://github.com/JT-Ushio/MLAfication (verificado en el README de ese repo).
- **Método:** *modality-adaptive partial-RoPE* (enmascara dimensiones no esenciales por separado para visión y texto) + *modality-decoupled SVD* (comprime los espacios KV visual y textual por separado) + PEFT; clave: minimizar el **error de activación de salida**, no la distancia entre parámetros.
- **Números:** LLaVA-1.5, LLaVA-NeXT y Qwen2.5-VL; "restaura el rendimiento original con pocos datos supervisados, reduce mucho la KV-cache e integra cuantización" — **el abstract no da % exactos de compresión ni benchmarks numéricos; no verificado en detalle** (flag).

### 2.7 MLAfication — *Efficiently Enabling High-Fidelity MLA for Modern LLMs*

- **Autores/año:** Zile Wang, Zhide Xie, Tao Ji — 12 oct. 2026 (preprint, en revisión). https://arxiv.org/abs/2610.17202
- **Código:** https://github.com/JT-Ushio/MLAfication (mismo repo consolidado; verificado).
- **Método:** extensión de MHA2MLA a los modelos modernos que los métodos de 2025 **no soportan: QKNorm y MoE**. (1) *Calibrated KNorm*: resuelve la incompatibilidad de QK-Norm con el decoding MLA eficiente sin cache extra ni kernels custom; (2) destilación parameter-efficient (evita romper el balanceo de carga de expertos aprendido en pre-entrenamiento, que el fine-tuning full-parameter de MHA2MLA sí perturba); (3) partial-RoPE contribution-aware; (4) inicialización conjunta KV activation-aware (de SVD-LLM V2).
- **Números:** Qwen3-8B: −43,75 % KV → −0,78 en commonsense (CS) y −0,62 en LongBench (LB); −68,75 % → −1,99 / −0,44; −81,25 % → −3,35 / −2,12. **Qwen3-30B-A3B (MoE+QKNorm): −68,75 % → CS −1,60, LB −0,13; hasta 5,35× throughput de serving vs. GQA BF16 en vLLM (H200) y −94,73 % de KV combinando con cuantización Int4** (LongBench 27,21 en esa configuración). Es el método a mirar si el objetivo es Qwen3 o un MoE.

### 2.8 FunctionalMLA — *Beyond KV Reconstruction: Functional Reconstruction for MLA Draft Models in Speculative Decoding*

- **Autores/año:** Weiye Shi, Fanxu Meng, Muhan Zhang (Pekín) — 29 jul. 2026. https://arxiv.org/abs/2607.27269
- **Código:** https://github.com/swyhahaha/FunctionalMLA — **⚠️ NO verificado en vivo** (solo citado en el paper).
- **Método:** **no es un conversor independiente**: es un paso *post-conversión*, agnóstico al conversor (TransMLA o MHA2MLA), pensado para **modelos draft de speculative decoding**. Diagnóstico: la conversión directa degrada el acuerdo draft↔target (los errores de factorización de bajo rango y de RoPE son tolerables para generación standalone, pero bajan mucho la aceptación de tokens del draft). Solución: optimizar cada módulo MLA convertido para **reproducir la respuesta post-`W_O` del bloque MHA/GQA original** sobre hidden states de calibración, sin logits del verificador.
- **Números:** 192 configuraciones (4 pares draft-target Llama/Qwen × 2 conversores × HF/vLLM × 4 tareas de 200 prompts): mejora material la aceptación en **37 de 64 celdas**, sin cambio práctico en 26 y empeora en 1 (tolerancia 0,5 pp).

### 2.9 ¿Y una conversión "oficial" de DeepSeek?

**No existe.** DeepSeek no ha publicado ningún método ni código para convertir un modelo MHA/GQA a MLA; MLA solo aparece en modelos entrenados desde cero (V2/V3/R1/V3.2). Todos los métodos de arriba son de terceros (Fudan, Pekín/Xiaomi, AMD, Together AI). Tampoco hay evidencia de "MLA-ization" como producto: ese término aparece en blogs, no como paper/repo verificado.

---

## 3. Otras técnicas de compresión de KV-cache (comparación)

| Técnica | Año / arXiv | Familia | Cómo comprime | Ratio típico | ¿Reentrenar? | ¿Sirve sobre modelo existente sin tocar pesos? |
|---|---|---|---|---|---|---|
| **MHA** (baseline) | 2017, 1706.03762 | — | 1 cabeza KV por cabeza Q | 1× | — | — |
| **MQA** | 2019, https://arxiv.org/abs/1911.02150 | Arquitectura | 1 sola cabeza KV compartida | nº cabezas (p. ej. 32×) | Sí (o uptraining) | No (cambia arquitectura) |
| **GQA** | 2023, https://arxiv.org/abs/2305.13245 | Arquitectura | G grupos, 1 KV por grupo | H/G (4–8× típico) | Uptraining ~5 % del cómputo original | Parcial (uptraining con mean-pooling) |
| **MLA** (from-scratch) | 2024, https://arxiv.org/abs/2405.04434 | Arquitectura | Latente conjunto K+V de bajo rango | 10–15× vs. MHA; 93,3 % en V2 | Sí, desde cero | No |
| **MLA conversión** | 2025–26, §2 | Arquitectura | Igual que MLA, inicializado por SVD | 6–13× (hasta 50× híbrido MLA+Mamba2) | Fine-tuning 0,3–6 B tokens | No (requiere conversión + fine-tuning) |
| **CLA** (Cross-Layer Attention) | 2024, https://arxiv.org/abs/2405.12981 | Arquitectura | Compartir KV entre capas adyacentes | +2× sobre MQA | Sí, desde cero (1B/3B probados) | No (sin receta de retrofit) |
| **KIVI** | 2024, ICML, arXiv:2402.02750 | Cuantización | K 2-bit por canal, V 2-bit por token | ~8× en KV (2,6× memoria pico total) | **No** | **Sí** |
| **KVQuant** | 2024, NeurIPS, arXiv:2401.18079 | Cuantización | Pre-RoPE, no-uniforme, outliers en FP16 | 3-bit con <0,1 PPL | No | Sí (calibración) |
| **H2O** | 2023, NeurIPS, arXiv:2306.14048 | Evicción | Conservar heavy-hitters + recientes | 2–5× (conservar ~20 %) | No | Sí, pero pierde contexto medio |
| **SnapKV** | 2024, arXiv:2404.14469 | Selección (prefill) | Ventana de observación vota tokens importantes por cabeza | 3–8× en prefill | No | Sí (compresión del prompt) |
| **StreamingLLM** | 2023 | Evicción | Attention sink + ventana deslizante | Cache acotada | No | Sí, pierde contexto lejano |

**Lectura práctica:** la compresión arquitectónica (MLA) da el mayor salto único y **se compone** con cuantización (MHA2MLA+Int4: −96,87 %; MLAfication+Int4: −94,73 %). Evicción/cuantización son training-free y se pueden aplicar hoy a cualquier modelo; la conversión a MLA es la única vía arquitectónica post-hoc, pero cuesta un fine-tuning. CLA es ortogonal (eje capas, no cabezas) y en principio se apilaría sobre MLA, pero no hay trabajo de conversión CLA→MLA verificado.

**Fórmula de referencia:** KV/token = `2 × capas × cabezas_KV × dim_cabeza × bytes/elem` (MHA/GQA); en MLA: `(d_c + d_rope) × capas × bytes/elem` (p. ej. V3: 576 × 61 × 2 B = 70 KB).

---

## 4. Estadísticas y benchmarks por método de conversión

### 4.1 Tabla comparativa (cifras de cada paper; no comparables entre sí sin cautela: distintos benchmarks, modelos y protocolos)

| Método | Modelo(s) | Reducción KV | Caída de calidad | Tokens / coste de conversión | Throughput / speedup |
|---|---|---|---|---|---|
| MHA2MLA | SmolLM 135M–1,7B, Llama2-7B/13B | −68,75 % a −92,19 % (−96,87 % +Int4) | −0,5 % LongBench (v1) / −1 % (v2) a −92,19 %; −0,71 % commonsense a −68,75 % (7B) | 0,3–0,6 % tokens pre-entren. (v1); 0,6–1 % (v2) ≈ 6–12 B tokens para Llama2-7B (2 T) | No reporta speedup de inferencia |
| TransMLA | SmolLM-1,7B, Llama-2-7B (+ equivalencia Qwen/Mistral/Gemma) | −68,75 % / −92,97 % | Sin entrenar −1,65 % media 6 bench a −68,75 %; con 6 B tokens: 58,68 vs. 59,85 orig. (−1,17) a −92,97 % | **6 B tokens** fine-tuning (8× GPU 40 GB, 312 TFLOPS c/u, según paper) | **10,6× @8K** en vLLM |
| X-EcoMLA | Llama-3.2-1B-Instruct | 6,4× (−84 %) / 10,6× (−91 %) | 0 % media a 6,4×; <0,1 % a 10,6× | 3,6 B tok / 70 GPU-h MI300 (6,4×); 7 B tok / 140 GPU-h (10,6×) | No reporta |
| Zebra-Llama | Llama-3.2-1B, 3B; Llama-3.1-8B | 25× / 50× / 36× (KV al 3,9 % / 2 % / 2,7 %) | 0 % / 0 % / <3 % media LM-Harness zero-shot | 7–11 B tokens, teacher 8B | 2,6–3,8× vs. MambaInLlama (v1) |
| CARE | Qwen3-4B/30B-A3B, Llama-3.1-8B/70B | **0 % (paridad de KV)** — no comprime más | PPL one-shot hasta 215× menor que SVD uniforme; healing FT recupera 100 % | "Breve" healing FT (sin nº de tokens en el abstract) | No reporta |
| MLAfication | Qwen3-8B, Qwen3-30B-A3B, + MHA/GQA clásicos | −43,75 % a −81,25 % (−94,73 % +Int4) | Qwen3-8B −0,78 CS a −43,75 %; Qwen3-30B −1,60 CS / −0,13 LB a −68,75 % | 300 M–1,2 B tokens CPT (según tabla del paper) | **5,35× vs. GQA BF16** (vLLM, H200) |
| MHA2MLA-VLM | LLaVA-1.5/NeXT, Qwen2.5-VL | "Significativa" (sin % en abstract) | "Restaura rendimiento original" (sin nº en abstract) | Pocos datos supervisados + PEFT | No reporta |

### 4.2 Detalle de benchmarks (qué mide cada uno)

- **MHA2MLA:** 6 benchmarks de commonsense (MMLU, ARC-Easy/Challenge, PIQA, HellaSwag, OpenBookQA, WinoGrande) + **LongBench** (generación long-context). Tabla 1 del paper (v2): Llama2-13B original media 62,65 → MHA2MLA `d_kv`=64: 60,45 (−0,27 en la tabla; la caída crece a `d_kv`=16: 59,68). **No reporta GSM8K, HumanEval ni Needle-in-a-Haystack.**
- **TransMLA:** mismos 6 benchmarks que MHA2MLA (para comparar directamente) + PPL en WikiText-2. **No reporta GSM8K/HumanEval/LongBench/Needle** en el abstract/tablas principales verificadas.
- **X-EcoMLA / Zebra-Llama:** media LM-Harness (8 tareas zero-shot incl. MMLU, ARC, HellaSwag) + GSM8K few-shot en Zebra (tabla: Zebra-1B 41,09 vs. X-EcoMLA-1B 38,06 vs. Llama-3.2-1B-Instruct 38,51, a 7 B tokens). **No hay MMLU/GSM8K/HumanEval desglosados por método de conversión en los abstracts.**
- **CARE / MLAfication:** commonsense (CS) + LongBench (LB) como métricas agregadas; MLAfication da la tabla más completa para Qwen3 (ver §2.7).
- **Vacío general:** ningún paper de conversión publica **Needle-in-a-Haystack** ni una evaluación long-context a 32K–128K sistemática tras la conversión; LongBench es el proxy más usado y solo MHA2MLA/MLAfication lo reportan con números.

### 4.3 Discrepancias entre versiones (importante para citar bien)

- **MHA2MLA v1 (feb. 2025) vs. v2 (oct. 2025):** datos 0,3–0,6 % → 0,6–1 %; caída LongBench 0,5 % → 1 %. Citar la v2 (ACL 2025) como definitiva.
- **X-EcoMLA:** abstract 3,6 B tokens / 70 GPU-h vs. una survey secundaria que resume 3,4 B tokens / ~9 h en 8× MI300X. La cifra del abstract (70 GPU-h totales) es la primaria; la de la survey implicaría ~72 GPU-h, consistente en orden de magnitud.
- **Zebra-Llama v1 vs. v2:** throughput 2,6–3,8× → 1,4–3,3× vs. MambaInLlama. Citar v2 (ene. 2026).

---

## 5. Repos / implementaciones prácticas

### 5.1 Para convertir (código de conversión)

| Repo | Verificado | Qué hace | Limitaciones conocidas |
|---|---|---|---|
| https://github.com/fxmeng/TransMLA | ✅ en vivo 2026-11-07 | `converter.py` (calibración WikiText-2/PTB/C4/Alpaca, freqfold auto, q/kv-lora-rank), salida formato DeepSeek, registro vLLM | Solo Llama/Qwen/Mistral; PPL training-free muy mala en Llama (ver §6); sin checkpoints publicados (to-do del README) |
| https://github.com/JT-Ushio/MLAfication (antes MHA2MLA) | ✅ en vivo 2026-11-07 | MHA2MLA + MHA2MLA-VLM + MLAfication en un repo; checkpoints en HF (colecciones MHA2MLA, Qwen2.5-VL MLA, MLAfication) | Cada proyecto con su propio entorno/README; el MHA2MLA original es de dic. 2024 |
| https://github.com/AMD-AGI/AMD-Hybrid-Models | ✅ en vivo 2026-11-07 | X-EcoMLA + Zebra-Llama; checkpoints HF `amd/X-EcoMLA-1B8B-fixed-kv64-DPO` (según survey) | Orientado a AMD MI300/ROCm; solo escala 1B–8B |
| https://github.com/FutureMLS-Lab/CARE | ⚠️ solo citado en paper | Conversión covariance-aware a paridad de KV | Sin verificación de existencia/contenido en esta investigación |
| https://github.com/swyhahaha/FunctionalMLA | ⚠️ solo citado en paper | Reconstrucción funcional post-conversión (drafts) | Sin verificación en vivo; no es conversor standalone |

### 5.2 Para ejecutar MLA (kernels / motores — no convierten, solo sirven modelos MLA)

| Proyecto | URL | Verificado / notas |
|---|---|---|
| **FlashMLA** (DeepSeek) | https://github.com/deepseek-ai/FlashMLA | ✅ en vivo 2026-11-07: 13.251★, MIT, creado 21 feb. 2025. **Ojo breaking change 2026-09-30:** el `main` actual solo soporta DeepSeek-V4.1 en Blackwell SM100 y Ascend 950, con formato FP8/FP4 nuevo; **eliminó Hopper y V3/V3.2/V4.0**. Para V3 o un modelo convertido a MLA estilo V2/V3, usar el commit `ba89a3466e9470ad08ab39738d4e7bb66989e1e7`. Kernels: decode/prefill sparse (DSA), fused Q-norm+RoPE+attn+RoPE+cast, dense prefill/backward. Originalmente kernel de decoding para Hopper (Open Source Week, feb. 2025). |
| **vLLM** | https://github.com/vllm-project/vllm | MLA nativo desde `config.json` (DeepSeek, Kimi K2…); MLA absorbido por defecto; kernels FP8 GEMM para up-projections y `concat_mla_k` de FlashInfer. Sin verificación en vivo en esta run (índice/secundario). |
| **SGLang** | https://github.com/sgl-project/sglang | MLA con backend FlashInfer (`--attention-backend flashinfer`, `--enable-flashinfer-mla`); FA3 es su backend MLA por defecto según guías 2026. Sin verificación en vivo (índice/secundario). |
| **FlashInfer** | https://github.com/flashinfer-ai/flashinfer | Librería de kernels (MLA decode/prefill, paged KV) usada por vLLM y SGLang. |
| **DeepInfer** (DeepSeek) | https://github.com/deepseek-ai/DeepInfer | Oct. 2026, Apache-2.0, **solo offline, solo DeepSeek-V4.1-Flash, SM100/Ascend 950**; incluye FlashMLA/DeepGEMM/DeepEP. No es un serving stack completo (sin sistema de KV-cache ni disaggregación). |
| **Transformers (HF)** | — | Los modelos convertidos por TransMLA se cargan con `AutoModelForCausalLM(..., trust_remote_code=True)` en formato DeepSeek; no hay soporte nativo genérico "convierte cualquier checkpoint a MLA" en HF. |
| **llama.cpp** | — | Soporta inferencia de modelos MLA nativos (DeepSeek/Kimi/GLM); no convierte MHA→MLA. No verificado en esta run. |

---

## 6. Limitaciones y problemas abiertos

1. **Sin fine-tuning, la conversión colapsa — y mucho más en Llama que en Qwen.** Model Zoo de TransMLA (PPL WikiText-2, training-free, `kv-lora-rank 512 + qk 64`): Llama-2-7B **5,47 → 41,61**; Llama-3-8B 6,14 → 25,80; Gemma-2-9B-it 10,16 → 21,63; Mistral-7B 5,32 → 9,58; **Qwen2.5-7B 6,85 → 8,41** y Qwen2.5-72B-Instruct 4,27 → 7,39. **Conclusión: Qwen convierte bien; Llama, mal.** El fine-tuning es obligatorio, no opcional.
2. **El obstáculo central es RoPE.** Todos los métodos gastan su innovación en aislar/eliminar RoPE (partial-RoPE, RoRoPE+FreqFold, RoPE compartido). Quitar demasiado RoPE destruye la información posicional; quitar poco impide la absorción y no hay ganancia de velocidad. No hay una solución universal: cada familia necesita su calibración (freqfold, nº de subespacios).
3. **La compresión agresiva (>90 %) sin entrenar es inviable; con entrenar, cara.** TransMLA a −93 % necesita 6 B tokens; MHA2MLA a −92 % necesita 0,6–1 % del pre-entrenamiento (~6–20 B tokens según el modelo base). Para un modelo de 30–70 B no hay evidencia publicada de conversión completa (CARE prueba 70B solo a paridad de KV, sin comprimir; MLAfication llega a 30B-A3B).
4. **Modelos modernos (QKNorm, MoE) no eran convertibles hasta oct. 2026** (MLAfication). Qwen3, con QKNorm, era el caso que rompía MHA2MLA/TransMLA; la solución (KNorm calibrada) es de hace semanas y aún es preprint.
5. **Long-context degradado y no medido.** La evidencia long-context se limita a LongBench (MHA2MLA: −0,5/−1 %; MLAfication: −0,13 a −2,12 según compresión). **Nadie publica Needle-in-a-Haystack, ni evaluación a 32K/128K, ni "lost-in-the-middle" tras conversión.** Es el mayor agujero empírico: la compresión de bajo rango es precisamente más arriesgada cuando el contexto es largo y hay que recuperar un dato puntual.
6. **La ganancia de velocidad no es automática.** Convertir a formato MLA sin más no acelera nada (TransMLA lo dice explícitamente): el speedup (10,6× TransMLA, 5,35× MLAfication) exige (a) compresión real del latente, (b) absorción efectiva y (c) un motor con kernel MLA (FlashMLA/FlashInfer) en el hardware adecuado (Hopper/Blackwell). En CPU o hardware sin kernel MLA, el modelo convertido puede ser *más lento* por las up-projections extra.
7. **Incompatibilidad de versiones de FlashMLA.** El `main` actual (sep. 2026) rompió compatibilidad con V3/V3.2/V4.0 y Hopper. Un modelo convertido al estilo DeepSeek-V2/V3 necesita el commit antiguo o un motor (vLLM/SGLang) que mantenga ese path.
8. **Speculative decoding se rompe.** FunctionalMLA (2026) demuestra que la conversión directa baja mucho la aceptación de tokens draft; hace falta una reconstrucción funcional adicional. Si el despliegue usa speculative decoding (EAGLE, etc.), la conversión estándar no basta.
9. **Sin checkpoints convertidos de referencia para modelos grandes.** Los repos publican a lo sumo checkpoints 1B–8B (AMD, MHA2MLA) o ninguno (TransMLA). No hay un "Llama-3.1-70B-MLA" o "Qwen2.5-72B-MLA" oficial y evaluado que se pueda descargar y servir.
10. **Sesgo de evaluación:** casi todos los benchmarks son commonsense de opción múltiple (MMLU/ARC/HellaSwag) o LongBench agregado. Faltan GSM8K, HumanEval, MATH y evaluación generativa larga por método (salvo Zebra-Llama en GSM8K few-shot).

---

## Could not verify / preguntas abiertas

- **Existencia/contenido de https://github.com/FutureMLS-Lab/CARE y https://github.com/swyhahaha/FunctionalMLA** — solo citados en sus papers; no se abrieron en esta investigación. Tratar como no confirmados hasta verificar.
- **Cifras exactas de MHA2MLA-VLM** (% de reducción de KV, benchmarks por VLM) — el abstract no las da; haría falta leer el PDF completo.
- **Coste en GPU-horas de TransMLA y MLAfication** — TransMLA da hardware (8× GPU 40 GB) pero no horas; MLAfication da tokens (300 M–1,2 B CPT) pero no GPU-horas.
- **Needle-in-a-Haystack / long-context ≥32K tras conversión** — no publicado por ningún método verificado.
- **Conversión de Mistral/Mixtral/Gemma a MLA con benchmarks** — TransMLA afirma la equivalencia, pero sus experimentos con números son solo SmolLM y Llama-2-7B.
- **vLLM/SGLang/llama.cpp** — soporte MLA descrito a partir de fuentes secundarias/búsqueda, no verificado abriendo sus repos en esta run.
- **DeepSeek-V3.2 / V4 / V4.1 y DSA** — fuera del alcance central; solo se citan para contextualizar FlashMLA actual.
- **"MLA-ization"** como término/producto — no se encontró paper ni repo con ese nombre; probablemente jerga de blogs.

## Sources

- DeepSeek-V2 (MLA original): https://arxiv.org/abs/2405.04434 — verificado en arXiv (abstract + HTML), 2026-11-07.
- DeepSeek-V3: https://arxiv.org/abs/2412.19437 — índice/búsqueda, 2026-11-07.
- TransMLA: https://arxiv.org/abs/2502.07864 ; repo https://github.com/fxmeng/TransMLA — ambos verificados, 2026-11-07.
- MHA2MLA: https://arxiv.org/abs/2502.14837 ; repo (hoy MLAfication) https://github.com/JT-Ushio/MLAfication — verificados, 2026-11-07. ACL: https://aclanthology.org/2025.acl-long.1597/
- X-EcoMLA: https://arxiv.org/abs/2503.11132 ; repo https://github.com/AMD-AGI/AMD-Hybrid-Models — verificados, 2026-11-07.
- Zebra-Llama: https://arxiv.org/abs/2505.17272 ; mismo repo AMD — verificado, 2026-11-07.
- CARE: https://arxiv.org/abs/2603.17946 ; repo claimed https://github.com/FutureMLS-Lab/CARE — paper verificado, repo no verificado.
- MHA2MLA-VLM: https://arxiv.org/abs/2601.11464 ; repo https://github.com/JT-Ushio/MHA2MLA-VLM — paper verificado; consolidación en MLAfication verificada en README.
- MLAfication: https://arxiv.org/abs/2610.17202 ; repo https://github.com/JT-Ushio/MLAfication — verificados, 2026-11-07.
- FunctionalMLA: https://arxiv.org/abs/2607.27269 ; repo claimed https://github.com/swyhahaha/FunctionalMLA — paper verificado, repo no verificado.
- CLA: https://arxiv.org/abs/2405.12981 — verificado (abstract/búsqueda).
- GQA: https://arxiv.org/abs/2305.13245 ; MQA: https://arxiv.org/abs/1911.02150 — búsqueda.
- KIVI arXiv:2402.02750; H2O arXiv:2306.14048; SnapKV arXiv:2404.14469; KVQuant arXiv:2401.18079 — búsqueda/secundario.
- FlashMLA: https://github.com/deepseek-ai/FlashMLA — verificado en vivo, 2026-11-07.
- DeepInfer: https://github.com/deepseek-ai/DeepInfer — búsqueda.
- vLLM: https://github.com/vllm-project/vllm ; SGLang: https://github.com/sgl-project/sglang ; FlashInfer: https://github.com/flashinfer-ai/flashinfer — búsqueda/secundario, no verificados en vivo.
- Notas de trabajo: `notes/source-01-deepseek-v2.md`, `notes/source-02-conversion-core.md`, `notes/source-03-kv-others-infra.md`.
