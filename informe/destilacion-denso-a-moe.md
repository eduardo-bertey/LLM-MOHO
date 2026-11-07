# Destilación y conversión de denso a MoE

**Fecha:** 7 de noviembre de 2026.
**Alcance:** destilación de conocimiento en LLMs y conversión de un transformer denso en Mixture-of-Experts (MoE), a escala 0,8–0,9B y a escala 27B.
**Origen:** investigación con fuentes en vivo (arXiv/ar5iv, informes técnicos, model cards y repos) del 6–7 de noviembre de 2026. La página interactiva con las mismas tablas está en el artefacto "Destilación y conversión denso a MoE".

> **Aviso importante:** no existe ningún paper, repo, blog ni modelo publicado que convierta un denso de 20–35B a MoE. Tampoco hay benchmarks publicados justo a escala 0,8–0,9B. Los números a 27B y a 0,9B son extrapolaciones de trabajos a 2B, 7B y 15B, marcadas como [EXTRAPOLADO] o [CALCULADO]. Los números tomados directo de un paper se marcan [MEDIDO].

## Resumen

1. **La destilación funciona y es hoy la forma estándar de hacer LLMs chicos.** La receta pasó de KL hacia adelante off-policy (Hinton 2015) a destilación on-policy con reverse-KL / JSD (GKD, MiniLLM, 2023–24): el estudiante entrena sobre sus propias generaciones, puntuadas token a token por el profesor.
2. **Denso a MoE no es una sola cosa. Hay dos familias y no hay que confundirlas:**
   - **Sparse upcycling (agranda):** copiar el FFN denso N veces como expertos + router aleatorio, y seguir preentrenando. **Requiere fine-tuning / continued pretraining.** Sin él no hay ganancia y el modelo se degrada. Con 10–60% del cómputo original supera al denso continuado.
   - **MoEfication / CMoE / ToMoE (reparte):** partir UN FFN en expertos disjuntos, mismos parámetros totales. **Puede ser casi sin entrenar** y retiene casi toda la calidad con 25–50% del FFN activo. Pero no agrega capacidad: solo ahorra FLOPs de inferencia.
3. **Para 0,8–0,9B no conviene el upcycling clásico (copiar FFN ×8).** El cruce de la ley de escala está en ~4B de tokens de upcycling, el salto de memoria es de 0,9B a ~3–7B totales, y a 7B el split con 7–200B de tokens perdió 8–16 puntos de MMLU. Mejor: autodestilación del bloque FFN + split estilo CMoE/ToMoE.
4. **Para 27B tampoco hay almuerzo gratis.** El padre más limpio es Gemma-2 27B (MMLU 75,2). Copiarlo ×8 da 191,3B totales y 383 GB en bf16. La ganancia esperada, extrapolada de Nemotron 15B, es de +0,5 a +1,5 puntos de MMLU (iso-FLOP) o +1 a +2,5 (top-2) con 100B–1T de tokens.

## 1. Destilación de conocimiento

### 1.1 Papers y métodos clave

| Paper / método | Año | Idea en una línea |
|---|---|---|
| Hinton et al., *Distilling the Knowledge in a Neural Network* | 2015 | Soft labels + temperatura + KL; el "conocimiento oscuro" está en las probabilidades no objetivo. arXiv:1503.02531 |
| DistilBERT (Sanh et al.) | 2019 | Estudiante de 6 capas desde BERT de 12; MLM + KD suave + pérdida coseno en estados ocultos |
| TinyBERT (Jiao et al.) | 2019 | Dos etapas (general + específica de tarea); MSE en atención y estados ocultos |
| MiniLM (Wang et al.) | 2020 | Destila solo la última capa de autoatención (distribuciones + value-relations) |
| SeqKD (Kim & Rush) | 2016 | El profesor genera secuencias completas; el estudiante hace CE/SFT sobre ellas |
| GKD (Agarwal et al.) | ICLR 2024 | On-policy: el estudiante entrena sobre sus propias generaciones, el profesor puntúa cada token; JSD generalizada |
| MiniLLM (Gu et al.) | ICLR 2024 | Reverse KL, policy-gradient, on-policy; busca el modo más nítido |
| DistiLLM (Ko et al.) | 2024 | Skew-KL, schedule adaptativo off → on-policy |
| DistillSpec (Zhou et al.) | ICLR 2024 | KD para alinear un modelo borrador con el objetivo en speculative decoding |
| DeepSeek-R1-Distill | 2025 | Black-box CoT: 800K trazas de razonamiento, solo SFT |

**Taxonomía.** White-box (logits / estados ocultos / atención del profesor; mismo tokenizer, profesor propio) transfiere mucho más por token. Black-box (solo texto generado por el profesor, p. ej. API) es SeqKD / SFT sobre salidas del profesor. Un pipeline típico de producción es black-box primero, luego white-box on-policy, luego RL.

**Token-level vs sequence-level.** Token-level: KL por posición entre profesor y estudiante con teacher forcing. Eficiente, conserva la señal suave, pero es off-policy: el estudiante nunca ve sus propios errores. Sequence-level (SeqKD): CE sobre secuencias muestreadas del profesor. On-policy (GKD/MiniLLM): se muestrea del estudiante y el profesor puntúa esos tokens. Esto cierra el mismatch entrenar–inferencia ("exposure bias"): el error off-policy crece como O(εT²), on-policy como O(εT).

### 1.2 Funciones de pérdida

| Pérdida | Forma | Cuándo usarla |
|---|---|---|
| Cross-entropy (etiquetas duras) | `CE = −log p_S(y_true)` | Siempre como ancla; sola es SFT |
| Forward KL (Hinton) | `T² · KL(softmax(z_T/T) ‖ softmax(z_S/T))`, T = 2–10 | KD clásica; cubre todos los modos del profesor, puede difuminar |
| Reverse KL (MiniLLM) | `KL(p_S ‖ p_T)` | LLMs generativos; busca el modo, más nítida |
| JSD generalizada (GKD) | β interpola FKL (β=0) ↔ JSD (β=0,5) ↔ RKL (β=1); λ mezcla off-policy ↔ on-policy | Ajustable; en TRL `GKDTrainer`: `lmbda`, `beta` |
| Skew-KL (DistiLLM) | Mezcla de FKL/RKL | Schedule adaptativo, estudiantes chicos |
| MSE en estados ocultos | `‖h_S − W·h_T‖²` (+ coseno) | DistilBERT/TinyBERT; necesita mapeo de capas + proyección W |
| KD de atención | KL/MSE en mapas de atención o value-relations | TinyBERT, MiniLM (solo última capa) |
| Combinada (Hinton) | `L = α·CE + (1−α)·T²·KL`, α típico 0,1–0,5 | Punto de partida estándar |
| SeqKD / SFT sobre trazas | CE plana sobre secuencias del profesor | Black-box / destilación de razonamiento |

El factor `T²` no es cosmético: suavizar con T achica los gradientes ~1/T², así que el término KL se multiplica por T² para mantener magnitudes comparables. Forward vs reverse KL es la decisión más importante para estudiantes generativos: un estudiante chico no puede cubrir toda la distribución de un profesor grande y le conviene igualar su modo más nítido.

### 1.3 Datos necesarios

- **KD por logits:** necesita pocos datos etiquetados pero muchos *prompts*; el profesor da la supervisión. MiniLLM usó solo el set databricks-dolly-15k de 12,5K prompts.
- **SeqKD / razonamiento:** necesita muchas trazas de alta calidad. DeepSeek-R1-Distill usó 800K trazas; la calidad (corrección, verificabilidad) importa más que el volumen.
- **KD de features/atención:** necesita el corpus de preentrenamiento o al menos datos de la tarea.
- **Mismo tokenizer es un requisito duro para KD por logits.** Vocabularios distintos hacen incomparables las distribuciones; por eso la destilación entre familias (p. ej. R1 → Qwen) se hace como SeqKD, no como KD por logits.
- **Para el caso LLM-MOHO** (destilar un bloque FFN en una copia MoE *dentro del mismo modelo*): el "profesor" es el propio bloque denso, así que basta un corpus de calibración/destilación en el idioma objetivo (español): miles a decenas de miles de secuencias alcanzan para router + fine-tune liviano (CMoE: 2k muestras; ToMoE: 0,02B de tokens). La pérdida típica es KL entre los logits del modelo completo con bloque denso vs bloque MoE, + CE de LM, + un MSE pequeño en la salida del bloque como estabilizador. Es autodestilación, no KD entre modelos.

### 1.4 Resultados [MEDIDO]

| Modelo | Profesor → estudiante | Compresión | Retención |
|---|---|---|---|
| DistilBERT | BERT-Base 110M → 66M | 1,67× | 97% GLUE, 60% más rápido, 40% más chico |
| TinyBERT (4 capas) | BERT-Base → 14,5M | 7,5× | 96,8% GLUE, 9,4× más rápido |
| MiniLM | BERT-Large → 22M | ~16× | 99%+ MNLI/SQuAD |
| MiniLLM GPT-2-760M | GPT-2-1,5B → 760M | 2× | GPT-4 score 54,7 / Rouge-L 26,4 vs SeqKD 52,0/25,6 y SFT 50,7/25,4 (DollyEval) |
| DeepSeek-R1-Distill-Qwen-1,5B | R1-671B → 1,5B | ~447× (parámetros) | AIME24 28,5; MATH-500 ~79,8; GSM8K ~75,7–84,1 (según evaluación) |
| DeepSeek-R1-Distill-Qwen-7B | R1-671B → 7B | ~96× | AIME24 52,8; MATH-500 91,7; GSM8K ~86,4–91,1 |
| DistillSpec (borrador) | T5-XL → T5-Small | — | +10–45% de speedup sobre speculative decoding estándar; 6,4× (XSum) / 10,7× (GSM8K) de latencia vs el objetivo solo |

**Funciona cuando:** profesor y estudiante comparten tokenizer y familia de arquitectura; la brecha de capacidad es moderada; la tarea es acotada o el estudiante entrena on-policy; las trazas están verificadas.
**Falla / degrada cuando:** (a) brecha de capacidad muy grande — forward KL fuerza cubrir modos imposibles y el estudiante se difumina, reverse KL colapsa a un modo; (b) exposure bias — SeqKD off-policy acumula sus propios errores en inferencia; (c) se heredan los errores del profesor — la destilación amplifica alucinaciones y sesgos; (d) cambio de distribución — se destila en un dominio/idioma y se evalúa en otro; (e) razonamiento sin verificación — trazas CoT no verificadas enseñan razonamiento fluido pero incorrecto; (f) KD por logits entre tokenizers distintos, que no está bien definida.

### 1.5 Repos / implementaciones

- **TRL (HF):** `GKDTrainer` / `GKDConfig` (GKD), `DistillationTrainer` — https://huggingface.co/docs/trl/v0.24.0/en/gkd_trainer
- **MiniLLM:** https://github.com/microsoft/LMOps/tree/main/minillm
- **DistillSpec / GKD de referencia:** código de DeepMind vía TRL; paper arXiv:2310.08461
- **HF general:** ejemplos de destilación en `transformers`, SFT + KD en `trl`, vLLM para generación rápida del profesor (rollouts on-policy)

## 2. Convertir un denso en MoE

### 2.1 Papers clave

| Paper | Año | Aporte |
|---|---|---|
| Shazeer et al., Sparsely-Gated MoE | 2017 | Top-k gating, plantilla de capa MoE. arXiv:1701.06538 |
| Fedus et al., Switch Transformer | 2021/22 | Routing top-1, aux loss de balanceo (α=0,01), Switch-C 1,6T. arXiv:2101.03961 |
| Zhang et al., **MoEfication** | 2021/22 | Partir el FFN en expertos + router; mismos parámetros totales; 10–30% del FFN activo, >95% de rendimiento. arXiv:2110.01786 |
| Komatsuzaki et al., **Sparse Upcycling** | 2022 | Copiar el FFN denso ×N como expertos, router aleatorio, seguir entrenando; ~50% de costo hundido. arXiv:2212.05055 (ojo: no 2212.02574) |
| Sukhbaatar et al., **Branch-Train-MiX (BTX)** | 2024 | Ramificar la semilla en expertos de dominio, instalar los FFN como expertos, promediar el resto, fine-tune MoE del router. arXiv:2403.07816 |
| Zhu et al., **LLaMA-MoE** | 2024 | Partir FFN de Llama-2-7B + 200B tokens de continual pretraining |
| He et al., **Upcycling LLMs into MoE (NVIDIA)** | 2024/25 | Virtual-group init + weight scaling; Nemotron-4 15B con 1T de tokens. arXiv:2410.07524 |
| Liew et al., **Scaling Laws for Upcycling** | ICML 2025 | Término de interacción D1×D2; cruce D*. arXiv:2502.03009 |
| Gao et al., **ToMoE** | TMLR 2026 | Solo router, pesos congelados, 0,02B de tokens. arXiv:2501.15316 |
| Pei et al., **CMoE** | 2025/26 | Router analítico, expertos compartidos + ruteados, minutos + 2k muestras. arXiv:2502.04416 |
| Dai et al., DeepSeekMoE | 2024 | Expertos fine-grained + compartidos. arXiv:2401.06066 |
| Jiang et al., Mixtral 8x7B | 2024 | Referencia MoE-from-scratch: 8 expertos, top-2. arXiv:2401.04088 |

**Referencias MoE-from-scratch:** Mixtral 8x7B — 46,7B totales / 12,9B activos, 8 expertos top-2, MMLU 70,6 vs Llama-2-70B 68,9 (vendor); DeepSeek-V3 — 671B / 37B activos, 1 compartido + 256 ruteados, top-8; Qwen3-30B-A3B — 30B / 3B activos, 128 expertos top-8; Qwen3-235B-A22B — 235B / 22B activos.

### 2.2 Cómo funciona la conversión

**Dos familias (no confundir):**
- **(A) Expande capacidad (sparse upcycling):** cada experto = una copia completa del FFN denso. Parámetros totales ×N, activos ≈ top-k × FFN. Router aleatorio. *Requiere* seguir entrenando.
- **(B) Preserva capacidad (MoEfication):** partir las neuronas intermedias del FFN en E grupos disjuntos; cada experto = un grupo (filas de gate/up/down recortadas). Parámetros totales sin cambio, activos = k/E del FFN. Router aprendido o analítico. *Puede* ser casi sin entrenar.

**Partición / clustering de expertos:**
- *Por pesos* (MoEfication): k-means balanceado sobre las filas de `W_gate`/`W_up` — neuronas con pesos de entrada parecidos responden a entradas parecidas.
- *Por activación* (CMoE): perfilar la frecuencia de activación en un set de calibración; neuronas de alta frecuencia → **experto(s) compartido(s)** (siempre activos), neuronas de baja frecuencia → expertos ruteados. CMoE muestra que esto supera al clustering por pesos por +5–7 puntos y que su router analítico suma +2–6 puntos.
- *Aleatorio* (baseline LLaMA-MoE): partición uniforme + reescalado; funciona pero es lo peor en calidad.
- *Aprendido* (DOT-MoE, 2026): transporte óptimo diferenciable (Sinkhorn) aprende a la vez asignación neurona→experto y token→experto; 3.500 pasos de alineación (<3 h en 8×H100) con pesos densos congelados.
- *Nota SwiGLU:* los FFN de Qwen/K2/Llama son gated (gate/up/down, SiLU/SwiGLU), no ReLU. Hay que recortar gate, up **y** down con el mismo índice intermedio; si no, se destruye el producto gate×up aprendido.

**Inicialización del router:**
- *Aleatoria* (sparse upcycling, NVIDIA): estándar; solo funciona con continued training y LR reseteado.
- *Virtual-group init* (NVIDIA, fine-grained): inicializar el router para que top-k seleccione exactamente una réplica de cada shard MLP en el paso 0, haciendo el MoE *funcionalmente idéntico* al denso al inicio; + weight scaling (factor 4 para E8G8T8; ∛(E·G²/T) en general) para arreglar el colapso de escala de salida. Solo el weight scaling dio 1,5% mejor loss.
- *Router analítico* (CMoE): pesos del router construidos desde neuronas representativas de cada cluster — no necesita entrenar el router; training-free MMLU 42,50 vs 30,33 de LLaMA-MoE-v2 training-free.
- *TopK-then-softmax vs softmax-then-topK:* Mixtral usa topK-then-softmax. NVIDIA encuentra **softmax-then-topK consistentemente mejor para upcycling**, una vez que el weight scaling arregla la escala.

**Balanceo de carga:**
- Aux loss de Switch: `L_aux = α·N·Σ_i f_i·P_i` (f_i = fracción de tokens enviados al experto i, P_i = probabilidad media del router), α = 0,01 en Switch; NVIDIA barre 1e-2–1e-3 como óptimo. Sin aux loss → expertos muertos y plateau temprano; demasiado alto → el aux loss tapa la loss de LM. Z-loss (1e-3) no tuvo efecto.
- Alternativas: Expert Choice routing, capacity factor + token dropping, bias por experto sin aux loss (DeepSeek-V3), group-limited routing. HF/TRL lo expone como `router_aux_loss_coef`.

### 2.3 Benchmarks de upcycling [MEDIDO]

**Tabla 1 — Sparse upcycling (expande capacidad)**

| Estudio / modelo | Padre denso | MoE upcycled | Cómputo / tokens extra | Resultado vs padre denso |
|---|---|---|---|---|
| Komatsuzaki 2022, T5-Base/Large | T5 Base/Large | 32 expertos, ~½ capas MoE, Expert Choice C=2 / Top-2 | 46% / 55% extra | +1,5–2,0 pts SuperGLUE |
| Komatsuzaki 2022, ViT-B/16 | ViT-B/16 | V-MoE | 13% de tiempo extra (vs 58% de continuación densa para el mismo +1% ImageNet 10-shot) | ≥+1% ImageNet 10-shot |
| Komatsuzaki 2022, general | T5/ViT | — | ~50% del costo hundido; efectivo +10–60% | Supera a la continuación densa y al MoE-from-scratch con el 100% del presupuesto denso |
| NVIDIA 2024, Nemotron-2B | 2B, 1,1T tokens | 8 expertos top-1/2; 64 expertos top-8 (E8G8T8) | 110B tokens (~10%) ablaciones | Loss de upcycling −1,1% vs denso continuado con los mismos 0,1T tokens |
| NVIDIA 2024/25, Nemotron-4 15B | 15B, 8T tokens | E8G8T8 (64 expertos, top-8, ⅛ hidden) | 1T tokens | Val loss 1,320 vs 1,377 denso continuado (−4,1%); MMLU 66,2 vs 65,3 |
| NVIDIA, Nemotron-4 15B | 15B, 8T tokens | E8G1T2 (8 expertos, top-2, 2× FLOPs) | 1T tokens | Val loss 1,306 (−5,2%); MMLU 67,6 vs 65,3 (+2,3) |
| NVIDIA, Nemotron-4 15B (base) | 15B | — | — | Val loss 1,623; MMLU 59,3 (antes del continued training) |

**Tabla 2 — Split-FFN que preserva capacidad (Llama-2-7B, 25% de sparsity salvo nota) [MEDIDO]**

| Método | Tokens para la conversión | MMLU | HumanEval | GSM8K | WikiText-2 PPL (denso 5,12) |
|---|---|---|---|---|---|
| Denso Llama-2-7B | — (2T pretrain) | ~46 | — | — | 5,12 |
| LLaMA-MoE (v1) | 200B | 35,09 | 7,58 | 7,41 | — |
| LLaMA-MoE-v2 | 7B | 38,02 | 9,32 | 10,09 | 8,68 (fine-tuneado) / >10k (training-free) |
| LLaMA-MoE-v2 training-free | 0 | 30,33 | — | — | >10k |
| CMoE training-free | 0 (+2k muestras de calibración) | 42,50 | — | — | 7,32 |
| CMoE fine-tuneado (2k muestras ≈ 4M tokens) | 0,004B | 44,02 | 11,22 | 13,01 | 5,92 |
| ToMoE (50% activo, Llama-2-7B) | 0,02B (solo router) | — | — | — | 8,36 |
| DOT-MoE (50% activo, Llama-2-7B) | 1,2B FT tras <3 h de alineación | ~66,6 promedio† | — | — | — |

† DOT-MoE promedio de 6 benchmarks 66,6 vs denso 76,6 y LLaMA-MoE-v2 48,1 con los mismos 1,2B tokens de FT — es decir ~87–90% del denso retenido. El MMLU del denso Llama-2-7B es ~45–46 en la literatura; a 25% de sparsity el paper de CMoE reporta PIQA 78,78 / HellaSwag 76,00 en denso vs CMoE 74,34 / 69,36.

**Tabla 3 — BTX (fusión de especialistas de dominio) [MEDIDO]**

| Setup | Resultado |
|---|---|
| Llama-2-7B semilla → 4 expertos (math, code, Wiki, semilla), top-2, MoE fine-tune | +18,8 pts math, +13,2 pts coding, +3,6 pts world knowledge vs semilla; se acerca a los especialistas (CodeLlama/Llemma) in-domain sin su olvido catastrófico |

**Tabla 4 — Velocidad / inferencia (conversión que preserva capacidad) [MEDIDO]**

| Método / modelo | Ahorro |
|---|---|
| MoEfication, T5-Large 700M | 20% de parámetros FFN activos, sin degradación; 25% de FFN → 2× speedup |
| CMoE, Llama-2-7B 25% sparsity | Hasta 1,17× speedup en escenarios compute-bound |
| ToMoE, 50% activo | ~50% FLOPs de FFN; costo en PPL 5,12 → 8,36 (Llama-2-7B) |
| MoE-from-scratch, Mixtral 8x7B | 46,7B totales a costo de ~13B activos; ~6× velocidad de inferencia vs Llama-2-70B a igual o mejor calidad (vendor) |

### 2.4 Ley de escala para upcycling (Liew et al. 2025) [MEDIDO, fitted ≤1B]

`L = A·D1^(−α1)·D2^(−α2 + α3·log D1) + E` — D1 = tokens densos (costo hundido), D2 = tokens de upcycling. Un D1 mayor da menor loss inicial pero *más lento* progreso en D2.

Cruce D* (el upcycling deja de ganarle al MoE-from-scratch): `D* ≃ 4·(N1/10^9)^(−0,7 + 0,04·log(N1/10^9))` mil millones de tokens.
- Para N1 = 1B, D* ≃ 4B tokens.
- Para N1 = 0,9B, D* ≃ 4,3B tokens.
- Para N1 = 27B, D* ≃ 0,5–0,6B tokens (0,62B con log natural, 0,48B con log10; la base del log no está declarada en el paper).

Cuidado: D* se define contra MoE-from-scratch, NO contra denso continuado. Nemotron entrenó 1T de tokens (≫ D* ≈ 0,8B a 15B) y aun así le ganó al denso continuado. El régimen de Komatsuzaki retiene ventaja hasta ~120% del costo hundido D1; el de Muennighoff muestra que from-scratch gana con <100% del costo hundido en su setting — depende del régimen, no es una contradicción.

### 2.5 Denso vs upcycled-MoE vs MoE-from-scratch

| Dimensión | Denso (continued training) | MoE upcycled | MoE from scratch |
|---|---|---|---|
| Parámetros totales | N1 | N1 × E (expertos) o N1 (split) | Libre, típicamente >> denso |
| Parámetros activos / token | N1 | ~top-k × FFN + compartido (iso-FLOP si es fine-grained) | Mismo punto de diseño |
| Costo para llegar a una loss dada | Baseline (se estanca rápido — Nemotron-2B se estancó con 0,1T extra) | El menor **si** D2 < D* y D2 es 10–60% de D1 | El menor **si** D2 > D* o D1 es muy grande |
| Calidad con iguales tokens *adicionales* | La menor de las tres (Nemotron-15B: MMLU 65,3 con +1T) | La mejor con presupuestos moderados (67,6 con +1T, E8G1T2) | La mejor con presupuestos grandes |
| Calidad con iguales tokens *totales* | — | Supera al MoE-from-scratch entrenado con el 100% del presupuesto denso (Komatsuzaki) | Con el tiempo lo alcanza y lo pasa (Liew et al.) |
| Riesgo | Olvido catastrófico si el LR es muy alto | Caída inicial de calidad tras la cirugía; colapso del router / expertos muertos sin aux loss; pérdida de diversidad si el LR es muy bajo | Inestabilidad, balanceo, alto costo de ingeniería |
| Costo de inferencia | Todos los parámetros cada token | Igual que el denso (iso-FLOP) o 2× (top-2 coarse) para *mejor* calidad; o 25–50% del denso para *igual* calidad (familia split) | Mismo trade-off, diseñado desde el inicio |
| Cuándo elegirlo | Presupuesto < ~5% de D1, o el modelo ya alcanza | Presupuesto 10–60% de D1, buen checkpoint denso, D2 < ~4B tokens a escala 1B | Presupuesto >> D1, o no hay checkpoint denso usable, o la escala objetivo >> padre |

### 2.6 ¿Requiere fine-tuning? Respuesta explícita

**Corto: el upcycling que expande capacidad requiere fine-tuning / continued pretraining. La MoEfication que preserva capacidad puede ser training-free, pero solo para *preservar* (no mejorar) calidad, y solo con un buen router.**

| Régimen | ¿Training-free? | ¿Qué pasa con cero entrenamiento? | Mínimo útil de entrenamiento |
|---|---|---|---|
| Sparse upcycling, coarse (copiar FFN ×8, top-2) | En principio idéntico al inicio (los pesos suman 1), pero expertos idénticos y router aleatorio | Funciona pero es solo el modelo denso con capacidad extra sin usar; sin ganancia | 10–60% del cómputo de pretraining denso (Komatsuzaki); 110B tokens para 2B, 1T para 15B (NVIDIA) |
| Sparse upcycling, fine-grained (E8G8T8) | No — la salida se achica ×8–64 al inicio sin virtual-group init + weight scaling | Loss "muy alta", la red "no convergió a la loss original" con sharding naive (NVIDIA §2.3) | Igual que arriba, *más* virtual-group init + weight scaling (factor 4) para siquiera arrancar |
| Split-FFN, router aleatorio (LLaMA-MoE-v2) | Corre, pero roto en la práctica | MMLU 30,33 (vs ~46 denso), WikiText-2 PPL >10.000 (vs 5,12) | 7B tokens → MMLU 38,02, PPL 8,68 — todavía lejos del denso |
| Split-FFN, router analítico (CMoE) | **Sí, usable** | MMLU 42,50, PPL 7,32 — ya supera a LLaMA-MoE-v2 fine-tuneado (38,02) | 2k muestras (~4M tokens, <1 h) → MMLU 44,02, PPL 5,92 |
| Split-FFN, solo router, pesos congelados (ToMoE) | El router mismo hay que entrenarlo (10k iters) | N/A (entrenar el router *es* la conversión) | 0,02B tokens, 1–4 A100s; PPL 5,12 → 8,36 con 50% activo |
| Split-FFN, asignación aprendida (DOT-MoE) | Solo alineación, pesos congelados | Alineación 3.500 pasos (<3 h, 8×H100) | +1,2B tokens de FT → 87–90% del denso retenido |
| BTX | No — el MoE fine-tuning es definitorio | Los expertos existen pero el router no está entrenado; no hay beneficio de mezcla | Branch training + MoE fine-tune (~2% del cómputo de pretraining, secundario) |

**Si hay que hacer fine-tuning, ¿cuánto y qué queda congelado?**
- **Sparse upcycling (receta NVIDIA, la mejor documentada):** entrena *todo* (expertos, atención, embeddings, router), nada congelado. LR: **resetear al pico de pretraining** (2e-4 para el modelo 2B; 3e-4 pico desde 4,5e-4 mínimo para la corrida 15B de 1T), cosine decay; LR bajo constante (2e-5) se estanca temprano y deja los expertos casi idénticos (similitud coseno ~1,0 vs 0,6–0,7 con LR reseteado). Batch: ~4M tokens (1024 × 4096). Aux loss 1e-2, sin z-loss. Tokens: 10% de los tokens de pretraining para ablaciones; 1T para el resultado principal. Datos: mismo blend de pretraining o blend de continued training.
- **Split-FFN (LLaMA-MoE):** entrena todo, SlimPajama, 200B (v1) / 7B (v2) tokens. Aun así rinde por debajo del denso — es el caso de precaución para modelos chicos.
- **Solo router (ToMoE / CMoE liviano):** congelar *todos* los pesos originales; entrenar solo router + proyecciones chicas / hypernetwork (+0,27% de parámetros). 0,02B tokens / 2k muestras. Es el único régimen barato como experimento lateral en un modelo 0,9B.

### 2.7 Limitaciones / modos de falla (denso → MoE)

1. **Caída inicial de calidad tras la cirugía** — todos los papers la reportan; la recuperación no está garantizada y es más lenta cuanto más entrenado estaba el padre denso (término de interacción de Liew).
2. **Colapso del router / expertos muertos** — sin aux loss (o bias de balanceo), pocos expertos toman todos los tokens; el resto nunca entrena. Monitorear utilización de expertos por capa, no solo loss.
3. **Sin diversidad de expertos con LR bajo** — los expertos quedan copias (similitud coseno ~1,0); el MoE es entonces solo un modelo denso más lento. El reset de LR es el arreglo, pero arriesga olvido catastrófico.
4. **MoE fine-grained es eficiente en FLOPs pero ineficiente en sistema** — GEMMs más chicas, overhead de permute/unpermute; NVIDIA nota menor utilización de FLOPs de GPU; 256 expertos fue *peor* que 64 en 15B.
5. **Explosión de memoria sin ganancia de inferencia (upcycling):** 0,9B → 8 expertos ≈ 0,9B + 7× parámetros FFN ≈ 3–4B totales (el FFN es ~⅔ de un decoder); a escala 0,9B esa memoria compra ~+2 MMLU como mucho (extrapolando Nemotron) y solo tras miles de millones de tokens.
6. **FFN SwiGLU / gated complican el split** — gate/up/down se deben recortar de forma consistente; las hipótesis de sparsity de ReLU fallan; los expertos no seleccionados necesitan valores representativos (G-MoEfication).
7. **Qwen3.5-0.8B es un modelo híbrido** — la mayoría de las capas son Gated DeltaNet (atención lineal), no atención estándar + FFN; un script naive de "reemplazar cada FFN por MoE" escrito para Llama/Qwen3-denso lo va a manejar mal. Revisar `config.json` primero.
8. **Trampa de evaluación:** la PPL puede recuperarse mientras el downstream (MMLU/GSM8K) sigue colapsado (LLaMA-MoE-v2: PPL 8,68 pero MMLU 38,02). Reportar siempre ambos, más una métrica de salud del routing (histograma de carga de expertos, % de tokens dropeados).
9. **Upcyclear un modelo chico ya muy destilado (K2-Horizon-0.9B) es doblemente riesgoso:** ya es un producto de destilación (multi-teacher); sus pesos son un óptimo comprimido, así que el "costo hundido" es alto en relación a su tamaño y el D* de Liew es chico — el continued training tiene más chances de *dañarlo* que de mejorarlo. Mejor destilar *desde* él, no upcyclearlo.

## 3. Escala 27B

**Hallazgo central: no hay ninguna conversión publicada de un denso de 20–35B a MoE.** Los upcycles densos→MoE publicados más grandes son Nemotron-4 15B (NVIDIA, 1T de tokens) y Skywork-MoE (13B → 146B totales / 22B activos). La ley de escala de Liew et al. solo se ajustó a modelos densos ≤1B. Todo a 27B es extrapolación o aritmética desde configs verificadas de los padres.

### 3.1 Padres densos de 27B [MEDIDO]

| Modelo | Parámetros (total / sin emb) | Capas | Hidden | FFN intermedio | Heads Q/KV | Vocab / tying | Tokens pretrain | Contexto |
|---|---|---|---|---|---|---|---|---|
| **Gemma-2 27B** (principal) | 27,2B / 26,05B | 46 | 4.608 | 36.864 (GeGLU) | 32 / 16, head 128 | 256.000, tied | **13T** | 8K |
| **Gemma-3 27B** | 27,0B / 25,6B | 62 | 5.376 | 21.504 (GeGLU) | 32 / 16, head 128 | 262.144, tied* | **14T** | 128K |
| Qwen2.5-32B | 32,5B / 31,0B | 64 | 5.120 | 27.648 (SwiGLU) | 40 / 8, head 128 | 151.936, untied | **18T** | 128K |
| Qwen3-32B | 32,8B / 31,2B | 64 | 5.120 | 25.600 (SwiGLU) | 64 / 8, head 128 | 151.936, untied | **36T** | 128K |

\* Gemma-3 27B también lleva un encoder de visión SigLIP de ~0,4B (congelado durante el entrenamiento de lenguaje); los totales de arriba lo excluyen. No existe un Llama/Mistral 27B (8B → 70B; 24B / 123B).

**Baselines preentrenados (vendor, no mezclar shots/protocolos entre filas) [MEDIDO]:**

| Modelo (base/PT) | MMLU (5-shot) | GSM8K | HumanEval (0-shot) | ARC-C | HellaSwag (10-shot) |
|---|---|---|---|---|---|
| Gemma-2 27B PT | **75,2** | **74,0** (5-shot, maj@1) | **51,8** | **71,4** (25-shot) | **86,4** |
| Gemma-3 27B PT | **78,6** | **82,6** (8-shot)† | **48,8** | n/v | n/v |
| Qwen2.5-32B Base | **83,32** | **92,87** (4-shot CoT) | 66,25 EvalPlus‡ | n/v | n/v |
| Qwen3-32B Base | **83,61** | **93,40** (4-shot CoT) | 72,05 EvalPlus‡ | n/v | n/v |

† Discrepancia Gemma-3 GSM8K: la model card reporta 82,6 (8-shot), mientras la Tabla 4 del informe Qwen3 da 81,20 para Gemma-3-27B (distinto shot/CoT). Usar 81–83 como rango, no como punto. ‡ EvalPlus es el promedio de HumanEval/MBPP/HumanEval+/MBPP+, no HumanEval solo.

**Qué padre elegir:** Gemma-2 27B es el padre de upcycling más limpio — texto puro, entrenado desde cero (no destilado; los 2B/9B se destilaron desde él), contexto 8K, FFN GeGLU, y la baseline más débil, así que el headroom es mayor. Gemma-3 27B es más fuerte pero agrega torre de visión y 62 capas de profundidad. Qwen3-32B es el padre más fuerte pero ya está entrenado con 36T de tokens — el más "osificado" por el término de interacción de Liew — y es 32B, no 27B.

### 3.2 Configuraciones MoE desde Gemma-2 27B [CALCULADO]

FFN 23,44B, atención 2,60B, embedding 1,18B; total 27,2B. Método: regla 6·N·D para FLOPs, H100 989 TFLOPS bf16 denso a 40% MFU.

| Configuración | Expertos × tamaño | Parámetros totales | Parámetros activos (sin emb) | Activo/total | Pesos bf16 | FLOPs forward / token |
|---|---|---|---|---|---|---|
| Denso (padre) | 1 × FFN completo | 27,2B | 26,0B | 100% | 54 GB | ~52 GFLOP |
| **8E top-2, copia** (E8G1T2) | 8 × completo | **191,3B** | **49,5B** | 26% | **383 GB** | ~99 GFLOP |
| **8E top-1, copia** (E8G1T1) | 8 × completo | **191,3B** | **26,0B** | 14% | **383 GB** | ~52 GFLOP |
| **64E top-8, tamaño 1/8** (E8G8T8, Nemotron fine-grained) | 64 × 1/8 | **191,3B** | **26,0B** | 14% | **383 GB** | ~52 GFLOP |
| **64E top-8, split** (cada uno 1/64) | 64 × 1/64 | 27,2B | 5,5B | 20% | 54 GB | ~11 GFLOP |
| **Split 8E top-2** (cada uno 1/8) | 8 × 1/8 | 27,2B | 8,5B | 31% | 54 GB | ~17 GFLOP |
| **Split 8E top-4** (cada uno 1/8) | 8 × 1/8 | 27,2B | 14,3B | 53% | 54 GB | ~29 GFLOP |

Notas: (a) "64 expertos fine-grained top-8" es ambiguo — se muestran las dos lecturas. La lectura Nemotron (64 expertos de 1/8, top-8) es iso-FLOP/iso-activo con 8× parámetros totales; la lectura MoEfication/split (64 shards de 1/64, top-8) mantiene el total en 27B y activa solo 12,5% del FFN. (b) Los parámetros del router son despreciables (<0,1B). (c) Mismos shapes para otros padres, escalados por tamaño de FFN: Gemma-3 27B copia → ~178B totales, top-2 activo ~47B; Qwen3-32B copia → ~209B totales, top-2 activo ~56B; Qwen2.5-32B copia → ~225B totales, top-2 activo ~58B.

**Memoria (solo pesos, bf16):** denso 27B = 54 GB (1× H100 80GB); cualquier copia/fine-grained upcycle = 383 GB (5× H100 80GB solo para pesos); split = 54 GB. En int8 la copia es ~191 GB; en int4 ~96 GB. **Estados de entrenamiento** (pesos bf16 + grads + master fp32 + Adam m/v ≈ 16 bytes/parámetro): denso ~435 GB vs copia ~3,1 TB — ZeRO-sharded en ≥40 H100s solo para estados, vs ~6 para el denso.

### 3.3 Tokens, FLOPs, horas GPU [CALCULADO]

Gemma-2 27B, H100, 40% MFU, $2,50/GPU-h ilustrativo. Pretraining de Gemma-2 ≈ 6·27,2e9·13e12 ≈ 2,1e24 FLOPs.

| Corrida | N activo | Tokens D₂ | FLOPs train (6·N·D) | Horas H100 | Días H100 | Costo @ $2,50 | % de FLOPs pretrain Gemma-2 |
|---|---|---|---|---|---|---|---|
| Umbral D* (Liew) | 26B | **0,62B** | 9,6e19 | ~67 | ~3 | ~$170 | 0,005% |
| Piloto: split 25%, 10B tok | 8,5B | 10B | 5,1e20 | ~360 | ~15 | ~$900 | 0,02% |
| Split 25%, 100B tok | 8,5B | 100B | 5,1e21 | ~3.600 | ~150 | ~$9k | 0,24% |
| Iso-FLOP (E8G8T8), 100B tok | 26B | 100B | 1,6e22 | ~11.000 | ~460 | ~$28k | 0,77% |
| Iso-FLOP, **1T tok** (escala Nemotron) | 26B | 1T | 1,6e23 | ~115.000 | ~4.800 | ~$287k | 7,7% |
| Top-2 (E8G1T2), 100B tok | 49,5B | 100B | 3,0e22 | ~21.000 | ~870 | ~$52k | 1,4% |
| Top-2, **1T tok** | 49,5B | 1T | 3,0e23 | ~209.000 | ~8.700 | ~$521k | 14% |
| Denso continuado, 1T tok (control) | 26B | 1T | 1,6e23 | ~115.000 | ~4.800 | ~$287k | 7,7% |

Wall-clock: dividir días-H100 por cantidad de GPUs (la corrida iso de 1T es ~19 días en 256 H100s, ~75 días en 64). Con 30% MFU multiplicar horas por 1,33; con 50% por 0,80. Excluye evals, restarts, ablaciones y trabajo de datos — presupuestar 1,5–2× para un proyecto real.

**¿Cuántos tokens hacen falta de verdad? Tres respuestas distintas, no confundirlas:**
1. **Para ganarle al MoE-from-scratch (Liew):** ≲0,6B tokens. Prácticamente irrelevante a 27B: nadie va a entrenar un MoE fresco de 191B totales por 0,6B tokens, y la ley no está validada acá.
2. **Para ganarle al denso continuado (Nemotron, medido a 15B):** 1T de tokens dio +0,9 pp (iso-FLOP) / +2,3 pp (top-2) de MMLU sobre el denso continuado con los mismos 1T; la ganancia *creció* con la cantidad de tokens. Un plan razonable a 27B es **100B tokens mínimo para ver algo, 1T para replicar Nemotron** — 0,8–7,7% de los FLOPs de pretraining de Gemma-2 para la variante iso-FLOP.
3. **Para ganarle al padre (Komatsuzaki, medido a ≤0,9B):** ~50% del cómputo de pretraining (≈6,5T tokens para Gemma-2) — **no aplicar esto a 27B**. Costaría ~1e24 FLOPs y contradice a Nemotron, donde 1T (12,5% de los 8T del padre 15B) ya dio una victoria clara sobre el denso continuado.

### 3.4 Comparación a 27B (modelos base)

Filas medidas = números vendor de modelos base; la fila "27B upcycled MoE" es [EXTRAPOLADO] (no existe ese modelo).

| Modelo | Arq. | Total / Activo | MMLU | GSM8K | HumanEval / EvalPlus | ARC-C | HellaSwag |
|---|---|---|---|---|---|---|---|
| Gemma-2 27B PT (padre) | Denso | 27B / 27B | 75,2 | 74,0 | 51,8 HE | 71,4 | 86,4 |
| Gemma-2 27B **continuado, 1T tok** [EXTRAPOLADO] | Denso | 27B / 27B | ~76–78? | ~76–80? | ~53–56? | ~72–74? | ~87? |
| Gemma-2 27B **upcycled E8G8T8, 1T tok** [EXTRAPOLADO] | MoE 64E top-8 | 191B / 26B | **~77–79?** (+0,9 pp sobre continuado, delta Nemotron) | ~77–81? | ~54–57? | ~73–75? | ~87–88? |
| Gemma-2 27B **upcycled E8G1T2, 1T tok** [EXTRAPOLADO] | MoE 8E top-2 | 191B / 50B | **~78–80?** (+2,3 pp sobre continuado) | ~78–82? | ~55–59? | ~74–76? | ~87–88? |
| Gemma-3 27B PT | Denso | 27B / 27B | 78,6 | 82,6† | 48,8 HE | n/v | n/v |
| Qwen2.5-32B Base | Denso | 32B / 32B | 83,32 | 92,87 | 66,25 EvalPlus | n/v | n/v |
| Qwen3-32B Base | Denso | 32B / 32B | 83,61 | 93,40 | 72,05 EvalPlus | n/v | n/v |
| Mixtral 8x22B Base (MoE-from-scratch) | MoE 8E top-2 | 141B / 39B | 77,6 | 80,3 | 53,1 HE | n/v | n/v |
| Qwen3-30B-A3B Base (MoE-from-scratch) | MoE 128E top-8 | 30,5B / 3,3B | 81,38 | 91,81 | 71,45 EvalPlus | n/v | n/v |
| DeepSeek-V3 Base (MoE-from-scratch) | MoE 256E top-8+1 compartido | 671B / 37B | 87,1 | 89,3 | 65,2 HE | 95,3 | 88,9 |

**Velocidad / costo de inferencia:** en decode batch-1 el costo sigue a los parámetros *activos* (iso-FLOP ≈ denso; top-2 ≈ 2× más lento por token; split-25% ≈ 3× más rápido). Con batch grande, casi todos los expertos se tocan por capa y el costo sigue a los parámetros *totales*: un upcycle de 191B totales necesita ~7× el provisioning de HBM del padre denso a escala de serving, aunque la latencia batch-1 parezca iso-FLOP. La residencia la fijan los parámetros totales: 383 GB bf16 para cualquier copia, vs 54 GB denso; el split queda en 54 GB. Qwen3-30B-A3B (from scratch, 36T tokens) llega a MMLU 81,38 con 3,3B activos — *por encima* de Gemma-2 27B (75,2) y por encima incluso del forecast optimista del upcycle top-2 (~78–80), con solo 61 GB bf16 de residencia. Es el argumento más fuerte contra upcyclear Gemma-2 en particular.

### 3.5 Recomendación práctica a 27B (por etapas, con criterios de corte)

**Etapa 0 — Piloto (días, ~$1–5k): split, no copiar.** Tomar Gemma-2 27B (base, no IT). Aplicar CMoE (restructuración analítica) o ToMoE con 50% de activación (o 25% si el objetivo es latencia). Costo: minutos (CMoE) a ~10k iteraciones en pocas GPUs (ToMoE, 0,02B tokens). Medir MMLU/GSM8K/HumanEval + PPL WikiText-2 vs denso. **Regla de corte:** si el MMLU training-free cae >5 pp desde 75,2, o la PPL más que se duplica, el FFN del padre no es limpiamente separable — no seguir a copy-upcycling por esperanza.

**Etapa 1 — Continued training barato (1–2 semanas, ~$10–30k): split + fine-tune liviano.** Si la Etapa 0 queda dentro de ~3 pp, fine-tunear el modelo split (estilo CMoE, 2k–100k muestras; o estilo LLaMA-MoE-v2, hasta 7B tokens a 27B ≈ 250 GPU-horas para el modelo 25%-activo). Es la jugada de mejor valor esperado: calidad casi densa con 25–50% de los FLOPs de inferencia del denso, parámetros totales sin cambio, sin explosión de memoria. **Ganancia esperada: −2 a +0 pp MMLU vs padre denso, pero 2–3× más barata la inferencia.** Si el objetivo era *ganar calidad*, esta etapa no lo va a dar — decirlo explícito y frenar acá si el costo de inferencia era el objetivo.

**Etapa 2 — Upcycle completo, solo si la ganancia de calidad es el objetivo Y la Etapa 1 pasó (3–6 semanas, ~$30–300k): Nemotron E8G8T8 iso-FLOP.** Copy-upcycle de Gemma-2 27B con virtual-group init + weight scaling ∛(E·G²/T)=4×, routing softmax-then-topK, aux loss 1e-2, LR reseteado (pico ≈ pico de pretraining; para Gemma-2 el pico no es público — arrancar en 2e-4, cosine a 2e-5, por analogía con el pico 2e-4 de Nemotron-2B; hacer un sweep de LR a 10B tokens con 1e-4 vs 2e-4 vs 3e-4 antes de comprometerse), batch 4M tokens, 100B tokens de datos *frescos o re-mezclados* (no una repetición pura de los datos de pretraining). Usar Megatron-LM/Megatron-Core `--moe-use-upcycling --moe-upcycling-granularity 8` con expert parallelism EP=8. **Ganancia esperada: +0,5 a +1,5 pp MMLU sobre un control denso continuado con los mismos 100B tokens** (el +0,9 pp de Nemotron a 1T, escalado hacia abajo por menos tokens y un padre más saturado). **Regla de corte:** a los 25B tokens, comparar val loss vs el control denso continuado corrido en paralelo con datos idénticos; si el upcycle no va por delante por ≥0,5% de val loss relativa, matarlo — el término de interacción de Liew dice que la brecha no se va a abrir después, y los 75B tokens restantes rinden más en el control denso. Si va por delante, seguir a 100B y re-evaluar; extender a 1T solo con una decisión de presupuesto fresca (×10 de costo por, a 15B, ~+1 pp más).

**No hacer:** copia 8E top-2 como primer intento (2× FLOPs, mismos 383 GB, +1,4 pp sobre iso-FLOP a 15B); upcycling training-free por copia (colapso garantizado); upcyclear el modelo IT (el router va a pelear con el instruction tuning — upcyclear el modelo base y luego re-hacer SFT/RLHF); upcyclear Gemma-3 27B primero (torre de visión + 62 capas agregan modos de falla; hacer Gemma-2 primero y portar la receta).

**Riesgos a 27B (ordenados):** 1) explosión de memoria (copia: 191B / 383 GB / ~3,1 TB de estados); 2) colapso del router / expertos muertos sin aux loss 1e-3–1e-2; 3) falta de diversidad de expertos (todos arrancan como copias idénticas; solo un LR reseteado rompe la simetría); 4) bugs de split SwiGLU/GeGLU (desalineación gate/up/down destruye el FFN en silencio — test unitario: el modelo split con top-K = todos los expertos debe reproducir los logits densos con precisión numérica antes de entrenar); 5) penalidad de granularidad/MFU (GEMMs más chicas, overhead de permute; presupuestar MFU 30–35%, no 40%, para E8G8T8); 6) osificación (Liew: Gemma-2 con 13T y Qwen3-32B con 36T están tan preentrenados que los tokens de upcycling rinden cada vez menos — D₂ ∝ N₁^1,8); 7) ruido de evaluación (diferencias de 1–2 pp de MMLU están dentro de la varianza entre corridas — pre-registrar el harness y el control denso continuado *antes* de entrenar); 8) licencia/tooling (pesos Gemma bajo licencia Gemma, no Apache-2.0; Qwen es Apache-2.0; Megatron Bridge lista el soporte Gemma-2 como deprecado — verificar antes de comprometerse).

## 4. Recomendación práctica a 0,8–0,9B (K2-Horizon-0.9B / Qwen3-0.8B)

**Paso 0 — Baselines (medio día, no negociable).** Evaluar el padre denso en: PPL en WikiText-2 / held-out en español, MMLU (5-shot, subset en español si hay), ARC, HellaSwag, GSM8K (o math en español). Sin esto, ninguna afirmación de conversión es interpretable.

**Paso 1 — Autodestilación del bloque FFN (la idea LLM-MOHO, bien hecha).**
- Profesor = bloque FFN denso (congelado); estudiante = copia MoE de *ese bloque* (split, no copia — ver Paso 2).
- Pérdida: KL entre los logits *del modelo completo* con bloque denso vs bloque MoE, + CE de LM en texto en español, + MSE chico en la salida del bloque como estabilizador. Es autodestilación white-box; profesor y estudiante comparten tokenizer por construcción.
- Datos: 2k–20k secuencias en español (CMoE usó 2k muestras; ToMoE 0,02B tokens). Congelar todo menos el bloque MoE + router primero; descongelar bloques vecinos solo si la PPL se estanca.

**Paso 2 — Split que preserva capacidad (la conversión para shippear).**
- Por FFN (hidden h, intermedio d_ff): partir d_ff en **8 expertos** (cada uno d_ff/8), **top-2**, **+1 experto compartido** con las neuronas de mayor frecuencia de activación (receta CMoE). Parámetros totales sin cambio; FFN activo 25%.
- Router: inicializar **analíticamente** desde centroides / neuronas representativas (CMoE), no aleatorio — el router aleatorio training-free con PPL >10k es el modo de falla publicado.
- SwiGLU: recortar gate/up/down por el mismo índice intermedio; el input del router es el input del bloque (hidden state post-norm).
- Balanceo: aux loss **0,01** (Switch/NVIDIA), monitorear share de tokens por experto; si algún experto <1% o >40%, subir el peso aux o usar Expert Choice en esa capa.
- Fine-tune: solo router + bloque MoE, LR ~1e-5–5e-5 (análogo a pequeña escala del "LR bajo constante" de NVIDIA — pero solo porque el *resto* del modelo está congelado; si se descongela todo, resetear el LR hacia el pico de pretraining, p. ej. 1e-4–2e-4, con cosine decay).
- **Criterio de éxito:** PPL dentro de ~10% del denso y MMLU dentro de ~2 puntos con 25% de FFN activo, con speedup medido (si no hay speedup en tu hardware/batch, la conversión no vale la pena shippearla — el overhead de routing domina a 0,9B).

**Paso 3 — Solo si el Paso 2 funciona: upcycling que expande capacidad (la apuesta de investigación).**
- Copiar FFN ×4 (no ×8 — menos expertos idénticos diversifican más rápido) como expertos, top-2, init estilo virtual-group si es fine-grained, weight scaling ∛(E·G²/T).
- Entrenar *todos* los pesos, LR reseteado al pico de pretraining, cosine decay, batch ~2–4M tokens, aux loss 0,01.
- Presupuesto: **1–4B tokens en español** — el D* de Liew para un modelo ~1B es ~4B tokens; más allá, entrenar un MoE desde cero (o seguir el modelo denso) se espera que gane. A 0,9B, 1B tokens ya es un experimento grande; no arrancar con corridas estilo LLaMA-MoE de 200B tokens (perdieron 8+ puntos de MMLU a escala 7B).
- **Regla de corte:** si tras ~0,5B tokens la loss del upcycled no está por debajo de la baseline del denso continuado entrenado con los mismos tokens, abortar — es la señal publicada de que el upcycling no está pagando en tu régimen.

**Qué NO hacer:** (a) copiar FFN×8 + router aleatorio + sin entrenamiento y esperar mejora; (b) destilar solo con forward-KL desde un profesor mucho más grande a un estudiante 0,9B — usar reverse-KL/JSD (GKD, β≈0,5–1,0) o SeqKD; (c) evaluar solo PPL; (d) convertir las capas DeltaNet de Qwen3.5-0.8B como si fueran FFN de atención.

**Resultado esperado (honesto):**
- Pasos 1–2: un modelo con el **mismo tamaño total, ~25–50% de cómputo FFN activo, ~90–95% de la calidad densa** (extrapolando CMoE/ToMoE/DOT-MoE a 7B; a 0,9B esperar el extremo bajo), y speedup real solo si el batch es lo bastante grande para amortizar el routing.
- Paso 3: **+0–2 puntos MMLU sobre el continued dense training con iguales tokens, si es que funciona** (Nemotron ganó +2,3 a 15B/1T tokens; el trabajo split-FFN a 7B *perdió* puntos). Presupuestar la posibilidad de que no funcione y que el entregable sea el modelo del Paso 2 + un resultado negativo con números.

## 5. Correcciones a premisas

- **Sparse Upcycling arXiv ID es 2212.05055, no 2212.02574.**
- **No existe un "Qwen3-0.8B" denso** en la familia Qwen3 original (abr 2025: 0,6B, 1,7B, 4B, 8B, 14B, 32B densos; 30B-A3B / 235B-A22B MoE). El modelo 0,8B es casi seguro **Qwen3.5-0.8B** (feb 2026: 24 capas, hidden 1024, intermedio 3584, vocab 248320 tied, híbrido Gated DeltaNet + atención, 262K contexto) — que *no* es un transformer denso plano (¾ de sus capas son DeltaNet de atención lineal, así que solo sus capas de atención tienen un FFN estándar para partir).
- **K2-Horizon-0.9B es real y muy nuevo** (IFM/MBZUAI, 3 sep 2026): denso 0,9B, contexto 128K con YaRN, Apache 2.0, "multi-teacher distillation" para math/code, STEM e instruction following; AIME 2026 48,5 y HumanEval+ 79,9 según el vendor (verificación independiente pendiente). HF: `IFM/K2-Horizon-0.9B`.
- **`eduardo-bertey/LLM-MOHO` no se pudo verificar vía búsqueda web** en la investigación inicial (7 nov 2026); ninguna afirmación de este informe depende de él. El repo sí existe y es el que Juls/Eduardo está trabajando (ver `llm-moho-revision.md` en la raíz).

## 6. Codebases y tooling práctico

| Codebase | Para qué | Link |
|---|---|---|
| Megatron-LM / Megatron-Core | Upcycling en producción a 27B; virtual-group init, weight scaling, EP | https://github.com/NVIDIA/Megatron-LM |
| t5x / vMoE (Google) | Código original de sparse-upcycling (lenguaje / visión) | https://github.com/google-research/t5x/tree/main/t5x/contrib/moe |
| LLaMA-MoE | Split-FFN + continual pretraining, Llama-2-7B | https://github.com/pjlab-sys4nlp/llama-moe |
| CMoE | Router analítico, expertos compartidos + ruteados, escala de minutos | https://github.com/JarvisPei/CMoE |
| ToMoE | Solo router, pesos congelados | https://github.com/gaosh/ToMoE |
| HF Transformers | Modelos MoE nativos (Mixtral, Qwen3-MoE, DeepSeek-V3); `output_router_logits`, `aux_loss` | https://huggingface.co/docs/transformers/main/model_doc/qwen3 |
| TRL | GKD / DistillationTrainer (incl. coef. de aux-loss MoE) | https://huggingface.co/docs/trl/v0.24.0/en/gkd_trainer |
| Liew scaling-law code | Código de ajuste de D* / joint-law y datos de loss | https://github.com/sbintuitions/sparse-upcycling-scaling-laws |
| Skywork-MoE | Referencia de upcycle 13B | https://github.com/SkyworkAI/Skywork-MoE |
| Tutel / MegaBlocks / ScatterMoE | Kernels MoE rápidos (expert-parallel dispatch) — necesarios una vez que el routing top-k se sirve de verdad; el MoE plano de HF es lento con batch chico | Verificar repos upstream antes de depender de ellos |
| vLLM / SGLang | Servir MoE (y los modelos K2/Qwen) | — |

## 7. No verificado / preguntas abiertas

- **No existe conversión a 27B** — el gap central. Cualquier benchmark a 27B en este informe es extrapolado; solo una corrida real lo cierra.
- **No hay resultado publicado denso→MoE a exactamente 0,8–0,9B** con MMLU/ARC/HellaSwag/GSM8K para ambas familias (split y copia); lo más cercano es 2B (NVIDIA), 7B (LLaMA-MoE/CMoE/ToMoE) y modelos densos ≤1B en el estudio de leyes de escala (solo loss, sin benchmarks downstream). Todas las expectativas a 0,9B son extrapolaciones.
- **Resultados en español:** ninguno encontrado; todos los números son benchmarks en inglés. La PPL / MMLU-es en español hay que medirla localmente.
- **LR pico de pretraining de Gemma-2** (necesario para el reset de la Etapa 2 a 27B) no es público. El 2e-4 (2B) / 3e-4 pico (15B, corrida 1T) de Nemotron es el proxy usado.
- **Gemma-3 27B PT ARC-C / HellaSwag y Qwen-32B ARC/HellaSwag** no se encontraron en las tablas consultadas.
- **Gemma-3 GSM8K 82,6 vs 81,20** — discrepancia de protocolo (8-shot vs 4-shot CoT), sin resolver; citar como rango.
- **Gemma-4 26B-A4B** (LMArena 1441, MMLU-Pro 82,6) son resúmenes secundarios; no había informe técnico completo disponible. Si Google publica uno, se vuelve el comparador clave de MoE nativo a 26B.
- **"0,3 yottaFLOPs" de Nemotron** para el programa de 1T se cita como está declarado; su alcance exacto (una variante vs ambas) es ambiguo en el paper.
- **Base del log en D* de Liew** (natural vs base-10) no está declarada; se dan ambas evaluaciones (0,62B vs 0,48B) — la diferencia no cambia ninguna conclusión.
- **Qwen3.8-27B** (ago 2026) benchmarks y tokens de pretraining no verificados; puede superar a Gemma-2 como mejor padre exacto de 27B — re-chequear antes de arrancar.
- **Throughput de inferencia a 27B** (tok/s, $/Mtok) para cualquier variante upcycled no está medido; solo razonamiento FLOPs/bytes.

## Fuentes

- Hinton et al. 2015 — arXiv:1503.02531
- Distillation tutorial/survey — https://llm-tutorial.darvinyi.com/ch16-distillation/
- Distillation in LLMs — https://prateeksinghphd.in/distillation-llms.html
- Model compression survey — https://ar5iv.labs.arxiv.org/html/2308.07633
- On-policy distillation survey — https://arxiv.org/html/2604.00626v1
- GKD / TRL — https://huggingface.co/docs/trl/v0.24.0/en/gkd_trainer
- MiniLLM — https://github.com/microsoft/LMOps/tree/main/minillm ; paper https://pdfs.assets.alphaxiv.org/2306.08543v6.pdf
- DistillSpec — https://arxiv.org/pdf/2310.08461
- Sparse Upcycling (Komatsuzaki et al.) — https://ar5iv.labs.arxiv.org/html/2212.05055
- Upcycling LLMs into MoE (He et al., NVIDIA) — https://arxiv.org/html/2410.07524v2
- Scaling Laws for Upcycling (Liew et al.) — https://arxiv.org/html/2502.03009v2
- MoEfication (Zhang et al.) — https://arxiv.org/abs/2110.01786v2
- Branch-Train-MiX (Sukhbaatar et al.) — https://ar5iv.labs.arxiv.org/html/2403.07816
- LLaMA-MoE — https://github.com/pjlab-sys4nlp/llama-moe
- CMoE — https://arxiv.org/html/2502.04416v1 ; código https://github.com/JarvisPei/CMoE
- ToMoE — https://arxiv.org/html/2501.15316v2 ; código https://github.com/gaosh/ToMoE
- DOT-MoE — https://pub-fad10fba280c45dba0823fd585acc08c.r2.dev/pdf/2606.01666.pdf
- Gemma 2 technical report — https://arxiv.org/html/2408.00118v1
- Gemma 3 technical report — https://arxiv.org/pdf/2503.19786v1.pdf
- Qwen2.5 technical report — https://arxiv.org/html/2412.15115v2
- Qwen3 technical report — https://arxiv.org/pdf/arXiv:2505.09388
- Skywork-MoE — https://arxiv.org/html/2406.06563v1/
- DeepSeek-V3 — https://github.com/Shyboy0499/DeepSeek-V3/blob/HEAD/README.md
- Megatron-Core MoE / upcycling — https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/moe.html
- Qwen3-30B-A3B serving / MoE math — https://www.aicassindra.com/blogs/llm/llm_moe_serving.html
- H100 training economics — https://www.aicassindra.com/blogs/gpu/flops_budget_training.html
