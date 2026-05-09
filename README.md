# Proyecto 7 — Codificación Posicional en Transformers

**Curso:** CC0C2 Procesamiento del Lenguaje Natural  
**Documentos base:** `Learner-Transformer-Workbook.xlsx` (hoja `Transformer_Blank`), Cuaderno 9 y Cuaderno 11

---

## Objetivo

Demostrar experimentalmente que **self-attention es invariante a la permutación**: sin información posicional, el modelo no puede distinguir una secuencia de tokens de su versión invertida. Se comparan tres configuraciones del mismo Transformer encoder, cambiando únicamente el módulo de codificación posicional.

---

## Cuaderno y workbook base usados

| Elemento | Origen |
|---|---|
| `SinusoidalPositionalEncoding` | Cuaderno 9, Sección 2.1 — código sin modificaciones |
| `LearnedPositionalEncoding` | Cuaderno 9, Sección 2.2 — código sin modificaciones |
| `SimpleVocab` y `basic_tokenize` | Cuaderno 9, Sección 1 — código sin modificaciones |
| Corpus de entrenamiento | Cuaderno 9, Sección 1 (llama, dog, horse…) |
| Arquitectura `Net` → `TransformerClassifier` | Cuaderno 11, Sección 5 — misma estructura |
| `AdamW` + `clip_grad_norm_` | Cuaderno 9, Sección 6 |
| Tabla PE sinusoidal | `Learner-Transformer-Workbook.xlsx`, hoja `Transformer_Blank` |

---

## Línea base implementada

**`SinusoidalPositionalEncoding` + Transformer encoder + clasificación binaria**

- **Tarea:** clasificar si una secuencia de tokens está en orden correcto (etiqueta=1) o invertida (etiqueta=0). Los mismos tokens en distinto orden tienen etiquetas opuestas.
- **Arquitectura** (igual a clase `Net` del Cuaderno 11):  
  `nn.Embedding → SinusoidalPositionalEncoding → nn.TransformerEncoder → mean(dim=1) → nn.Linear`
- **Hiperparámetros:** `d_model=32`, `n_heads=4`, `d_ff=64`, `n_layers=1`, `seq_len=6`
- **Optimizador:** `AdamW(lr=3e-3, weight_decay=0.01)` + `clip_grad_norm_(max_norm=1.0)`
- **Pérdida:** `CrossEntropyLoss`

---

## Modificación realizada

Se reemplaza el módulo de PE en dos variantes adicionales, manteniendo **idéntica** la arquitectura del encoder:

| Configuración | Módulo PE | Parámetros extra |
|---|---|---|
| **Línea base** | `SinusoidalPositionalEncoding` (Cuaderno 9) | 0 (vectores fijos) |
| **Variante 1** | `NoPositionalEncoding` (sin PE) | 0 |
| **Variante 2** | `LearnedPositionalEncoding` (Cuaderno 9) | `(SEQ_LEN+1) × d_model = 224` |

---

## Resultados principales

| Modelo | Acc Train | **Acc Test** | Conclusión |
|---|---|---|---|
| Sinusoidal (línea base) | 100.00 % | **100.00 %** | Captura el orden correctamente |
| Sin PE (variante 1) | 53.31 % | **46.15 %** | Azar — self-attention es invariante a la permutación |
| Aprendida (variante 2) | 99.61 % | **100.00 %** | Aprende la posición desde los datos |

El modelo sin PE alcanza 46 % (azar puro), confirmando que sin información posicional el encoder no distingue el orden de los tokens.

---

## Errores y limitaciones

**Errores identificados en test:**
- El modelo sin PE comete errores balanceados en ambas clases, confirmando que opera al azar.
- Los modelos con PE cometen 0 errores en test; los errores durante el entrenamiento se concentran al inicio, antes de que el encoder aprenda a usar la señal posicional.

**Limitaciones técnicas:**
1. **PE aprendida no generaliza** a posiciones mayores que `SEQ_LEN` — no tiene vectores entrenados para esas posiciones.
2. **Mean pooling** sobre la salida del encoder diluye la información posicional aprendida.
3. **Dataset sintético** — en lenguaje natural, "orden correcto" es más complejo que una simple inversión de ids.
4. **Una sola capa encoder** puede no capturar dependencias posicionales de largo alcance.

**Qué superaría estas limitaciones:**
- **RoPE / ALiBi** (Cuaderno 9, Secciones 2.3 / 2.4): codifican posición dentro de $QK^\top$, generalizan mejor a secuencias más largas.
- Más capas encoder y más cabeceras para representaciones más ricas.

---

## Conclusión técnica

La diferencia entre un Transformer que aprende y uno que opera al azar está en la codificación posicional. Self-attention calcula $\text{softmax}(QK^\top / \sqrt{d_k})V$: si permutamos los tokens, la salida se permuta igual y el modelo no detecta el cambio. La PE sinusoidal y la aprendida resuelven esto de formas distintas — la primera con una fórmula fija que generaliza; la segunda con parámetros que se adaptan al corpus. Ambas superan en 50+ puntos porcentuales al modelo sin PE en esta tarea.
