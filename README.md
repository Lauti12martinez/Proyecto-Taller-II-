# Desafío Taller II - Modelo de Análisis de Sentimiento

**Materia:** Taller II — Ingeniería en Inteligencia Artificial (UNSTA)  
**Proyecto:** Desafío Taller II - Modelo de Análisis de Sentimiento (Strata)

---

## 📌 Descripción del Proyecto

El objetivo del proyecto consiste en construir un pipeline integral de Machine Learning para análisis de sentimiento: desde la obtención y preprocesamiento de un corpus de reseñas, pasando por diversas representaciones numéricas del texto (BoW, TF-IDF unigrama/bigrama, embeddings pre-entrenados y pipelines de Hugging Face), hasta el entrenamiento, optimización, evaluación sobre un banco adversarial curado por el curso y el despliegue de una aplicación web básica para inferencia ante el cliente (Strata).

El repositorio está organizado en **dos flujos experimentales independientes** que comparten la misma ingesta de datos crudos y vectores pre-entrenados:

* **`biclase/` (Enfoque Binario):** Descarta las reseñas neutras y clasifica en 2 polaridades (`0`: Negativo, `1`: Positivo).
* **`triclase/` (Enfoque Multiclase):** Conserva la totalidad del corpus e incorpora la categoría neutral (`0`: Negativo, `1`: Neutral, `2`: Positivo).

---

## 📁 Estructura del Repositorio

```text
Proyecto-Taller-II-/
├── data/
│   ├── raw/                        # Compartido: Dataset crudo descargado de la API (ignorado en Git)
│   └── external/                   # Compartido: Vectores pre-entrenados GloVe 100d (ignorado en Git)
├── src/
│   └── extract_reviews.py          # Script único de ingesta masiva desde la API de Amazon Reviews
├── biclase/                        # Experimento de 2 categorías (Negativo y Positivo)
│   ├── data/
│   │   ├── processed/              # Dataset limpio (~168k reseñas) y split train/test (ignorado en Git)
│   │   └── vectorized/             # Matrices X_train/X_test e y_train/y_test binarias (ignorado en Git)
│   ├── models/                     # Vectorizadores y modelos binarios serializados (ignorado en Git)
│   ├── notebooks/
│   │   └── 01_eda.ipynb            # Análisis exploratorio (EDA) para 2 clases
│   ├── reports/
│   │   └── vectorization_summary.md # Tamaño de vocabulario y dimensiones (Biclase)
│   └── src/
│       ├── preprocess.py           # Limpieza y binarización de etiquetas (descarta neutras)
│       └── vectorize.py            # Vectorización BoW, TF-IDF, GloVe y tokenizador HF (SST-2)
├── triclase/                       # Experimento de 3 categorías (Negativo, Neutral y Positivo)
│   ├── data/
│   │   ├── processed/              # Dataset limpio (~210k reseñas) y split train/test (ignorado en Git)
│   │   └── vectorized/             # Matrices X_train/X_test e y_train/y_test triclase (ignorado en Git)
│   ├── models/                     # Vectorizadores y modelos triclase serializados (ignorado en Git)
│   ├── notebooks/
│   │   ├── 01_eda.ipynb            # Análisis exploratorio (EDA) para 3 clases
│   │   └── 02_evaluation.ipynb     # Comparativa F1, matrices de confusión y curvas de aprendizaje
│   ├── reports/
│   │   ├── vectorization_summary.md # Tamaño de vocabulario y dimensiones (Triclase)
│   │   ├── metrics_<bloque>.csv    # Métricas por bloque (linear, opt, hf, dense) con esquema común
│   │   └── figures/                # Gráficos exportados por los scripts y el notebook
│   └── src/
│       ├── preprocess.py           # Limpieza y mapeo en 3 clases (incluye categoría neutral)
│       ├── vectorize.py            # Vectorización BoW, TF-IDF, GloVe y tokenizador HF (RoBERTa 3 clases)
│       ├── train_linear.py         # NB, LogReg, LinearSVC y SGD sobre matrices dispersas
│       ├── optimize_cv.py          # Optimización CV de los modelos clásicos
│       ├── eval_hf.py              # Inferencia del pipeline de Hugging Face sobre el test
│       ├── train_dense.py          # RF, XGBoost y MLP optimizados sobre embeddings GloVe
│       └── evaluate_adversarial.py # Evaluación de todos los modelos sobre el banco adversarial
├── .gitignore                      # Exclusión de archivos pesados y entornos locales
├── pyproject.toml                  # Especificación de dependencias y configuración Poetry
├── poetry.lock                     # Registro determinista de versiones
└── README.md
```

---

## ⚙️ Requisitos e Instalación

El proyecto gestiona su entorno y dependencias mediante **Poetry**.

### 1. Clonar el repositorio

```bash
git clone git@github.com:Lauti12martinez/Proyecto-Taller-II-.git
cd Proyecto-Taller-II-
```

### 2. Instalar el entorno y dependencias

Asegurarse de contar con Poetry instalado localmente y ejecutar en la raíz del proyecto:

```bash
poetry install
```

> 💡 **Nota para usuarios de Windows (PyTorch DLLs):**  
> Si al ejecutar `vectorize.py` aparece el error `OSError: [WinError 126] ... Error loading "shm.dll"`, instalar las librerías oficiales de C++ de Microsoft ejecutando en PowerShell:  
> `winget install Microsoft.VCRedist.2015+.x64` y reiniciar VS Code.

---

## 📥 Ingesta de Datos (API de Amazon Reviews)

El corpus crudo se descarga una única vez en la raíz del proyecto consultando el servidor de la API provista por la cátedra (`amazon-reviews-api-g5ae.onrender.com`):

```bash
poetry run python src/extract_reviews.py
```

### Especificaciones del Dataset Crudo

* **Total de registros:** 210.000 reseñas.
* **Archivo de salida compartido:** `data/raw/amazon_reviews_raw.csv`.
* **Distribución de clases:** Completamente uniforme y balanceada (42.000 reseñas por cada una de las 5 calificaciones originales, con etiquetas `label` del `0` al `4`).
* **Esquema devuelto:** `id`, `text`, `label`, `label_text`.
* **Resiliencia:** Incluye control de *Cold Start* en Render, reintentos automáticos ante cortes de red y paginación por lotes de 1.000 registros con barra de progreso interactiva (`tqdm`).

> ⚠️ **Importante sobre el control de versiones (.gitignore):**  
> El archivo `data/raw/amazon_reviews_raw.csv`, los embeddings GloVe, las matrices vectorizadas y los modelos `.joblib` **no deben subirse al historial de GitHub**. El `.gitignore` omite esos artefactos pesados y únicamente habilita el versionado de archivos livianos compartidos por el equipo (`data/external/adversarial_test.csv`, `**/data/vectorized/y_pred_*.npy` y `**/reports/*.csv`). Para obtener los datasets base y modelos serializados, utilizar la sección de **GitHub Releases** descrita más abajo.

---

## 🧹 Preprocesamiento (Biclase vs. Triclase)

Ambos pipelines leen el archivo compartido `data/raw/amazon_reviews_raw.csv` y aplican exactamente las mismas reglas de normalización léxica para garantizar una comparativa justa:

* **Limpieza de `text` → `cleaned_text`:** Conversión a minúsculas, sin HTML, sin dígitos numéricos, supresión de signos de puntuación extendidos, tokenización estructurada con NLTK, filtrado de stopwords y eliminación de tokens de un solo carácter, y lematización morfológica con WordNet.
* **Tratamiento enriquecido de negaciones:** No solo se conservan `not`, `no` y `nor`, sino un vocabulario exhaustivo de términos que invierten o restringen polaridad (*never*, *without*, *hardly*, *barely*, *rarely*, *nobody*, *neither*, etc.) y sus variantes informales de tipeo (*dont*, *didnt*, *wont*). Las contracciones negativas estándar se expanden antes de quitar la puntuación (`don't` → `do not`, `can't` → `can not`, `won't` → `will not`), evitando que expresiones como *"not good"* se reduzcan a *"good"*.
* La columna `text` original se mantiene intacta: los embeddings contextuales y tokenizadores de Hugging Face trabajan sobre el texto crudo.

### 1. Ejecución Enfoque Biclase (~168.000 reseñas)

```bash
poetry run python biclase/src/preprocess.py
```

* **Salida:** `biclase/data/processed/amazon_reviews_clean.csv`.
* **Etiquetas:** Se descartan las reseñas neutras (`label == 2`) y se binariza en `sentiment` (`0` = negativa: labels 0-1; `1` = positiva: labels 3-4). Clases balanceadas (50% / 50%).

### 2. Ejecución Enfoque Triclase (~210.000 reseñas)

```bash
poetry run python triclase/src/preprocess.py
```

* **Salida:** `triclase/data/processed/amazon_reviews_clean.csv`.
* **Etiquetas:** Se conservan todas las reseñas y se recodifican en 3 clases dentro de `sentiment` (`0` = negativa: labels 0-1; `1` = neutral: label 2; `2` = positiva: labels 3-4). Distribución 40% / 20% / 40%.

---

## 📊 Análisis Exploratorio de Datos (EDA)

Cada flujo cuenta con su propio notebook de análisis exploratorio adaptado a la cantidad de categorías del experimento (balance de clases, distribución de longitud de reseñas, métricas léxicas del corpus y Top 20 palabras por polaridad):

* **Notebook Biclase:** `biclase/notebooks/01_eda.ipynb`.
* **Notebook Triclase:** `triclase/notebooks/01_eda.ipynb`.

---

## 🔢 Vectorización

Para ejecutar el pipeline de vectorización en cada experimento (siempre desde la raíz del proyecto):

```bash
# Vectorización para el enfoque de 2 categorías
poetry run python biclase/src/vectorize.py

# Vectorización para el enfoque de 3 categorías
poetry run python triclase/src/vectorize.py
```

Requiere haber ejecutado previamente el `preprocess.py` correspondiente. La primera ejecución descarga los vectores GloVe (~820 MB, con soporte de reanudación automática ante cortes de red y extracción de `data/external/glove.6B.100d.txt` compartido por ambos flujos) y el tokenizador del modelo de Hugging Face.

1. **Partición train/test estratificada 80/20 (`random_state=42`):** Los IDs quedan registrados en `<enfoque>/data/processed/train_test_split.csv` y las etiquetas en `<enfoque>/data/vectorized/y_train.npy` e `y_test.npy`, asegurando que **todos los modelos de cada flujo compitan sobre exactamente la misma partición**.
2. **Prevención de Data Leakage:** Los vectorizadores se ajustan únicamente con los datos de entrenamiento (`fit_transform` sobre train) y luego transforman el conjunto de prueba (`transform` sobre test).

| Representación | Implementación | Entrada | Salida en `<enfoque>/data/vectorized/` |
|---|---|---|---|
| BoW | `CountVectorizer(min_df=5)` | `cleaned_text` | `X_{train,test}_bow.npz` (dispersa) |
| TF-IDF unigrama | `TfidfVectorizer(min_df=5, ngram_range=(1,1))` | `cleaned_text` | `X_{train,test}_tfidf_unigram.npz` |
| TF-IDF bigrama | `TfidfVectorizer(min_df=5, ngram_range=(1,2))` | `cleaned_text` | `X_{train,test}_tfidf_bigram.npz` |
| Embeddings | GloVe 6B 100d, promedio de los vectores de las palabras de cada reseña | `cleaned_text` | `X_{train,test}_glove.npy` (densa, 100 dims) |
| Pipeline HF (Biclase) | Tokenizador de `distilbert-base-uncased-finetuned-sst-2-english` | `text` (crudo) | Estadísticas de tokens; inferencia en etapa de evaluación |
| Pipeline HF (Triclase) | Tokenizador de `cardiffnlp/twitter-roberta-base-sentiment-latest` | `text` (crudo) | Estadísticas de tokens; inferencia en etapa de evaluación |

Las matrices se guardan en `<enfoque>/data/vectorized/` y los vectorizadores ajustados en `<enfoque>/models/vectorizer_*.joblib` (para reutilizarlos en el banco adversarial y la app web). El **tamaño de vocabulario de cada representación** queda documentado en:

* **Resumen Biclase:** `biclase/reports/vectorization_summary.md`.
* **Resumen Triclase:** `triclase/reports/vectorization_summary.md`.

Para cargar una representación desde un notebook o script de modelado:

```python
import numpy as np
from scipy import sparse

# Elegir el flujo experimental: "biclase" o "triclase"
ENFOQUE = "triclase"

X_train = sparse.load_npz(f"{ENFOQUE}/data/vectorized/X_train_tfidf_bigram.npz")
X_test = sparse.load_npz(f"{ENFOQUE}/data/vectorized/X_test_tfidf_bigram.npz")
y_train = np.load(f"{ENFOQUE}/data/vectorized/y_train.npy")
y_test = np.load(f"{ENFOQUE}/data/vectorized/y_test.npy")
```

---

## 📦 Distribución de Datasets y Modelos (GitHub Releases)

Los artefactos pesados (dataset crudo, embeddings GloVe y modelos/vectorizadores `.joblib`) **no se suben al historial de Git**, sino como assets adjuntos en **GitHub Releases** del repositorio:

* **Durante la semana:** se publican **pre-releases** por bloque de trabajo para compartir datasets base y modelos intermedios con el equipo.
* **Viernes 20/11:** se publica el **Release consolidado en `main`** con todos los modelos finales listos para la entrega parcial del **23/11**.

### Rutas exactas para descomprimir los archivos del Release

Desde la raíz del repositorio (`Proyecto-Taller-II-/`), extraer cada archivo descargado del Release en su ruta correspondiente:

| Asset del Release | Ruta exacta de destino en el repo | Uso |
|---|---|---|
| `amazon_reviews_raw.zip` | `data/raw/amazon_reviews_raw.csv` | Corpus crudo de entrada para `preprocess.py` |
| `glove.6B.100d.zip` | `data/external/glove.6B.100d.txt` | Embeddings pre-entrenados compartidos por `vectorize.py` |
| Modelos y vectorizadores (`*.joblib`) | `<enfoque>/models/*.joblib` (`triclase/models/*.joblib` y `biclase/models/*.joblib`) | Vectorizadores y clasificadores entrenados para inferencia y evaluación adversarial |

---

## 🤖 Entrenamiento, Comparación y Evaluación de Modelos

Para la entrega del **16/11/2026**, el objetivo principal consiste en entrenar y comparar modelos de clasificación sobre las matrices estratificadas al 80/20 (`<enfoque>/data/vectorized/`) y evaluar su robustez ante el **test set adversarial** curado en conjunto por el curso:

1. **Cuatro enfoques de representación a contrastar:**
   - BoW / TF-IDF unigrama
   - TF-IDF bigrama
   - Embeddings pre-entrenados (GloVe 100d)
   - Pipeline pre-entrenado de Hugging Face (SST-2 para biclase / RoBERTa para triclase)
2. **Algoritmos sugeridos por la cátedra:** Naive Bayes (NB), Regresión Logística (LogReg), Random Forest (RF) y Perceptrón Multicapa (MLP). Cada equipo selecciona libremente la combinación de clasificadores para encontrar la solución más óptima.
3. **Métricas de evaluación:** Accuracy, Precision, Recall, F1-Score (macro para triclase) y Matriz de Confusión.
4. **Optimización e interpretación:** Ajuste de hiperparámetros (`GridSearchCV` / `RandomizedSearchCV`) sobre los modelos líderes, análisis de palabras relevantes y justificación explícita del modelo final seleccionado para la defensa ante Strata.

### Modelos densos sobre GloVe (Random Forest, XGBoost y MLP)

```bash
# Triclase (por defecto)
poetry run python triclase/src/train_dense.py

# Biclase, con el mismo script parametrizado
poetry run python triclase/src/train_dense.py --enfoque biclase
```

Requiere `X_train_glove.npy` / `X_test_glove.npy` generados por `vectorize.py`. `MultinomialNB` se excluye porque los embeddings GloVe contienen valores negativos. Para cada algoritmo:

1. **Búsqueda de hiperparámetros:** `RandomizedSearchCV` (`cv=5`, `scoring="f1_macro"`) sobre una submuestra estratificada del train (`--search-size`, 40.000 reseñas por defecto; `0` usa el train completo) para acotar el costo en CPU.
2. **Curva de aprendizaje** con los mejores hiperparámetros sobre el train completo (5 tamaños × 5 folds). Su último punto es la validación cruzada de 5 folds que se informa como `cv_f1_mean` / `cv_f1_std`.
3. **Ajuste final** sobre todo el train y evaluación en el 20% de test.

| Salida | Contenido |
|---|---|
| `<enfoque>/models/model_{rf,xgb,mlp}_glove.joblib` | Modelos finales (el MLP incluye su `StandardScaler` como pipeline). `evaluate_adversarial.py` los detecta solo. |
| `<enfoque>/reports/metrics_dense.csv` | Métricas con el esquema común del contrato de integración. |
| `<enfoque>/reports/cv_search_dense.csv` | Historial de combinaciones probadas en la búsqueda. |
| `<enfoque>/reports/learning_curve_dense.csv` | Datos de las curvas de aprendizaje. |
| `<enfoque>/data/vectorized/y_pred_dense_{rf,xgb,mlp}.npy` | Predicciones sobre el test (para las matrices de confusión). |

Opciones: `--models rf xgb mlp`, `--n-iter 10`, `--search-size 40000`, `--n-jobs -1`.

### Notebook de evaluación consolidada

`triclase/notebooks/02_evaluation.ipynb` lee dinámicamente todos los `metrics_*.csv`, `adversarial_results.csv`, `y_pred_*.npy` y `learning_curve_*.csv` del enfoque elegido (`ENFOQUE = "triclase"` o `"biclase"` en la primera celda) y genera en `<enfoque>/reports/figures/`:

* Comparativa de **F1 macro** de todos los modelos y dispersión **CV vs test** (estabilidad).
* **Test vs banco adversarial** (robustez ante negaciones e inversiones).
* **Matrices de confusión** normalizadas del mejor modelo de cada bloque. Si un bloque no guardó sus predicciones, se calculan con su `.joblib` y se cachean como `y_pred_*.npy`.
* **Curvas de aprendizaje** de cada `learning_curve_*.csv` (y la del líder lineal `logreg + tfidf_bigram`, calculada y cacheada la primera vez).

Un bloque nuevo que respete el contrato de nombres aparece en el notebook sin modificar código.

---

## 🗓️ Cronograma de Hitos del Proyecto

* **28/09/2026:** Entrega parcial: Dataset extraído y preprocesado para análisis de sentimiento.
* **05/10/2026:** Word Embeddings y aporte de 2-3 oraciones al banco adversarial común del curso.
* **19/10/2026:** Entrega parcial: Dataset vectorizado (BoW, TF-IDF unigrama/bigrama, embeddings, Hugging Face).
* **16/11/2026:** Entrega parcial: Comparativa y evaluación de modelos sobre el test set adversarial.
* **14/12/2026:** Entrega final: Informe integrador, repositorio actualizado, app web y defensa técnica con Strata.

---

## 👥 Integrantes del Equipo

* Patricio Sebastián Castillo Upton
* Lautaro Martínez Serrano
* Juan Pablo Rébora
* Lautaro Rivadeneira
* Joaquin Mecle
* Germán Rodriguez Vaquero