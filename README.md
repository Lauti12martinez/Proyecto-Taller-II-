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
│   │   └── 01_eda.ipynb            # Análisis exploratorio (EDA) para 3 clases
│   ├── reports/
│   │   └── vectorization_summary.md # Tamaño de vocabulario y dimensiones (Triclase)
│   └── src/
│       ├── preprocess.py           # Limpieza y mapeo en 3 clases (incluye categoría neutral)
│       └── vectorize.py            # Vectorización BoW, TF-IDF, GloVe y tokenizador HF (RoBERTa 3 clases)
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

El corpus crudo se descarga una única vez en la raíz del proyecto consultando la API provista por la cátedra (`[https://amazon-reviews-api-g5ae.onrender.com](https://amazon-reviews-api-g5ae.onrender.com)`):

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
> El archivo `data/raw/amazon_reviews_raw.csv` supera los 100 MB y **no debe subirse a GitHub**. Ya está configurada la regla en el `.gitignore` para omitir `*.csv`. Cada integrante debe generar su copia local corriendo el script de extracción o solicitando el CSV por el almacenamiento compartido del grupo.

---

## 🧹 Preprocesamiento (Biclase vs. Triclase)

Ambos pipelines leen el archivo compartido `data/raw/amazon_reviews_raw.csv` y aplican exactamente las mismas reglas de normalización léxica para garantizar una comparativa justa:

* **Limpieza de `text` → `cleaned_text`:** minúsculas, sin HTML, sin dígitos, sin puntuación, tokenización (NLTK), sin stopwords ni tokens de una letra, lematización (WordNet).
* **Negaciones:** `not`, `no` y `nor` **se conservan** (se quitaron de las stopwords) y las contracciones negativas se expanden antes de quitar la puntuación (`don't` → `do not`, `can't` → `can not`, `won't` → `will not`). Sin esto, "not good" quedaba como "good", lo que invierte el sentimiento y rompe el banco adversarial.
* La columna `text` original se mantiene: los embeddings contextuales (Hugging Face) trabajan sobre el texto crudo.

### 1. Ejecución Enfoque Biclase (~168.000 reseñas)

```bash
poetry run python biclase/src/preprocess.py
```

* **Salida:** `biclase/data/processed/amazon_reviews_clean.csv`.
* **Etiquetas:** se descartan las reseñas neutras (`label == 2`) y se binariza en `sentiment` (`0` = negativa: labels 0-1; `1` = positiva: labels 3-4). Clases balanceadas (50% / 50%).

### 2. Ejecución Enfoque Triclase (~210.000 reseñas)

```bash
poetry run python triclase/src/preprocess.py
```

* **Salida:** `triclase/data/processed/amazon_reviews_clean.csv`.
* **Etiquetas:** se conservan todas las reseñas y se recodifican en 3 clases dentro de `sentiment` (`0` = negativa: labels 0-1; `1` = neutral: label 2; `2` = positiva: labels 3-4). Distribución 40% / 20% / 40%.

---

## 📊 Análisis Exploratorio de Datos (EDA)

Cada flujo cuenta con su propio notebook de análisis exploratorio adaptado a la cantidad de categorías del experimento (balance de clases, distribución de longitud de reseñas, métricas léxicas del corpus y Top 20 palabras por polaridad):

* **Notebook Biclase:** [`biclase/notebooks/01_eda.ipynb`](biclase/notebooks/01_eda.ipynb)
* **Notebook Triclase:** [`triclase/notebooks/01_eda.ipynb`](triclase/notebooks/01_eda.ipynb)

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

1. **Partición train/test** estratificada 80/20 (`random_state=42`). Los ids quedan en `<enfoque>/data/processed/train_test_split.csv` y las etiquetas en `<enfoque>/data/vectorized/y_train.npy` / `y_test.npy`, para que **todos los modelos de cada flujo usen la misma partición**.
2. **Los vectorizadores se ajustan solo con train** (para no filtrar información del test) y se aplican a ambos conjuntos.

| Representación | Implementación | Entrada | Salida en `<enfoque>/data/vectorized/` |
|---|---|---|---|
| BoW | `CountVectorizer(min_df=5)` | `cleaned_text` | `X_{train,test}_bow.npz` (dispersa) |
| TF-IDF unigrama | `TfidfVectorizer(min_df=5, ngram_range=(1,1))` | `cleaned_text` | `X_{train,test}_tfidf_unigram.npz` |
| TF-IDF bigrama | `TfidfVectorizer(min_df=5, ngram_range=(1,2))` | `cleaned_text` | `X_{train,test}_tfidf_bigram.npz` |
| Embeddings | GloVe 6B 100d, promedio de los vectores de las palabras de cada reseña | `cleaned_text` | `X_{train,test}_glove.npy` (densa, 100 dims) |
| Pipeline HF (Biclase) | Tokenizador de `distilbert-base-uncased-finetuned-sst-2-english` | `text` (crudo) | Solo estadísticas; la inferencia se hace en la etapa de evaluación |
| Pipeline HF (Triclase) | Tokenizador de `cardiffnlp/twitter-roberta-base-sentiment-latest` | `text` (crudo) | Solo estadísticas; la inferencia se hace en la etapa de evaluación |

Las matrices se guardan en `<enfoque>/data/vectorized/` y los vectorizadores ajustados en `<enfoque>/models/vectorizer_*.joblib` (para reutilizarlos en el banco adversarial y la app web). El **tamaño de vocabulario de cada representación** queda documentado en:

* **Resumen Biclase:** [`biclase/reports/vectorization_summary.md`](biclase/reports/vectorization_summary.md)
* **Resumen Triclase:** [`triclase/reports/vectorization_summary.md`](triclase/reports/vectorization_summary.md)

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

## 🗓️ Cronograma de Hitos del Proyecto

* **28/09/2026:** Entrega parcial: Dataset extraído y preprocesado para análisis de sentimiento.
* **05/10/2026:** Word Embeddings y aporte de 2-3 oraciones al banco adversarial común del curso.
* **19/10/2026:** Entrega parcial: Dataset vectorizado (BoW, TF-IDF unigrama/bigrama, embeddings, Hugging Face).
* **16/11/2026:** Entrega parcial: Comparativa y evaluación de modelos sobre el test set adversarial.
* **A definir:** Entrega final (informe integrador, repositorio actualizado, app web y defensa técnica con Strata).

---

## 👥 Integrantes del Equipo

* Patricio Sebastián Castillo Upton
* Lautaro Martínez Serrano
* Juan Pablo Rébora
* Lautaro Rivadeneira
* Joaquin Mecle
* Germán Rodriguez Vaquero