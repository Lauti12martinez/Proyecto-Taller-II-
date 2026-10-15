# Desafío Taller II - Modelo de Análisis de Sentimiento

**Materia:** Taller II — Ingeniería en Inteligencia Artificial (UNSTA)  
**Proyecto:** Desafío Taller II - Modelo de Análisis de Sentimiento (Strata)

---

## 📌 Descripción del Proyecto

El objetivo del proyecto consiste en construir un pipeline integral de Machine Learning para análisis de sentimiento: desde la obtención y preprocesamiento de un corpus de reseñas, pasando por diversas representaciones numéricas del texto (BoW, TF-IDF unigrama/bigrama, embeddings pre-entrenados y pipelines de Hugging Face), hasta el entrenamiento, optimización, evaluación sobre un banco adversarial curado por el curso y el despliegue de una aplicación web básica para inferencia ante el cliente (Strata).

---

## 📁 Estructura del Repositorio

```text
Proyecto-Taller-II-/
├── data/
│   ├── raw/                 # Dataset crudo descargado de la API (ignorado en Git)
│   ├── processed/           # Dataset limpio y partición train/test (ignorado en Git)
│   ├── vectorized/          # Matrices vectorizadas y etiquetas (ignorado en Git)
│   └── external/            # Vectores pre-entrenados GloVe descargados (ignorado en Git)
├── models/                  # Vectorizadores y modelos serializados (ignorado en Git)
├── notebooks/               # Análisis exploratorio (EDA) y experimentación
├── reports/
│   └── vectorization_summary.md  # Tamaño de vocabulario y dimensiones de cada representación
├── src/
│   ├── extract_reviews.py   # Ingesta masiva desde la API de Amazon Reviews
│   ├── preprocess.py        # Pipeline de limpieza, tokenización y normalización
│   └── vectorize.py         # BoW, TF-IDF uni/bigrama, embeddings GloVe y tokenizador HF
├── .gitignore               # Exclusión de archivos pesados y entornos locales
├── pyproject.toml           # Especificación de dependencias y configuración Poetry
├── poetry.lock              # Registro determinista de versiones
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

Asegurarse de contar con Poetry instalado localmente y ejecutá en la raíz del proyecto:

```bash
poetry install
```

---

## 📥 Ingesta de Datos (API de Amazon Reviews)

El corpus se descarga de forma automatizada consultando la API provista por la cátedra (`https://amazon-reviews-api-g5ae.onrender.com`).

Para ejecutar el proceso de descarga masiva:

```bash
poetry run python src/extract_reviews.py
```

### Especificaciones del Dataset Crudo

* **Total de registros:** 210.000 reseñas.
* **Archivo de salida:** `data/raw/amazon_reviews_raw.csv`.
* **Distribución de clases:** Completamente uniforme y balanceada (42.000 reseñas por cada una de las 5 calificaciones, con etiquetas del `0` al `4`).
* **Esquema devuelto:** `id`, `text`, `label`, `label_text`.
* **Resiliencia:** El script incluye control de *Cold Start* en Render, reintentos automáticos ante cortes de red y paginación masiva por lotes de 1.000 registros con barra de progreso interactiva (`tqdm`).

> ⚠️ **Importante sobre el control de versiones (.gitignore):**  
> El archivo `data/raw/amazon_reviews_raw.csv` supera los 100 MB y **no debe subirse a GitHub**. Ya está configurada la regla en el `.gitignore` para omitir `*.csv`. Cada integrante debe generar su copia local corriendo el script de extracción o solicitando el CSV por el almacenamiento compartido del grupo.

---

## 🧹 Preprocesamiento

```bash
poetry run python src/preprocess.py
```

Lee `data/raw/amazon_reviews_raw.csv` y genera `data/processed/amazon_reviews_clean.csv` (~168.000 reseñas):

* **Etiquetas:** se descartan las reseñas neutras (`label == 2`) y se binariza en `sentiment` (0 = negativa: labels 0-1; 1 = positiva: labels 3-4). Clases balanceadas.
* **Limpieza de `text` → `cleaned_text`:** minúsculas, sin HTML, sin dígitos, sin puntuación, tokenización (NLTK), sin stopwords ni tokens de una letra, lematización (WordNet).
* **Negaciones:** `not`, `no` y `nor` **se conservan** (se quitaron de las stopwords) y las contracciones negativas se expanden antes de quitar la puntuación (`don't` → `do not`, `can't` → `can not`). Sin esto, "not good" quedaba como "good", lo que invierte el sentimiento y rompe el banco adversarial.
* La columna `text` original se mantiene: los embeddings contextuales (Hugging Face) trabajan sobre el texto crudo.

---

## 🔢 Vectorización

```bash
poetry run python src/vectorize.py
```

Requiere `data/processed/amazon_reviews_clean.csv`. La primera ejecución descarga los vectores GloVe de Stanford (~820 MB, se extrae solo el archivo de 100 dimensiones a `data/external/`) y el tokenizador del modelo de Hugging Face.

1. **Partición train/test** estratificada 80/20 (`random_state=42`). Los ids quedan en `data/processed/train_test_split.csv` y las etiquetas en `data/vectorized/y_train.npy` / `y_test.npy`, para que **todos los modelos usen la misma partición**.
2. **Los vectorizadores se ajustan solo con train** (para no filtrar información del test) y se aplican a ambos conjuntos.

| Representación | Implementación | Entrada | Salida |
|---|---|---|---|
| BoW | `CountVectorizer(min_df=5)` | `cleaned_text` | `X_{train,test}_bow.npz` (dispersa) |
| TF-IDF unigrama | `TfidfVectorizer(min_df=5, ngram_range=(1,1))` | `cleaned_text` | `X_{train,test}_tfidf_unigram.npz` |
| TF-IDF bigrama | `TfidfVectorizer(min_df=5, ngram_range=(1,2))` | `cleaned_text` | `X_{train,test}_tfidf_bigram.npz` |
| Embeddings | GloVe 6B 100d, promedio de los vectores de las palabras de cada reseña | `cleaned_text` | `X_{train,test}_glove.npy` (densa, 100 dims) |
| Pipeline HF | Tokenizador de `distilbert-base-uncased-finetuned-sst-2-english` | `text` (crudo) | Solo estadísticas; la inferencia se hace en la etapa de evaluación |

Las matrices se guardan en `data/vectorized/` y los vectorizadores ajustados en `models/vectorizer_*.joblib` (para reutilizarlos en el banco adversarial y la app web). El **tamaño de vocabulario de cada representación** queda en [`reports/vectorization_summary.md`](reports/vectorization_summary.md).

Para cargar una representación desde un notebook o script de modelado:

```python
import numpy as np
from scipy import sparse

X_train = sparse.load_npz("data/vectorized/X_train_tfidf_bigram.npz")
X_test = sparse.load_npz("data/vectorized/X_test_tfidf_bigram.npz")
y_train = np.load("data/vectorized/y_train.npy")
y_test = np.load("data/vectorized/y_test.npy")
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
