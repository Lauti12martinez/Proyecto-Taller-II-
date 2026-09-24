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
│   └── processed/           # Dataset limpio y preprocesado para modelado
├── notebooks/               # Análisis exploratorio (EDA) y experimentación
├── src/
│   ├── extract_reviews.py   # Ingesta masiva desde la API de Amazon Reviews
│   └── preprocess.py        # Pipeline de limpieza, tokenización y normalización
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
> El archivo `data/raw/amazon_reviews_raw.csv` supera los 100 MB y **no debe subirse a GitHub**. Ya está configurada la regla en el `.gitignore` para omitir `data/raw/*.csv` y `*.csv`. Cada integrante debe generar su copia local corriendo el script de extracción o solicitando el CSV por el almacenamiento compartido del grupo.

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
