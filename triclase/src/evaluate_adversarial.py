import argparse
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import pipeline

# Configuracion base del evaluador adversarial y rutas compartidas.
DEFAULT_APPROACH = "triclase"
ADVERSARIAL_PATH = Path("data/external/adversarial_test.csv")
GLOVE_PATH = Path("data/external/glove.6B.100d.txt")

# Modelos de Hugging Face asignados a cada enfoque del proyecto.
HF_MODELS = {
    "triclase": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "biclase": "distilbert-base-uncased-finetuned-sst-2-english",
}

# Banco adversarial de respaldo en memoria por si aun no existe el archivo CSV externo.
FALLBACK_ADVERSARIAL_DATA = [
    {"text": "This product is not bad at all, I actually like it.", "sentiment": 2},
    {"text": "I expected it to be terrible, but it works great.", "sentiment": 2},
    {"text": "Never thought a cheap cable could last this long.", "sentiment": 2},
    {"text": "Not the best design, yet it gets the job done reliably.", "sentiment": 2},
    {"text": "I could not be happier with this purchase.", "sentiment": 2},
    {"text": "The packaging was damaged, however the device itself is amazing.", "sentiment": 2},
    {"text": "It is okay, nothing special to write home about.", "sentiment": 1},
    {"text": "The item works as described, neither great nor terrible.", "sentiment": 1},
    {"text": "Average quality for the price, just acceptable.", "sentiment": 1},
    {"text": "It arrived on time and does what it is supposed to do.", "sentiment": 1},
    {"text": "Not terrible, not amazing either, just a standard charger.", "sentiment": 1},
    {"text": "I have mixed feelings about this keyboard after two weeks.", "sentiment": 1},
    {"text": "I wanted to love this headphones, but the sound is awful.", "sentiment": 0},
    {"text": "Not good at all, stopped working after two days.", "sentiment": 0},
    {"text": "Looks great in photos, unfortunately it feels super cheap.", "sentiment": 0},
    {"text": "I cannot recommend this to anyone, complete waste of money.", "sentiment": 0},
    {"text": "Hardly works and customer support never replied.", "sentiment": 0},
    {"text": "Great, another charger that breaks in less than a week.", "sentiment": 0},
]


def parse_arguments():
    # Lectura del argumento de linea de comandos para seleccionar triclase o biclase.
    parser = argparse.ArgumentParser(
        description="Evaluacion unificada de modelos clasicos y Transformers sobre banco adversarial."
    )
    parser.add_argument(
        "--enfoque",
        dest="approach",
        default=DEFAULT_APPROACH,
        choices=["triclase", "biclase"],
        help="Enfoque del experimento a evaluar (triclase o biclase).",
    )
    return parser.parse_args()


def load_preprocess_function(approach):
    # Carga dinamica de la funcion preprocess_text desde el modulo del enfoque activo.
    preprocess_file = Path(approach) / "src" / "preprocess.py"
    if not preprocess_file.exists():
        raise FileNotFoundError(f"No se encontro el modulo de preprocesamiento en {preprocess_file}.")

    spec = importlib.util.spec_from_file_location(f"{approach}_preprocess", preprocess_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.preprocess_text


def load_adversarial_dataset(approach):
    # Carga del CSV adversarial externo o uso del conjunto de respaldo en memoria.
    if ADVERSARIAL_PATH.exists():
        print(f"Cargando banco adversarial desde: {ADVERSARIAL_PATH}")
        df = pd.read_csv(ADVERSARIAL_PATH)
    else:
        print(
            f"[Aviso] No se encontro {ADVERSARIAL_PATH}. "
            "Usando banco adversarial de respaldo en memoria."
        )
        df = pd.DataFrame(FALLBACK_ADVERSARIAL_DATA)

    df["text"] = df["text"].fillna("").astype(str)
    df["sentiment"] = df["sentiment"].astype(int)

    # Adaptacion automatica de etiquetas si se ejecuta el enfoque binario sobre tres clases.
    if approach == "biclase" and df["sentiment"].nunique() > 2:
        df = df[df["sentiment"] != 1].copy()
        df["sentiment"] = df["sentiment"].replace({2: 1})
        df = df.reset_index(drop=True)

    return df


def load_glove_embeddings():
    # Carga de vectores preentrenados GloVe 100d solo si existen modelos densos para evaluar.
    if not GLOVE_PATH.exists():
        raise FileNotFoundError(f"No se encontro el archivo de embeddings GloVe en {GLOVE_PATH}.")

    words = []
    vectors = []
    with open(GLOVE_PATH, encoding="utf8") as file:
        for line in file:
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            vectors.append(np.asarray(parts[1:], dtype=np.float32))

    word_index = {word: idx for idx, word in enumerate(words)}
    return word_index, np.vstack(vectors)


def embed_texts_glove(cleaned_texts, word_index, matrix):
    # Calculo del vector promedio de palabras conocidas por cada oracion limpia.
    embeddings = np.zeros((len(cleaned_texts), matrix.shape[1]), dtype=np.float32)
    for row, text in enumerate(cleaned_texts):
        indexes = [word_index[token] for token in text.split() if token in word_index]
        if indexes:
            embeddings[row] = matrix[indexes].mean(axis=0)
    return embeddings


def parse_model_filename(model_path):
    # Extraccion del alias de algoritmo y representacion desde model_<algoritmo>_<repr>.joblib.
    stem = model_path.stem
    if not stem.startswith("model_"):
        return None, None
    remainder = stem[len("model_"):]
    if "_" not in remainder:
        return None, None
    algorithm, representation = remainder.split("_", 1)
    return algorithm, representation


LABEL_TO_ID = {
    "triclase": {"negative": 0, "label_0": 0, "neutral": 1, "label_1": 1, "positive": 2, "label_2": 2},
    "biclase": {"negative": 0, "label_0": 0, "positive": 1, "label_1": 1},
}


def map_hf_label(raw_label, approach):
    # Normalizacion y validacion de etiquetas de Hugging Face al esquema numerico del proyecto.
    key = str(raw_label).strip().lower()
    mapping = LABEL_TO_ID[approach]
    if key not in mapping:
        raise ValueError(f"Etiqueta inesperada de Hugging Face para {approach}: {raw_label}")
    return mapping[key]


def predict_huggingface(raw_texts, approach):
    # Ejecucion de inferencia en CPU con el pipeline de Hugging Face sobre texto crudo.
    hf_model_name = HF_MODELS[approach]
    print(f"Ejecutando inferencia Hugging Face en CPU ({hf_model_name})...")
    classifier = pipeline(
        "sentiment-analysis",
        model=hf_model_name,
        tokenizer=hf_model_name,
        device=-1,
        truncation=True,
        max_length=512,
    )
    outputs = classifier(raw_texts, batch_size=16)
    predictions = [map_hf_label(item["label"], approach) for item in outputs]
    return np.array(predictions, dtype=int)


def compute_metrics(y_true, y_pred):
    # Calculo de metricas globales y macro sobre el conjunto adversarial.
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
    }


def main():
    args = parse_arguments()
    approach = args.approach

    # Definicion de directorios de modelos y reportes del enfoque seleccionado.
    base_dir = Path(approach)
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    results_path = reports_dir / "adversarial_results.csv"
    predictions_path = reports_dir / "adversarial_predictions.csv"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f">> EVALUACION ADVERSARIAL UNIFICADA (Enfoque: {approach})")
    print("=" * 75)

    preprocess_text = load_preprocess_function(approach)
    adv_df = load_adversarial_dataset(approach)
    raw_texts = adv_df["text"].tolist()
    y_true = adv_df["sentiment"].to_numpy()
    cleaned_texts = [preprocess_text(text) for text in raw_texts]

    print(f"Oraciones adversariales cargadas: {len(adv_df)}")
    print("Distribucion de clases reales:")
    print(adv_df["sentiment"].value_counts().sort_index().to_string())

    # Descubrimiento dinamico de todos los modelos serializados en la carpeta models.
    model_files = sorted(models_dir.glob("model_*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No se encontraron modelos entrenados en {models_dir}.")

    vectorizer_cache = {}
    glove_cache = None
    summary_records = []
    predictions_df = adv_df[["text", "sentiment"]].copy()

    print(f"\nEvaluando {len(model_files)} modelos clasicos encontrados en {models_dir}...")
    for model_path in model_files:
        algorithm, representation = parse_model_filename(model_path)
        if algorithm is None:
            continue

        model = joblib.load(model_path)

        # Transformacion de texto segun el tipo de representacion del modelo.
        if representation == "glove":
            if glove_cache is None:
                print("Cargando vectores GloVe para evaluar modelos densos...")
                glove_cache = load_glove_embeddings()
            word_index, matrix = glove_cache
            x_adv = embed_texts_glove(cleaned_texts, word_index, matrix)
        else:
            if representation not in vectorizer_cache:
                vec_path = models_dir / f"vectorizer_{representation}.joblib"
                if not vec_path.exists():
                    raise FileNotFoundError(f"No se encontro el vectorizador {vec_path}.")
                vectorizer_cache[representation] = joblib.load(vec_path)
            x_adv = vectorizer_cache[representation].transform(cleaned_texts)

        y_pred = model.predict(x_adv)
        scores = compute_metrics(y_true, y_pred)

        summary_records.append(
            {
                "model": algorithm,
                "representation": representation,
                **scores,
            }
        )
        predictions_df[f"{algorithm}_{representation}"] = y_pred

    # Evaluacion del modelo preentrenado de Hugging Face sobre las oraciones crudas.
    try:
        hf_model_name = HF_MODELS[approach]
        y_pred_hf = predict_huggingface(raw_texts, approach)
        hf_scores = compute_metrics(y_true, y_pred_hf)
        summary_records.append(
            {
                "model": hf_model_name,
                "representation": "hf_pipeline",
                **hf_scores,
            }
        )
        predictions_df[f"{hf_model_name.split('/')[-1]}_hf_pipeline"] = y_pred_hf
    except Exception as err:
        print(f"[Aviso] No se pudo ejecutar Hugging Face en esta corrida ({err}).")

    # Ordenamiento del ranking adversarial por F1-Macro descendente.
    summary_df = (
        pd.DataFrame(summary_records)
        .sort_values(by="f1_macro", ascending=False)
        .reset_index(drop=True)
    )
    summary_df.to_csv(results_path, index=False)

    # Calculo de errores por oracion para facilitar el analisis cualitativo de casos dificiles.
    model_columns = [col for col in predictions_df.columns if col not in ("text", "sentiment")]
    error_matrix = predictions_df[model_columns].ne(predictions_df["sentiment"], axis=0)
    predictions_df["models_failed"] = error_matrix.sum(axis=1)
    predictions_df["total_models"] = len(model_columns)
    predictions_df.to_csv(predictions_path, index=False)

    print("\n" + "=" * 75)
    print(f">> RANKING ADVERSARIAL POR F1-MACRO ({approach.upper()})")
    print("=" * 75)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 75)
    print(">> DETALLE ORACION POR ORACION (ORDENADO POR CANTIDAD DE ERRORES)")
    print("=" * 75)
    detailed_view = predictions_df.sort_values(by="models_failed", ascending=False)
    for _, row in detailed_view.iterrows():
        print(
            f"[Real: {row['sentiment']} | Fallaron: {row['models_failed']}/{row['total_models']}] "
            f"{row['text']}"
        )

    print(f"\nTabla resumen guardada en: {results_path}")
    print(f"Predicciones detalladas guardadas en: {predictions_path}")


# Condicional estandar de Python para ejecutar el script de forma directa.
if __name__ == "__main__":
    main()