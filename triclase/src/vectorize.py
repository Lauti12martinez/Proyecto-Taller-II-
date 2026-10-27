import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer

# Rutas del proyecto.
PROCESSED_PATH = Path("data/processed/amazon_reviews_clean.csv")
SPLIT_PATH = Path("data/processed/train_test_split.csv")
VECTORIZED_DIR = Path("data/vectorized")
EXTERNAL_DIR = Path("data/external")
MODELS_DIR = Path("models")
REPORT_PATH = Path("reports/vectorization_summary.md")

# Particion train/test. Los vectorizadores se ajustan SOLO con train para no filtrar info del test.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Frecuencia minima de documentos para que un termino entre al vocabulario (descarta ruido y typos).
MIN_DF = 5

# Configuracion de las representaciones dispersas: nombre -> vectorizador.
SPARSE_VECTORIZERS = {
    "bow": lambda: CountVectorizer(min_df=MIN_DF),
    "tfidf_unigram": lambda: TfidfVectorizer(min_df=MIN_DF, ngram_range=(1, 1)),
    "tfidf_bigram": lambda: TfidfVectorizer(min_df=MIN_DF, ngram_range=(1, 2)),
}

# Vectores pre-entrenados GloVe (Stanford, Wikipedia + Gigaword, 400k palabras).
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_DIM = 100
GLOVE_FILE = EXTERNAL_DIR / f"glove.6B.{GLOVE_DIM}d.txt"

# Modelo de Hugging Face pre-entrenado para sentimiento (se usa su tokenizador en esta etapa).
HF_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
HF_MAX_LENGTH = 512


def load_dataset():
    df = pd.read_csv(PROCESSED_PATH)
    # Resenas cuyo texto limpio quedo vacio o se leyo como nulo no aportan informacion.
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    return df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)


def split_dataset(df):
    # Particion estratificada para mantener la proporcion de clases en train y test.
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["sentiment"], random_state=RANDOM_STATE
    )
    # Se guardan los ids de cada particion para que todos los modelos usen exactamente la misma.
    split = pd.concat([
        pd.DataFrame({"id": train_df["id"], "split": "train"}),
        pd.DataFrame({"id": test_df["id"], "split": "test"}),
    ])
    split.to_csv(SPLIT_PATH, index=False)
    np.save(VECTORIZED_DIR / "y_train.npy", train_df["sentiment"].to_numpy())
    np.save(VECTORIZED_DIR / "y_test.npy", test_df["sentiment"].to_numpy())
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def vectorize_sparse(name, train_texts, test_texts):
    vectorizer = SPARSE_VECTORIZERS[name]()
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)

    sparse.save_npz(VECTORIZED_DIR / f"X_train_{name}.npz", x_train)
    sparse.save_npz(VECTORIZED_DIR / f"X_test_{name}.npz", x_test)
    # Se guarda el vectorizador ajustado para reutilizarlo en inferencia (app web y banco adversarial).
    joblib.dump(vectorizer, MODELS_DIR / f"vectorizer_{name}.joblib")

    density = x_train.nnz / (x_train.shape[0] * x_train.shape[1])
    return {
        "representation": name,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "dimensions": x_train.shape[1],
        "train_shape": x_train.shape,
        "test_shape": x_test.shape,
        "notes": f"min_df={MIN_DF}, densidad={density:.5%}",
    }


def download_glove():
    if GLOVE_FILE.exists():
        return
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = EXTERNAL_DIR / "glove.6B.zip"

    # Descarga por partes con barra de progreso (el zip pesa ~820 MB).
    print("Descargando vectores GloVe de Stanford (solo la primera vez)...")
    with requests.get(GLOVE_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(zip_path, "wb") as file, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                file.write(chunk)
                bar.update(len(chunk))

    # Del zip solo se extrae la dimension usada y se borra el resto para ahorrar espacio.
    with zipfile.ZipFile(zip_path) as archive:
        archive.extract(GLOVE_FILE.name, EXTERNAL_DIR)
    zip_path.unlink()


def load_glove():
    words = []
    vectors = []
    with open(GLOVE_FILE, encoding="utf8") as file:
        for line in tqdm(file, desc="Cargando GloVe", unit=" palabras"):
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            vectors.append(np.asarray(parts[1:], dtype=np.float32))
    word_index = {word: i for i, word in enumerate(words)}
    return word_index, np.vstack(vectors)


def embed_texts(texts, word_index, matrix):
    # Cada resena se representa como el promedio de los vectores de sus palabras conocidas.
    # Las resenas sin ninguna palabra conocida quedan como vector de ceros.
    result = np.zeros((len(texts), matrix.shape[1]), dtype=np.float32)
    for row, text in enumerate(tqdm(texts, desc="Promediando embeddings")):
        indexes = [word_index[w] for w in text.split() if w in word_index]
        if indexes:
            result[row] = matrix[indexes].mean(axis=0)
    return result


def vectorize_glove(train_texts, test_texts):
    download_glove()
    word_index, matrix = load_glove()

    x_train = embed_texts(train_texts, word_index, matrix)
    x_test = embed_texts(test_texts, word_index, matrix)
    np.save(VECTORIZED_DIR / "X_train_glove.npy", x_train)
    np.save(VECTORIZED_DIR / "X_test_glove.npy", x_test)

    # Cobertura: que parte del vocabulario del corpus (train) tiene vector pre-entrenado.
    corpus_vocab = {w for text in train_texts for w in text.split()}
    covered = sum(1 for w in corpus_vocab if w in word_index)
    empty_docs = int((~x_train.any(axis=1)).sum())
    return {
        "representation": "glove",
        "vocabulary_size": len(word_index),
        "dimensions": GLOVE_DIM,
        "train_shape": x_train.shape,
        "test_shape": x_test.shape,
        "notes": (
            f"cobertura del vocabulario del corpus: {covered:,}/{len(corpus_vocab):,} "
            f"({covered / len(corpus_vocab):.1%}); reseñas sin palabras conocidas: {empty_docs}"
        ),
    }


def describe_hf_tokenizer(train_texts):
    # El pipeline de HF trabaja con texto crudo: tiene su propio tokenizador (WordPiece) y vocabulario.
    # En esta etapa solo se tokeniza para medir el vocabulario y cuanto texto se trunca;
    # la inferencia con el modelo completo se hace en la etapa de evaluacion.
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    lengths = []
    batch_size = 1000
    for start in tqdm(range(0, len(train_texts), batch_size), desc="Tokenizando (HF)"):
        batch = list(train_texts[start:start + batch_size])
        lengths.extend(len(ids) for ids in tokenizer(batch)["input_ids"])
    lengths = np.array(lengths)
    return {
        "representation": "hf_pipeline",
        "vocabulary_size": tokenizer.vocab_size,
        "dimensions": 768,
        "train_shape": (len(train_texts), HF_MAX_LENGTH),
        "test_shape": "-",
        "notes": (
            f"{HF_MODEL}; tokens por reseña: media {lengths.mean():.0f}, "
            f"max {lengths.max()}; truncadas (>{HF_MAX_LENGTH}): {(lengths > HF_MAX_LENGTH).mean():.2%}"
        ),
    }


def write_report(rows, n_train, n_test):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Resumen de vectorizacion",
        "",
        f"Generado por `src/vectorize.py`. Particion estratificada: {n_train:,} reseñas de train "
        f"y {n_test:,} de test (test_size={TEST_SIZE}, random_state={RANDOM_STATE}).",
        "",
        "| Representacion | Tamaño de vocabulario | Dimensiones | Train | Test | Notas |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['representation']} | {row['vocabulary_size']:,} | {row['dimensions']:,} "
            f"| {row['train_shape']} | {row['test_shape']} | {row['notes']} |"
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    VECTORIZED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando dataset preprocesado...")
    df = load_dataset()
    train_df, test_df = split_dataset(df)
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    rows = []
    for name in SPARSE_VECTORIZERS:
        print(f"\nVectorizando con {name}...")
        rows.append(vectorize_sparse(name, train_df["cleaned_text"], test_df["cleaned_text"]))

    print("\nVectorizando con embeddings GloVe...")
    rows.append(vectorize_glove(train_df["cleaned_text"].tolist(), test_df["cleaned_text"].tolist()))

    print("\nAnalizando tokenizador del pipeline de Hugging Face...")
    rows.append(describe_hf_tokenizer(train_df["text"].tolist()))

    write_report(rows, len(train_df), len(test_df))

    print("\n" + "=" * 50)
    print(">> VECTORIZACION COMPLETADA")
    print("=" * 50)
    for row in rows:
        print(f"{row['representation']:<15} vocabulario: {row['vocabulary_size']:>9,}  dims: {row['dimensions']:>9,}")
    print(f"\nResumen guardado en: {REPORT_PATH}")


# Condicional estandar de Python: asegura que main() solo se ejecute si corremos el script directamente.
if __name__ == "__main__":
    main()
