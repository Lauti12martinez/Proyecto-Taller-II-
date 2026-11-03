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

# Rutas del proyecto para el experimento triclase.
# EXTERNAL_DIR apunta a la raiz para compartir los vectores GloVe sin descargarlos dos veces.
PROCESSED_PATH = Path("triclase/data/processed/amazon_reviews_clean.csv")
SPLIT_PATH = Path("triclase/data/processed/train_test_split.csv")
VECTORIZED_DIR = Path("triclase/data/vectorized")
EXTERNAL_DIR = Path("data/external")
MODELS_DIR = Path("triclase/models")
REPORT_PATH = Path("triclase/reports/vectorization_summary.md")

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
# Se usa el espejo oficial en Hugging Face para evitar caidas y timeouts del servidor de Stanford.
GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_DIM = 100
GLOVE_FILE = EXTERNAL_DIR / f"glove.6B.{GLOVE_DIM}d.txt"

# Modelo de Hugging Face pre-entrenado especificamente para 3 clases (0: Negativo, 1: Neutral, 2: Positivo).
# En biclase se usaba SST-2 (estrictamente binario); este modelo RoBERTa incluye la clase neutral.
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
HF_MAX_LENGTH = 512


def load_dataset():
    df = pd.read_csv(PROCESSED_PATH)
    # Resenas cuyo texto limpio quedo vacio o se leyo como nulo no aportan informacion.
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    return df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)


def split_dataset(df):
    # Particion estratificada para mantener la proporcion de las 3 clases (0, 1 y 2) en train y test.
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["sentiment"], random_state=RANDOM_STATE
    )
    # Se guardan los ids de cada particion para que todos los modelos triclase usen exactamente la misma.
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

    # Descarga con soporte de reanudacion automatica (Range HTTP) y reintentos ante microcortes.
    print("Descargando vectores GloVe (reanudable si se corta la red)...")
    while True:
        downloaded = zip_path.stat().st_size if zip_path.exists() else 0
        headers = {"Range": f"bytes={downloaded}-"} if downloaded > 0 else {}

        try:
            with requests.get(GLOVE_URL, headers=headers, stream=True, timeout=60) as response:
                # 416 significa que el archivo ya se habia terminado de descargar por completo.
                if response.status_code == 416:
                    break
                response.raise_for_status()

                # Si el servidor acepta Range (206), continuamos agregando ("ab"); si no, empezamos de cero ("wb").
                mode = "ab" if response.status_code == 206 else "wb"
                if mode == "wb":
                    downloaded = 0

                total = int(response.headers.get("content-length", 0)) + downloaded
                with open(zip_path, mode) as file, tqdm(
                    total=total, initial=downloaded, unit="B", unit_scale=True, desc="GloVe zip"
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            file.write(chunk)
                            bar.update(len(chunk))
            break  # Salimos del bucle cuando la descarga llega al 100%.

        except (requests.exceptions.RequestException, TimeoutError) as err:
            print(f"\n[Aviso] Microcorte de red ({err}). Reanudando desde donde quedo...")

    # Del zip solo se extrae la dimension usada (100d) y se borra el zip para ahorrar espacio.
    print("Extrayendo glove.6B.100d.txt...")
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

    # Cobertura: que parte del vocabulario del corpus triclase (train) tiene vector pre-entrenado.
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
    # El pipeline de HF trabaja con texto crudo: tiene su propio tokenizador y vocabulario.
    # En esta etapa solo se tokeniza para medir el vocabulario y cuanto texto se trunca;
    # la inferencia con el modelo de 3 clases completo se hace en la etapa de evaluacion.
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    lengths = []
    batch_size = 1000
    for start in tqdm(range(0, len(train_texts), batch_size), desc="Tokenizando (HF Triclase)"):
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
        "# Resumen de vectorizacion (Triclase)",
        "",
        f"Generado por `triclase/src/vectorize.py`. Particion estratificada en 3 clases: {n_train:,} reseñas de train "
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

    print("Cargando dataset preprocesado (Triclase)...")
    df = load_dataset()
    train_df, test_df = split_dataset(df)
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    rows = []
    for name in SPARSE_VECTORIZERS:
        print(f"\nVectorizando con {name}...")
        rows.append(vectorize_sparse(name, train_df["cleaned_text"], test_df["cleaned_text"]))

    print("\nVectorizando con embeddings GloVe...")
    rows.append(vectorize_glove(train_df["cleaned_text"].tolist(), test_df["cleaned_text"].tolist()))

    print("\nAnalizando tokenizador del pipeline de Hugging Face (3 clases)...")
    rows.append(describe_hf_tokenizer(train_df["text"].tolist()))

    write_report(rows, len(train_df), len(test_df))

    print("\n" + "=" * 50)
    print(">> VECTORIZACION TRICLASE COMPLETADA")
    print("=" * 50)
    for row in rows:
        print(f"{row['representation']:<15} vocabulario: {row['vocabulary_size']:>9,}  dims: {row['dimensions']:>9,}")
    print(f"\nResumen guardado en: {REPORT_PATH}")


# Condicional estandar de Python: asegura que main() solo se ejecute si corremos el script directamente.
if __name__ == "__main__":
    main()