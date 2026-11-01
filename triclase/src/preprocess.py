import re
import string
from pathlib import Path
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Descarga silenciosa de paquetes necesarios para que no falle en entornos limpios.
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)  # Requerido por word_tokenize en NLTK >= 3.9
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)  # Agregado para lematizar

# Rutas del proyecto:
# RAW_PATH lee el dataset crudo compartido en la raiz del repositorio.
# PROCESSED_PATH guarda el dataset limpio dentro de la carpeta madre triclase/.
RAW_PATH = Path("data/raw/amazon_reviews_raw.csv")
PROCESSED_PATH = Path("triclase/data/processed/amazon_reviews_clean.csv")

# Tabla para remover signos de puntuacion extendidos.
CUSTOM_PUNCTUATION = string.punctuation + '¿¡“”‘’«»—–…！？：，（）'
REMOVE_PUNCT_TABLE = str.maketrans("", "", CUSTOM_PUNCTUATION)

# Negaciones que NO se descartan: invierten el sentimiento ("not good" != "good").
NEGATION_WORDS = {"no", "not", "nor"}

# Lista de palabras vacias en ingles (sin las negaciones) y Lematizador.
STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS
LEMMATIZER = WordNetLemmatizer()

# Contracciones negativas que se expanden antes de quitar la puntuacion.
# Ej: "don't" -> "do not", "can't" -> "can not", "won't" -> "will not".
CONTRACTION_PATTERNS = [
    (re.compile(r"\bwon['’]t\b"), "will not"),
    (re.compile(r"\bcan['’]t\b"), "can not"),
    (re.compile(r"n['’]t\b"), " not"),
]


def expand_negations(text):
    for pattern, replacement in CONTRACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def map_sentiment(label):
    """
    Convierte la calificacion original de la API (0 a 4) en 3 categorias de sentimiento:
    - 0 (Negativo): reseñas con label 0 o 1.
    - 1 (Neutral):  reseñas con label 2 (antes descartadas en el enfoque biclase).
    - 2 (Positivo): reseñas con label 3 o 4.
    """
    if label <= 1:
        return 0   
    if label == 2:
        return 1
    return 2


def preprocess_text(text):
    # Validacion de tipo para evitar errores con nulos.
    if not isinstance(text, str):
        return ""

    # Conversion a minusculas.
    text = text.lower()

    # Eliminacion de etiquetas html para no dejar palabras sueltas como br.
    text = re.sub(r"<.*?>", " ", text)

    # Eliminacion de digitos numericos.
    text = re.sub(r"\d+", " ", text)

    # Expansion de contracciones negativas para no perder la negacion al quitar el apostrofo.
    text = expand_negations(text)

    # Supresion de puntuacion mediante la tabla de caracteres.
    text = text.translate(REMOVE_PUNCT_TABLE)

    # Tokenizacion por palabras.
    tokens = word_tokenize(text)

    # Filtrado de palabras vacias, tokens de longitud uno y Lematizacion (raiz de la palabra).
    clean_tokens = [
        LEMMATIZER.lemmatize(w)
        for w in tokens
        if w not in STOP_WORDS and len(w) > 1
    ]

    # Retorno del texto procesado en una sola cadena.
    return " ".join(clean_tokens)


def main():
    print("Iniciando pipeline de preprocesamiento (Enfoque Triclase)...")
    df = pd.read_csv(RAW_PATH)

    # En el enfoque triclase NO se descartan las reseñas neutras (label == 2).
    # Se recodifican las 5 etiquetas originales en 3 clases (0: Negativo, 1: Neutral, 2: Positivo).
    df["sentiment"] = df["label"].apply(map_sentiment)

    # Aplicacion de limpieza con barra de progreso.
    tqdm.pandas(desc="Limpiando texto (Triclase)")
    df["cleaned_text"] = df["text"].progress_apply(preprocess_text)

    # Eliminacion de filas que hayan quedado sin texto util tras la limpieza.
    df = df[df["cleaned_text"].str.strip() != ""].copy()

    # Creacion de la carpeta de destino y guardado en CSV.
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("\n" + "=" * 50)
    print(">> PREPROCESAMIENTO TRICLASE COMPLETADO")
    print("=" * 50)
    print(f"Total de reseñas útiles guardadas: {len(df):,}")
    print("\nDistribución de clases (sentiment):")
    print(df["sentiment"].value_counts().sort_index())
    print(f"\nDataset procesado guardado en: {PROCESSED_PATH}")


# Condicional estandar de Python: asegura que main() solo se ejecute si corremos el script directamente.
if __name__ == "__main__":
    main()