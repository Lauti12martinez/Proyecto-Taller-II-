import re
import string
from pathlib import Path
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Estructura general del script basada en el que arme para el 1er cuatri.

# Descarga silenciosa de paquetes necesarios para que no falle en entornos limpios.
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Rutas del proyecto.
RAW_PATH = Path("data/raw/amazon_reviews_raw.csv")
PROCESSED_PATH = Path("data/processed/amazon_reviews_clean.csv")

# Tabla para remover signos de puntuacion extendidos.
CUSTOM_PUNCTUATION = string.punctuation + '¿¡“”‘’«»—–…！？：，（）'
REMOVE_PUNCT_TABLE = str.maketrans("", "", CUSTOM_PUNCTUATION)

# Lista de palabras vacias en ingles.
STOP_WORDS = set(stopwords.words("english"))


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

    # Supresion de puntuacion mediante la tabla de caracteres.
    text = text.translate(REMOVE_PUNCT_TABLE)

    # Tokenizacion por palabras.
    tokens = word_tokenize(text)

    # Filtrado de palabras vacias y tokens de longitud uno.
    clean_tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]

    # Retorno del texto procesado en una sola cadena.
    return " ".join(clean_tokens)


def main():
    print("Iniciando pipeline de preprocesamiento...")
    df = pd.read_csv(RAW_PATH)

    # Descarte de reseñas neutras con etiqueta dos.
    df = df[df["label"] != 2].copy()

    # Binarizacion de etiquetas asignando uno a positivas y cero a negativas.
    df["sentiment"] = df["label"].apply(lambda x: 1 if x >= 3 else 0)

    # Aplicacion de limpieza con barra de progreso.
    tqdm.pandas(desc="Limpiando texto")
    df["cleaned_text"] = df["text"].progress_apply(preprocess_text)

    # Eliminacion de filas que hayan quedado sin texto util.
    df = df[df["cleaned_text"].str.strip() != ""].copy()

    # Creacion de la carpeta de destino y guardado en csv.
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Dataset procesado guardado en: {PROCESSED_PATH}")

# Condicional estandar de Python: asegura que main() solo se ejecute si corremos el script directamente.
if __name__ == "__main__":
    main()