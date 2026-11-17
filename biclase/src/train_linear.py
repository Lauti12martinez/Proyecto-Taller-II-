import argparse
import gc
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Configuracion base por defecto para el experimento y semilla de reproducibilidad.
DEFAULT_APPROACH = "biclase"
RANDOM_STATE = 42
CV_FOLDS = 5

# Representaciones dispersas generadas en la etapa de vectorizacion.
SPARSE_REPRESENTATIONS = [
    "bow",
    "tfidf_unigram",
    "tfidf_bigram",
]


def parse_arguments():
    # Lectura del argumento de linea de comandos para alternar entre triclase y biclase.
    parser = argparse.ArgumentParser(
        description="Entrenamiento de clasificadores lineales y probabilisticos sobre matrices dispersas."
    )
    parser.add_argument(
        "--enfoque",
        dest="approach",
        default=DEFAULT_APPROACH,
        choices=["triclase", "biclase"],
        help="Enfoque del experimento a ejecutar (triclase o biclase).",
    )
    return parser.parse_args()


def build_models():
    # Diccionario de constructores de modelos rapidos en CPU con alias corto estandarizado.
    return {
        "nb": lambda: MultinomialNB(),
        "logreg": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "linearsvc": lambda: LinearSVC(random_state=RANDOM_STATE),
        "sgd": lambda: SGDClassifier(loss="log_loss", max_iter=1000, random_state=RANDOM_STATE),
    }


def load_labels(vectorized_dir):
    # Carga de vectores de etiquetas de entrenamiento y prueba previamente particionados.
    y_train_path = vectorized_dir / "y_train.npy"
    y_test_path = vectorized_dir / "y_test.npy"
    if not y_train_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            f"No se encontraron las etiquetas en {vectorized_dir}. Ejecuta vectorize.py primero."
        )
    return np.load(y_train_path), np.load(y_test_path)


def load_sparse_matrices(vectorized_dir, representation):
    # Carga de matrices dispersas en formato CSR para una representacion especifica.
    x_train_path = vectorized_dir / f"X_train_{representation}.npz"
    x_test_path = vectorized_dir / f"X_test_{representation}.npz"
    if not x_train_path.exists() or not x_test_path.exists():
        raise FileNotFoundError(
            f"No se encontraron las matrices para {representation} en {vectorized_dir}."
        )
    x_train = sparse.load_npz(x_train_path)
    x_test = sparse.load_npz(x_test_path)
    return x_train, x_test


def train_and_evaluate(model, x_train, y_train, x_test, y_test, cv_splitter):
    # Validacion cruzada estratificada usando exclusivamente el conjunto de entrenamiento.
    cv_scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=cv_splitter,
        scoring="f1_macro",
        n_jobs=-1,
    )

    # Ajuste final del clasificador sobre todo el conjunto de entrenamiento.
    model.fit(x_train, y_train)

    # Prediccion y calculo de metricas sobre el conjunto de prueba reservado.
    y_pred = model.predict(x_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "cv_f1_mean": float(np.mean(cv_scores)),
        "cv_f1_std": float(np.std(cv_scores)),
    }


def main():
    args = parse_arguments()
    approach = args.approach

    # Definicion dinamica de rutas segun el enfoque seleccionado.
    base_dir = Path(approach)
    vectorized_dir = base_dir / "data" / "vectorized"
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    metrics_path = reports_dir / "metrics_linear.csv"

    # Creacion de directorios de salida en caso de que no existan.
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f">> INICIANDO ENTRENAMIENTO DE MODELOS DISPERSOS (Enfoque: {approach})")
    print("=" * 70)

    y_train, y_test = load_labels(vectorized_dir)
    print(f"Etiquetas cargadas -> Train: {len(y_train):,} | Test: {len(y_test):,}")

    # Configuracion del particionador estratificado para las cinco particiones de validacion.
    cv_splitter = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model_builders = build_models()
    metrics_records = []

    for representation in SPARSE_REPRESENTATIONS:
        print(f"\n--- Representacion: {representation} ---")
        x_train, x_test = load_sparse_matrices(vectorized_dir, representation)
        print(f"Dimensiones -> Train: {x_train.shape} | Test: {x_test.shape}")

        for model_alias, builder in model_builders.items():
            start_time = time.perf_counter()
            print(f"Entrenando y validando [{model_alias}] con [{representation}]...")

            model = builder()
            scores = train_and_evaluate(
                model,
                x_train,
                y_train,
                x_test,
                y_test,
                cv_splitter,
            )

            # Serializacion del modelo entrenado respetando el contrato de nombres del equipo.
            model_filename = f"model_{model_alias}_{representation}.joblib"
            model_path = models_dir / model_filename
            joblib.dump(model, model_path)

            elapsed = time.perf_counter() - start_time
            print(
                f"   -> Completado en {elapsed:.1f}s | "
                f"CV F1-Macro: {scores['cv_f1_mean']:.4f} (+/- {scores['cv_f1_std']:.4f}) | "
                f"Test F1-Macro: {scores['f1_macro']:.4f} | "
                f"Guardado: {model_filename}"
            )

            # Registro de metricas con el esquema exacto de columnas requerido.
            metrics_records.append(
                {
                    "model": model_alias,
                    "representation": representation,
                    "accuracy": round(scores["accuracy"], 6),
                    "precision_macro": round(scores["precision_macro"], 6),
                    "recall_macro": round(scores["recall_macro"], 6),
                    "f1_macro": round(scores["f1_macro"], 6),
                    "cv_f1_mean": round(scores["cv_f1_mean"], 6),
                    "cv_f1_std": round(scores["cv_f1_std"], 6),
                }
            )

        # Liberacion explicita de memoria RAM al finalizar cada matriz dispersa.
        del x_train, x_test
        gc.collect()

    # Exportacion de la tabla consolidada de metricas a CSV.
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(metrics_path, index=False)

    # Ordenamiento para mostrar el ranking de rendimiento en consola.
    summary_df = metrics_df.sort_values(by="f1_macro", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print(f">> RESUMEN DE MODELOS LINEALES Y PROBABILISTICOS ({approach.upper()})")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nTabla de metricas exportada en: {metrics_path}")


# Condicional estandar de Python para ejecutar el flujo principal solo de forma directa.
if __name__ == "__main__":
    main()
