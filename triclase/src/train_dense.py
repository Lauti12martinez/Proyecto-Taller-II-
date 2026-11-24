import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
CV_FOLDS = 5
REPRESENTATION = "glove"
LEARNING_CURVE_SIZES = np.linspace(0.1, 1.0, 5)


def build_search_spaces():
    # Estimador base y espacio de busqueda por algoritmo denso.
    # MultinomialNB queda excluido porque los embeddings GloVe tienen valores negativos.
    return {
        "rf": (
            RandomForestClassifier(n_jobs=1, random_state=RANDOM_STATE),
            {
                # max_samples acota el bootstrap de cada arbol: es el parametro que mas reduce el costo en CPU.
                "n_estimators": [150, 300],
                "max_depth": [None, 25],
                "min_samples_leaf": [1, 3],
                "max_features": ["sqrt", 0.2],
                "max_samples": [0.3, 0.5],
                "class_weight": [None, "balanced_subsample"],
            },
        ),
        "xgb": (
            XGBClassifier(tree_method="hist", n_jobs=1, random_state=RANDOM_STATE),
            {
                "n_estimators": [300, 500, 800],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.7, 0.9, 1.0],
                "min_child_weight": [1, 5],
            },
        ),
        "mlp": (
            # El escalado estandar acelera la convergencia del perceptron sobre vectores promediados.
            make_pipeline(StandardScaler(), MLPClassifier(early_stopping=True, max_iter=200, random_state=RANDOM_STATE)),
            {
                "mlpclassifier__hidden_layer_sizes": [(128,), (256,), (128, 64), (256, 128)],
                "mlpclassifier__alpha": [1e-4, 1e-3, 1e-2],
                "mlpclassifier__learning_rate_init": [5e-4, 1e-3],
                "mlpclassifier__batch_size": [256, 512],
            },
        ),
    }


def load_dense_data(vectorized_dir):
    # Carga de embeddings GloVe promediados y etiquetas del split oficial 80/20.
    files = {name: vectorized_dir / f"{name}.npy" for name in (f"X_train_{REPRESENTATION}", f"X_test_{REPRESENTATION}", "y_train", "y_test")}
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Faltan archivos vectorizados: {missing}. Ejecuta vectorize.py primero.")
    return [np.load(path) for path in files.values()]


def search_best_params(estimator, params, x_train, y_train, cv_splitter, args):
    # Busqueda aleatoria de hiperparametros sobre una submuestra estratificada del train para acotar el costo en CPU.
    if 0 < args.search_size < len(y_train):
        x_train, _, y_train, _ = train_test_split(
            x_train, y_train, train_size=args.search_size, stratify=y_train, random_state=RANDOM_STATE
        )
    search = RandomizedSearchCV(
        estimator, params, n_iter=args.n_iter, scoring="f1_macro", cv=cv_splitter,
        n_jobs=args.n_jobs, random_state=RANDOM_STATE, refit=False, verbose=1,
    )
    search.fit(x_train, y_train)
    return search


def main():
    parser = argparse.ArgumentParser(description="Entrena y optimiza RF, XGBoost y MLP sobre embeddings GloVe.")
    parser.add_argument("--enfoque", dest="approach", choices=["triclase", "biclase"], default="triclase")
    parser.add_argument("--models", nargs="+", choices=["rf", "xgb", "mlp"], default=["rf", "xgb", "mlp"])
    parser.add_argument("--n-iter", type=int, default=10, help="Combinaciones evaluadas por RandomizedSearchCV.")
    parser.add_argument("--search-size", type=int, default=40000, help="Resenas de train usadas en la busqueda (0 = todas).")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    base_dir = Path(args.approach)
    vectorized_dir, models_dir, reports_dir = base_dir / "data" / "vectorized", base_dir / "models", base_dir / "reports"
    for folder in (models_dir, reports_dir):
        folder.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = load_dense_data(vectorized_dir)
    print(f">> MODELOS DENSOS SOBRE {REPRESENTATION.upper()} ({args.approach}) | Train: {x_train.shape} | Test: {x_test.shape}")

    cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search_spaces = build_search_spaces()
    metrics_rows, search_tables, curve_tables = [], [], []

    for model_name in args.models:
        estimator, params = search_spaces[model_name]
        t0 = time.perf_counter()
        print(f"\nOptimizando {model_name}...")
        search = search_best_params(estimator, params, x_train, y_train, cv_splitter, args)
        model = estimator.set_params(**search.best_params_)
        print(f"  Mejores parametros: {search.best_params_}")

        # Curva de aprendizaje sobre el train completo; su ultimo punto equivale al CV de 5 folds del modelo final.
        sizes, train_scores, val_scores = learning_curve(
            model, x_train, y_train, train_sizes=LEARNING_CURVE_SIZES, cv=cv_splitter,
            scoring="f1_macro", n_jobs=args.n_jobs, shuffle=True, random_state=RANDOM_STATE,
        )
        curve_tables.append(pd.DataFrame({
            "model": model_name, "representation": REPRESENTATION, "train_size": sizes,
            "train_f1_mean": train_scores.mean(axis=1), "train_f1_std": train_scores.std(axis=1),
            "val_f1_mean": val_scores.mean(axis=1), "val_f1_std": val_scores.std(axis=1),
        }))

        # Ajuste final sobre todo el train (con todos los nucleos) y evaluacion en el test reservado.
        model.set_params(**{key: args.n_jobs for key in model.get_params() if key.endswith("n_jobs")})
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        joblib.dump(model, models_dir / f"model_{model_name}_{REPRESENTATION}.joblib")
        np.save(vectorized_dir / f"y_pred_dense_{model_name}.npy", y_pred)

        metrics_rows.append({
            "model": model_name,
            "representation": REPRESENTATION,
            "accuracy": report["accuracy"],
            "precision_macro": report["macro avg"]["precision"],
            "recall_macro": report["macro avg"]["recall"],
            "f1_macro": report["macro avg"]["f1-score"],
            "cv_f1_mean": val_scores[-1].mean(),
            "cv_f1_std": val_scores[-1].std(),
        })
        search_df = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
        search_tables.append(search_df.assign(model=model_name).sort_values("rank_test_score"))
        print(f"  [OK] {model_name} en {time.perf_counter() - t0:.0f}s | CV F1: {val_scores[-1].mean():.4f} | Test F1: {report['macro avg']['f1-score']:.4f}")

    # Exportacion con el esquema de columnas del contrato de integracion.
    metrics_df = pd.DataFrame(metrics_rows).sort_values("f1_macro", ascending=False).round(6)
    metrics_df.to_csv(reports_dir / "metrics_dense.csv", index=False)
    pd.concat(search_tables, ignore_index=True).to_csv(reports_dir / "cv_search_dense.csv", index=False)
    pd.concat(curve_tables, ignore_index=True).round(6).to_csv(reports_dir / "learning_curve_dense.csv", index=False)

    print(f"\n>> MODELOS DENSOS COMPLETADOS ({args.approach.upper()})")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
