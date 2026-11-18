import argparse
import time
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
CV_FOLDS = 5
CLASS_NAMES = {
    "triclase": ["Negativo (0)", "Neutral (1)", "Positivo (2)"],
    "biclase": ["Negativo (0)", "Positivo (1)"],
}
WEIGHT_OPTIONS = {
    "triclase": [None, "balanced"],
    "biclase": [None, "balanced"],
}


def load_train_test_texts(approach):
    # Carga textos limpios y etiquetas de train y test siguiendo el split oficial.
    df = pd.read_csv(f"{approach}/data/processed/amazon_reviews_clean.csv", usecols=["id", "cleaned_text", "sentiment"])
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
    split_df = pd.read_csv(f"{approach}/data/processed/train_test_split.csv")

    train_df = split_df[split_df["split"] == "train"][["id"]].merge(df, on="id", how="left", sort=False)
    test_df = split_df[split_df["split"] == "test"][["id"]].merge(df, on="id", how="left", sort=False)
    return train_df["cleaned_text"].tolist(), train_df["sentiment"].to_numpy(), test_df["cleaned_text"].tolist(), test_df["sentiment"].to_numpy()


def build_search_spaces(approach):
    # Define pipelines y grillas combinando BoW (CountVectorizer), TF-IDF y pesos de clase.
    weights = WEIGHT_OPTIONS[approach]
    vec_grid = [
        {
            "vectorizer": [CountVectorizer()],
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "vectorizer__min_df": [5, 10],
            "vectorizer__max_df": [0.85, 0.95],
        },
        {
            "vectorizer": [TfidfVectorizer()],
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "vectorizer__min_df": [5, 10],
            "vectorizer__max_df": [0.85, 0.95],
            "vectorizer__sublinear_tf": [True],
        },
    ]

    def expand(clf_params):
        return [{**vec_params, **clf_params} for vec_params in vec_grid]

    return {
        "logregopt": (
            "logreg",
            Pipeline([("vectorizer", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1500, solver="lbfgs", random_state=RANDOM_STATE))]),
            expand({"clf__C": [0.15, 0.25, 0.5, 1.0], "clf__class_weight": weights}),
        ),
        "linearsvcopt": (
            "svc",
            Pipeline([
                ("vectorizer", TfidfVectorizer()),
                ("clf", CalibratedClassifierCV(LinearSVC(dual=False, max_iter=3000, random_state=RANDOM_STATE), cv=3, n_jobs=1)),
            ]),
            expand({
                "clf__method": ["isotonic", "sigmoid"],
                "clf__estimator__penalty": ["l1", "l2"],
                "clf__estimator__C": [0.05, 0.15, 0.25, 0.5],
                "clf__estimator__class_weight": weights,
            }),
        ),
        "sgdopt": (
            "sgd",
            Pipeline([
                ("vectorizer", TfidfVectorizer()),
                ("clf", SGDClassifier(loss="log_loss", max_iter=1000, early_stopping=True, random_state=RANDOM_STATE)),
            ]),
            expand({
                "clf__penalty": ["l1", "l2", "elasticnet"],
                "clf__l1_ratio": [0.15, 0.5, 0.85],
                "clf__alpha": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
                "clf__class_weight": weights,
            }),
        ),
        "compnbopt": (
            "nb",
            Pipeline([("vectorizer", TfidfVectorizer()), ("clf", ComplementNB())]),
            expand({"clf__alpha": [0.1, 0.3, 0.5, 1.0], "clf__fit_prior": [True, False], "clf__norm": [True, False]}),
        ),
    }


def save_plots(approach, best_model, best_pred, y_test, metrics_df, class_names, reports_dir):
    # Guarda matriz de confusion, metricas por clase del ganador y comparativa CV vs Test.
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, best_pred, display_labels=class_names, cmap="Blues", values_format="d", ax=ax
    )
    ax.set_title(f"Matriz de Confusión - Ganador: {best_model} ({approach.capitalize()})")
    ax.set_xlabel("Predicción del Modelo")
    ax.set_ylabel("Clase Real")
    plt.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix_opt_best.png", dpi=150)
    plt.close(fig)

    report = classification_report(y_test, best_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T.loc[class_names, ["precision", "recall", "f1-score"]]
    ax = report_df.plot(kind="bar", figsize=(8, 5), rot=0, ylim=(0, 1.15), color=["#60A5FA", "#FDBA74", "#4ADE80"])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
    ax.set_title(f"Métricas por Clase - {best_model} ({approach.capitalize()})")
    ax.set_xlabel("Clase de Sentimiento")
    ax.set_ylabel("Puntaje")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(["Precision", "Recall", "F1-Score"], title="Métrica", loc="upper center", ncol=3)
    plt.tight_layout()
    ax.get_figure().savefig(figures_dir / "class_metrics_opt_best.png", dpi=150)
    plt.close(ax.get_figure())

    comp_df = metrics_df.set_index("model")[["cv_f1_mean", "f1_macro"]].rename(
        columns={"cv_f1_mean": "CV (5 folds)", "f1_macro": "Test (20%)"}
    )
    hf_csv = reports_dir / "metrics_hf.csv"
    if hf_csv.exists():
        hf_label = "hf_roberta" if approach == "triclase" else "hf_distilbert"
        comp_df.loc[hf_label] = [np.nan, float(pd.read_csv(hf_csv)["f1_macro"].iloc[0])]

    ax = comp_df.plot(kind="bar", figsize=(9, 5), rot=0, ylim=(0, 1.05), color=["#60A5FA", "#FDBA74"])
    for container in ax.containers:
        labels = [f"{v:.3f}" if v > 0 else "" for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=3, fontsize=9)
    ax.set_title(f"Comparativa F1 Macro: CV vs Test ({approach.capitalize()})")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("F1 Macro")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Evaluación", loc="upper right")
    plt.tight_layout()
    ax.get_figure().savefig(figures_dir / "model_comparison_opt.png", dpi=150)
    plt.close(ax.get_figure())


def main():
    # Ejecuta la optimizacion CV en train, exporta modelos, metricas y genera graficos.
    parser = argparse.ArgumentParser(description="Optimiza modelos clasicos con Pipeline y RandomizedSearchCV sobre train.")
    parser.add_argument("--enfoque", dest="approach", choices=["triclase", "biclase"], default="triclase")
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--models", nargs="+", default=["logregopt", "linearsvcopt", "sgdopt", "compnbopt"])
    parser.add_argument("--only-plots", action="store_true")
    args = parser.parse_args()

    approach = args.approach
    class_names = CLASS_NAMES[approach]
    models_dir = Path(f"{approach}/models")
    vectorized_dir = Path(f"{approach}/data/vectorized")
    reports_dir = Path(f"{approach}/reports")
    for folder in (models_dir, vectorized_dir, reports_dir):
        folder.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_train_test_texts(approach)

    if args.only_plots:
        metrics_df = pd.read_csv(reports_dir / "metrics_opt.csv")
        best_model = metrics_df.iloc[0]["model"]
        best_pred = np.load(vectorized_dir / f"y_pred_opt_{best_model}.npy")
        save_plots(approach, best_model, best_pred, y_test, metrics_df, class_names, reports_dir)
        print(f"Graficos regenerados desde cache en {reports_dir / 'figures'}")
        return

    cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search_spaces = build_search_spaces(approach)
    metrics_rows, cv_tables = [], []
    best_model, best_f1, best_pred = None, -1.0, None

    for model_name in args.models:
        repr_suffix, pipeline, params = search_spaces[model_name]
        print(f"\nOptimizando {model_name}...")
        t0 = time.perf_counter()
        search = RandomizedSearchCV(
            pipeline, params, n_iter=args.n_iter, scoring="f1_macro", cv=cv_splitter,
            n_jobs=args.n_jobs, random_state=RANDOM_STATE, verbose=1, refit=True, return_train_score=True,
        )
        search.fit(x_train, y_train)

        best_est = search.best_estimator_
        # Detecta el vectorizador ganador (TfidfVectorizer hereda de CountVectorizer en sklearn).
        vec_prefix = "tfidfopt" if isinstance(best_est.named_steps["vectorizer"], TfidfVectorizer) else "bowopt"
        repr_name = f"{vec_prefix}{repr_suffix}"
        y_pred = best_est.predict(x_test)
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        macro = report["macro avg"]
        cv_mean = float(search.cv_results_["mean_test_score"][search.best_index_])
        cv_std = float(search.cv_results_["std_test_score"][search.best_index_])

        # Guarda pipeline completo para Gradio y componentes separados para evaluate_adversarial.
        joblib.dump(best_est, models_dir / f"pipeline_{model_name}_{repr_name}.joblib")
        joblib.dump(best_est.named_steps["vectorizer"], models_dir / f"vectorizer_{repr_name}.joblib")
        joblib.dump(best_est.named_steps["clf"], models_dir / f"model_{model_name}_{repr_name}.joblib")
        np.save(vectorized_dir / f"y_pred_opt_{model_name}.npy", y_pred)

        metrics_rows.append({
            "model": model_name,
            "representation": repr_name,
            "accuracy": report["accuracy"],
            "precision_macro": macro["precision"],
            "recall_macro": macro["recall"],
            "f1_macro": macro["f1-score"],
            "cv_f1_mean": cv_mean,
            "cv_f1_std": cv_std,
        })
        cv_df = pd.DataFrame(search.cv_results_)
        cv_df.insert(0, "model", model_name)
        cv_df["overfit_gap"] = cv_df["mean_train_score"] - cv_df["mean_test_score"]
        cv_tables.append(cv_df)

        print(f"  [OK] {model_name} ({repr_name}) en {time.perf_counter() - t0:.1f}s | CV F1: {cv_mean:.4f} (+/- {cv_std:.4f}) | Test F1: {macro['f1-score']:.4f}")
        print(f"  Mejores parametros: {search.best_params_}")

        if macro["f1-score"] > best_f1:
            best_model, best_f1, best_pred = model_name, macro["f1-score"], y_pred

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="f1_macro", ascending=False).round(6).reset_index(drop=True)
    metrics_df.to_csv(reports_dir / "metrics_opt.csv", index=False)
    pd.concat(cv_tables, ignore_index=True).sort_values(by="mean_test_score", ascending=False).to_csv(reports_dir / "cv_search_history.csv", index=False)

    save_plots(approach, best_model, best_pred, y_test, metrics_df, class_names, reports_dir)
    print(f"\n>> OPTIMIZACION {approach.upper()} COMPLETADA - GANADOR: {best_model}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
