import argparse
from contextlib import nullcontext
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configuracion por enfoque usando los mismos modelos definidos en vectorize.py.
APPROACH_CONFIG = {
    "triclase": {
        "hf_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "label_map": {"negative": 0, "neutral": 1, "positive": 2},
        "class_names": ["Negativo (0)", "Neutral (1)", "Positivo (2)"],
    },
    "biclase": {
        "hf_model": "distilbert-base-uncased-finetuned-sst-2-english",
        "label_map": {"negative": 0, "positive": 1},
        "class_names": ["Negativo (0)", "Positivo (1)"],
    },
}

MAX_LENGTH = 512


def load_test_split(enfoque):
    processed_path = Path(f"{enfoque}/data/processed/amazon_reviews_clean.csv")
    split_path = Path(f"{enfoque}/data/processed/train_test_split.csv")
    y_test_path = Path(f"{enfoque}/data/vectorized/y_test.npy")

    df = pd.read_csv(processed_path, usecols=["id", "text", "sentiment"])
    df["text"] = df["text"].fillna("").astype(str)

    split_df = pd.read_csv(split_path)
    test_split = split_df[split_df["split"] == "test"][["id"]].copy()

    # Mantiene el orden exacto de train_test_split.csv para coincidir con y_test.npy.
    test_df = test_split.merge(df, on="id", how="left", sort=False)
    if test_df["sentiment"].isna().any():
        raise ValueError("Se encontraron ids del split test sin coincidencia en amazon_reviews_clean.csv.")

    y_true = test_df["sentiment"].to_numpy()
    if y_test_path.exists():
        y_test_npy = np.load(y_test_path)
        if not np.array_equal(y_true, y_test_npy):
            raise ValueError("Las etiquetas del split test no coinciden fila por fila con y_test.npy.")

    return test_df["text"].tolist(), y_true


def build_id_remap(id2label, label_map):
    # Traduce indices de salida del modelo hacia los enteros de sentiment del proyecto.
    remap = np.zeros(len(id2label), dtype=np.int64)
    for idx, label_name in id2label.items():
        normalized = str(label_name).strip().lower()
        if normalized not in label_map:
            raise ValueError(f"Etiqueta inesperada en model.config.id2label: {label_name}")
        remap[int(idx)] = label_map[normalized]
    return remap


def predict_batches(texts, model_name, label_map, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo de inferencia: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    print(f"Mapeo model.config.id2label verificado: {model.config.id2label}")
    id_remap = build_id_remap(model.config.id2label, label_map)

    preds = []
    # Usa precision mixta fp16 en GPU para acelerar inferencia y ahorrar VRAM.
    autocast_ctx = torch.amp.autocast("cuda") if device == "cuda" else nullcontext()

    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode(), autocast_ctx:
        for start in tqdm(range(0, len(texts), batch_size), desc="Evaluando HF"):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.append(id_remap[pred_ids])

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    return np.concatenate(preds), elapsed_seconds, model.config.id2label, device


def save_plots(cm, y_true, y_pred, class_names, enfoque, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Grafico tradicional de matriz de confusion con mapa de calor.
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 12, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(f"Matriz de Confusión - Hugging Face ({enfoque.capitalize()})", fontsize=13, pad=12)
    ax.set_xlabel("Predicción del Modelo", fontsize=11)
    ax.set_ylabel("Clase Real", fontsize=11)
    plt.tight_layout()
    cm_fig_path = figures_dir / "confusion_matrix_hf.png"
    fig.savefig(cm_fig_path, dpi=150)
    plt.close(fig)

    # Grafico de barras agrupadas para precision, recall y f1 por clase.
    prec, rec, f1_vals, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    rows = []
    for idx, name in enumerate(class_names):
        rows.append({"class_name": name, "metric": "Precision", "score": prec[idx]})
        rows.append({"class_name": name, "metric": "Recall", "score": rec[idx]})
        rows.append({"class_name": name, "metric": "F1-Score", "score": f1_vals[idx]})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=plot_df,
        x="class_name",
        y="score",
        hue="metric",
        palette="Set2",
        ax=ax,
    )
    ax.set_title(f"Métricas por Clase - Hugging Face ({enfoque.capitalize()})", fontsize=13, pad=12)
    ax.set_xlabel("Clase de Sentimiento", fontsize=11)
    ax.set_ylabel("Puntaje", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(title="Métrica", loc="upper center", ncol=3, frameon=True)
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.3f}",
                (patch.get_x() + patch.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )
    plt.tight_layout()
    metrics_fig_path = figures_dir / "class_metrics_hf.png"
    fig.savefig(metrics_fig_path, dpi=150)
    plt.close(fig)

    return cm_fig_path, metrics_fig_path


def save_outputs(enfoque, model_name, y_true, y_pred, class_names):
    vectorized_dir = Path(f"{enfoque}/data/vectorized")
    reports_dir = Path(f"{enfoque}/reports")
    figures_dir = reports_dir / "figures"
    vectorized_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    np.save(vectorized_dir / "y_pred_hf.npy", y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics_df = pd.DataFrame([{
        "model": model_name,
        "representation": "hf_pipeline",
        "accuracy": round(float(accuracy), 6),
        "precision_macro": round(float(precision_macro), 6),
        "recall_macro": round(float(recall_macro), 6),
        "f1_macro": round(float(f1_macro), 6),
        "cv_f1_mean": np.nan,
        "cv_f1_std": np.nan,
    }])
    metrics_csv_path = reports_dir / "metrics_hf.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    report_text = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_fig_path, metrics_fig_path = save_plots(
        cm, y_true, y_pred, class_names, enfoque, figures_dir
    )

    return accuracy, f1_macro, report_text, cm_df, metrics_csv_path, cm_fig_path, metrics_fig_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evalua modelo preentrenado de Hugging Face sobre test.")
    parser.add_argument("--enfoque", choices=["triclase", "biclase"], default="triclase")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = APPROACH_CONFIG[args.enfoque]
    y_pred_path = Path(f"{args.enfoque}/data/vectorized/y_pred_hf.npy")

    print(f"Cargando split test ({args.enfoque})...")
    texts, y_true = load_test_split(args.enfoque)
    print(f"Resenas a evaluar: {len(texts):,}")

    # Reutiliza predicciones guardadas si existen para regenerar graficos sin recalcular GPU.
    if y_pred_path.exists() and not args.force:
        print(f"Cargando predicciones existentes desde {y_pred_path} (usa --force para recalcular)...")
        y_pred = np.load(y_pred_path)
        if len(y_pred) != len(y_true):
            raise ValueError("La cantidad de predicciones en y_pred_hf.npy no coincide con y_true.")
        elapsed_seconds = None
    else:
        y_pred, elapsed_seconds, _, _ = predict_batches(
            texts, config["hf_model"], config["label_map"], args.batch_size
        )

    accuracy, f1_macro, report_text, cm_df, metrics_csv_path, cm_fig_path, metrics_fig_path = save_outputs(
        args.enfoque,
        config["hf_model"],
        y_true,
        y_pred,
        config["class_names"],
    )

    print("\n" + "=" * 50)
    print(f">> EVALUACION HF {args.enfoque.upper()} COMPLETADA")
    print("=" * 50)
    if elapsed_seconds is not None:
        print(f"Accuracy: {accuracy:.4f} | F1 Macro: {f1_macro:.4f} | Tiempo: {elapsed_seconds:.2f} s")
    else:
        print(f"Accuracy: {accuracy:.4f} | F1 Macro: {f1_macro:.4f}")
    print("\nReporte por clase:")
    print(report_text)
    print("Matriz de confusion:")
    print(cm_df)
    print(f"\nCSV guardado en: {metrics_csv_path}")
    print(f"Matriz de confusion guardada en: {cm_fig_path}")
    print(f"Metricas por clase guardadas en: {metrics_fig_path}")


if __name__ == "__main__":
    main()
