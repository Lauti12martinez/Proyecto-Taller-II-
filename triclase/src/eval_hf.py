import argparse
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HF_MODELS = {
    "triclase": ("cardiffnlp/twitter-roberta-base-sentiment-latest", ["Negativo (0)", "Neutral (1)", "Positivo (2)"]),
    "biclase": ("distilbert-base-uncased-finetuned-sst-2-english", ["Negativo (0)", "Positivo (1)"]),
}
LABEL_TO_ID = {
    "triclase": {"negative": 0, "label_0": 0, "neutral": 1, "label_1": 1, "positive": 2, "label_2": 2},
    "biclase": {"negative": 0, "label_0": 0, "positive": 1, "label_1": 1},
}
MAX_LENGTH = 512


def load_test_texts(approach):
    # Lee las resenas crudas del split test respetando el orden exacto de train_test_split.csv.
    df = pd.read_csv(f"{approach}/data/processed/amazon_reviews_clean.csv", usecols=["id", "text", "sentiment"])
    df["text"] = df["text"].fillna("").astype(str)
    split_df = pd.read_csv(f"{approach}/data/processed/train_test_split.csv")
    test_df = split_df[split_df["split"] == "test"][["id"]].merge(df, on="id", how="left", sort=False)
    if test_df["sentiment"].isna().any():
        raise ValueError("Se encontraron ids del split test sin coincidencia en amazon_reviews_clean.csv.")

    # Verifica alineacion exacta fila por fila contra y_test.npy si existe.
    y_true = test_df["sentiment"].to_numpy()
    y_test_path = Path(f"{approach}/data/vectorized/y_test.npy")
    if y_test_path.exists():
        y_test_npy = np.load(y_test_path)
        if not np.array_equal(y_true, y_test_npy):
            raise ValueError("Las etiquetas del split test no coinciden fila por fila con y_test.npy.")

    return test_df["text"].tolist(), y_true


def predict_gpu(texts, model_name, approach, batch_size):
    # Ejecuta inferencia por lotes en GPU usando precision mixta fp16 sobre Tensor Cores.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    print(f"Dispositivo: {device} | id2label: {model.config.id2label}")

    # Mapea las etiquetas reales de model.config.id2label hacia los ids de sentimiento del proyecto.
    label_map = LABEL_TO_ID[approach]
    id_remap = np.zeros(len(model.config.id2label), dtype=np.int64)
    for idx, label_name in model.config.id2label.items():
        normalized = str(label_name).strip().lower()
        if normalized not in label_map:
            raise ValueError(f"Etiqueta inesperada en model.config.id2label: {label_name}")
        id_remap[int(idx)] = label_map[normalized]

    preds = []
    start_time = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
        for start in tqdm(range(0, len(texts), batch_size), desc="Evaluando HF"):
            encoded = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            pred_ids = torch.argmax(model(**encoded).logits, dim=-1).cpu().numpy()
            preds.append(id_remap[pred_ids])

    elapsed = time.perf_counter() - start_time
    print(f"Tiempo de inferencia: {elapsed:.2f} s ({elapsed / len(texts) * 1000:.2f} ms/resena)")
    return np.concatenate(preds)


def save_plots(y_true, y_pred, report_dict, class_names, approach, figures_dir):
    # Genera la matriz de confusion y el grafico de barras por clase con pandas y sklearn.
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, cmap="Blues", values_format="d", ax=ax
    )
    ax.set_title(f"Matriz de Confusión - Hugging Face ({approach.capitalize()})")
    ax.set_xlabel("Predicción del Modelo")
    ax.set_ylabel("Clase Real")
    plt.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix_hf.png", dpi=150)
    plt.close(fig)

    report_df = pd.DataFrame(report_dict).T.loc[class_names, ["precision", "recall", "f1-score"]]
    ax = report_df.plot(kind="bar", figsize=(8, 5), rot=0, ylim=(0, 1.15), color=["#60A5FA", "#FDBA74", "#4ADE80"])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
    ax.set_title(f"Métricas por Clase - Hugging Face ({approach.capitalize()})")
    ax.set_xlabel("Clase de Sentimiento")
    ax.set_ylabel("Puntaje")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(["Precision", "Recall", "F1-Score"], title="Métrica", loc="upper center", ncol=3)
    plt.tight_layout()
    ax.get_figure().savefig(figures_dir / "class_metrics_hf.png", dpi=150)
    plt.close(ax.get_figure())


def main():
    # Coordina la carga de test, inferencia o lectura de cache, exportacion CSV y graficos.
    parser = argparse.ArgumentParser(description="Evalua modelo preentrenado de Hugging Face sobre test.")
    parser.add_argument("--enfoque", dest="approach", choices=["triclase", "biclase"], default="triclase")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    approach = args.approach
    model_name, class_names = HF_MODELS[approach]
    y_pred_path = Path(f"{approach}/data/vectorized/y_pred_hf.npy")
    reports_dir = Path(f"{approach}/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    y_pred_path.parent.mkdir(parents=True, exist_ok=True)

    texts, y_true = load_test_texts(approach)
    print(f"Evaluando {len(texts):,} resenas de test ({approach})...")

    if y_pred_path.exists() and not args.force:
        print(f"Cargando predicciones guardadas en {y_pred_path} (usa --force para recalcular)...")
        y_pred = np.load(y_pred_path)
        if len(y_pred) != len(y_true):
            raise ValueError("La cantidad de predicciones en y_pred_hf.npy no coincide con y_true.")
    else:
        y_pred = predict_gpu(texts, model_name, approach, args.batch_size)
        np.save(y_pred_path, y_pred)

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    macro = report_dict["macro avg"]
    pd.DataFrame([{
        "model": model_name,
        "representation": "hf_pipeline",
        "accuracy": report_dict["accuracy"],
        "precision_macro": macro["precision"],
        "recall_macro": macro["recall"],
        "f1_macro": macro["f1-score"],
        "cv_f1_mean": np.nan,
        "cv_f1_std": np.nan,
    }]).round(6).to_csv(reports_dir / "metrics_hf.csv", index=False)

    save_plots(y_true, y_pred, report_dict, class_names, approach, reports_dir / "figures")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
