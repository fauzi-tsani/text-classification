import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from metrics_utils import classification_report_np, confusion_matrix_np

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


LABELS_5CLASS = [
    "Liga Inggris",
    "Liga Indonesia",
    "Liga Spanyol",
    "Liga Italia",
    "Olahraga non-sepakbola",
]


def load_split(split_dir: str = "dataset") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(os.path.join(split_dir, "train_5class.csv"))
    val = pd.read_csv(os.path.join(split_dir, "val_5class.csv"))
    test = pd.read_csv(os.path.join(split_dir, "test_5class.csv"))
    return train, val, test


def prepare_texts_and_labels(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    texts = df["text"].astype(str).tolist()
    label_map = {name: idx for idx, name in enumerate(LABELS_5CLASS)}
    labels = df["label_5class"].map(label_map).values
    return texts, labels


def tokenize_dataset(tokenizer, texts: List[str], labels: np.ndarray) -> Dataset:
    enc = tokenizer(texts, truncation=True, padding=False, max_length=256)
    enc["labels"] = labels.tolist()
    return Dataset.from_dict(enc)


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_experiments(
    model_name: str = "xlm-roberta-base",
    lr_grid=(2e-5, 5e-5),
    batch_grid=(8, 16),
    num_epochs: int = 3,
    out_dir: str = "experiments/llm_1stage",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    train_df, val_df, test_df = load_split()

    train_texts, y_train = prepare_texts_and_labels(train_df)
    val_texts, y_val = prepare_texts_and_labels(val_df)
    test_texts, y_test = prepare_texts_and_labels(test_df)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = tokenize_dataset(tokenizer, train_texts, y_train)
    val_ds = tokenize_dataset(tokenizer, val_texts, y_val)
    test_ds = tokenize_dataset(tokenizer, test_texts, y_test)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    summary_rows = []

    for lr in lr_grid:
        for bs in batch_grid:
            run_name = f"lr{lr}_bs{bs}"
            run_dir = os.path.join(out_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)

            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=len(LABELS_5CLASS)
            )

            args = TrainingArguments(
                output_dir=run_dir,
                learning_rate=lr,
                per_device_train_batch_size=bs,
                per_device_eval_batch_size=bs,
                num_train_epochs=num_epochs,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                logging_steps=10,
                seed=SEED,
                report_to=[],
                fp16=torch.cuda.is_available(),
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

            trainer.train()

            # Predictions for validation and test
            val_pred = trainer.predict(val_ds)
            test_pred = trainer.predict(test_ds)

            y_val_pred = np.argmax(val_pred.predictions, axis=1)
            y_test_pred = np.argmax(test_pred.predictions, axis=1)

            report_val = classification_report_np(y_val, y_val_pred, LABELS_5CLASS)
            report_test = classification_report_np(y_test, y_test_pred, LABELS_5CLASS)

            with open(os.path.join(run_dir, "report_val.json"), "w", encoding="utf-8") as f:
                json.dump(report_val, f, ensure_ascii=False, indent=2)
            with open(os.path.join(run_dir, "report_test.json"), "w", encoding="utf-8") as f:
                json.dump(report_test, f, ensure_ascii=False, indent=2)

            cm_val = confusion_matrix_np(y_val, y_val_pred, len(LABELS_5CLASS))
            cm_test = confusion_matrix_np(y_test, y_test_pred, len(LABELS_5CLASS))
            plot_confusion_matrix(cm_val, LABELS_5CLASS, os.path.join(run_dir, "cm_val.png"))
            plot_confusion_matrix(cm_test, LABELS_5CLASS, os.path.join(run_dir, "cm_test.png"))

            summary_rows.append({
                "run": run_name,
                "learning_rate": lr,
                "batch_size": bs,
                "val_accuracy": report_val.get("accuracy", 0),
                "val_f1_weighted": report_val.get("weighted avg", {}).get("f1-score", 0),
                "test_accuracy": report_test.get("accuracy", 0),
                "test_f1_weighted": report_test.get("weighted avg", {}).get("f1-score", 0),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("Saved LLM experiment summary to:", summary_path)


if __name__ == "__main__":
    run_experiments(
        model_name="xlm-roberta-base",
        lr_grid=(2e-5, 5e-5),
        batch_grid=(8, 16),
        num_epochs=3,
    )
