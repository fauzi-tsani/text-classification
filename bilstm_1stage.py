import os
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from metrics_utils import classification_report_np, confusion_matrix_np

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


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


def vectorize_texts(
    train_texts: List[str], val_texts: List[str], test_texts: List[str],
    num_words: int = 20000, max_len: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tokenizer]:
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    x_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=max_len, padding="post")
    x_val = pad_sequences(tokenizer.texts_to_sequences(val_texts), maxlen=max_len, padding="post")
    x_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=max_len, padding="post")
    return x_train, x_val, x_test, tokenizer


def build_model(vocab_size: int, max_len: int, num_classes: int, embed_dim: int, lstm_units: int) -> Sequential:
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len, mask_zero=True),
        Bidirectional(LSTM(lstm_units, return_sequences=False)),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_experiments(
    embed_grid=(64, 128), lstm_grid=(64, 128), num_words=20000, max_len=200, batch_size=32, epochs=10,
    out_dir: str = "experiments/bilstm_1stage"
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    train_df, val_df, test_df = load_split()

    train_texts, y_train = prepare_texts_and_labels(train_df)
    val_texts, y_val = prepare_texts_and_labels(val_df)
    test_texts, y_test = prepare_texts_and_labels(test_df)

    x_train, x_val, x_test, tokenizer = vectorize_texts(train_texts, val_texts, test_texts, num_words=num_words, max_len=max_len)
    vocab_size = min(num_words, len(tokenizer.word_index) + 1)
    num_classes = len(LABELS_5CLASS)

    summary_rows = []

    for embed_dim in embed_grid:
        for lstm_units in lstm_grid:
            run_name = f"embed{embed_dim}_lstm{lstm_units}"
            run_dir = os.path.join(out_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)

            model = build_model(vocab_size=vocab_size, max_len=max_len, num_classes=num_classes, embed_dim=embed_dim, lstm_units=lstm_units)

            # Callbacks
            ckpt_path = os.path.join(run_dir, "best.keras")
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
                ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
            ]

            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks,
            )

            # Save history
            hist_df = pd.DataFrame(history.history)
            hist_df.to_csv(os.path.join(run_dir, "history.csv"), index=False)

            # Evaluate on validation and test
            y_val_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
            y_test_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

            report_val = classification_report_np(y_val, y_val_pred, LABELS_5CLASS)
            report_test = classification_report_np(y_test, y_test_pred, LABELS_5CLASS)

            with open(os.path.join(run_dir, "report_val.json"), "w", encoding="utf-8") as f:
                json.dump(report_val, f, ensure_ascii=False, indent=2)
            with open(os.path.join(run_dir, "report_test.json"), "w", encoding="utf-8") as f:
                json.dump(report_test, f, ensure_ascii=False, indent=2)

            # Confusion matrices
            cm_val = confusion_matrix_np(y_val, y_val_pred, num_classes)
            cm_test = confusion_matrix_np(y_test, y_test_pred, num_classes)
            plot_confusion_matrix(cm_val, LABELS_5CLASS, os.path.join(run_dir, "cm_val.png"))
            plot_confusion_matrix(cm_test, LABELS_5CLASS, os.path.join(run_dir, "cm_test.png"))

            # Summary row
            summary_rows.append({
                "run": run_name,
                "embed_dim": embed_dim,
                "lstm_units": lstm_units,
                "val_accuracy": report_val.get("accuracy", 0),
                "val_f1_weighted": report_val.get("weighted avg", {}).get("f1-score", 0),
                "test_accuracy": report_test.get("accuracy", 0),
                "test_f1_weighted": report_test.get("weighted avg", {}).get("f1-score", 0),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("Saved experiment summary to:", summary_path)


if __name__ == "__main__":
    run_experiments(
        embed_grid=(64, 128),
        lstm_grid=(64, 128),
        num_words=20000,
        max_len=200,
        batch_size=32,
        epochs=10,
    )
