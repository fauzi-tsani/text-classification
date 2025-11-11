import os
import re
import json
import random
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np


FOOTBALL_KEYWORDS = [
    # general football terms
    "sepak bola", "sepakbola", "football", "soccer", "gol", "pelatih", "striker", "penyerang",
    "bek", "gelandang", "kiper", "offside", "assist", "penalti", "tendangan bebas", "corner", "liga",
]

NON_FOOTBALL_KEYWORDS = [
    # other sports
    "badminton", "bulu tangkis", "bulutangkis", "basket", "bola basket", "voli", "tenis", "renang",
    "atletik", "cricket", "rugby", "golf", "tinju", "mma", "panco", "balap", "motogp", "formula",
    "f1", "nascar", "valentino rossi", "marquez", "marc marquez", "lebron", "curry",
]

# League keyword sets (Indonesian + common entities)
LEAGUE_KEYWORDS = {
    "Liga Inggris": [
        "liga inggris", "premier league", "epl", "fa cup", "carabao cup",
        # clubs
        "manchester united", "manchester city", "liverpool", "chelsea", "arsenal", "tottenham",
        "newcastle", "everton", "aston villa", "west ham", "brighton", "bournemouth", "fulham",
        "brentford", "wolves", "burnley", "sheffield united", "nottingham forest", "crystal palace",
    ],
    "Liga Indonesia": [
        "liga 1", "liga indonesia", "liga 2", "bri liga 1", "piala menpora", "piala presiden",
        # clubs
        "persib", "persija", "arema", "psm makassar", "persebaya", "bali united", "bhayangkara",
        "barito putera", "madura united", "pss sleman", "persik", "psis semarang", "rans", "dewa united",
        "persita", "persikabo",
    ],
    "Liga Spanyol": [
        "la liga", "liga spanyol", "primera division", "copa del rey",
        # clubs
        "barcelona", "real madrid", "atletico madrid", "sevilla", "valencia", "real sociedad",
        "villarreal", "athletic bilbao", "betis", "celta vigo", "espanyol",
    ],
    "Liga Italia": [
        "serie a", "liga italia", "coppa italia", "supercoppa",
        # clubs
        "juventus", "inter", "milan", "roma", "lazio", "napoli", "fiorentina", "atalanta",
        "udinese", "sassuolo", "torino", "bologna",
    ],
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


def label_5class(text: str) -> Optional[str]:
    """Return one of 5 labels or None if unknown.
    Labels: Liga Inggris, Liga Indonesia, Liga Spanyol, Liga Italia, Olahraga non-sepakbola
    """
    if not isinstance(text, str) or not text:
        return None
    t = normalize_text(text)

    # Non-football first
    if contains_any(t, NON_FOOTBALL_KEYWORDS) and not contains_any(t, FOOTBALL_KEYWORDS):
        return "Olahraga non-sepakbola"

    # Football leagues
    for league, keys in LEAGUE_KEYWORDS.items():
        if contains_any(t, keys):
            return league

    # If football but unknown league, return None to exclude from 5-class training
    if contains_any(t, FOOTBALL_KEYWORDS):
        return None

    # Default to non-football if neither set matches clearly
    return "Olahraga non-sepakbola"


def label_stage1_binary(text: str) -> Optional[int]:
    """Binary label: 1=sepakbola, 0=non-sepakbola. None if unknown."""
    if not isinstance(text, str) or not text:
        return None
    t = normalize_text(text)
    if contains_any(t, NON_FOOTBALL_KEYWORDS) and not contains_any(t, FOOTBALL_KEYWORDS):
        return 0
    if contains_any(t, FOOTBALL_KEYWORDS):
        return 1
    # default non-football
    return 0


def label_stage2_league(text: str) -> Optional[str]:
    """Stage-2 label among 4 leagues for football texts. None if non-football or unknown league."""
    if not isinstance(text, str) or not text:
        return None
    t = normalize_text(text)
    if not contains_any(t, FOOTBALL_KEYWORDS):
        return None
    for league, keys in LEAGUE_KEYWORDS.items():
        if league == "Olahraga non-sepakbola":
            continue
        if contains_any(t, keys):
            return league
    return None


def stratified_split(
    df: pd.DataFrame, label_col: str, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed=random_state)
    classes = df[label_col].unique()
    train_parts = []
    val_parts = []
    test_parts = []

    for c in classes:
        sub = df[df[label_col] == c].sample(frac=1.0, random_state=random_state)  # shuffle
        n = len(sub)
        n_test = int(round(test_size * n))
        n_val = int(round(val_size * n))
        # ensure bounds
        n_test = min(n_test, n)
        n_val = min(n_val, n - n_test)
        n_train = n - n_test - n_val
        test_parts.append(sub.iloc[:n_test])
        val_parts.append(sub.iloc[n_test:n_test + n_val])
        train_parts.append(sub.iloc[n_test + n_val:])

    train = pd.concat(train_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val = pd.concat(val_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return train, val, test


def prepare_dataset(input_csv: str = "dataset/cleaned_news_articles.csv", output_dir: str = "dataset") -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    # Gunakan kolom cleaned_content sebagai teks utama bila tersedia
    text_col = "cleaned_content" if "cleaned_content" in df.columns else "content"
    df["text"] = df[text_col].fillna("")

    # Gunakan kolom category sebagai label utama 5 kelas
    def map_category(cat: Optional[str]) -> Optional[str]:
        if cat is None or not isinstance(cat, str) or not cat.strip():
            return None
        t = cat.lower().replace("_", " ").replace("-", " ")
        t = re.sub(r"\s+", " ", t).strip()
        if "inggris" in t or "premier" in t or "epl" in t:
            return "Liga Inggris"
        if "indonesia" in t:
            return "Liga Indonesia"
        if "spanyol" in t or "la liga" in t:
            return "Liga Spanyol"
        if "italia" in t or "serie a" in t:
            return "Liga Italia"
        # default ke non-sepakbola
        if "non" in t or "lain" in t:
            return "Olahraga non-sepakbola"
        # jika kategori di luar empat liga, anggap non-sepakbola
        return "Olahraga non-sepakbola"

    if "category" in df.columns:
        df["label_5class"] = df["category"].apply(map_category)
    else:
        # fallback: gunakan heuristik teks (lama)
        df["label_5class"] = df["text"].apply(label_5class)

    # Label untuk 2-Stage berdasarkan label_5class
    df["label_stage1"] = df["label_5class"].apply(lambda x: 1 if x in {"Liga Inggris", "Liga Indonesia", "Liga Spanyol", "Liga Italia"} else 0)
    df["label_stage2"] = df["label_5class"].apply(lambda x: x if x in {"Liga Inggris", "Liga Indonesia", "Liga Spanyol", "Liga Italia"} else None)

    # Filter rows for 5-class (exclude unknown football league)
    df_5 = df[df["label_5class"].notna()].copy()
    print(f"Total usable for 5-class: {len(df_5)} / {len(df)}")

    # Split 5-class
    train5, val5, test5 = stratified_split(df_5, label_col="label_5class")
    train5.to_csv(os.path.join(output_dir, "train_5class.csv"), index=False)
    val5.to_csv(os.path.join(output_dir, "val_5class.csv"), index=False)
    test5.to_csv(os.path.join(output_dir, "test_5class.csv"), index=False)

    # Stage-1 binary split on all rows with label present
    df_s1 = df[df["label_stage1"].notna()].copy()
    train1, val1, test1 = stratified_split(df_s1, label_col="label_stage1")
    train1.to_csv(os.path.join(output_dir, "train_stage1.csv"), index=False)
    val1.to_csv(os.path.join(output_dir, "val_stage1.csv"), index=False)
    test1.to_csv(os.path.join(output_dir, "test_stage1.csv"), index=False)

    # Stage-2 4-league only for football rows with known league
    df_s2 = df[df["label_stage2"].notna()].copy()
    if len(df_s2) >= 5:
        train2, val2, test2 = stratified_split(df_s2, label_col="label_stage2")
        train2.to_csv(os.path.join(output_dir, "train_stage2.csv"), index=False)
        val2.to_csv(os.path.join(output_dir, "val_stage2.csv"), index=False)
        test2.to_csv(os.path.join(output_dir, "test_stage2.csv"), index=False)
    else:
        print("Warning: Not enough data for stage-2 stratified split. Skipping.")

    # Save label distribution summaries
    summary = {
        "5class_distribution": df_5["label_5class"].value_counts().to_dict(),
        "stage1_distribution": df_s1["label_stage1"].value_counts().to_dict(),
        "stage2_distribution": df_s2["label_stage2"].value_counts().to_dict(),
    }
    with open(os.path.join(output_dir, "label_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Saved label distributions and splits to:", output_dir)


if __name__ == "__main__":
    prepare_dataset()
