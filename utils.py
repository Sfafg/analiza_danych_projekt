import os
import pandas as pd
import ast
import json
from pathlib import Path


def prepare_build_dir(directory_name):
    folder = f"build/{directory_name}"
    os.makedirs(folder, exist_ok=True)
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)

    return folder


def get_data():
    folder = "data"
    os.makedirs("data", exist_ok=True)

    def get_raw_data(data):
        return pd.read_csv("games_march2025_cleaned.csv")

    def select_data(data):
        return data[
            [
                "appid",
                "name",
                "price",
                "metacritic_score",
                "achievements",
                "recommendations",
                "categories",
                "genres",
                "positive",
                "negative",
                "tags",
            ]
        ]

    def simplify_tags(data):
        def extract_tags(x):
            if isinstance(x, str):
                x = ast.literal_eval(x)

            if isinstance(x, dict):
                x = list(x.keys())

            if isinstance(x, list):
                return str(sorted(x, key=str.lower))

            return x

        data["tags"] = data["tags"].apply(extract_tags)
        return data

    def encode_tags(data):
        def parse_tags(
            value,
        ):  # the function scrapes tags, sorts them in alphabetical order and adds multi-hot encoding for each row in the tags column based on that alphabetically ordered tags list
            if pd.isna(value) or not str(value).strip():
                return set()

            parsed = ast.literal_eval(value)

            if not isinstance(parsed, (dict, list)):
                raise ValueError(f"Unexpected tags format: {value}")

            return set(parsed)  # słownik: klucze; lista: jej elementy

        game_tags = data["tags"].apply(parse_tags)

        all_tags = set()
        for tags in game_tags:
            all_tags.update(tags)

        tag_order = sorted(all_tags, key=str.casefold)

        tag_order_path = Path(folder) / "kolejnosc_tagow.txt"
        tag_order_path.write_text(", ".join(tag_order), encoding="utf-8")

        result = data.copy()
        result["tags"] = game_tags.apply(
            lambda tags: json.dumps(
                [int(tag in tags) for tag in tag_order], separators=(",", ":")
            )
        )

        print(f"Saved {len(tag_order)} tag names to {tag_order_path}")
        return result

    pipeline = [
        ("games_march2025_cleaned", get_raw_data),
        ("games_march2025_cleaned_selected_columns", select_data),
        ("games_march2025_cleaned_simplified_tags", simplify_tags),
        ("games_march2025_cleaned_multi_hot_tags", encode_tags),
    ]

    latest_data = None
    latest_step = -1
    for step in reversed(range(len(pipeline))):
        file_name, func = pipeline[step]
        path = f"{folder}/{file_name}.csv"
        if os.path.exists(path):
            latest_data = pd.read_csv(path)
            latest_step = step
            break

    for step in range(latest_step + 1, len(pipeline)):
        file_name, func = pipeline[step]
        path = f"{folder}/{file_name}.csv"
        data = func(latest_data)
        data.to_csv(path, index=False)
        latest_data = data

    return latest_data


TAG_ORDER = None


def decode_tags(vector):
    global TAG_ORDER

    if TAG_ORDER is None:
        tag_order_path = (
            Path(__file__).resolve().parent / "data" / "kolejnosc_tagow.txt"
        )
        TAG_ORDER = [
            tag.strip()
            for tag in tag_order_path.read_text(encoding="utf-8").split(",")
            if tag.strip()
        ]

    # Przydatne, gdy wektor pochodzi z komórki pliku CSV.
    if isinstance(vector, str):
        vector = json.loads(vector)

    if len(vector) != len(TAG_ORDER):
        raise ValueError(f"Expected {len(TAG_ORDER)} values, got {len(vector)}")

    if any(value not in (0, 1) for value in vector):
        raise ValueError("The vector must contain only 0 and 1")

    return [tag for tag, value in zip(TAG_ORDER, vector) if value == 1]

