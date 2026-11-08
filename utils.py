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

    def decode_tags(data): # does the inverse of encode_tags
        tag_order_path = Path(folder) / "kolejnosc_tagow.txt"
        tag_order = [
            tag.strip()
            for tag in tag_order_path.read_text(encoding="utf-8").split(",")
            if tag.strip()
        ]

        def decode_row(value):
            vector = json.loads(value) if isinstance(value, str) else value

            if len(vector) != len(tag_order):
                raise ValueError(
                    f"Expected {len(tag_order)} values, got {len(vector)}"
                )

            return [
                tag
                for tag, present in zip(tag_order, vector)
                if present == 1
            ]

        result = data.copy()
        result["tags"] = result["tags"].apply(decode_row)
        return result
    
    pipeline = [
        ("games_march2025_cleaned", get_raw_data),
        ("games_march2025_cleaned_selected_columns", select_data),
        ("games_march2025_cleaned_multi_hot_tags", encode_tags),
        ("games_march2025_cleaned_tags_decoded", decode_tags)
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