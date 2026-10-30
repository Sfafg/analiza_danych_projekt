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

    def encode_tags(data): # the function scrapes tags, sorts them in alphabetical order and adds multi-hot encoding for each row in the tags column based on that alphabetically ordered tags list
        tag_order_path = Path(__file__).resolve().parent / "kolejnosc_tagow.txt"

        tag_order = [
            tag.strip()
            for tag in tag_order_path.read_text(encoding="utf-8-sig").split(",")
            if tag.strip()
        ]

        if len(tag_order) != len(set(tag_order)):
            raise ValueError("The tag order file contains duplicate tags")

        def parse_tags(value):
            if pd.isna(value) or not str(value).strip():
                return set()

            parsed = ast.literal_eval(value)

            if not isinstance(parsed, (dict, list)):
                raise ValueError(f"Unexpected tags format: {value}")

            return set(parsed)

        game_tags = data["tags"].apply(parse_tags)

        found_tags = set()
        for tags in game_tags:
            found_tags.update(tags)

        ordered_found_tags = [
            tag for tag in tag_order
            if tag in found_tags
        ]

        output_path = Path(folder) / "tagi_w_danych.txt"
        output_path.write_text(
            ", ".join(ordered_found_tags),
            encoding="utf-8"
        )

        unknown_tags = found_tags - set(tag_order)
        if unknown_tags:
            print(f"Skipped {len(unknown_tags)} tags not present in the tag order")

        result = data.copy()
        result["tags"] = game_tags.apply(
            lambda tags: json.dumps(
                [int(tag in tags) for tag in tag_order],
                separators=(",", ":")
            )
        )

        return result

    pipeline = [
        ("games_march2025_cleaned", get_raw_data),
        ("games_march2025_cleaned_selected_columns", select_data),
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

get_data()