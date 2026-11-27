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

def encode_multi_hot(data, column):
        def parse_values(value):
            if pd.isna(value) or not str(value).strip():
                return set()

            parsed = ast.literal_eval(value)

            if not isinstance(parsed, (dict, list)):
                raise ValueError(f"Unexpected column format: {value}")

            return set(parsed)

        values  = data[column].apply(parse_values)

        all_values = set()
        for v in values:
            all_values.update(v)

        value_order = sorted(all_values, key=str.casefold)

        value_order_path = Path("data") / f"kolejnosc_{column}.txt"
        value_order_path.write_text(", ".join(value_order), encoding="utf-8")

        data[column] = values.apply(lambda v: [int(val in v) for val in value_order])

        return data

orders = {}

def decode_multi_hot(vector, column):
    global orders
    if column not in orders:
        order_path = (
            Path(__file__).resolve().parent / "data" / f"kolejnosc_{column}.txt"
        )
        orders[column] = [
            v.strip()
            for v in order_path.read_text(encoding="utf-8").split(",")
            if v.strip()
        ]
    order = orders[column]

    if isinstance(vector, str):
        vector = json.loads(vector)

    if len(vector) != len(order):
        raise ValueError(f"Expected {len(order)} values, got {len(vector)}")

    if any(value not in (0, 1) for value in vector):
        raise ValueError("The vector must contain only 0 and 1")

    return [tag for tag, value in zip(order, vector) if value == 1]

def get_data(target_pipeline_stage=None):
    folder = "data"
    os.makedirs("data", exist_ok=True)

    def get_raw_data(data):
        return pd.read_csv("games_march2025_cleaned.csv")

    def select_data(data):
        return data[
            [
                "appid",
                "name",
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
        return encode_multi_hot(data, "tags")
        
    def encode_genres(data):
        return encode_multi_hot(data, "genres")

    def encode_categories(data):
        return encode_multi_hot(data, "categories")

    def filter_low_number_of_reviews(data):
        data = data[
            (data["positive"] + data["negative"]) >= 10
        ]
        return data

    def review_score(data):
        def score(row):

            positive_ratio = row["positive"] / (row["positive"] + row["negative"])
            if positive_ratio >= 0.95:
                return 5
            elif positive_ratio >= 0.8:
                return 4
            elif positive_ratio >= 0.7:
                return 3
            elif positive_ratio >= 0.4:
                return 2
            elif positive_ratio >= 0.2:
                return 1
            else:
                return 0

        data["review_score"] = data.apply(score, axis=1)
        return data.drop(columns=["positive_ratio"])
    
    def expand_columns(data):
        for column in ["tags", "genres", "categories"]:
            data[column] = data[column].apply(lambda v: json.loads(v))
            expanded = pd.DataFrame(
                data[column].tolist(),
                index=data.index
            ).add_prefix(f"{column}_")
            data = data.drop(columns=[column]).join(expanded)

        return data

    pipeline = [
        ("games_march2025_cleaned", get_raw_data),
        ("games_march2025_cleaned_selected_columns", select_data),
        ("games_march2025_cleaned_simplified_tags", simplify_tags),
        ("games_march2025_cleaned_filter_low_number_of_reviews", filter_low_number_of_reviews),
        ("games_march2025_cleaned_multi_hot_tags", encode_tags),
        ("games_march2025_cleaned_multi_hot_genres", encode_genres),
        ("games_march2025_cleaned_multi_hot_categories", encode_categories),
        ("games_march2025_cleaned_review_scored", review_score),
        ("games_march2025_cleaned_expanded_columns", expand_columns)
    ]

    if target_pipeline_stage is None:
        target_step = len(pipeline) - 1
    else:
        pipeline_stages = [stage for stage, _ in pipeline]

        if target_pipeline_stage not in pipeline_stages:
            raise ValueError(f"Unknown pipeline stage: {target_pipeline_stage}")

        target_step = pipeline_stages.index(target_pipeline_stage)

    latest_data = None
    latest_step = -1
    for step in reversed(range(target_step + 1)):
        file_name, func = pipeline[step]
        path = f"{folder}/{file_name}.csv"
        if os.path.exists(path):
            latest_data = pd.read_csv(path)
            latest_step = step
            break

    for step in range(latest_step + 1, target_step + 1):
        file_name, func = pipeline[step]
        path = f"{folder}/{file_name}.csv"
        data = func(latest_data)
        data.to_csv(path, index=False)
        latest_data = data

    return latest_data
