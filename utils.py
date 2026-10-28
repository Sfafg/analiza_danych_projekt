import os
import pandas as pd


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

    pipeline = [
        ("games_march2025_cleaned", get_raw_data),
        ("games_march2025_cleaned_selected_columns", select_data),
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
