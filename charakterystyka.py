import ast
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import utils


def format_label(value):
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, list):
        text = ", ".join(map(str, parsed))
    elif isinstance(parsed, dict):
        text = ", ".join(f"{key}: {count}" for key, count in parsed.items())

    return textwrap.fill(text, width=45)


folder = utils.prepare_build_dir("charakterystyka")
folder_wykresy = utils.prepare_build_dir("wykresy")
data = utils.get_data()


with open(f"{folder}/charakterystyka.txt", "w") as f:
    for column in data.columns:
        if column == "appid":
            continue

        column_data = data[column]

        min = None
        max = None
        mean = None
        median = None
        zero_values = None
        if column_data.dtype != "object":
            min = column_data.min()
            max = column_data.max()
            mean = column_data.mean()
            median = column_data.median()
            zero_values = len(column_data[column_data == 0])

        print(column, file=f)
        print("Minimum:", min, file=f)
        print("Maximum:", max, file=f)
        print("Średnia:", mean, file=f)
        print("Mediana:", median, file=f)
        print("Ilość wartości zerowych:", zero_values, file=f)
        print("", file=f)

folder = folder_wykresy
for column in data.columns:
    if column == "appid" or column == "name":
        continue

    if data[column].dtype == "object" or data[column].dtype == "string":
        unique_count = data[column].nunique()
        print(f"{column}: {unique_count} unique elements")

        counts = data[column][data[column] != "[]"].value_counts().head(100)
        labels = [format_label(label) for label in counts.index]
        total_lines = sum(label.count("\n") + 1 for label in labels)
        height = 6 if total_lines <= 20 else total_lines * 0.3
        plt.figure(figsize=(16, height))
        counts.plot(kind="barh")

        plt.title(f"Top 30 frequencies of {column}")
        plt.xlabel("Count")
        plt.ylabel(column)
        plt.yticks(range(len(counts)), labels, fontsize=8)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{folder}/bar_{column}.png", dpi=150)
        plt.close()

        continue

    values = data[column][data[column] != 0]
    log_values = np.log1p(values)

    plt.figure()
    plt.hist(log_values, bins="auto", edgecolor="black")
    plt.title(f"Histogram {column} (w skali logarytmicznej)")
    plt.xlabel(f"log(1 + {column})")
    plt.ylabel("Frequency")
    plt.savefig(f"{folder}/histogram_{column}_log.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.boxplot(log_values)
    plt.title(f"Wykres pudełkowy {column} (w skali logarytmicznej)")
    plt.ylabel(f"log(1 + {column})")
    plt.savefig(f"{folder}/boxplot_{column}_log.png", dpi=300, bbox_inches="tight")
    plt.close()
