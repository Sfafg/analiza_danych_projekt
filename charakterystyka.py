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

    if isinstance(parsed, (list, tuple)):
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
        column_data = data[column][data[column] != "[]"]
        if column == "tags":
            column_data = column_data.apply(utils.decode_tags)
            column_data = column_data.apply(tuple)
            column_data = column_data[column_data != ()]

        counts = column_data.value_counts().head(30)
        labels = [format_label(label) for label in counts.index]
        total_lines = sum(label.count("\n") + 1 for label in labels)
        height = 6 if total_lines <= 20 else total_lines * 0.3
        plt.figure(figsize=(16, height))
        counts.plot(kind="barh")

        plt.title(f"Najczęstrze 30 {column}")
        plt.xlabel("Częstość")
        plt.ylabel(column)
        plt.yticks(range(len(counts)), labels, fontsize=8)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{folder}/bar_{column}.png", dpi=150)
        plt.close()

        continue

    use_log_scale = False if column in {"price"} else True

    values = data[column][data[column] != 0]

    n_bins = len(np.histogram_bin_edges(np.log10(values[values > 0]), bins="auto"))
    bins = np.logspace(np.log10(values.min()), np.log10(values.max()), n_bins)
    bins = bins if use_log_scale else "auto"

    plt.figure()
    plt.hist(values, bins=bins, edgecolor="black")
    plt.title(f"Histogram {column} (w skali logarytmicznej)")
    if use_log_scale:
        plt.xscale("log")
    else:
        plt.yscale("log")
    plt.xlabel(f"{column}")
    plt.ylabel("Liczebność")
    plt.savefig(f"{folder}/histogram_{column}_log.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.boxplot(values)
    plt.title(f"Wykres pudełkowy {column} (w skali logarytmicznej)")
    if use_log_scale:
        plt.yscale("log")
    plt.ylabel(f"{column}")
    plt.savefig(f"{folder}/boxplot_{column}_log.png", dpi=300, bbox_inches="tight")
    plt.close()
