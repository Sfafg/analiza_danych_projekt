import matplotlib.pyplot as plt
import numpy as np
import utils

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
    if column == "appid":
        continue

    if data[column].dtype == "object":
        continue
        plt.figure()
        data[column].value_counts().plot(kind="bar")
        plt.title(f"Frequency of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{folder}/bar_{column}.png", dpi=300)
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
