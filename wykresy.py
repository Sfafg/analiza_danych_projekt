import matplotlib.pyplot as plt
import numpy as np
import utils

folder = utils.prepare_build_dir("wykresy")
data = utils.get_data()

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
