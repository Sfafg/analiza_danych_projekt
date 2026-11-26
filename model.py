
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

labels = [5, 4, 3, 2, 1, 0]

class_names = [
    "Overwhelmingly Positive",
    "Very Positive",
    "Mostly Positive",
    "Mixed",
    "Mostly Negative",
    "Overwhelmingly Negative"
]

def load_data():
    data = utils.get_data()
    X = data.drop(columns=["review_score", "appid", "name"])
    y = data["review_score"]
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_and_evaluate_model(model, train_data):
    X_train, X_test, y_train, y_test = train_data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "F1": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        )
    }

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0
    )
    return y_pred, metrics, report

def log_report(file, report, metrics):
    with open(file, "a", encoding="utf-8") as log:
        print(f"Report for {metrics["Model"]}", file=log)
        print(report, file=log)
        print("Metrics:", file=log)
        for metric, value in metrics.items():
            if metric != "Model":
                print(f"{metric}: {value:.4f}", file=log)
        print("",file=log)

def log_metrics(file, results):
    with open(file, "a", encoding="utf-8") as log:
        results_df = pd.DataFrame(results).sort_values(
            by="F1",
            ascending=False
        )
        print(results_df.to_string(index=False), file=log)

def plot_comparison(file, results):
    results_df = pd.DataFrame(results).sort_values(
        by="F1",
        ascending=False
    )

    results_df.set_index("Model").plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title("Classification Models Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(
        file,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

def plot_confusion_matrices(file, predictions, train_data):
    X_train, X_test, y_train, y_test = train_data
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for ax, (name, y_pred) in zip(axes.flat, predictions.items()):
        sns.heatmap(
            confusion_matrix(y_test, y_pred, labels=labels),
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )

        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        file,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()