
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
import utils

# ==========================================
# 1. LOAD DATA
# ==========================================

data = utils.get_data()

# ==========================================
# 2. CREATE TARGET CLASSES
# ==========================================

data["target"] = pd.cut(
    data["positive_ratio"],
    bins=[-np.inf, 0.4, 0.7, np.inf],
    labels=[0, 1, 2],
    right=False
)

# Usunięcie wierszy bez określonej wartości docelowej
data = data.dropna(subset=["target"])

data["target"] = data["target"].astype(int)

print("Class distribution:")
print(data["target"].value_counts().sort_index())


# ==========================================
# 3. PREPARE FEATURES
# ==========================================

X = data.drop(columns=[
    "target",
    "positive_ratio",
    "appid",
    "name"
])

y = data["target"]

# Zamiana ewentualnych wartości nieskończonych na NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Usunięcie kolumn tekstowych, które nie zostały zakodowane
X = X.select_dtypes(include=[np.number])

# Uzupełnienie brakujących wartości medianą
X = X.fillna(X.median())


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. DEFINE MODELS
# ==========================================

models = {
    "k-NN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf")
    ),

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    )
}


# ==========================================
# 6. TRAIN AND EVALUATE MODELS
# ==========================================

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining: {name}")

    # Training
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    trained_models[name] = model

    # Metrics
    results.append({
        "Model": name,
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
    })

    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Negative", "Mixed", "Positive"],
        zero_division=0
    ))


# ==========================================
# 7. COMPARE MODELS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
)

print("\nMODEL COMPARISON:")
print(results_df.to_string(index=False))


# ==========================================
# 8. PLOT MODEL COMPARISON
# ==========================================

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1"]
].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Classification Models Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# ==========================================
# 9. CONFUSION MATRICES
# ==========================================

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for ax, (name, model) in zip(axes.flat, trained_models.items()):

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1, 2]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Mixed", "Positive"],
        yticklabels=["Negative", "Mixed", "Positive"],
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

# Ukrycie niewykorzystanego wykresu
axes.flat[-1].axis("off")

plt.tight_layout()
plt.show()


# ==========================================
# 10. BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print(f"\nBest model: {best_model_name}")
print(f"F1 macro: {results_df.iloc[0]['F1']:.4f}")
