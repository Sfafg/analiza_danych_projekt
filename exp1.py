from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import model
import utils

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

folder = utils.prepare_build_dir("exp1")
results = []
predictions = {}
data = model.load_data()
for name, m in models.items():
    y_pred, metrics, report = model.train_and_evaluate_model(m, data)
    metrics["Model"] = name

    results.append(metrics)
    predictions[name] = y_pred
    model.log_report(folder+"/reports.txt", report, metrics)
    
model.log_metrics(folder+"/metryki.txt", results)
model.plot_comparison(folder+"/porownanie.png", results)
model.plot_confusion_matrices(folder+"/confusion_matrices.png", predictions, data)