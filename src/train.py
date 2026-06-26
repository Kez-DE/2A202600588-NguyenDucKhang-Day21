import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

EVAL_THRESHOLD = 0.65
DRIFT_MIN_RATIO = 0.10  # Bonus 5: nguong canh bao lech lac du lieu


def build_model(params: dict):
    """
    Bonus 2: Chon thuat toan theo params['model_type'].
    Ho tro: random_forest (mac dinh), gradient_boosting, logistic_regression.
    Tra ve (model, model_type).
    """
    p = dict(params)
    model_type = p.pop("model_type", "random_forest")

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 100)),
            max_depth=p.get("max_depth", None),
            min_samples_split=int(p.get("min_samples_split", 2)),
            random_state=42,
        )
    elif model_type == "gradient_boosting":
        md = p.get("max_depth", 3)
        model = GradientBoostingClassifier(
            n_estimators=int(p.get("n_estimators", 100)),
            max_depth=int(md) if md is not None else 3,
            random_state=42,
        )
    elif model_type == "logistic_regression":
        # Chuan hoa dac trung truoc khi hoi quy logistic
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42),
        )
    else:
        raise ValueError(f"model_type khong ho tro: {model_type}")

    return model, model_type


def check_data_drift(y_train) -> dict:
    """
    Bonus 5: Tinh phan phoi nhan trong tap huan luyen va canh bao
    neu bat ky lop nao chiem < 10% tong mau.
    """
    dist = y_train.value_counts(normalize=True).sort_index()
    label_distribution = {str(int(k)): round(float(v), 4) for k, v in dist.items()}

    for cls, ratio in label_distribution.items():
        if ratio < DRIFT_MIN_RATIO:
            print(
                f"[CANH BAO DATA DRIFT] Lop {cls} chi chiem {ratio:.1%} "
                f"(< {DRIFT_MIN_RATIO:.0%}) tong mau huan luyen!"
            )
    print(f"Phan phoi nhan (train): {label_distribution}")
    return label_distribution


def write_perf_report(y_eval, preds, path="outputs/report.txt"):
    """
    Bonus 3: Ghi confusion matrix + precision/recall theo tung lop ra file van ban.
    """
    cm = confusion_matrix(y_eval, preds)
    cls_report = classification_report(y_eval, preds, digits=4, zero_division=0)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("=== Confusion Matrix (hang = thuc te, cot = du doan) ===\n")
        f.write(str(cm) + "\n\n")
        f.write("=== Classification Report (precision / recall / f1 theo lop) ===\n")
        f.write(cls_report + "\n")

    print("Confusion matrix:\n" + str(cm))
    print(cls_report)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """Huan luyen mo hinh va ghi nhan ket qua vao MLflow."""

    # Bonus 1: dung MLflow tracking server tu xa (vd DagsHub) neu co bien moi truong.
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
        print(f"MLflow tracking URI: {uri}")

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: kiem tra phan phoi nhan truoc khi huan luyen
    label_distribution = check_data_drift(y_train)

    with mlflow.start_run():
        mlflow.log_params(params)

        # Bonus 2: chon thuat toan theo model_type
        model, model_type = build_model(params)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.set_tag("model_type", model_type)
        mlflow.sklearn.log_model(model, "model")

        print(f"[{model_type}] Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Bonus 3: bao cao hieu suat (confusion matrix + precision/recall)
        os.makedirs("outputs", exist_ok=True)
        write_perf_report(y_eval, preds, "outputs/report.txt")
        mlflow.log_artifact("outputs/report.txt")

        # Luu metrics.json (Bonus 5: kem label_distribution)
        with open("outputs/metrics.json", "w") as f:
            json.dump(
                {
                    "accuracy": acc,
                    "f1_score": f1,
                    "model_type": model_type,
                    "label_distribution": label_distribution,
                },
                f,
                indent=2,
            )

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
