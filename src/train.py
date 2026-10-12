import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.dataset import load_data
from src.features.build_features import build_preprocessor, extract_target_and_features
def train(args):
    # 1. Cargar y preparar datos
    df_raw = load_data(args.data_path)
    X, y = extract_target_and_features(df_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # 2. Construir Pipeline
    preprocessor = build_preprocessor()
    classifier = LogisticRegression(
        C=args.c,
        solver=args.solver,
        penalty=args.penalty,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

    # 3. Configurar y registrar en MLflow
    mlflow.set_experiment(args.experiment_name)
    run_name = f"run_C{args.c}_{args.solver}_{args.penalty}_seed{args.seed}"

    with mlflow.start_run(run_name=run_name):
        # Registrar hiperparámetros
        mlflow.log_params({
            "C": args.c, "solver": args.solver, "penalty": args.penalty,
            "max_iter": args.max_iter, "seed": args.seed, "test_size": args.test_size,
        })

        # Entrenar
        pipeline.fit(X_train, y_train)

        # Predecir y evaluar
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        # Artefacto: Matriz de Confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Sano", "Cardiopatía"], yticklabels=["Sano", "Cardiopatía"])
        plt.title(f"Matriz de Confusión ({run_name})")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        artifact_img = "confusion_matrix.png"
        plt.savefig(artifact_img, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(artifact_img)
        Path(artifact_img).unlink(missing_ok=True)

        # Registrar Modelo
        mlflow.sklearn.log_model(
            pipeline, 
            "model_pipeline",
            serialization_format="cloudpickle"
        )
        print(f"\\n[ÉXITO] Corrida '{run_name}' registrada en MLflow.")
        print(f"        Recall: {metrics['recall']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline CLI con MLflow")
    parser.add_argument("--c", type=float, default=1.0, help="Inverso de regularización (float)")
    parser.add_argument("--solver", type=str, default="liblinear", choices=["liblinear", "lbfgs", "saga"])
    parser.add_argument("--penalty", type=str, default="l2", choices=["l1", "l2"])
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--data_path", type=str, default="data/raw/heart_disease.csv")
    parser.add_argument("--experiment_name", type=str, default="Heart_Disease_Experiment")

    args = parser.parse_args()
    train(args)