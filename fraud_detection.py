import warnings
warnings.filterwarnings("ignore")

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from pathlib import Path
from typing import Any, cast

import pandas as pd

try:
    import matplotlib
    matplotlib.use("TkAgg", force=True)
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
# Removed unused import: Colormap

from sklearn.model_selection import train_test_split, GridSearchCV  # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.metrics import (
    classification_report,  # type: ignore
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

from imblearn.over_sampling import SMOTE  # type: ignore
from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore

# -------------------------------
# ==================== Project Configuration ====================
# -------------------------------

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "creditcard.csv"
OUTPUT_DIR = BASE_DIR

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# -------------------------------
# ==================== Dataset Loading ====================
# -------------------------------

def load_data(path: Path) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"{path} not found. Please place creditcard.csv inside the project folder."
        )

    df = pd.read_csv(path)

    print("\nDataset loaded successfully.")
    print("Shape :", df.shape)

    return df


# -------------------------------
# Data Exploration
# -------------------------------

def explore_data(df: pd.DataFrame) -> None:

    print("\nFirst 5 Rows")
    print(df.head())

    print("\nDataset Shape")
    print(df.shape)

    print("\nDataset Information")
    df.info()

    print("\nStatistical Summary")
    print(df.describe())

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Records :", df.duplicated().sum())

    print("\nFraud Distribution")
    print(df["Class"].value_counts())

    print("\nPercentage Distribution")
    print(df["Class"].value_counts(normalize=True) * 100)


# -------------------------------
# Exploratory Data Analysis
# -------------------------------

def plot_eda(df: pd.DataFrame) -> None:

    fig: Figure
    ax: Axes

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.countplot(data=df, x="Class", ax=ax)
    ax.set_title("Fraud vs Genuine Transactions")  # type: ignore
    fig.savefig(str(OUTPUT_DIR / "class_distribution.png"))  # type: ignore
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="Amount", bins=50, kde=True, ax=ax)
    ax.set_title("Transaction Amount Distribution")  # type: ignore
    fig.savefig(str(OUTPUT_DIR / "amount_distribution.png"))  # type: ignore
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="Time", bins=50, kde=True, ax=ax)
    ax.set_title("Transaction Time Distribution")  # type: ignore
    fig.savefig(str(OUTPUT_DIR / "time_distribution.png"))  # type: ignore
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Amount", ax=ax)
    ax.set_title("Transaction Amount Boxplot")  # type: ignore
    fig.savefig(str(OUTPUT_DIR / "amount_boxplot.png"))  # type: ignore
    plt.show()
    plt.close(fig)

    corr = df.corr().round(2)
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)  # type: ignore
    ax.set_title("Correlation Heatmap")  # type: ignore
    fig.savefig(str(OUTPUT_DIR / "correlation_heatmap.png"))  # type: ignore
    plt.show()
    plt.close(fig)

    print("\nEDA graphs saved successfully.")


# -------------------------------
# Data Preparation
# -------------------------------

def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    df = df.drop_duplicates()

    X: pd.DataFrame = df.drop("Class", axis=1)

    y: pd.Series = df["Class"]

    # Provide explicit type annotations so static type checkers can infer types
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    X_train, X_test, y_train, y_test = cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    # Returned values already have correct types; remove redundant casts

    print("\nTraining Shape :", X_train.shape)
    print("Testing Shape :", X_test.shape)

    return X_train, X_test, y_train, y_test
# -------------------------------
# Machine Learning Pipelines
# -------------------------------

def build_pipelines():

    logistic_pipeline = ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE
                ),
            ),
        ]
    )

    random_forest_pipeline = ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                ),
            ),
        ]
    )

    return logistic_pipeline, random_forest_pipeline


# -------------------------------
# Hyperparameter Tuning
# -------------------------------

def tune_random_forest(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:

    param_grid = {
        "classifier__n_estimators": [100],
        "classifier__max_depth": [10],
        "classifier__min_samples_split": [2],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=2,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)  # type: ignore

    best_params: dict[str, Any] = grid_search.best_params_  # type: ignore

    print("\nBest Parameters")
    print(best_params)

    return grid_search.best_estimator_


# -------------------------------
# Model Evaluation
# -------------------------------

def evaluate_model(model_name: str, model: Any, X_test: Any, y_test: Any, cmap: str) -> dict[str, Any]:

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    report_str = cast(str, classification_report(y_test, predictions, output_dict=False))
    print(report_str)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(  # type: ignore
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        ax=ax
    )

    ax.set_title(f"{model_name} Confusion Matrix")  # type: ignore
    ax.set_xlabel("Predicted")  # type: ignore
    ax.set_ylabel("Actual")  # type: ignore

    fig.tight_layout()

    fig.savefig(  # type: ignore
        str(
            OUTPUT_DIR /
            f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        )
    )

    plt.show()  # type: ignore
    plt.close(fig)

    # ---------------- ROC Curve ----------------
    fpr, tpr, _ = roc_curve(y_test, probabilities)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")  # type: ignore
    ax.plot([0, 1], [0, 1], "r--", label="Random")  # type: ignore

    ax.set_xlabel("False Positive Rate")  # type: ignore
    ax.set_ylabel("True Positive Rate")  # type: ignore
    ax.set_title(f"{model_name} ROC Curve")  # type: ignore
    ax.legend()  # type: ignore

    fig.tight_layout()

    fig.savefig(  # type: ignore
        str(
            OUTPUT_DIR /
            f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
        )
    )

    plt.show()  # type: ignore
    plt.close(fig)

    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
    }
# -------------------------------
# ==================== Feature Importance Analysis ====================
# -------------------------------

def plot_feature_importance(model: Any, feature_names: Any) -> None:

    classifier_model: Any = model

    if hasattr(model, "named_steps"):
        classifier_model = model.named_steps["classifier"]

    feature_importances = getattr(classifier_model, "feature_importances_", None)
    if feature_importances is None:
        raise AttributeError("The selected model does not expose feature_importances_.")

    importance = pd.Series(
        feature_importances,
        index=list(feature_names),
    ).sort_values(ascending=False)

    # CSV Save
    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        header=["Importance"]
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    importance.head(15).plot(
        kind="bar",
        ax=ax,
        color="steelblue"
    )

    ax.set_title("Top 15 Feature Importances")
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")

    fig.tight_layout()

    # Save Image
    fig.savefig(str(OUTPUT_DIR / "feature_importance.png"))

    # Show Graph
    plt.show(block=True)

    # Close Figure
    plt.close(fig)


# -------------------------------
# Main Function
# -------------------------------

def main():

    print("=" * 60)
    print("Fraud Detection using Machine Learning")
    print("=" * 60)

    df = load_data(DATA_PATH)

    explore_data(df)

    plot_eda(df)

    X_train, X_test, y_train, y_test = prepare_data(df)

    logistic_model, random_forest_model = build_pipelines()

    print("\nTraining Logistic Regression Model...")
    logistic_model.fit(X_train, y_train)  # type: ignore

    logistic_results = evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_test,
        y_test,
        "Blues"
    )

    print("\nTraining Random Forest Model...")
    random_forest_model.fit(X_train, y_train)  # type: ignore

    random_forest_results = evaluate_model(
        "Random Forest",
        random_forest_model,
        X_test,
        y_test,
        "Greens"
    )

    print("\nRunning GridSearchCV...")

    best_model = tune_random_forest(
        random_forest_model,
        X_train,
        y_train
    )

    tuned_results = evaluate_model(
        "Tuned Random Forest",
        best_model,
        X_test,
        y_test,
        "Purples"
    )

    comparison = pd.DataFrame([
        logistic_results,
        random_forest_results,
        tuned_results
    ])

    comparison = comparison.round(4)

    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    print("\nModel Comparison")
    print(comparison)

    plot_feature_importance(
        best_model,
        list(X_train.columns)
    )

    best = comparison.loc[
        comparison["roc_auc"].idxmax()
    ]

    print("\n" + "=" * 60)
    print("PROJECT SUMMARY")
    print("=" * 60)

    print(f"Best Model : {best['model']}")
    print(f"Precision : {best['precision']}")
    print(f"Recall    : {best['recall']}")
    print(f"F1 Score  : {best['f1_score']}")
    print(f"ROC-AUC   : {best['roc_auc']}")

    print("\nFiles Generated Successfully")
    print("- class_distribution.png")
    print("- amount_distribution.png")
    print("- time_distribution.png")
    print("- amount_boxplot.png")
    print("- correlation_heatmap.png")
    print("- feature_importance.png")
    print("- feature_importance.csv")
    print("- model_comparison.csv")

print("\n" + "=" * 60)
print("Fraud Detection Pipeline Executed Successfully!")
print("All analyses, visualizations, and model evaluations completed.")
print("=" * 60)
# -------------------------------
# Run Project
# -------------------------------

if __name__ == "__main__":
    main()