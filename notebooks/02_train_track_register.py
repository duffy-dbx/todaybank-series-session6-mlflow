# Databricks notebook source
# MAGIC %md
# MAGIC # TodayBank Session 6 - Notebook 02: Train, Track & Register
# MAGIC
# MAGIC **Stage 01 CRAWL - Train once, track everything**
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads labeled loan data from Unity Catalog
# MAGIC 2. Prepares features (one-hot encode categoricals, scale numerics)
# MAGIC 3. Trains **3 MLflow-tracked runs** - each with different hyperparameters
# MAGIC 4. Compares runs in the MLflow Experiment UI
# MAGIC 5. Registers the best model to UC Model Registry as
# MAGIC    `todaybank_mlflow101.models.loan_default_risk` with alias `@champion`
# MAGIC
# MAGIC **Key concept:** Every training run is automatically logged - hyperparameters,
# MAGIC metrics, model artifacts, and data lineage. Examiners can reproduce any historical run.

# COMMAND ----------

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

spark = SparkSession.builder.getOrCreate()

CATALOG  = "todaybank_mlflow101"
SCHEMA   = "lending"
MODEL_UC = f"{CATALOG}.models.loan_default_risk"
EXPERIMENT_PATH = "/Users/duffy.walsh@databricks.com/todaybank-loan-default"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 - Load feature data from Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data preview - governed feature table in Unity Catalog
# MAGIC Scroll the grid: 10,000 labeled historical loan applications. This is the same
# MAGIC governed `lending.loan_applications` table from Session 1's medallion gold layer.

# COMMAND ----------

# Rich, scrollable preview of the governed feature table (for the live walkthrough)
display(spark.table(f"{CATALOG}.{SCHEMA}.loan_applications"))

# COMMAND ----------

df = spark.table(f"{CATALOG}.{SCHEMA}.loan_applications").toPandas()
print(f"Loaded {len(df):,} rows")
print(df["defaulted"].value_counts(normalize=True).rename("pct").to_frame())

FEATURES = [
    "credit_score", "annual_income", "dti_ratio", "loan_amount",
    "loan_term_months", "interest_rate", "employment_years",
    "num_prior_delinquencies", "home_ownership", "loan_purpose",
]
TARGET = "defaulted"

X = df[FEATURES]
y = df[TARGET].astype(int)

# Train / test split - stratified to preserve default rate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 - Build the feature preprocessing pipeline

# COMMAND ----------

# Numeric features: standard scaling
numeric_features = [
    "credit_score", "annual_income", "dti_ratio", "loan_amount",
    "loan_term_months", "interest_rate", "employment_years",
    "num_prior_delinquencies",
]

# Categorical features: one-hot encoding
categorical_features = ["home_ownership", "loan_purpose"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
])

print("Preprocessor configured.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 - Define the ProbaPredictor wrapper
# MAGIC
# MAGIC **Why?** Mosaic AI Model Serving uses the sklearn `predict()` method by default,
# MAGIC which returns class labels (0/1) - not probabilities. We want to return the
# MAGIC Probability of Default (PD) as a continuous score (0.0 - 1.0). The wrapper
# MAGIC overrides `predict()` to return `predict_proba(X)[:, 1]`.
# MAGIC
# MAGIC This uses the **prebuilt sklearn serving image** (~13 min deploy) instead of
# MAGIC a slow custom container build.

# COMMAND ----------

class ProbaPredictor:
    """
    Sklearn-compatible wrapper that makes predict() return default probabilities.
    Required for Mosaic AI Model Serving to return PD scores (not 0/1 labels).
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        # predict_proba returns [[p_no_default, p_default], ...]
        # We return the probability of default (column index 1)
        return self.pipeline.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def get_params(self, deep=True):
        return self.pipeline.get_params(deep=deep)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 - Train 3 runs with MLflow tracking
# MAGIC
# MAGIC Each run varies the GradientBoosting hyperparameters.
# MAGIC MLflow auto-logs: parameters, metrics, model artifacts, Python env.

# COMMAND ----------

# Set a named experiment so all runs are grouped together
mlflow.set_experiment(EXPERIMENT_PATH)
print(f"MLflow experiment: {EXPERIMENT_PATH}")

# --- Three hyperparameter configurations to compare ---
CONFIGS = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,  "subsample": 0.8},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
    {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.08, "subsample": 0.9},
]

run_results = []

for i, params in enumerate(CONFIGS, start=1):
    run_name = f"gbm_run_{i}_n{params['n_estimators']}_d{params['max_depth']}"
    print(f"\n--- Starting Run {i}: {run_name} ---")

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "GradientBoostingClassifier")
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("features", ",".join(FEATURES))
        mlflow.log_param("use_case", "loan_default_risk")

        # Build and train the wrapped pipeline
        gbm = GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            random_state=42,
        )
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", gbm),
        ])

        wrapper = ProbaPredictor(pipeline)
        wrapper.fit(X_train, y_train)

        # Evaluate on held-out test set
        y_pred_proba = wrapper.predict(X_test)
        y_pred_class = (y_pred_proba >= 0.5).astype(int)

        auc_roc  = roc_auc_score(y_test, y_pred_proba)
        avg_prec = average_precision_score(y_test, y_pred_proba)
        brier    = brier_score_loss(y_test, y_pred_proba)

        print(f"  AUC-ROC: {auc_roc:.4f}  |  Avg Precision: {avg_prec:.4f}  |  Brier: {brier:.4f}")

        # Log metrics
        mlflow.log_metric("auc_roc", auc_roc)
        mlflow.log_metric("avg_precision", avg_prec)
        mlflow.log_metric("brier_score", brier)

        # Log model - use artifact_path= (not name=) for serverless MLflow
        # The wrapper has a sklearn-compatible interface so use the sklearn flavor
        # to get the prebuilt serving image (avoids custom container build)
        mlflow.sklearn.log_model(
            sk_model=wrapper,
            artifact_path="loan_default_model",
            input_example=X_test.iloc[:5],
        )

        run_id = mlflow.active_run().info.run_id

        run_results.append({
            "run": i,
            "run_id": run_id,
            "run_name": run_name,
            "params": params,
            "auc_roc": auc_roc,
            "avg_precision": avg_prec,
            "brier": brier,
        })

        print(f"  Run ID: {run_id}")

print("\n=== Run comparison ===")
results_df = pd.DataFrame(run_results)[["run", "run_name", "auc_roc", "avg_precision", "brier"]]
print(results_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 - Select the best run and register to UC Model Registry
# MAGIC
# MAGIC Best run = highest AUC-ROC.
# MAGIC We register it as `todaybank_mlflow101.models.loan_default_risk`
# MAGIC and assign the `@champion` alias - the pointer that serving and scoring notebooks use.

# COMMAND ----------

best_run = max(run_results, key=lambda r: r["auc_roc"])
print(f"Best run: {best_run['run_name']}  (AUC-ROC = {best_run['auc_roc']:.4f})")
print(f"Run ID:   {best_run['run_id']}")

# Register the model to Unity Catalog
model_uri = f"runs:/{best_run['run_id']}/loan_default_model"
print(f"\nRegistering {model_uri} -> {MODEL_UC}")

mv = mlflow.register_model(model_uri=model_uri, name=MODEL_UC)
print(f"Registered: version {mv.version}")

# COMMAND ----------

# Assign the @champion alias so downstream code can always reference "champion"
# regardless of the version number.
#
# ROOT CAUSE NOTE: MlflowClient().set_registered_model_alias() sets the alias only
# in the MLflow tracking layer (/api/2.0/mlflow/unity-catalog/...).  The UC catalog
# API (/api/2.1/unity-catalog/models/{name}?include_aliases=true) requires a separate
# PUT to /api/2.1/unity-catalog/models/{name}/aliases/{alias}.  We call BOTH so the
# alias is visible in every surface (MLflow client, UC Explorer, and the UC API).
from mlflow import MlflowClient
import requests as _requests

client = MlflowClient()

# 1. MLflow tracking layer (needed for mlflow.sklearn.load_model("models:/...@champion"))
client.set_registered_model_alias(
    name=MODEL_UC,
    alias="champion",
    version=mv.version,
)
print(f"MLflow alias @champion -> version {mv.version} of {MODEL_UC}")

# 2. UC catalog API layer (needed for the UC catalog UI and /api/2.1/unity-catalog/models API)
_workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
_alias_url = (
    f"https://{_workspace_host}/api/2.1/unity-catalog/models"
    f"/{MODEL_UC}/aliases/champion"
)
_r = _requests.put(
    _alias_url,
    headers={"Authorization": f"Bearer {_token}", "Content-Type": "application/json"},
    json={"alias_name": "champion", "version_num": int(mv.version)},
)
if _r.status_code in (200, 201):
    print(f"UC alias @champion -> version {mv.version}  (UC catalog layer confirmed)")
else:
    raise RuntimeError(f"UC alias PUT failed {_r.status_code}: {_r.text}")

# Verify via include_aliases=true
_verify = _requests.get(
    f"https://{_workspace_host}/api/2.1/unity-catalog/models/{MODEL_UC}?include_aliases=true",
    headers={"Authorization": f"Bearer {_token}"},
).json()
_aliases = _verify.get("aliases", [])
print(f"Verified aliases on model: {_aliases}")

# COMMAND ----------

# Optionally add a tag and description so the registry UI shows context
client.update_registered_model(
    name=MODEL_UC,
    description=(
        "TodayBank consumer loan default risk model. "
        "Predicts probability of default (PD) for personal loan applicants. "
        "Session 6 MLflow 101 demo - for demonstration purposes only."
    ),
)
client.set_model_version_tag(
    name=MODEL_UC,
    version=mv.version,
    key="use_case",
    value="loan_default_risk",
)
client.set_model_version_tag(
    name=MODEL_UC,
    version=mv.version,
    key="audience",
    value="zero_ml_intro_demo",
)
print("Model description and tags set.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Item | Value |
# MAGIC |---|---|
# MAGIC | Experiment | /Users/duffy.walsh@databricks.com/todaybank-loan-default |
# MAGIC | Runs trained | 3 (GradientBoosting, varying n_estimators / max_depth / learning_rate) |
# MAGIC | Best run AUC-ROC | see output above |
# MAGIC | Registered model | todaybank_mlflow101.models.loan_default_risk |
# MAGIC | Active alias | @champion |
# MAGIC
# MAGIC **Next:** Notebook 03 will deploy this model to Mosaic AI Model Serving.

# COMMAND ----------

print("=== Notebook 02 complete ===")
print(f"Best model: {MODEL_UC}  version {mv.version}  alias @champion")
print(f"AUC-ROC: {best_run['auc_roc']:.4f}  |  Best run ID: {best_run['run_id']}")
