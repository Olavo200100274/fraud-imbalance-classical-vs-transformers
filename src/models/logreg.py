from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def get_pipeline_and_params(preprocessor):
    """Return (pipeline, suggest_fn) for Logistic Regression.

    Search space (Optuna):
    - C: log-uniform [1e-4, 100] — regularisation strength.
    - l1_ratio: uniform [0, 1] — 0 = L2, 1 = L1, in between = ElasticNet.
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        # scikit-learn 1.8 uses l1_ratio to select L2/L1/Elastic Net; omitting
        # the deprecated penalty argument is deliberate and version-pinned.
        ('classifier', LogisticRegression(
            max_iter=5000, solver='saga', l1_ratio=0.0, random_state=42,
        )),
    ])

    def suggest_params(trial):
        return {
            "classifier__C": trial.suggest_float("C", 1e-4, 100, log=True),
            "classifier__l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }

    return pipeline, suggest_params
