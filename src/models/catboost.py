from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline


def get_pipeline_and_params(preprocessor):
    """Return (pipeline, suggest_fn) for CatBoost.

    Search space (Optuna):
    - learning_rate: log-uniform [0.005, 0.3] — step size shrinkage.
    - depth: [4, 10] — tree depth.
    - l2_leaf_reg: log-uniform [0.1, 10] — L2 regularisation.

    iterations is set high (2000); actual count determined by
    early stopping during Optuna tuning.
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', CatBoostClassifier(
            iterations=2000, silent=True, random_state=42)),
    ])

    def suggest_params(trial):
        return {
            "classifier__learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "classifier__depth": trial.suggest_int("depth", 4, 10),
            "classifier__l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        }

    return pipeline, suggest_params
