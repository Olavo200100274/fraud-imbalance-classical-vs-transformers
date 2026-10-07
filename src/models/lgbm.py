import lightgbm as lgb
from sklearn.pipeline import Pipeline


def get_pipeline_and_params(preprocessor):
    """Return (pipeline, suggest_fn) for LightGBM.

    Search space (Optuna):
    - num_leaves: [15, 127] — tree complexity.
    - learning_rate: log-uniform [0.01, 0.3] — step size shrinkage.
    - min_child_samples: [5, 100] — min data in a leaf.
    - reg_alpha: log-uniform [1e-8, 10] — L1 regularisation.
    - reg_lambda: log-uniform [1e-8, 10] — L2 regularisation.
    - colsample_bytree: [0.5, 1.0] — column subsampling per tree.

    n_estimators is set high (2000); actual count determined by
    early stopping during Optuna tuning.
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(
            n_estimators=2000, random_state=42, verbosity=-1)),
    ])

    def suggest_params(trial):
        return {
            "classifier__num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "classifier__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "classifier__min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "classifier__reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "classifier__reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "classifier__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

    return pipeline, suggest_params
