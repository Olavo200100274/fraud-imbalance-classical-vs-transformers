"""
Preprocessing module — builds a ColumnTransformer appropriate for the
features present in the training data.

- Numeric features → mean imputation + StandardScaler
- Categorical features (if any) → most-frequent imputation + OneHotEncoder
"""

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import sklearn
from missing_values import numeric_pipeline
sklearn.set_config(transform_output="pandas")


def get_preprocessor(X_train, missing_policy="preserve"):
    """
    Build a ColumnTransformer that handles both numeric and categorical
    features.  If the data has no categorical columns (e.g. ULB), only
    the numeric branch is created.
    """
    numeric_features = X_train.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Numeric branch: impute + scale
    numeric_transformer = numeric_pipeline(missing_policy)

    transformers = [
        ("num", numeric_transformer, numeric_features),
    ]

    # Categorical branch (only if categorical columns exist)
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(
            ("cat", categorical_transformer, categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        verbose_feature_names_out=False,
    )
    return preprocessor
