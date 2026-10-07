"""Fold-fitted categorical-aware resampling with unchanged CatBoost OHE inputs.

The intermediate ordinal codes are identifiers, never continuous predictors
for SMOTE-NC. The final classifier still receives standardised numerical
features and one-hot categorical features, as in the primary CatBoost study.
These classes live in an importable module so saved pipelines remain loadable.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from missing_values import numeric_pipeline


class MixedCategoryPreprocessor(TransformerMixin, BaseEstimator):
    """Standardise numerical fields and identify categories using train only."""

    def fit(self, X, y=None):
        if not hasattr(X, "columns") or not X.columns.is_unique:
            raise TypeError("Categorical control requires a named, unique-column DataFrame.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.numeric_features_ = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
        self.categorical_features_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if not self.numeric_features_ or not self.categorical_features_:
            raise ValueError("SMOTE-NC control requires both numerical and categorical features.")
        if set(self.numeric_features_ + self.categorical_features_) != set(X.columns):
            raise ValueError("A predictor dtype differs from the primary preprocessing schema.")
        self.numeric_transformer_ = numeric_pipeline("preserve").set_output(transform="default")
        self.numeric_transformer_.fit(X[self.numeric_features_])
        self.categorical_imputer_ = SimpleImputer(strategy="most_frequent").set_output(transform="default")
        categories = self.categorical_imputer_.fit_transform(X[self.categorical_features_])
        self.ordinal_encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float64,
        ).set_output(transform="default")
        self.ordinal_encoder_.fit(categories)
        self.categorical_cardinalities_ = [len(values) for values in self.ordinal_encoder_.categories_]
        self.categorical_indices_ = list(range(len(self.numeric_features_), self.n_features_in_))
        return self

    def transform(self, X):
        check_is_fitted(self, "ordinal_encoder_")
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Predictor order differs from the fitted categorical-control schema.")
        numerical = self.numeric_transformer_.transform(X[self.numeric_features_])
        categories = self.categorical_imputer_.transform(X[self.categorical_features_])
        categorical = self.ordinal_encoder_.transform(categories)
        result = np.hstack((numerical, categorical)).astype(np.float64, copy=False)
        if not np.isfinite(result).all():
            raise ValueError("Non-finite predictor in the mixed categorical representation.")
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "ordinal_encoder_")
        return np.asarray(self.numeric_features_ + self.categorical_features_, dtype=object)


def validate_category_codes(values, categorical_indices, cardinalities, *, allow_unknown=False):
    """Reject fractional, out-of-vocabulary or silently truncated categories."""
    values = np.asarray(values)
    categorical = values[:, list(categorical_indices)]
    if not np.isfinite(categorical).all() or not np.all(categorical == np.rint(categorical)):
        raise ValueError("Categorical resampling produced fractional or non-finite category codes.")
    lower = -1 if allow_unknown else 0
    for column, cardinality in enumerate(cardinalities):
        if np.any(categorical[:, column] < lower) or np.any(categorical[:, column] >= cardinality):
            raise ValueError("Categorical resampling produced an out-of-vocabulary code.")
    return categorical


class CategoryOneHotRepresentation(TransformerMixin, BaseEstimator):
    """Restore primary OHE representation after nominal-aware resampling.

The complete vocabulary comes from the original fold training data. Unknown
validation/test codes -1 map to all-zero OHE blocks, matching the primary
OneHotEncoder(handle_unknown='ignore') behaviour. No validation/test fitting.
"""

    def __init__(self, numeric_features, categorical_features, cardinalities):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.cardinalities = cardinalities

    def fit(self, X, y=None):
        values = np.asarray(X)
        self.n_features_in_ = len(self.numeric_features) + len(self.categorical_features)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Mixed predictor matrix differs from its categorical-control schema.")
        self.categorical_indices_ = list(range(len(self.numeric_features), self.n_features_in_))
        categories = validate_category_codes(values, self.categorical_indices_, self.cardinalities)
        self.encoder_ = OneHotEncoder(
            categories=[np.arange(size, dtype=np.float64) for size in self.cardinalities],
            handle_unknown="ignore", sparse_output=False,
        ).set_output(transform="default")
        self.encoder_.fit(categories)
        return self

    def transform(self, X):
        check_is_fitted(self, "encoder_")
        values = np.asarray(X)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Mixed predictor matrix differs from its fitted OHE schema.")
        categorical = validate_category_codes(
            values, self.categorical_indices_, self.cardinalities, allow_unknown=True,
        )
        return np.hstack((values[:, :len(self.numeric_features)], self.encoder_.transform(categorical)))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "encoder_")
        return np.concatenate((np.asarray(self.numeric_features, dtype=object),
                               self.encoder_.get_feature_names_out(self.categorical_features)))
