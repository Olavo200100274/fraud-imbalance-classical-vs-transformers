"""Explicit BAF numerical absence coding and predefined sensitivity policy.

Only fields documented as using absence codes are modified. Negative values in
credit_risk_score and velocity fields are not automatically treated as missing.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BAF_MINUS_ONE_FIELDS = (
    "prev_address_months_count", "current_address_months_count",
    "bank_months_count", "session_length_in_minutes", "device_distinct_emails_8w",
)
BAF_NEGATIVE_MISSING_FIELDS = ("intended_balcon_amount",)
MISSING_POLICIES = ("preserve", "nan_indicators")


class BafAbsenceToNaN(TransformerMixin, BaseEstimator):
    """Convert only documented absence codes, preserving names and row order."""

    def fit(self, X, y=None):
        if not hasattr(X, "columns"):
            raise TypeError("BAF absence conversion requires a named DataFrame.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Numerical feature order differs from the fitted schema.")
        result = X.copy()
        for field in BAF_MINUS_ONE_FIELDS:
            if field in result:
                result[field] = result[field].mask(result[field].eq(-1), np.nan)
        for field in BAF_NEGATIVE_MISSING_FIELDS:
            if field in result:
                result[field] = result[field].mask(result[field].lt(0), np.nan)
        return result

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_.copy()


def numeric_pipeline(missing_policy="preserve"):
    """Build a fold-fitted numerical pipeline for either predefined policy."""
    if missing_policy not in MISSING_POLICIES:
        raise ValueError(f"Unknown missing policy: {missing_policy}")
    steps = []
    if missing_policy == "nan_indicators":
        steps.append(("absence_codes", BafAbsenceToNaN()))
    steps.extend([
        ("imputer", SimpleImputer(strategy="mean", add_indicator=(missing_policy == "nan_indicators"))),
        ("scaler", StandardScaler()),
    ])
    return Pipeline(steps)
