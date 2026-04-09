"""
training.py — sklearn wrappers and chronological evaluation split
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

import config


def fit_gradient_boosting_rate_limit_classifier(
    feature_matrix: np.ndarray, labels_batch_had_429: np.ndarray
) -> GradientBoostingClassifier:
    gradient_boosting_classifier = GradientBoostingClassifier(random_state=config.RANDOM_SEED)
    gradient_boosting_classifier.fit(feature_matrix, labels_batch_had_429)
    return gradient_boosting_classifier


def fit_logistic_rate_limit_classifier(
    feature_matrix: np.ndarray, labels_batch_had_429: np.ndarray, max_iter: int = 500
) -> LogisticRegression:
    logistic_classifier = LogisticRegression(max_iter=max_iter, random_state=config.RANDOM_SEED)
    logistic_classifier.fit(feature_matrix, labels_batch_had_429)
    return logistic_classifier


def evaluate_classifier_with_chronological_split(
    feature_matrix: np.ndarray, labels_batch_had_429: np.ndarray, train_frac: float = 0.7
):
    train_row_count = int(train_frac * feature_matrix.shape[0])
    X_train, X_test = feature_matrix[:train_row_count], feature_matrix[train_row_count:]
    y_train, y_test = labels_batch_had_429[:train_row_count], labels_batch_had_429[train_row_count:]
    holdout_classifier = GradientBoostingClassifier(random_state=config.RANDOM_SEED)
    holdout_classifier.fit(X_train, y_train)
    y_predicted_on_test = holdout_classifier.predict(X_test)
    print("\n=== Time-split holdout report ===")
    print(classification_report(y_test, y_predicted_on_test, digits=4))
    if len(set(y_test.tolist())) > 1:
        positive_class_column_index = list(holdout_classifier.classes_).index(1)
        positive_class_probability_on_test = holdout_classifier.predict_proba(X_test)[
            :, positive_class_column_index
        ]
        print(
            "time-split ROC-AUC (positive=429): "
            f"{roc_auc_score(y_test, positive_class_probability_on_test):.4f}"
        )
    return holdout_classifier
