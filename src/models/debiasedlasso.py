import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from .weightedlasso import WeightedLasso

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"You are solving a parameterized problem that is not DPP.",
)


class DebiasedLasso(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        weights: np.ndarray,
        alpha: float = 1.0,
        max_iter: int = 100000,
        threshhold: float = 1e-4,
    ):
        num_problems = weights.shape[0]

        self.weights = weights
        self.alpha = alpha
        self.coef_ = np.array([])
        self.max_iter = max_iter
        self.threshhold = threshhold
        self.model = WeightedLasso(
            weights=weights,
            alpha=alpha,
            max_iter=max_iter,
        )
        self.support_model = [LinearRegression() for _ in range(num_problems)]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        self.model.fit(X, y)

        support = np.abs(self.model.coef_) > self.threshhold

        for i, sup in enumerate(support):
            if np.sum(sup) == 0:
                self.coef_ = (
                    np.vstack([self.coef_, np.zeros(X.shape[1])])
                    if self.coef_.size
                    else np.zeros(X.shape[1])
                )
                continue

            self.support_model[i].fit(X[:, sup], y[:, i])
            coef = np.zeros(X.shape[1])
            coef[sup] = self.support_model[i].coef_
            self.coef_ = np.vstack([self.coef_, coef]) if self.coef_.size else coef
            print(f"Problem {i+1}/{len(support)} fitted.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_.size == 0:
            raise ValueError("Model is not fitted yet. Please call 'fit' first.")
        return X @ self.coef_.T
