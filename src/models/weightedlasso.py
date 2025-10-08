import warnings

import cvxpy as cp
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"You are solving a parameterized problem that is not DPP.",
)


class WeightedLasso(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        weights: np.ndarray,
        alpha: float = 1.0,
        max_iter: int = 100000,
    ):
        self.weights: np.ndarray = weights
        self.alpha: float = alpha
        self.coef_: np.ndarray = np.array([])
        self.max_iter: int = max_iter

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        norm = np.linalg.norm(X, axis=0).reshape(-1, 1)
        x_normed = X / norm.T

        coef = cp.Variable((n_features, n_targets))
        weights_param = cp.Parameter((n_features, n_targets), nonneg=True)
        alpha_param = cp.Parameter(nonneg=True)

        error = (1 / (2 * n_samples)) * cp.sum_squares(x_normed @ coef - y)
        weighted_l1 = cp.norm1(cp.multiply(weights_param, coef), axis=0)
        # print(f"Weighted L1 shape: {weighted_l1.shape}")
        obj = cp.Minimize(cp.norm1(error + alpha_param * weighted_l1))
        prob = cp.Problem(obj)

        weights_param.value = self.weights
        alpha_param.value = self.alpha

        prob.solve(max_iter=self.max_iter)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            print("Problem status did not solve to optimality:", prob.status)

        self.coef_ = (coef.value / norm).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_.size == 0:
            raise ValueError("Model is not fitted yet. Please call 'fit' first.")
        return X @ self.coef_.T
