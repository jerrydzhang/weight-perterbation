import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, Tuple

import numpy as np
import yaml
from jernerics.experiment import Experiment
from jernerics.generate.generator import DataGenerator
from sklearn.model_selection import KFold

import weighting
from models import WeightedLasso

GRAPH_DIR = Path(__file__).parent / "data"


@dataclass
class WeightedLassoExperiment(Experiment):
    random_state: int
    metrics: dict[str, float] = field(default_factory=dict)

    def setup_data(
        self: Self, config: dict
    ) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        graph_definition_file = config["graph_file"]
        graph_definition_file = GRAPH_DIR / graph_definition_file
        with open(graph_definition_file, "r") as f:
            data_config = yaml.safe_load(f)

        n_samples = config["n_samples"]
        n_features = config["n_features"]

        data_generator = DataGenerator.from_config(data_config)
        X, y, theta = data_generator.generate(
            n_samples=n_samples,
            n_features=n_features,
            random_state=self.random_state,
        )

        return X, y, theta

    def save_model(self: Self, result_path: Path, model: WeightedLasso) -> None:
        model_path = result_path / f"{self.task_id}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    def train(self: Self, data: np.ndarray, config: dict) -> WeightedLasso:
        alpha = config["alpha"]

        X, y, theta = data
        weights = weighting.reciprocal(theta, eps=float(config["eps"]))
        weights = weighting.preterb(
            weights,
            std=config["preterb_weight_std"],
            random_state=self.random_state,
        )
        weights = weights.reshape(-1, 1)

        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = WeightedLasso(alpha=alpha, weights=weights)
            model.fit(X_train, y_train)

            fold_metrics = self._compute_metrics(model, X_test, y_test)

            for key, value in fold_metrics.items():
                if key in self.metrics:
                    self.metrics[key] += value / 5.0
                else:
                    self.metrics[key] = value / 5.0

        model = WeightedLasso(alpha=alpha, weights=weights)
        model.fit(X, y)

        return model

    def _compute_metrics(
        self: Self, model: WeightedLasso, X: np.ndarray, y: np.ndarray
    ) -> dict:
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return {"mse": mse}

    def evaluate(
        self: Self,
        model: WeightedLasso,
        data: Tuple[np.ndarray, np.ndarray | None, np.ndarray | None],
        config: dict,
    ) -> dict:
        print("Evaluation metrics:", self.metrics)
        return self.metrics


def get_experiment(config: dict) -> Experiment:
    return WeightedLassoExperiment(**config)
