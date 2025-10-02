import csv
import importlib
import pickle
import signal
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, Tuple

import numpy as np
import pysindy as ps
from jernerics.experiment import Experiment
from utils.training.sliding_split import SlidingWindowSplit

import weighting
from models import WeightedLasso
from utils.ode import map_equation

DATA_DIR = Path(__file__).parent / "data"


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


@dataclass
class WeightedLassoExperiment(Experiment):
    random_state: int
    metrics: dict[str, float] = field(default_factory=dict)

    def setup_data(self: Self, config: dict) -> Tuple[np.ndarray, np.ndarray]:
        data_path = DATA_DIR / config["data_file"]
        with open(data_path, "r") as f:
            reader = csv.reader(f)
            # header = next(reader)
            data = np.array([[float(value) for value in row] for row in reader])

        X = data[:, 1:]
        t = data[:, 0]

        return X, t

    def save_model(self: Self, result_path: Path, model: ps.SINDy) -> None:
        model_path = result_path / f"{self.task_id}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    def train(
        self: Self, data: Tuple[np.ndarray, np.ndarray], config: dict
    ) -> ps.SINDy:
        print("Training with config:", config)
        alpha = config["model_alpha"]

        X, t = data
        library = ps.PolynomialLibrary(degree=3)
        library.fit(X)  # type: ignore

        equation_module = importlib.import_module(f"utils.ode._{config['equation']}")
        equation_func = getattr(equation_module, config["equation"])

        true_parameters = map_equation(equation_func, library)
        weights = weighting.reciprocal(true_parameters, eps=float(config["weight_eps"]))
        weights = weighting.preterb(
            weights,
            std=config["weight_std"],
            random_state=self.random_state,
        ).T

        kf = SlidingWindowSplit(n_splits=5, train_size=10, test_size=1)
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            print(f"Fold {i + 1}/5")
            X_train, t_train = X[train_index], t[train_index]
            X_test, t_test = X[test_index], t[test_index]

            try:
                model = ps.SINDy(
                    feature_library=library,
                    optimizer=WeightedLasso(
                        alpha=alpha,
                        weights=weights,
                        max_iter=100000,
                    ),  # type: ignore
                    differentiation_method=ps.FiniteDifference(),
                )
                model.fit(X_train, t=t_train, x_dot=np.asarray(equation_func(t, X.T)).T)
            except Exception as e:
                print(f"An error occurred during model fitting: {e}")
                self.metrics = {f"nmse_{j}": float("inf") for j in range(X.shape[1])}
                continue

            fold_metrics = self._compute_metrics(model, X_test, t_test)

            for key, value in fold_metrics.items():
                if key in self.metrics:
                    self.metrics[key] += value / 5.0
                else:
                    self.metrics[key] = value / 5.0

        model = ps.SINDy(
            feature_library=library,
            optimizer=WeightedLasso(
                alpha=alpha,
                weights=weights,
                max_iter=100000,
            ),  # type: ignore
            differentiation_method=ps.FiniteDifference(),
        )
        model.fit(X, t=t, x_dot=np.asarray(equation_func(t, X.T)).T)

        return model

    def _compute_metrics(
        self: Self, model: ps.SINDy, X: np.ndarray, y: np.ndarray
    ) -> dict:
        print("Computing metrics...")
        timeout = 300
        with time_limit(timeout):
            try:
                trajectory = model.simulate(X[0], t=y)
                normalized_mse = np.mean((X - trajectory) ** 2, axis=0) / np.var(
                    X, axis=0
                )
            except TimeoutException:
                print(f"Simulation timed out after {timeout} seconds.")
                normalized_mse = np.array([float("inf")] * X.shape[1])
            except Exception as e:
                print(f"An error occurred during simulation: {e}")
                normalized_mse = np.array([float("inf")] * X.shape[1])

        nmse_dict = {f"nmse_{i}": nmse for i, nmse in enumerate(normalized_mse)}
        return nmse_dict

    def evaluate(
        self: Self,
        model: ps.SINDy,
        data: np.ndarray,
        config: dict,
    ) -> dict:
        print("Evaluation metrics:", self.metrics)
        self.metrics["nmse_mean"] = np.mean(
            [v for k, v in self.metrics.items() if k.startswith("nmse_")]
        ).item()
        return self.metrics


def get_experiment(config: dict) -> Experiment:
    return WeightedLassoExperiment(**config)
