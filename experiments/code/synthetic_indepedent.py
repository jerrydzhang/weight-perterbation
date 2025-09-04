from utils.generators.generator import DataGenerator
from utils.generators.data import GaussianDataGenerator
from utils.generators.parameters import (
    GaussianParametersGenerator,
    SparseParametersTransformer,
)
from utils.generators.labels import LinearLabelComputer
from models import WeightedLasso

import sys
import json
import numpy as np
from sklearn.model_selection import GridSearchCV

from shared import compute_metrics, save_results

if len(sys.argv) > 1:
    graph_path = sys.argv[1]
    experiment_setting = sys.argv[2]
else:
    raise ValueError("Please provide a graph path and experiment setting")

with open(graph_path, "r") as f:
    data_config = json.load(f)

with open(experiment_setting, "r") as f:
    experiment_config = json.load(f)

    random_state = experiment_config["random_state"]
    n_samples = experiment_config["n_samples"]
    n_features = experiment_config["n_features"]


data_generator = DataGenerator.from_config(data_config)
X, y, theta = data_generator.generate(
    n_samples=n_samples,
    n_features=n_features,
    random_state=random_state,
)


# Do stuff
params = {
    "alpha": np.logspace(-4, 4, 9),
}

# weighted_lasso = WeightedLasso()
# model = GridSearchCV(weighted_lasso, params, cv=5, n_jobs=-1)

# TODO: Fixure out good metric
# Save best parameters
results = compute_metrics()
save_results(model, results, "synthetic_independent")
