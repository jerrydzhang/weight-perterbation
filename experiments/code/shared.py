import os
import json
import datetime
import pickle as pkl
from typing import Any


def compute_metrics(data):
    pass


def save_results(model: Any, results: dict, experiment_name: str) -> None:
    dest_dir = (
        f"results/{experiment_name}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(
        dest_dir,
        exist_ok=True,
    )

    with open(os.path.join(dest_dir, "model.pkl"), "wb") as f:
        pkl.dump(model, f)

    with open(os.path.join(dest_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
