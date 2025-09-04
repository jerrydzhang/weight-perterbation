from utils.generators.generator import DataGenerator
from utils.generators.data import GaussianDataGenerator, LinearDependentTransformer
from utils.generators.parameters import (
    GaussianParametersGenerator,
    SparseParametersTransformer,
)
from utils.generators.labels import LinearLabelComputer
from models import WeightedLasso

data_generator = (
    DataGenerator()
    .add_step(
        label="gaussian_params",
        step=GaussianParametersGenerator(mean=0, std=1, random_state=42),
    )
    .add_step(
        label="sparse_params",
        step=SparseParametersTransformer(sparsity=0.9, random_state=42),
        deps=["gaussian_params"],
    )
    .add_step(
        label="gaussian_data",
        step=GaussianDataGenerator(mean=0, std=1, random_state=42),
    )
    .add_step(
        label="linear_dependent_data",
        step=LinearDependentTransformer(
            percent_independent=0.1, std=1e0, noise_std=0, random_state=42
        ),
        deps=["gaussian_data"],
    )
    .add_step(
        label="linear_labels",
        step=LinearLabelComputer(),
        deps=["linear_dependent_data", "sparse_params"],
    )
)
