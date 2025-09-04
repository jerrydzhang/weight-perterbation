import numpy as np


def reciprocal(theta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute weights based on the reciprocal weighting scheme.

    Args:
        theta (np.ndarray): Coefficient array.
        eps (float): Small constant to avoid division by zero.

    Returns:
        np.ndarray: Weights computed using the reciprocal scheme.
    """
    weights = 1.0 / (np.abs(theta) + eps)
    return weights


def preterb(
    weights: np.ndarray,
    std: float = 0.1,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Compute weights based on the preterb weighting scheme.

    Args:
        theta (np.ndarray): Coefficient array.
        std (float): Standard deviation for random perturbation.
        eps (float): Small constant to avoid division by zero.
        random_state (int | None): Seed for random number generator.

    Returns:
        np.ndarray: Weights computed using the preterb scheme.
    """
    rng = np.random.default_rng(random_state)
    perturbation = rng.normal(loc=0.0, scale=std, size=weights.shape)
    weights = weights + perturbation

    return weights
