import unittest
from pathlib import Path
import numpy as np

import pysindy as ps
from utils.weighting.gen_weights import map_equation

assets_path = Path(__file__).parent.parent.parent / "assets"


class TestGenWeights(unittest.TestCase):
    def test_map_equation(self):
        library = ps.feature_library.PolynomialLibrary(degree=3)
        library.fit(np.random.rand(10, 6))
        equation = map_equation(assets_path / "ode.py", "hydrogen_bromine", library)

        # TODO: Put in a real test
        self.assertIsNotNone(equation)


if __name__ == "__main__":
    unittest.main()
