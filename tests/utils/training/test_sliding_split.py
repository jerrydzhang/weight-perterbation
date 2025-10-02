from utils.training.sliding_split import SlidingWindowSplit
import numpy as np


class TestSlidingWindowSplit:
    def test_base(self):
        X = np.arange(20).reshape(20, 1)
        splitter = SlidingWindowSplit(n_splits=3, train_size=1, test_size=1)
        splits = list(splitter.split(X))

        expected_splits = [
            (list(range(0, 5)), list(range(5, 10))),
            (list(range(5, 10)), list(range(10, 15))),
            (list(range(10, 15)), list(range(15, 20))),
        ]

        assert splits == expected_splits, (
            f"Expected {expected_splits}, but got {splits}"
        )

    def test_larger_train(self):
        X = np.arange(40).reshape(40, 1)
        splitter = SlidingWindowSplit(n_splits=3, train_size=2, test_size=1)
        splits = list(splitter.split(X))

        expected_splits = [
            (list(range(0, 16)), list(range(16, 24))),
            (list(range(8, 24)), list(range(24, 32))),
            (list(range(16, 32)), list(range(32, 40))),
        ]

        assert splits == expected_splits, (
            f"Expected {expected_splits}, but got {splits}"
        )

    def test_larger_test(self):
        X = np.arange(40).reshape(40, 1)
        splitter = SlidingWindowSplit(n_splits=3, train_size=1, test_size=2)
        splits = list(splitter.split(X))

        expected_splits = [
            (list(range(0, 8)), list(range(8, 24))),
            (list(range(8, 16)), list(range(16, 32))),
            (list(range(16, 24)), list(range(24, 40))),
        ]

        assert splits == expected_splits, (
            f"Expected {expected_splits}, but got {splits}"
        )
