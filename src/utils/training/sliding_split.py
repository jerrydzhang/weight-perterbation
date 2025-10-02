from dataclasses import dataclass


@dataclass
class SlidingWindowSplit:
    """Custom sliding window splitter for time series data.

    Data is split into `n_splits + test_size + train_size - 1` segments,

    Parameters:
        n_splits (int): Number of splits.
        train_size (int): Number of training windows.
        test_size (int): Number of testing windows.
    """

    n_splits: int
    train_size: int
    test_size: int

    def split(self, X):
        n_samples = X.shape[0]
        step_size = n_samples // (self.n_splits + self.test_size + self.train_size - 1)
        for i in range(self.n_splits):
            start_train = i * step_size
            end_train = start_train + self.train_size * step_size
            start_test = end_train
            end_test = start_test + self.test_size * step_size

            if end_test > n_samples:
                break

            train_indices = list(range(start_train, end_train))
            test_indices = list(range(start_test, end_test))

            yield train_indices, test_indices
