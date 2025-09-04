import numpy as np
import pysindy as ps

from utils.ode import hydrogen_bromine, hydrogen_bromine_init, map_equation

from utils.generators.generator import DataGenerator
from utils.generators.data import ODEDataGenerator

from models import WeightedLasso

dt = 0.002
t_train = np.arange(0, 10, dt)
t_train_span = (t_train[0], t_train[-1])
data_generator = DataGenerator().add_step(
    label="ode_data",
    step=ODEDataGenerator(
        equation=hydrogen_bromine,
        initial_state=hydrogen_bromine_init,
        t_span=t_train_span,
        random_state=42,
    ),
)

X, y, theta = data_generator.generate(n_samples=t_train.shape[0], n_features=6)

print("Generated X shape:", X.shape)
print("t_train shape:", t_train.shape)

library = ps.PolynomialLibrary(degree=3)
library.fit(X)

true_parameters = map_equation(hydrogen_bromine, library)

model = ps.SINDy(
    feature_library=library,
    optimizer=WeightedLasso(
        alpha=1e0, weights=(1 / (np.abs(true_parameters) + 1e-6)).T, max_iter=10000
    ),
    differentiation_method=ps.FiniteDifference(),
)


model.fit(x=X, t=t_train)
model.print()
