#!/usr/bin/env python

import csv
import pathlib

import numpy as np
from scipy.integrate import solve_ivp

from utils.ode import hydrogen_bromine, hydrogen_bromine_init, rober, rober_init

data_dir = pathlib.Path(__file__).parent.parent / "data"

equation = hydrogen_bromine
initial_conditions = hydrogen_bromine_init
# equation = rober
# initial_conditions = rober_init

integrator_options = {
    "method": "LSODA",
    "rtol": 1e-12,
    "atol": 1e-12,
}

dt = 0.002
t_train = np.arange(0, 1e1, dt)
t_span = (t_train[0], t_train[-1])
sol = solve_ivp(
    equation, t_span, initial_conditions, **integrator_options, t_eval=t_train
)

with open(data_dir / f"{equation.__name__}_ode.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time"] + [f"x{i}" for i in range(len(initial_conditions))])
    for t, y in zip(sol.t, sol.y.T):
        writer.writerow([t] + list(y))
