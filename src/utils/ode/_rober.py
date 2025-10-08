rober_init = [1.0, 0.0, 0.0]
rober_init_past_stiff = [9.99902462e-01, 1.64381452e-05, 8.10998717e-05]


def rober(t, x, k=[0.04, 3e7, 1e4]):
    return [
        -k[0] * x[0] + k[1] * x[1] * x[2],
        k[0] * x[0] - k[1] * x[1] * x[2] - k[2] * x[1] ** 2,
        k[2] * x[1] ** 2,
    ]
