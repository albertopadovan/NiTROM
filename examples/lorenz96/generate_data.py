"""Generate eight transient-to-attractor trajectories and independent test data."""

import time as clock

import numpy as np
from _utils import prepare, write_json
from model import Lorenz96, input_functions
from settings import (
    TEST_INPUTS,
    TEST_LABELS,
    TRAIN_INPUTS,
    arguments,
    configuration,
    time_grid,
)

from nitrom.backend import set_backend
from nitrom.time_steppers import solve_ivp


def main(argv=None):
    settings, directory, _ = arguments(__doc__, argv)
    set_backend("numpy")
    start = clock.perf_counter()
    prepare(directory)
    fom = Lorenz96(settings.n, settings.F)
    train_seed, test_seed = np.random.SeedSequence(settings.seed).spawn(2)
    rng = np.random.default_rng(train_seed)
    amplitudes = np.repeat(TRAIN_INPUTS, 2)
    initial = settings.F + rng.standard_normal((len(amplitudes), settings.n))
    time = time_grid(settings.train_end, settings.sample_dt)
    X = solve_ivp(
        fom.rhs,
        initial,
        0,
        time[-1],
        settings.reference_dt,
        time,
        "rk4",
        forcing=amplitudes[:, None],
    )
    dX = fom.rhs(0, X.transpose(0, 2, 1), amplitudes[:, None, None]).transpose(0, 2, 1)
    mean = X.mean(axis=(0, 2))
    np.savez(
        directory / "data" / "training.npz",
        X=X,
        dX=dX,
        time=time,
        amplitudes=amplitudes,
        mean=mean,
    )

    rng = np.random.default_rng(test_seed)
    test_amplitudes = np.array(TEST_INPUTS)
    sinusoidal = np.array([False, False, False, True])
    inputs = input_functions(
        test_amplitudes, sinusoidal=sinusoidal, period=settings.input_period
    )
    initial = settings.F + rng.standard_normal((len(inputs), settings.n))
    long_time = time_grid(settings.long_end, settings.sample_dt)
    long_X = solve_ivp(
        lambda t, x: fom.rhs(t, x, np.array([fn(t)[0] for fn in inputs])[:, None]),
        initial,
        0,
        long_time[-1],
        settings.reference_dt,
        long_time,
        "rk4",
    )
    short_time = time_grid(settings.test_end, settings.sample_dt)
    np.savez(
        directory / "data" / "testing.npz",
        X=long_X[:, :, : len(short_time)],
        time=short_time,
        long_X=long_X,
        long_time=long_time,
        amplitudes=test_amplitudes,
        sinusoidal=sinusoidal,
        labels=np.array(TEST_LABELS),
        period=settings.input_period,
    )
    write_json(directory / "data" / "settings.json", configuration(settings))
    print(f"Data: {directory / 'data'} ({clock.perf_counter() - start:.2f} s)")


if __name__ == "__main__":
    main()
