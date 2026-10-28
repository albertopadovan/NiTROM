"""Generate training trajectories and held-out step responses in physical units."""

import time as clock

import numpy as np
from _utils import prepare, write_json
from model import ToyModel
from settings import arguments, configuration

from nitrom.backend import set_backend
from nitrom.time_steppers import solve_ivp


def main(argv=None):
    settings, directory, _ = arguments(__doc__, argv)
    set_backend("numpy")
    start = clock.perf_counter()
    prepare(directory)
    data = directory / "data"
    fom = ToyModel(settings.beta)
    time = np.linspace(0, settings.train_end, settings.train_snapshots)
    amplitudes = np.array([0.01, 0.1, 0.2, 0.248])
    trajectories, derivatives = [], []
    for b in amplitudes:
        X = solve_ivp(
            fom.rhs,
            np.zeros(3),
            0,
            time[-1],
            (time[1] - time[0]) / 100,
            time,
            "rk4",
            forcing=b * fom.B[:, 0],
        )
        trajectories.append(X)
        derivatives.append(fom.rhs(0, X.T, b * fom.B[:, 0]).T)
    np.savez(
        data / "training.npz",
        X=np.array(trajectories),
        dX=np.array(derivatives),
        time=time,
        amplitudes=amplitudes,
        weights=fom.steady_weights(amplitudes),
    )
    test_time = np.linspace(0, settings.test_end, settings.test_snapshots)
    test_amplitudes = 0.24 * np.random.default_rng(settings.seed).random(
        settings.n_test
    )
    X_test = solve_ivp(
        fom.rhs,
        np.zeros((settings.n_test, 3)),
        0,
        test_time[-1],
        (test_time[1] - test_time[0]) / 5,
        test_time,
        "rk4",
        forcing=test_amplitudes[:, None] * fom.B[:, 0],
    )
    np.savez(
        data / "testing.npz",
        time=test_time,
        X=X_test,
        amplitudes=test_amplitudes,
        weights=fom.steady_weights(test_amplitudes),
    )
    write_json(data / "settings.json", configuration(settings))
    print(f"Data: {data} ({clock.perf_counter() - start:.2f} s)")


if __name__ == "__main__":
    main()
