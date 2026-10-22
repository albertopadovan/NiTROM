"""Generate training trajectories and held-out step responses in physical units."""

import time as clock

import dill
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
    for j, b in enumerate(amplitudes):
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
        np.save(data / f"train_traj_{j:03d}.npy", X)
        np.save(data / f"train_deriv_{j:03d}.npy", fom.rhs(0, X.T, b * fom.B[:, 0]).T)
        np.save(data / f"train_weight_{j:03d}.npy", np.array([fom.steady_weights(b)]))
        with (data / f"train_forcing_{j:03d}.pkl").open("wb") as stream:
            dill.dump(lambda t, b=float(b): np.array([b]), stream, recurse=True)
    np.save(data / "time.npy", time)
    np.save(data / "train_amplitudes.npy", amplitudes)
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
