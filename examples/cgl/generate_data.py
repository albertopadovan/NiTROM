"""Generate physical CGL training, impulse, sinusoidal, and ensemble responses."""

import time as clock

import numpy as np
from _utils import prepare, write_json
from cgl import CGL, TimeStepper, input_output
from settings import (
    AMPLITUDES,
    FORCE_AMPLITUDES,
    FORCE_FREQUENCIES,
    arguments,
    configuration,
)

from nitrom.numerics import time_derivative


def main(argv=None):
    settings, directory, _ = arguments(__doc__, argv)
    start = clock.perf_counter()
    prepare(directory)
    data = directory / "data"
    fom = CGL(nx=settings.nx)
    columns, direction, omega, C = input_output(fom, settings.seed)
    stepper = TimeStepper(fom, settings.dt)
    amplitudes = np.repeat(AMPLITUDES, 2)
    training = []
    for j, amplitude in enumerate(amplitudes):
        time, X = stepper.integrate(
            amplitude * columns[j % 2], settings.train_end, settings.save_every
        )
        training.append(X)
    np.savez(
        data / "training.npz",
        X=np.array(training),
        dX=np.array([time_derivative(X, time) for X in training]),
        time=time,
        amplitudes=amplitudes,
    )
    impulses = [training[2], training[7]]
    for initial in (0.3 * columns[0], 0.3 * direction):
        _, X = stepper.integrate(initial, settings.train_end, settings.save_every)
        impulses.append(X)
    betas = np.random.default_rng(settings.seed + 1).uniform(-1, 1, settings.n_ensemble)
    ensemble = []
    for j, beta in enumerate(betas):
        _, X = stepper.integrate(
            beta * columns[j % 2], settings.train_end, settings.save_every
        )
        ensemble.append(X)
    forced, force_amplitudes, force_frequencies = [], [], []
    for amplitude in FORCE_AMPLITUDES:
        for frequency in FORCE_FREQUENCIES:
            tf, X = stepper.integrate(
                np.zeros(2 * settings.nx),
                settings.forced_end,
                settings.save_every,
                forcing=lambda t, a=amplitude, k=frequency: (
                    a * np.sin(k * omega * t) * direction
                ),
            )
            forced.append(X)
            force_amplitudes.append(amplitude)
            force_frequencies.append(frequency)
    np.savez(
        data / "testing.npz",
        time=time,
        time_forced=tf,
        impulses=np.array(impulses),
        impulse_amplitudes=[amplitudes[2], amplitudes[7], 0.3, 0.3],
        impulse_labels=[
            "Training, beta=0.01 (Re)",
            "Training, beta=1 (Im)",
            "Testing, beta=0.3 (Re)",
            "Testing, beta=0.3 (random actuator)",
        ],
        ensemble=np.array(ensemble),
        ensemble_amplitudes=betas,
        forced=np.array(forced),
        force_amplitudes=force_amplitudes,
        force_frequencies=force_frequencies,
        direction=direction,
        omega=omega,
        C=C,
        x=fom.x,
        xbar=fom.xbar,
    )
    write_json(data / "settings.json", configuration(settings))
    print(f"Data: {data} ({clock.perf_counter() - start:.2f} s)")


if __name__ == "__main__":
    main()
