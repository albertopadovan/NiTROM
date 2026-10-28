"""Lorenz-96 experiment settings; the latent dimension is chosen explicitly."""

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

LABELS = {
    "opinf": "OpInf",
    "atr_opinf": "ATR-OpInf",
    "nitrom": "NiTROM",
    "atr_nitrom": "ATR-NiTROM",
}
TRAIN_INPUTS = (-2.0, -1.0, 0.0, 2.0)
TEST_INPUTS = (-1.5, 0.5, 1.5, 2.0)
TEST_LABELS = (r"$u=-1.5$", r"$u=0.5$", r"$u=1.5$", r"$u=2\sin(2\pi t/5)$")


@dataclass(frozen=True)
class Settings:
    profile: str = "full"
    seed: int = 0
    n: int = 40
    F: float = 8.0
    r: int | None = None
    sample_dt: float = 0.02
    reference_dt: float = 0.002
    train_end: float = 30.0
    nitrom_end: float = 1.0
    test_end: float = 2.0
    long_end: float = 100.0
    statistics_start: float = 20.0
    input_bound: float = 2.0
    input_period: float = 5.0
    atr_opinf_epochs: int = 200
    nitrom_epochs: int = 200
    opinf_reg: float = 1e-6
    n_substeps: int = 4
    adjoint_method: str = "discrete"


def smoke_settings():
    return replace(
        Settings(),
        profile="smoke",
        train_end=2.0,
        nitrom_end=0.2,
        test_end=0.4,
        long_end=4.0,
        statistics_start=1.0,
        atr_opinf_epochs=2,
        nitrom_epochs=2,
    )


def time_grid(end, dt):
    count = round(end / dt)
    if count < 1 or not np.isclose(count * dt, end, rtol=0, atol=1e-10):
        raise ValueError(
            "The horizon must be a positive multiple of the sample interval."
        )
    return np.linspace(0.0, end, count + 1)


def arguments(
    description, argv=None, *, training=False, plotting=False, analysis=False
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--smoke", action="store_true", help="small workflow check")
    parser.add_argument("--output-dir", type=Path, help="override this run's directory")
    if training:
        parser.add_argument(
            "--r", type=int, required=True, help="chosen latent dimension"
        )
        parser.add_argument(
            "--fresh", action="store_true", help="ignore saved refinements"
        )
    if plotting or analysis:
        parser.add_argument("--tex", action="store_true", help="enable LaTeX figures")
    if analysis:
        parser.add_argument(
            "--energy",
            type=float,
            nargs="+",
            default=[],
            help="optional retained-energy fractions in (0, 1]; no default target",
        )
    args = parser.parse_args(argv)
    settings = smoke_settings() if args.smoke else Settings()
    if training:
        if not 1 <= args.r <= settings.n:
            parser.error(f"--r must be between 1 and {settings.n}")
        settings = replace(settings, r=args.r)
    if analysis and any(not np.isfinite(e) or not 0 < e <= 1 for e in args.energy):
        parser.error("--energy fractions must be finite and in (0, 1]")
    directory = (
        args.output_dir
        or Path(__file__).resolve().parent / "outputs" / settings.profile
    )
    return settings, directory.resolve(), args


def configuration(settings):
    return asdict(settings)
