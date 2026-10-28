"""Full CGL benchmark defaults and the small smoke profile."""

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path

LABELS = {
    "balanced_opinf": "Balanced OpInf",
    "pod_opinf": "POD OpInf",
    "nitrom_lall": "NiTROM (Lall init)",
    "nitrom_pod": "NiTROM (POD init)",
}
AMPLITUDES = (-1.0, 0.01, 0.1, 1.0)
FORCE_AMPLITUDES = (0.01, 0.05, 0.2)
FORCE_FREQUENCIES = (1, 2)


@dataclass(frozen=True)
class Settings:
    profile: str = "full"
    seed: int = 0
    r: int = 5
    nx: int = 256
    dt: float = 0.01
    save_every: int = 20
    train_end: float = 500.0
    forced_end: float = 300.0
    nitrom_end: float = 200.0
    n_ensemble: int = 50
    nitrom_epochs: int = 300
    n_substeps: int = 3
    adjoint_method: str = "discrete"
    balancing_checkpoints: int = 10
    balancing_stride: int = 5


def smoke_settings():
    return replace(
        Settings(),
        profile="smoke",
        nx=64,
        train_end=40.0,
        forced_end=20.0,
        nitrom_end=20.0,
        n_ensemble=2,
        nitrom_epochs=2,
    )


def arguments(description, argv=None, *, training=False, plotting=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--smoke", action="store_true", help="small installation/test run"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="override this run's output directory"
    )
    if training:
        parser.add_argument(
            "--fresh", action="store_true", help="ignore saved NiTROM parameters"
        )
    if plotting:
        parser.add_argument(
            "--tex", action="store_true", help="use LaTeX for publication figures"
        )
    args = parser.parse_args(argv)
    settings = smoke_settings() if args.smoke else Settings()
    directory = (
        args.output_dir
        or Path(__file__).resolve().parent / "outputs" / settings.profile
    )
    return settings, directory.resolve(), args


def configuration(settings):
    return asdict(settings)
