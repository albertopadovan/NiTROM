"""Full showcase defaults and the small smoke profile. Edit settings here."""

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path

LABELS = {
    "galerkin": "POD-Galerkin",
    "opinf": "OpInf",
    "gas_opinf": "GAS-OpInf",
    "nitrom": "NiTROM",
    "gas_nitrom": "GAS-NiTROM",
}


@dataclass(frozen=True)
class Settings:
    profile: str = "full"
    seed: int = 0
    r: int = 2
    beta: float = 20.0
    train_end: float = 10.0
    train_snapshots: int = 100
    test_end: float = 10.0
    test_snapshots: int = 200
    n_test: int = 100
    gas_opinf_epochs: int = 1000
    nitrom_epochs: int = 400
    n_substeps: int = 15
    adjoint_method: str = "continuous"
    opinf_reg: float = 2.983647e-7


def smoke_settings():
    return replace(
        Settings(),
        profile="smoke",
        train_end=2.0,
        train_snapshots=21,
        test_end=2.0,
        test_snapshots=41,
        n_test=4,
        gas_opinf_epochs=2,
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
