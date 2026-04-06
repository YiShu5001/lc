from __future__ import annotations

from pathlib import Path
from datetime import datetime

from lc.common.io import ensure_dir, write_json, write_metrics_csv as write_csv_rows
from lc.control.configs import PyBulletControlExperimentConfig
from lc.control.reference_generators.piecewise_velocity import ReferenceBundle


def build_run_directory(
    config: PyBulletControlExperimentConfig,
    mode: str,
    axis: str,
    controller_variant: str,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(config.artifact.output_root) / mode / controller_variant / axis / stamp)


def write_summary_json(path: str | Path, payload: object) -> Path:
    return write_json(path, payload)


def write_metrics_csv(path: str | Path, rows: list[dict[str, object]]) -> Path:
    return write_csv_rows(path, rows)


def write_timeseries_csv(path: str | Path, rows: list[dict[str, object]]) -> Path:
    return write_csv_rows(path, rows)


def write_reference_csv(path: str | Path, bundle: ReferenceBundle) -> Path:
    rows = []
    for index, (pos, vel) in enumerate(zip(bundle.positions, bundle.velocities)):
        rows.append(
            {
                "step": float(index),
                "target_x": float(pos[0]),
                "target_y": float(pos[1]),
                "target_z": float(pos[2]),
                "target_vx": float(vel[0]),
                "target_vy": float(vel[1]),
                "target_vz": float(vel[2]),
            }
        )
    return write_csv_rows(path, rows)


def export_legacy_logger_artifacts(rows: list[dict[str, float]], run_dir: str | Path, prefix: str) -> Path:
    target = ensure_dir(Path(run_dir) / "legacy_logger")
    return write_csv_rows(target / f"{prefix}_logger.csv", rows)
