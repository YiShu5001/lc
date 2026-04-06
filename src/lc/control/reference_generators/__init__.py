"""Reference generators for chapter 3 control experiments."""

from .piecewise_velocity import (
    ReferenceBundle,
    build_axis_piecewise_velocity_profile,
    build_xyz_reference_trajectory,
    integrate_velocity_profile,
    summarize_reference_segments,
)

__all__ = [
    "ReferenceBundle",
    "build_axis_piecewise_velocity_profile",
    "integrate_velocity_profile",
    "build_xyz_reference_trajectory",
    "summarize_reference_segments",
]
