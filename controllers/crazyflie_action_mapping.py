from dataclasses import dataclass
from math import degrees

import numpy as np

from controllers.crazyflie_runtime import (
    CrazyfliePositionCommand,
    CrazyflieVelocityCommand,
)


@dataclass(frozen=True)
class PositionActionBounds:
    minimum: tuple[float, float, float, float]
    maximum: tuple[float, float, float, float]


@dataclass(frozen=True)
class VelocityActionLimits:
    maximum_absolute: tuple[float, float, float, float]
    body_frame: bool = False


def map_normalized_position_action(
    normalized_action: np.ndarray,
    bounds: PositionActionBounds,
) -> CrazyfliePositionCommand:
    """Map [-1, 1]^4 to [x, y, z, yaw] using SI-unit workspace bounds."""
    action = _validate_normalized_action(normalized_action)
    lower = np.asarray(bounds.minimum, dtype=np.float64)
    upper = np.asarray(bounds.maximum, dtype=np.float64)

    if lower.shape != (4,) or upper.shape != (4,):
        raise ValueError("Position action bounds must contain four values.")
    if np.any(upper <= lower):
        raise ValueError("Each position action upper bound must exceed its lower bound.")

    physical_command = lower + 0.5 * (action + 1.0) * (upper - lower)

    return CrazyfliePositionCommand(
        x=float(physical_command[0]),
        y=float(physical_command[1]),
        z=float(physical_command[2]),
        yaw_degrees=degrees(float(physical_command[3])),
    )


def map_normalized_velocity_action(
    normalized_action: np.ndarray,
    limits: VelocityActionLimits,
) -> CrazyflieVelocityCommand:
    """Map [-1, 1]^4 to [vx, vy, vz, yaw_rate] using symmetric SI-unit limits."""
    action = _validate_normalized_action(normalized_action)
    maximum_absolute = np.asarray(limits.maximum_absolute, dtype=np.float64)

    if maximum_absolute.shape != (4,):
        raise ValueError("Velocity action limits must contain four values.")
    if np.any(maximum_absolute <= 0.0):
        raise ValueError("Velocity action limits must be strictly positive.")

    physical_command = action * maximum_absolute

    return CrazyflieVelocityCommand(
        vx=float(physical_command[0]),
        vy=float(physical_command[1]),
        vz=float(physical_command[2]),
        yaw_rate_degrees_per_second=degrees(float(physical_command[3])),
        body_frame=limits.body_frame,
    )


def _validate_normalized_action(normalized_action: np.ndarray) -> np.ndarray:
    action = np.asarray(normalized_action, dtype=np.float64).reshape(-1)

    if action.shape != (4,):
        raise ValueError("Normalized high-level actions must contain four values.")
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ValueError("Normalized high-level actions must remain within [-1, 1].")

    return action
