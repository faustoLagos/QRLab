from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass(frozen=True)
class QuadrotorControlAllocation:
    wrench_from_rotor_thrust: np.ndarray
    rotor_thrust_from_wrench: np.ndarray
    thrust_coefficient: float
    torque_coefficient: float
    max_rpm: float
    max_rotor_thrust_newtons: float


def _validate_positive_scalar(name: str, value: float) -> float:
    scalar_value = float(value)
    if not np.isfinite(scalar_value) or scalar_value <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar.")
    return scalar_value


def create_cf2x_control_allocation(
    arm_length_m: float,
    thrust_coefficient: float,
    torque_coefficient: float,
    max_rpm: float,
) -> QuadrotorControlAllocation:
    """Create the CF2X thrust-to-wrench allocation used by BaseAviary physics."""
    arm_length = _validate_positive_scalar("arm_length_m", arm_length_m)
    thrust_coefficient_value = _validate_positive_scalar(
        "thrust_coefficient",
        thrust_coefficient,
    )
    torque_coefficient_value = _validate_positive_scalar(
        "torque_coefficient",
        torque_coefficient,
    )
    maximum_rpm = _validate_positive_scalar("max_rpm", max_rpm)

    arm_projection = arm_length / sqrt(2.0)
    yaw_torque_per_thrust = torque_coefficient_value / thrust_coefficient_value

    wrench_from_rotor_thrust = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [-arm_projection, -arm_projection, arm_projection, arm_projection],
            [-arm_projection, arm_projection, arm_projection, -arm_projection],
            [
                -yaw_torque_per_thrust,
                yaw_torque_per_thrust,
                -yaw_torque_per_thrust,
                yaw_torque_per_thrust,
            ],
        ],
        dtype=np.float64,
    )
    rotor_thrust_from_wrench = np.linalg.inv(wrench_from_rotor_thrust)

    return QuadrotorControlAllocation(
        wrench_from_rotor_thrust=wrench_from_rotor_thrust,
        rotor_thrust_from_wrench=rotor_thrust_from_wrench,
        thrust_coefficient=thrust_coefficient_value,
        torque_coefficient=torque_coefficient_value,
        max_rpm=maximum_rpm,
        max_rotor_thrust_newtons=(maximum_rpm**2) * thrust_coefficient_value,
    )


def convert_rotor_thrust_to_wrench(
    rotor_thrust_newtons: np.ndarray,
    allocation: QuadrotorControlAllocation,
) -> np.ndarray:
    """Map four rotor thrusts to total thrust and body torques."""
    rotor_thrust = np.asarray(rotor_thrust_newtons, dtype=np.float64).reshape(-1)
    if rotor_thrust.shape != (4,):
        raise ValueError("rotor_thrust_newtons must contain exactly four values.")
    return allocation.wrench_from_rotor_thrust @ rotor_thrust


def allocate_wrench_to_rotor_thrust(
    wrench: np.ndarray,
    allocation: QuadrotorControlAllocation,
) -> np.ndarray:
    """Map [thrust, tau_x, tau_y, tau_z] to feasible rotor thrusts."""
    requested_wrench = np.asarray(wrench, dtype=np.float64).reshape(-1)
    if requested_wrench.shape != (4,):
        raise ValueError("wrench must contain exactly four values.")

    rotor_thrust = allocation.rotor_thrust_from_wrench @ requested_wrench
    return np.clip(
        rotor_thrust,
        0.0,
        allocation.max_rotor_thrust_newtons,
    )


def convert_rotor_thrust_to_rpm(
    rotor_thrust_newtons: np.ndarray,
    allocation: QuadrotorControlAllocation,
) -> np.ndarray:
    """Convert feasible rotor thrusts to rotor RPM."""
    rotor_thrust = np.asarray(rotor_thrust_newtons, dtype=np.float64).reshape(-1)
    if rotor_thrust.shape != (4,):
        raise ValueError("rotor_thrust_newtons must contain exactly four values.")

    clipped_rotor_thrust = np.clip(
        rotor_thrust,
        0.0,
        allocation.max_rotor_thrust_newtons,
    )
    return np.sqrt(clipped_rotor_thrust / allocation.thrust_coefficient)


def allocate_wrench_to_rpm(
    wrench: np.ndarray,
    allocation: QuadrotorControlAllocation,
) -> np.ndarray:
    """Map a requested wrench to feasible CF2X rotor RPM values."""
    rotor_thrust = allocate_wrench_to_rotor_thrust(
        wrench=wrench,
        allocation=allocation,
    )
    return convert_rotor_thrust_to_rpm(
        rotor_thrust_newtons=rotor_thrust,
        allocation=allocation,
    )


def create_delta_wrench_rotor_constraints(
    hover_thrust_newtons: float,
    allocation: QuadrotorControlAllocation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create linear rotor limits for delta-thrust/body-torque MPC inputs."""
    hover_thrust = _validate_positive_scalar(
        "hover_thrust_newtons",
        hover_thrust_newtons,
    )
    hover_wrench = np.array([hover_thrust, 0.0, 0.0, 0.0], dtype=np.float64)
    hover_rotor_thrust = allocation.rotor_thrust_from_wrench @ hover_wrench

    constraint_matrix = allocation.rotor_thrust_from_wrench.copy()
    lower_bound = -hover_rotor_thrust
    upper_bound = allocation.max_rotor_thrust_newtons - hover_rotor_thrust
    return constraint_matrix, lower_bound, upper_bound
