from dataclasses import dataclass
from math import pi

import numpy as np
from scipy.signal import cont2discrete


STATE_SIZE = 12
CONTROL_SIZE = 4

POSITION_SLICE = slice(0, 3)
LINEAR_VELOCITY_SLICE = slice(3, 6)
ATTITUDE_SLICE = slice(6, 9)
BODY_RATE_SLICE = slice(9, 12)

ROLL_INDEX = 6
PITCH_INDEX = 7
YAW_INDEX = 8


@dataclass(frozen=True)
class LinearQuadrotorModel:
    continuous_state_matrix: np.ndarray
    continuous_input_matrix: np.ndarray
    discrete_state_matrix: np.ndarray
    discrete_input_matrix: np.ndarray
    sample_time_seconds: float
    mass_kg: float
    inertia_diagonal_kg_m2: np.ndarray
    gravity_m_s2: float
    hover_thrust_newtons: float


def _validate_positive_scalar(name: str, value: float) -> float:
    scalar_value = float(value)
    if not np.isfinite(scalar_value) or scalar_value <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar.")
    return scalar_value


def _validate_inertia_diagonal(inertia_diagonal_kg_m2: np.ndarray) -> np.ndarray:
    inertia = np.asarray(inertia_diagonal_kg_m2, dtype=np.float64).reshape(-1)
    if inertia.shape != (3,):
        raise ValueError("The inertia diagonal must contain exactly three values.")
    if not np.all(np.isfinite(inertia)) or np.any(inertia <= 0.0):
        raise ValueError("All principal moments of inertia must be finite and positive.")
    return inertia


def create_continuous_hover_model(
    mass_kg: float,
    inertia_diagonal_kg_m2: np.ndarray,
    gravity_m_s2: float = 9.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the hover-linearized CF2X model in native PyBullet coordinates."""
    mass = _validate_positive_scalar("mass_kg", mass_kg)
    gravity = _validate_positive_scalar("gravity_m_s2", gravity_m_s2)
    inertia = _validate_inertia_diagonal(inertia_diagonal_kg_m2)

    state_matrix = np.zeros((STATE_SIZE, STATE_SIZE), dtype=np.float64)
    input_matrix = np.zeros((STATE_SIZE, CONTROL_SIZE), dtype=np.float64)

    state_matrix[POSITION_SLICE, LINEAR_VELOCITY_SLICE] = np.eye(3)
    state_matrix[3, PITCH_INDEX] = gravity
    state_matrix[4, ROLL_INDEX] = -gravity
    state_matrix[ATTITUDE_SLICE, BODY_RATE_SLICE] = np.eye(3)

    input_matrix[5, 0] = 1.0 / mass
    input_matrix[9, 1] = 1.0 / inertia[0]
    input_matrix[10, 2] = 1.0 / inertia[1]
    input_matrix[11, 3] = 1.0 / inertia[2]

    return state_matrix, input_matrix


def discretize_state_space_model(
    continuous_state_matrix: np.ndarray,
    continuous_input_matrix: np.ndarray,
    sample_time_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize a continuous state-space model with zero-order hold."""
    sample_time = _validate_positive_scalar(
        "sample_time_seconds",
        sample_time_seconds,
    )
    state_matrix = np.asarray(continuous_state_matrix, dtype=np.float64)
    input_matrix = np.asarray(continuous_input_matrix, dtype=np.float64)

    if state_matrix.shape != (STATE_SIZE, STATE_SIZE):
        raise ValueError(f"State matrix must have shape {(STATE_SIZE, STATE_SIZE)}.")
    if input_matrix.shape != (STATE_SIZE, CONTROL_SIZE):
        raise ValueError(f"Input matrix must have shape {(STATE_SIZE, CONTROL_SIZE)}.")

    output_matrix = np.eye(STATE_SIZE, dtype=np.float64)
    feedthrough_matrix = np.zeros((STATE_SIZE, CONTROL_SIZE), dtype=np.float64)
    discrete_state_matrix, discrete_input_matrix, _, _, _ = cont2discrete(
        (
            state_matrix,
            input_matrix,
            output_matrix,
            feedthrough_matrix,
        ),
        sample_time,
        method="zoh",
    )

    return (
        np.asarray(discrete_state_matrix, dtype=np.float64),
        np.asarray(discrete_input_matrix, dtype=np.float64),
    )


def create_hover_linear_model(
    mass_kg: float,
    inertia_diagonal_kg_m2: np.ndarray,
    sample_time_seconds: float,
    gravity_m_s2: float = 9.8,
) -> LinearQuadrotorModel:
    """Create continuous and discrete hover-linearized quadrotor dynamics."""
    mass = _validate_positive_scalar("mass_kg", mass_kg)
    gravity = _validate_positive_scalar("gravity_m_s2", gravity_m_s2)
    inertia = _validate_inertia_diagonal(inertia_diagonal_kg_m2)
    continuous_state_matrix, continuous_input_matrix = create_continuous_hover_model(
        mass_kg=mass,
        inertia_diagonal_kg_m2=inertia,
        gravity_m_s2=gravity,
    )
    discrete_state_matrix, discrete_input_matrix = discretize_state_space_model(
        continuous_state_matrix=continuous_state_matrix,
        continuous_input_matrix=continuous_input_matrix,
        sample_time_seconds=sample_time_seconds,
    )

    return LinearQuadrotorModel(
        continuous_state_matrix=continuous_state_matrix,
        continuous_input_matrix=continuous_input_matrix,
        discrete_state_matrix=discrete_state_matrix,
        discrete_input_matrix=discrete_input_matrix,
        sample_time_seconds=float(sample_time_seconds),
        mass_kg=mass,
        inertia_diagonal_kg_m2=inertia.copy(),
        gravity_m_s2=gravity,
        hover_thrust_newtons=mass * gravity,
    )


def wrap_angle_radians(angle_radians: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((float(angle_radians) + pi) % (2.0 * pi) - pi)


def create_nearest_yaw_reference(
    current_yaw_radians: float,
    target_yaw_radians: float,
) -> float:
    """Return the target-yaw equivalent requiring the shortest angular motion."""
    yaw_error = wrap_angle_radians(target_yaw_radians - current_yaw_radians)
    return float(current_yaw_radians + yaw_error)


def create_reference_state(
    target_position_m: np.ndarray,
    target_yaw_radians: float,
) -> np.ndarray:
    """Create a stationary position-and-yaw reference state."""
    target_position = np.asarray(target_position_m, dtype=np.float64).reshape(-1)
    if target_position.shape != (3,):
        raise ValueError("target_position_m must contain exactly three values.")
    if not np.all(np.isfinite(target_position)) or not np.isfinite(target_yaw_radians):
        raise ValueError("Reference position and yaw must be finite.")

    reference_state = np.zeros(STATE_SIZE, dtype=np.float64)
    reference_state[POSITION_SLICE] = target_position
    reference_state[YAW_INDEX] = float(target_yaw_radians)
    return reference_state


def create_state_vector(
    position_m: np.ndarray,
    linear_velocity_world_m_s: np.ndarray,
    rpy_radians: np.ndarray,
    angular_velocity_body_rad_s: np.ndarray,
) -> np.ndarray:
    """Create the MPC state vector from native PyBullet measurements."""
    state_components = tuple(
        np.asarray(component, dtype=np.float64).reshape(-1)
        for component in (
            position_m,
            linear_velocity_world_m_s,
            rpy_radians,
            angular_velocity_body_rad_s,
        )
    )
    if any(component.shape != (3,) for component in state_components):
        raise ValueError("Each state component must contain exactly three values.")
    if not all(np.all(np.isfinite(component)) for component in state_components):
        raise ValueError("MPC state components must be finite.")

    return np.concatenate(state_components)
