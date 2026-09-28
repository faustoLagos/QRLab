from dataclasses import dataclass, field, replace
from math import pi
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import sparse
from scipy.linalg import solve_discrete_are

from controllers.mpc_model import CONTROL_SIZE, STATE_SIZE, LinearQuadrotorModel


MPC_WEIGHT_SECTIONS = ("state_weights", "input_weights")
MPC_STATE_WEIGHT_NAMES = (
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "roll",
    "pitch",
    "yaw",
    "rate_roll",
    "rate_pitch",
    "rate_yaw",
)
MPC_INPUT_WEIGHT_NAMES = (
    "thrust",
    "torque_roll",
    "torque_pitch",
    "torque_yaw",
)


@dataclass(frozen=True)
class MPCStateWeights:
    position_x: float = 40.0
    position_y: float = 40.0
    position_z: float = 60.0
    velocity_x: float = 6.0
    velocity_y: float = 6.0
    velocity_z: float = 10.0
    roll: float = 12.0
    pitch: float = 12.0
    yaw: float = 15.0
    rate_roll: float = 0.5
    rate_pitch: float = 0.5
    rate_yaw: float = 0.5


@dataclass(frozen=True)
class MPCInputWeights:
    thrust: float = 0.25
    torque_roll: float = 0.08
    torque_pitch: float = 0.08
    torque_yaw: float = 0.05


@dataclass(frozen=True)
class MPCWeights:
    state: MPCStateWeights = field(default_factory=MPCStateWeights)
    input: MPCInputWeights = field(default_factory=MPCInputWeights)


@dataclass(frozen=True)
class MPCStateWeightOverrides:
    position_x: float | None = None
    position_y: float | None = None
    position_z: float | None = None
    velocity_x: float | None = None
    velocity_y: float | None = None
    velocity_z: float | None = None
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    rate_roll: float | None = None
    rate_pitch: float | None = None
    rate_yaw: float | None = None


@dataclass(frozen=True)
class MPCInputWeightOverrides:
    thrust: float | None = None
    torque_roll: float | None = None
    torque_pitch: float | None = None
    torque_yaw: float | None = None


@dataclass(frozen=True)
class MPCWeightOverrides:
    state: MPCStateWeightOverrides = field(default_factory=MPCStateWeightOverrides)
    input: MPCInputWeightOverrides = field(default_factory=MPCInputWeightOverrides)


@dataclass(frozen=True)
class MPCStateLimits:
    minimum_altitude_m: float = 0.0
    maximum_altitude_m: float = 3.0
    maximum_horizontal_speed_m_s: float = 3.0
    maximum_vertical_speed_m_s: float = 3.0
    maximum_tilt_radians: float = pi / 6.0
    maximum_body_rate_rad_s: float = 8.0


@dataclass(frozen=True)
class MPCConfig:
    horizon_steps: int = 20
    weights: MPCWeights = field(default_factory=MPCWeights)
    state_limits: MPCStateLimits = field(default_factory=MPCStateLimits)
    absolute_tolerance: float = 1e-4
    relative_tolerance: float = 1e-4
    maximum_iterations: int = 4000
    polishing: bool = True
    verbose: bool = False


@dataclass
class MPCWorkspace:
    solver: Any
    config: MPCConfig
    model: LinearQuadrotorModel
    state_weight_matrix: np.ndarray
    terminal_weight_matrix: np.ndarray
    input_weight_matrix: np.ndarray
    lower_constraint_bound: np.ndarray
    upper_constraint_bound: np.ndarray
    state_variable_count: int
    input_variable_count: int
    equality_constraint_count: int


@dataclass(frozen=True)
class MPCSolution:
    delta_wrench: np.ndarray
    predicted_states: np.ndarray
    predicted_delta_wrenches: np.ndarray
    status: str
    status_value: int
    objective_value: float
    solve_time_seconds: float
    iterations: int


def _validate_weight_value(
    section_name: str,
    weight_name: str,
    raw_value: Any,
    allow_zero: bool,
) -> float:
    if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
        raise ValueError(
            f"MPC weight '{section_name}.{weight_name}' must be a finite numeric scalar."
        )

    weight_value = float(raw_value)
    if not np.isfinite(weight_value):
        raise ValueError(
            f"MPC weight '{section_name}.{weight_name}' must be finite."
        )

    minimum_is_invalid = weight_value < 0.0 if allow_zero else weight_value <= 0.0
    if minimum_is_invalid:
        requirement = "non-negative" if allow_zero else "strictly positive"
        raise ValueError(
            f"MPC weight '{section_name}.{weight_name}' must be {requirement}."
        )

    return weight_value


def _validate_weight_mapping(
    section_name: str,
    raw_mapping: Any,
    allowed_weight_names: tuple[str, ...],
    allow_zero: bool,
) -> dict[str, float]:
    if not isinstance(raw_mapping, dict):
        raise ValueError(
            f"MPC section '{section_name}' must contain a mapping of weights."
        )

    non_string_names = [
        weight_name
        for weight_name in raw_mapping
        if not isinstance(weight_name, str)
    ]
    if non_string_names:
        raise ValueError(
            f"MPC section '{section_name}' contains non-string weight name(s): "
            f"{', '.join(map(str, non_string_names))}."
        )

    unknown_weight_names = sorted(set(raw_mapping) - set(allowed_weight_names))
    if unknown_weight_names:
        raise ValueError(
            f"Unknown MPC weight(s) in '{section_name}': "
            f"{', '.join(unknown_weight_names)}. Allowed weights: "
            f"{', '.join(allowed_weight_names)}."
        )

    return {
        weight_name: _validate_weight_value(
            section_name=section_name,
            weight_name=weight_name,
            raw_value=raw_value,
            allow_zero=allow_zero,
        )
        for weight_name, raw_value in raw_mapping.items()
    }


def _validate_mpc_weight_document(
    raw_document: Any,
) -> dict[str, dict[str, float]]:
    if raw_document is None:
        return {}
    if not isinstance(raw_document, dict):
        raise ValueError(
            "The MPC weight YAML root must be a mapping of weight sections."
        )

    non_string_sections = [
        section_name
        for section_name in raw_document
        if not isinstance(section_name, str)
    ]
    if non_string_sections:
        raise ValueError(
            "MPC weight section names must be strings. Invalid key(s): "
            f"{', '.join(map(str, non_string_sections))}."
        )

    unknown_sections = sorted(set(raw_document) - set(MPC_WEIGHT_SECTIONS))
    if unknown_sections:
        raise ValueError(
            f"Unknown MPC weight section(s): {', '.join(unknown_sections)}. "
            f"Allowed sections: {', '.join(MPC_WEIGHT_SECTIONS)}."
        )

    validators = {
        "state_weights": (MPC_STATE_WEIGHT_NAMES, True),
        "input_weights": (MPC_INPUT_WEIGHT_NAMES, False),
    }
    return {
        section_name: _validate_weight_mapping(
            section_name=section_name,
            raw_mapping=raw_mapping,
            allowed_weight_names=validators[section_name][0],
            allow_zero=validators[section_name][1],
        )
        for section_name, raw_mapping in raw_document.items()
    }


def load_mpc_weight_overrides(
    yaml_path: Path | None,
) -> MPCWeightOverrides:
    """Load sparse simulation-only MPC Q/R weight overrides from YAML."""
    if yaml_path is None:
        return MPCWeightOverrides()

    resolved_path = Path(yaml_path).expanduser()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"MPC weight YAML file not found: {resolved_path}")

    try:
        with resolved_path.open("r", encoding="utf-8") as yaml_file:
            raw_document = yaml.safe_load(yaml_file)
    except yaml.YAMLError as error:
        raise ValueError(
            f"Invalid YAML in MPC weight file '{resolved_path}': {error}"
        ) from error

    validated_document = _validate_mpc_weight_document(raw_document)
    return MPCWeightOverrides(
        state=MPCStateWeightOverrides(
            **validated_document.get("state_weights", {})
        ),
        input=MPCInputWeightOverrides(
            **validated_document.get("input_weights", {})
        ),
    )


def _apply_weight_override(
    default_weights: Any,
    overrides: Any,
    weight_names: tuple[str, ...],
) -> Any:
    replacements = {
        weight_name: getattr(overrides, weight_name)
        for weight_name in weight_names
        if getattr(overrides, weight_name) is not None
    }
    return replace(default_weights, **replacements)


def create_state_weight_vector(weights: MPCStateWeights) -> np.ndarray:
    """Return the Q diagonal in MPC state-vector order."""
    return np.asarray(
        [getattr(weights, weight_name) for weight_name in MPC_STATE_WEIGHT_NAMES],
        dtype=np.float64,
    )


def create_input_weight_vector(weights: MPCInputWeights) -> np.ndarray:
    """Return the normalized-input R diagonal in MPC input-vector order."""
    return np.asarray(
        [getattr(weights, weight_name) for weight_name in MPC_INPUT_WEIGHT_NAMES],
        dtype=np.float64,
    )


def validate_mpc_weights(weights: MPCWeights) -> None:
    """Validate a complete effective MPC Q/R weight configuration."""
    state_weights = create_state_weight_vector(weights.state)
    input_weights = create_input_weight_vector(weights.input)

    if not np.all(np.isfinite(state_weights)) or np.any(state_weights < 0.0):
        raise ValueError("All MPC state weights must be finite and non-negative.")
    if not np.any(state_weights > 0.0):
        raise ValueError("At least one MPC state weight must be strictly positive.")
    if not np.all(np.isfinite(input_weights)) or np.any(input_weights <= 0.0):
        raise ValueError("All MPC input weights must be finite and strictly positive.")


def create_default_mpc_weights() -> MPCWeights:
    """Create the immutable QRLab baseline MPC Q/R weights."""
    return MPCWeights()


def apply_mpc_weight_overrides(
    default_weights: MPCWeights,
    overrides: MPCWeightOverrides,
) -> MPCWeights:
    """Apply only explicitly specified YAML values to the MPC defaults."""
    effective_weights = MPCWeights(
        state=_apply_weight_override(
            default_weights=default_weights.state,
            overrides=overrides.state,
            weight_names=MPC_STATE_WEIGHT_NAMES,
        ),
        input=_apply_weight_override(
            default_weights=default_weights.input,
            overrides=overrides.input,
            weight_names=MPC_INPUT_WEIGHT_NAMES,
        ),
    )
    validate_mpc_weights(effective_weights)
    return effective_weights


def _validate_positive_vector(
    name: str,
    values: tuple[float, ...] | np.ndarray,
    expected_size: int,
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (expected_size,):
        raise ValueError(f"{name} must contain exactly {expected_size} values.")
    if not np.all(np.isfinite(vector)) or np.any(vector <= 0.0):
        raise ValueError(f"All {name} values must be finite and positive.")
    return vector


def _validate_mpc_config(config: MPCConfig) -> None:
    if config.horizon_steps < 1:
        raise ValueError("MPC horizon_steps must be at least 1.")
    if config.absolute_tolerance < 0.0 or config.relative_tolerance < 0.0:
        raise ValueError("OSQP tolerances must be non-negative.")
    if config.maximum_iterations < 1:
        raise ValueError("maximum_iterations must be at least 1.")

    limits = config.state_limits
    if limits.minimum_altitude_m >= limits.maximum_altitude_m:
        raise ValueError("minimum_altitude_m must be below maximum_altitude_m.")
    if min(
        limits.maximum_horizontal_speed_m_s,
        limits.maximum_vertical_speed_m_s,
        limits.maximum_tilt_radians,
        limits.maximum_body_rate_rad_s,
    ) <= 0.0:
        raise ValueError("Symmetric MPC state limits must be positive.")


def create_state_bounds(config: MPCConfig) -> tuple[np.ndarray, np.ndarray]:
    """Create state constraints for the hover-linearized MPC model."""
    _validate_mpc_config(config)
    limits = config.state_limits

    lower_bound = np.array(
        [
            -np.inf,
            -np.inf,
            limits.minimum_altitude_m,
            -limits.maximum_horizontal_speed_m_s,
            -limits.maximum_horizontal_speed_m_s,
            -limits.maximum_vertical_speed_m_s,
            -limits.maximum_tilt_radians,
            -limits.maximum_tilt_radians,
            -np.inf,
            -limits.maximum_body_rate_rad_s,
            -limits.maximum_body_rate_rad_s,
            -limits.maximum_body_rate_rad_s,
        ],
        dtype=np.float64,
    )
    upper_bound = np.array(
        [
            np.inf,
            np.inf,
            limits.maximum_altitude_m,
            limits.maximum_horizontal_speed_m_s,
            limits.maximum_horizontal_speed_m_s,
            limits.maximum_vertical_speed_m_s,
            limits.maximum_tilt_radians,
            limits.maximum_tilt_radians,
            np.inf,
            limits.maximum_body_rate_rad_s,
            limits.maximum_body_rate_rad_s,
            limits.maximum_body_rate_rad_s,
        ],
        dtype=np.float64,
    )
    return lower_bound, upper_bound


def create_mpc_cost_matrices(
    model: LinearQuadrotorModel,
    config: MPCConfig,
    input_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create state, terminal, and normalized physical-input cost matrices."""
    _validate_mpc_config(config)
    validate_mpc_weights(config.weights)
    state_weights = create_state_weight_vector(config.weights.state)
    input_weights = create_input_weight_vector(config.weights.input)
    scales = _validate_positive_vector(
        "input scales",
        input_scales,
        CONTROL_SIZE,
    )

    state_weight_matrix = np.diag(state_weights)
    input_weight_matrix = np.diag(input_weights / np.square(scales))
    terminal_weight_matrix = solve_discrete_are(
        model.discrete_state_matrix,
        model.discrete_input_matrix,
        state_weight_matrix,
        input_weight_matrix,
    )
    terminal_weight_matrix = 0.5 * (
        terminal_weight_matrix + terminal_weight_matrix.T
    )

    return state_weight_matrix, terminal_weight_matrix, input_weight_matrix


def _create_quadratic_cost_matrix(
    state_weight_matrix: np.ndarray,
    terminal_weight_matrix: np.ndarray,
    input_weight_matrix: np.ndarray,
    horizon_steps: int,
) -> sparse.csc_matrix:
    state_blocks = [state_weight_matrix] * horizon_steps + [terminal_weight_matrix]
    input_blocks = [input_weight_matrix] * horizon_steps
    return sparse.block_diag(
        [2.0 * block for block in state_blocks + input_blocks],
        format="csc",
    )


def _create_linear_cost_vector(
    reference_state: np.ndarray,
    state_weight_matrix: np.ndarray,
    terminal_weight_matrix: np.ndarray,
    horizon_steps: int,
) -> np.ndarray:
    stage_cost = -2.0 * state_weight_matrix @ reference_state
    terminal_cost = -2.0 * terminal_weight_matrix @ reference_state
    return np.concatenate(
        [
            np.tile(stage_cost, horizon_steps),
            terminal_cost,
            np.zeros(horizon_steps * CONTROL_SIZE, dtype=np.float64),
        ]
    )


def _create_dynamics_constraint_matrix(
    model: LinearQuadrotorModel,
    horizon_steps: int,
) -> sparse.csc_matrix:
    discrete_state_matrix = sparse.csc_matrix(model.discrete_state_matrix)
    discrete_input_matrix = sparse.csc_matrix(model.discrete_input_matrix)

    state_dynamics = (
        sparse.kron(
            sparse.eye(horizon_steps + 1, format="csc"),
            -sparse.eye(STATE_SIZE, format="csc"),
            format="csc",
        )
        + sparse.kron(
            sparse.eye(horizon_steps + 1, k=-1, format="csc"),
            discrete_state_matrix,
            format="csc",
        )
    )
    input_dynamics = sparse.kron(
        sparse.vstack(
            [
                sparse.csc_matrix((1, horizon_steps)),
                sparse.eye(horizon_steps, format="csc"),
            ],
            format="csc",
        ),
        discrete_input_matrix,
        format="csc",
    )
    return sparse.hstack([state_dynamics, input_dynamics], format="csc")


def _create_state_constraint_matrix(
    horizon_steps: int,
) -> sparse.csc_matrix:
    state_variable_count = (horizon_steps + 1) * STATE_SIZE
    input_variable_count = horizon_steps * CONTROL_SIZE
    return sparse.hstack(
        [
            sparse.eye(state_variable_count, format="csc"),
            sparse.csc_matrix((state_variable_count, input_variable_count)),
        ],
        format="csc",
    )


def _create_actuator_constraint_matrix(
    actuator_constraint_matrix: np.ndarray,
    horizon_steps: int,
) -> sparse.csc_matrix:
    actuator_matrix = np.asarray(actuator_constraint_matrix, dtype=np.float64)
    if actuator_matrix.shape[1] != CONTROL_SIZE:
        raise ValueError(
            f"actuator_constraint_matrix must have {CONTROL_SIZE} columns."
        )

    state_variable_count = (horizon_steps + 1) * STATE_SIZE
    horizon_actuator_matrix = sparse.kron(
        sparse.eye(horizon_steps, format="csc"),
        sparse.csc_matrix(actuator_matrix),
        format="csc",
    )
    return sparse.hstack(
        [
            sparse.csc_matrix(
                (horizon_actuator_matrix.shape[0], state_variable_count)
            ),
            horizon_actuator_matrix,
        ],
        format="csc",
    )


def _create_constraint_bounds(
    config: MPCConfig,
    actuator_lower_bound: np.ndarray,
    actuator_upper_bound: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    horizon_steps = config.horizon_steps
    equality_constraint_count = (horizon_steps + 1) * STATE_SIZE
    state_lower_bound, state_upper_bound = create_state_bounds(config)
    actuator_lower = np.asarray(actuator_lower_bound, dtype=np.float64).reshape(-1)
    actuator_upper = np.asarray(actuator_upper_bound, dtype=np.float64).reshape(-1)

    if actuator_lower.shape != actuator_upper.shape:
        raise ValueError("Actuator lower and upper bounds must have the same shape.")
    if np.any(actuator_lower > actuator_upper):
        raise ValueError("Actuator lower bounds cannot exceed upper bounds.")

    equality_bound = np.zeros(equality_constraint_count, dtype=np.float64)
    horizon_state_lower = np.tile(state_lower_bound, horizon_steps + 1)
    horizon_state_upper = np.tile(state_upper_bound, horizon_steps + 1)
    horizon_state_lower[:STATE_SIZE] = -np.inf
    horizon_state_upper[:STATE_SIZE] = np.inf

    lower_bound = np.concatenate(
        [
            equality_bound,
            horizon_state_lower,
            np.tile(actuator_lower, horizon_steps),
        ]
    )
    upper_bound = np.concatenate(
        [
            equality_bound,
            horizon_state_upper,
            np.tile(actuator_upper, horizon_steps),
        ]
    )
    return lower_bound, upper_bound, equality_constraint_count


def _load_osqp() -> Any:
    try:
        import osqp
    except ImportError as error:
        raise ImportError(
            "The MPC baseline requires OSQP. Install QRLab with the 'mpc' extra."
        ) from error
    return osqp


def create_mpc_workspace(
    model: LinearQuadrotorModel,
    config: MPCConfig,
    actuator_constraint_matrix: np.ndarray,
    actuator_lower_bound: np.ndarray,
    actuator_upper_bound: np.ndarray,
    input_scales: np.ndarray,
) -> MPCWorkspace:
    """Create and factorize the sparse constrained MPC QP once."""
    _validate_mpc_config(config)
    state_weight_matrix, terminal_weight_matrix, input_weight_matrix = (
        create_mpc_cost_matrices(
            model=model,
            config=config,
            input_scales=input_scales,
        )
    )
    quadratic_cost = _create_quadratic_cost_matrix(
        state_weight_matrix=state_weight_matrix,
        terminal_weight_matrix=terminal_weight_matrix,
        input_weight_matrix=input_weight_matrix,
        horizon_steps=config.horizon_steps,
    )
    zero_reference = np.zeros(STATE_SIZE, dtype=np.float64)
    linear_cost = _create_linear_cost_vector(
        reference_state=zero_reference,
        state_weight_matrix=state_weight_matrix,
        terminal_weight_matrix=terminal_weight_matrix,
        horizon_steps=config.horizon_steps,
    )
    dynamics_constraints = _create_dynamics_constraint_matrix(
        model=model,
        horizon_steps=config.horizon_steps,
    )
    state_constraints = _create_state_constraint_matrix(config.horizon_steps)
    actuator_constraints = _create_actuator_constraint_matrix(
        actuator_constraint_matrix=actuator_constraint_matrix,
        horizon_steps=config.horizon_steps,
    )
    constraint_matrix = sparse.vstack(
        [dynamics_constraints, state_constraints, actuator_constraints],
        format="csc",
    )
    lower_bound, upper_bound, equality_constraint_count = _create_constraint_bounds(
        config=config,
        actuator_lower_bound=actuator_lower_bound,
        actuator_upper_bound=actuator_upper_bound,
    )

    osqp = _load_osqp()
    solver = osqp.OSQP()
    solver.setup(
        P=quadratic_cost,
        q=linear_cost,
        A=constraint_matrix,
        l=lower_bound,
        u=upper_bound,
        warm_starting=True,
        polishing=config.polishing,
        eps_abs=config.absolute_tolerance,
        eps_rel=config.relative_tolerance,
        max_iter=config.maximum_iterations,
        verbose=config.verbose,
    )

    return MPCWorkspace(
        solver=solver,
        config=config,
        model=model,
        state_weight_matrix=state_weight_matrix,
        terminal_weight_matrix=terminal_weight_matrix,
        input_weight_matrix=input_weight_matrix,
        lower_constraint_bound=lower_bound,
        upper_constraint_bound=upper_bound,
        state_variable_count=(config.horizon_steps + 1) * STATE_SIZE,
        input_variable_count=config.horizon_steps * CONTROL_SIZE,
        equality_constraint_count=equality_constraint_count,
    )


def reset_mpc_workspace(workspace: MPCWorkspace) -> None:
    """Clear the solver warm start between independent simulation episodes."""
    total_variables = workspace.state_variable_count + workspace.input_variable_count
    workspace.solver.warm_start(
        x=np.zeros(total_variables, dtype=np.float64),
        y=np.zeros_like(workspace.lower_constraint_bound),
    )


def solve_mpc(
    workspace: MPCWorkspace,
    current_state: np.ndarray,
    reference_state: np.ndarray,
) -> MPCSolution:
    """Update only QP vectors, solve, and return the first optimal control input."""
    current = np.asarray(current_state, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_state, dtype=np.float64).reshape(-1)
    if current.shape != (STATE_SIZE,) or reference.shape != (STATE_SIZE,):
        raise ValueError(f"MPC state and reference must both have shape ({STATE_SIZE},).")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(reference)):
        raise ValueError("MPC state and reference must be finite.")

    linear_cost = _create_linear_cost_vector(
        reference_state=reference,
        state_weight_matrix=workspace.state_weight_matrix,
        terminal_weight_matrix=workspace.terminal_weight_matrix,
        horizon_steps=workspace.config.horizon_steps,
    )
    lower_bound = workspace.lower_constraint_bound.copy()
    upper_bound = workspace.upper_constraint_bound.copy()
    lower_bound[:STATE_SIZE] = -current
    upper_bound[:STATE_SIZE] = -current

    workspace.solver.update(q=linear_cost, l=lower_bound, u=upper_bound)
    result = workspace.solver.solve()
    status = str(result.info.status)

    if result.x is None or not status.lower().startswith("solved"):
        raise RuntimeError(f"OSQP failed to solve the MPC problem: {status}.")

    state_solution = np.asarray(
        result.x[: workspace.state_variable_count],
        dtype=np.float64,
    ).reshape(workspace.config.horizon_steps + 1, STATE_SIZE)
    input_solution = np.asarray(
        result.x[workspace.state_variable_count :],
        dtype=np.float64,
    ).reshape(workspace.config.horizon_steps, CONTROL_SIZE)

    return MPCSolution(
        delta_wrench=input_solution[0].copy(),
        predicted_states=state_solution,
        predicted_delta_wrenches=input_solution,
        status=status,
        status_value=int(result.info.status_val),
        objective_value=float(result.info.obj_val),
        solve_time_seconds=float(result.info.solve_time),
        iterations=int(result.info.iter),
    )
