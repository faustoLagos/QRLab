from dataclasses import dataclass, field, replace
from math import pi
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from gymnasium import spaces

from controllers import (
    CrazyflieAttitudeRateParameters,
    CrazyflieOuterLoopParameters,
    PidConfig,
    PositionActionBounds,
    VelocityActionLimits,
    create_cf2_attitude_rate_parameters,
    create_cf2_outer_loop_parameters,
)
from environments.BaseRLAviary import BaseRLAviary
from environments.utils.enums import ActionType, DroneModel, Physics


PYB_FREQUENCY_HZ = 1000
CONTROL_FREQUENCY_HZ = 100

PID_LOOP_NAMES = (
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "attitude_roll",
    "attitude_pitch",
    "attitude_yaw",
    "rate_roll",
    "rate_pitch",
    "rate_yaw",
)

PID_GAIN_NAMES = (
    "kp",
    "ki",
    "kd",
    "kff",
)


@dataclass(frozen=True)
class PidGainOverride:
    """Optional simulation-only overrides for one firmware PID configuration."""

    kp: float | None = None
    ki: float | None = None
    kd: float | None = None
    kff: float | None = None


@dataclass(frozen=True)
class PIDSimulationGainOverrides:
    """Simulation-only gain overrides; unspecified values use firmware defaults."""

    position_x: PidGainOverride = field(default_factory=PidGainOverride)
    position_y: PidGainOverride = field(default_factory=PidGainOverride)
    position_z: PidGainOverride = field(default_factory=PidGainOverride)
    velocity_x: PidGainOverride = field(default_factory=PidGainOverride)
    velocity_y: PidGainOverride = field(default_factory=PidGainOverride)
    velocity_z: PidGainOverride = field(default_factory=PidGainOverride)
    attitude_roll: PidGainOverride = field(default_factory=PidGainOverride)
    attitude_pitch: PidGainOverride = field(default_factory=PidGainOverride)
    attitude_yaw: PidGainOverride = field(default_factory=PidGainOverride)
    rate_roll: PidGainOverride = field(default_factory=PidGainOverride)
    rate_pitch: PidGainOverride = field(default_factory=PidGainOverride)
    rate_yaw: PidGainOverride = field(default_factory=PidGainOverride)


@dataclass(frozen=True)
class PIDSimulationConfig:
    """Configuration owned by the standalone firmware-like PID simulation."""

    battery_voltage: float = 3.7
    physics: Physics = Physics.PYB
    position_action_bounds: PositionActionBounds = field(
        default_factory=lambda: PositionActionBounds(
            minimum=(-2.0, -2.0, 0.1, -pi),
            maximum=(2.0, 2.0, 2.0, pi),
        )
    )
    velocity_action_limits: VelocityActionLimits = field(
        default_factory=lambda: VelocityActionLimits(
            maximum_absolute=(1.0, 1.0, 1.0, pi),
            body_frame=False,
        )
    )


def _validate_gain_value(
    loop_name: str,
    gain_name: str,
    raw_value: Any,
) -> float:
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(
            f"PID gain '{loop_name}.{gain_name}' must be a finite numeric scalar."
        )

    gain_value = float(raw_value)

    if not np.isfinite(gain_value):
        raise ValueError(
            f"PID gain '{loop_name}.{gain_name}' must be finite."
        )

    return gain_value


def _validate_pid_gain_mapping(
    loop_name: str,
    raw_mapping: Any,
) -> dict[str, float]:
    if not isinstance(raw_mapping, dict):
        raise ValueError(
            f"PID loop '{loop_name}' must contain a mapping of gains."
        )

    non_string_gain_names = [
        gain_name
        for gain_name in raw_mapping
        if not isinstance(gain_name, str)
    ]
    if non_string_gain_names:
        raise ValueError(
            f"PID loop '{loop_name}' contains non-string gain name(s): "
            f"{', '.join(map(str, non_string_gain_names))}."
        )

    unknown_gain_names = sorted(
        set(raw_mapping) - set(PID_GAIN_NAMES)
    )

    if unknown_gain_names:
        raise ValueError(
            f"Unknown gain(s) for PID loop '{loop_name}': "
            f"{', '.join(unknown_gain_names)}. "
            f"Allowed gains: {', '.join(PID_GAIN_NAMES)}."
        )

    return {
        gain_name: _validate_gain_value(
            loop_name=loop_name,
            gain_name=gain_name,
            raw_value=raw_value,
        )
        for gain_name, raw_value in raw_mapping.items()
    }


def _validate_pid_gain_document(
    raw_document: Any,
) -> dict[str, dict[str, float]]:
    if raw_document is None:
        return {}

    if not isinstance(raw_document, dict):
        raise ValueError(
            "The PID gain YAML root must be a mapping of PID loop names."
        )

    non_string_loop_names = [
        loop_name
        for loop_name in raw_document
        if not isinstance(loop_name, str)
    ]
    if non_string_loop_names:
        raise ValueError(
            "PID loop names must be strings. Invalid key(s): "
            f"{', '.join(map(str, non_string_loop_names))}."
        )

    unknown_loop_names = sorted(
        set(raw_document) - set(PID_LOOP_NAMES)
    )

    if unknown_loop_names:
        raise ValueError(
            "Unknown PID loop(s): "
            f"{', '.join(unknown_loop_names)}. "
            "Allowed loops: "
            f"{', '.join(PID_LOOP_NAMES)}."
        )

    return {
        loop_name: _validate_pid_gain_mapping(
            loop_name=loop_name,
            raw_mapping=raw_mapping,
        )
        for loop_name, raw_mapping in raw_document.items()
    }


def load_pid_gain_overrides(
    yaml_path: Path | None,
) -> PIDSimulationGainOverrides:
    """Load sparse simulation-only PID gain overrides from YAML."""
    if yaml_path is None:
        return PIDSimulationGainOverrides()

    resolved_path = Path(yaml_path).expanduser()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"PID gain YAML file not found: {resolved_path}"
        )

    try:
        with resolved_path.open("r", encoding="utf-8") as yaml_file:
            raw_document = yaml.safe_load(yaml_file)
    except yaml.YAMLError as error:
        raise ValueError(
            f"Invalid YAML in PID gain file '{resolved_path}': {error}"
        ) from error

    validated_document = _validate_pid_gain_document(raw_document)

    return PIDSimulationGainOverrides(
        **{
            loop_name: PidGainOverride(**gain_values)
            for loop_name, gain_values in validated_document.items()
        }
    )


def apply_pid_gain_override(
    pid_config: PidConfig,
    gain_override: PidGainOverride,
) -> PidConfig:
    """Return a PID config with only explicitly requested gains replaced."""
    replacements = {
        gain_name: gain_value
        for gain_name, gain_value in (
            ("kp", gain_override.kp),
            ("ki", gain_override.ki),
            ("kd", gain_override.kd),
            ("kff", gain_override.kff),
        )
        if gain_value is not None
    }

    return replace(pid_config, **replacements)


def create_simulation_outer_loop_parameters(
    gain_overrides: PIDSimulationGainOverrides,
) -> CrazyflieOuterLoopParameters:
    """Derive simulation outer-loop parameters from immutable firmware defaults."""
    firmware_defaults = create_cf2_outer_loop_parameters()

    return replace(
        firmware_defaults,
        position_x=apply_pid_gain_override(
            firmware_defaults.position_x,
            gain_overrides.position_x,
        ),
        position_y=apply_pid_gain_override(
            firmware_defaults.position_y,
            gain_overrides.position_y,
        ),
        position_z=apply_pid_gain_override(
            firmware_defaults.position_z,
            gain_overrides.position_z,
        ),
        velocity_x=apply_pid_gain_override(
            firmware_defaults.velocity_x,
            gain_overrides.velocity_x,
        ),
        velocity_y=apply_pid_gain_override(
            firmware_defaults.velocity_y,
            gain_overrides.velocity_y,
        ),
        velocity_z=apply_pid_gain_override(
            firmware_defaults.velocity_z,
            gain_overrides.velocity_z,
        ),
    )


def create_simulation_attitude_rate_parameters(
    gain_overrides: PIDSimulationGainOverrides,
) -> CrazyflieAttitudeRateParameters:
    """Derive simulation inner-loop parameters from immutable firmware defaults."""
    firmware_defaults = create_cf2_attitude_rate_parameters()

    return replace(
        firmware_defaults,
        attitude_roll=apply_pid_gain_override(
            firmware_defaults.attitude_roll,
            gain_overrides.attitude_roll,
        ),
        attitude_pitch=apply_pid_gain_override(
            firmware_defaults.attitude_pitch,
            gain_overrides.attitude_pitch,
        ),
        attitude_yaw=apply_pid_gain_override(
            firmware_defaults.attitude_yaw,
            gain_overrides.attitude_yaw,
        ),
        rate_roll=apply_pid_gain_override(
            firmware_defaults.rate_roll,
            gain_overrides.rate_roll,
        ),
        rate_pitch=apply_pid_gain_override(
            firmware_defaults.rate_pitch,
            gain_overrides.rate_pitch,
        ),
        rate_yaw=apply_pid_gain_override(
            firmware_defaults.rate_yaw,
            gain_overrides.rate_yaw,
        ),
    )


def normalize_position_command(
    target_xyz: np.ndarray,
    target_yaw_radians: float,
    bounds: PositionActionBounds,
) -> np.ndarray:
    """Map a physical position/yaw command to the normalized action space."""
    physical_command = np.array(
        [target_xyz[0], target_xyz[1], target_xyz[2], target_yaw_radians],
        dtype=np.float64,
    )
    lower = np.asarray(bounds.minimum, dtype=np.float64)
    upper = np.asarray(bounds.maximum, dtype=np.float64)

    if np.any(physical_command < lower) or np.any(physical_command > upper):
        raise ValueError(
            f"Position command {physical_command} lies outside [{lower}, {upper}]."
        )

    normalized_action = 2.0 * (physical_command - lower) / (upper - lower) - 1.0
    return normalized_action.reshape(1, 4).astype(np.float32)


def normalize_velocity_command(
    target_velocity: np.ndarray,
    target_yaw_rate_radians_per_second: float,
    limits: VelocityActionLimits,
) -> np.ndarray:
    """Map a physical velocity/yaw-rate command to the normalized action space."""
    physical_command = np.array(
        [
            target_velocity[0],
            target_velocity[1],
            target_velocity[2],
            target_yaw_rate_radians_per_second,
        ],
        dtype=np.float64,
    )
    maximum_absolute = np.asarray(limits.maximum_absolute, dtype=np.float64)

    if np.any(np.abs(physical_command) > maximum_absolute):
        raise ValueError(
            f"Velocity command {physical_command} exceeds symmetric limits "
            f"{maximum_absolute}."
        )

    normalized_action = physical_command / maximum_absolute
    return normalized_action.reshape(1, 4).astype(np.float32)


def create_pid_simulation_config(
    battery_voltage: float = 3.7,
    body_frame_velocity: bool = False,
) -> PIDSimulationConfig:
    """Create the standard standalone PID simulation configuration."""
    default_config = PIDSimulationConfig()

    return replace(
        default_config,
        battery_voltage=float(battery_voltage),
        velocity_action_limits=replace(
            default_config.velocity_action_limits,
            body_frame=body_frame_velocity,
        ),
    )


class PIDSimulationEnv(BaseRLAviary):
    """Simulation-only environment for the firmware-like Crazyflie PID stack."""

    def __init__(
        self,
        action_type: ActionType,
        initial_xyz: np.ndarray | None = None,
        initial_rpy: np.ndarray | None = None,
        config: PIDSimulationConfig | None = None,
        gain_overrides: PIDSimulationGainOverrides | None = None,
        gui: bool = False,
        record: bool = False,
    ) -> None:
        simulation_config = config or PIDSimulationConfig()
        simulation_gain_overrides = gain_overrides or PIDSimulationGainOverrides()
        initial_position = (
            np.array([0.0, 0.0, 0.1], dtype=np.float64)
            if initial_xyz is None
            else np.asarray(initial_xyz, dtype=np.float64)
        )
        initial_orientation = (
            np.zeros(3, dtype=np.float64)
            if initial_rpy is None
            else np.asarray(initial_rpy, dtype=np.float64)
        )

        self.PID_SIMULATION_CONFIG = simulation_config
        self.PID_SIMULATION_GAIN_OVERRIDES = simulation_gain_overrides

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=initial_position.reshape(1, 3),
            initial_rpys=initial_orientation.reshape(1, 3),
            physics=simulation_config.physics,
            pyb_freq=PYB_FREQUENCY_HZ,
            ctrl_freq=CONTROL_FREQUENCY_HZ,
            gui=gui,
            record=record,
            act=action_type,
            firmware_battery_voltage=simulation_config.battery_voltage,
            position_action_bounds=(
                simulation_config.position_action_bounds
                if action_type == ActionType.POSITION
                else None
            ),
            velocity_action_limits=(
                simulation_config.velocity_action_limits
                if action_type == ActionType.VELOCITY
                else None
            ),
        )

        self.CRAZYFLIE_OUTER_PARAMETERS = create_simulation_outer_loop_parameters(
            simulation_gain_overrides
        )
        self.CRAZYFLIE_INNER_PARAMETERS = create_simulation_attitude_rate_parameters(
            simulation_gain_overrides
        )
        self._initializeCrazyflieRuntimeStates()

    def create_position_action(
        self,
        target_xyz: np.ndarray,
        target_yaw_radians: float,
    ) -> np.ndarray:
        """Create a normalized action from a physical position setpoint."""
        if self.ACT_TYPE != ActionType.POSITION:
            raise ValueError("Position commands require ActionType.POSITION.")

        return normalize_position_command(
            target_xyz=np.asarray(target_xyz, dtype=np.float64),
            target_yaw_radians=target_yaw_radians,
            bounds=self.PID_SIMULATION_CONFIG.position_action_bounds,
        )

    def create_velocity_action(
        self,
        target_velocity: np.ndarray,
        target_yaw_rate_radians_per_second: float,
    ) -> np.ndarray:
        """Create a normalized action from a physical velocity setpoint."""
        if self.ACT_TYPE != ActionType.VELOCITY:
            raise ValueError("Velocity commands require ActionType.VELOCITY.")

        return normalize_velocity_command(
            target_velocity=np.asarray(target_velocity, dtype=np.float64),
            target_yaw_rate_radians_per_second=target_yaw_rate_radians_per_second,
            limits=self.PID_SIMULATION_CONFIG.velocity_action_limits,
        )

    def _observationSpace(self) -> spaces.Box:
        lower = np.full((1, 16), -np.inf, dtype=np.float32)
        upper = np.full((1, 16), np.inf, dtype=np.float32)
        return spaces.Box(low=lower, high=upper, dtype=np.float32)

    def _computeObs(self) -> np.ndarray:
        return self._getDroneStateVector(0)[:16].reshape(1, 16).astype(np.float32)

    def _computeReward(self) -> float:
        return 0.0

    def _computeTerminated(self) -> bool:
        return False

    def _computeTruncated(self) -> bool:
        return False

    def _computeInfo(self) -> dict:
        return {}
