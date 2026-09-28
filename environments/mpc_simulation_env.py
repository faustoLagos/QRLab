from dataclasses import dataclass, field, replace
from math import pi

import numpy as np
from gymnasium import spaces

from controllers.mpc import (
    MPCConfig,
    MPCSolution,
    MPCWeights,
    MPCWorkspace,
    create_mpc_workspace,
    reset_mpc_workspace,
    solve_mpc,
)
from controllers.mpc_model import (
    create_hover_linear_model,
    create_nearest_yaw_reference,
    create_reference_state,
    create_state_vector,
)
from controllers.quadrotor_control_allocation import (
    QuadrotorControlAllocation,
    allocate_wrench_to_rpm,
    create_cf2x_control_allocation,
    create_delta_wrench_rotor_constraints,
)
from environments.BaseAviary import BaseAviary
from environments.utils.enums import DroneModel, Physics


PYB_FREQUENCY_HZ = 1000
CONTROL_FREQUENCY_HZ = 100


@dataclass(frozen=True)
class MPCSimulationConfig:
    """Configuration for the standalone simulation-only MPC baseline."""

    physics: Physics = Physics.PYB
    controller: MPCConfig = field(default_factory=MPCConfig)
    target_minimum: tuple[float, float, float, float] = (
        -2.0,
        -2.0,
        0.1,
        -pi,
    )
    target_maximum: tuple[float, float, float, float] = (
        2.0,
        2.0,
        2.0,
        pi,
    )


def create_mpc_simulation_config(
    horizon_steps: int = 20,
    weights: MPCWeights | None = None,
) -> MPCSimulationConfig:
    """Create the standard MPC simulation configuration."""
    default_config = MPCSimulationConfig()
    return replace(
        default_config,
        controller=replace(
            default_config.controller,
            horizon_steps=int(horizon_steps),
            weights=weights or default_config.controller.weights,
        ),
    )


def _validate_target_bounds(config: MPCSimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(config.target_minimum, dtype=np.float64)
    upper = np.asarray(config.target_maximum, dtype=np.float64)
    if lower.shape != (4,) or upper.shape != (4,):
        raise ValueError("MPC target bounds must contain four values each.")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("MPC target bounds must be finite.")
    if np.any(lower >= upper):
        raise ValueError("Every MPC target lower bound must be below its upper bound.")
    return lower, upper


class MPCSimulationEnv(BaseAviary):
    """Simulation-only constrained linear MPC baseline for the CF2X plant."""

    def __init__(
        self,
        initial_xyz: np.ndarray | None = None,
        initial_rpy: np.ndarray | None = None,
        config: MPCSimulationConfig | None = None,
        gui: bool = False,
        record: bool = False,
    ) -> None:
        simulation_config = config or MPCSimulationConfig()
        _validate_target_bounds(simulation_config)
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
        if initial_position.shape != (3,) or initial_orientation.shape != (3,):
            raise ValueError("initial_xyz and initial_rpy must both have shape (3,).")

        self.MPC_SIMULATION_CONFIG = simulation_config
        self.MPC_WORKSPACE: MPCWorkspace | None = None
        self.MPC_ALLOCATION: QuadrotorControlAllocation | None = None
        self.LAST_MPC_SOLUTION: MPCSolution | None = None
        self.LAST_COMMANDED_WRENCH = np.zeros(4, dtype=np.float64)

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
        )

        self._initialize_mpc()

    def _initialize_mpc(self) -> None:
        model = create_hover_linear_model(
            mass_kg=self.M,
            inertia_diagonal_kg_m2=np.diag(self.J),
            sample_time_seconds=self.CTRL_TIMESTEP,
            gravity_m_s2=self.G,
        )
        allocation = create_cf2x_control_allocation(
            arm_length_m=self.L,
            thrust_coefficient=self.KF,
            torque_coefficient=self.KM,
            max_rpm=self.MAX_RPM,
        )
        constraint_matrix, lower_bound, upper_bound = (
            create_delta_wrench_rotor_constraints(
                hover_thrust_newtons=model.hover_thrust_newtons,
                allocation=allocation,
            )
        )
        input_scales = np.array(
            [
                model.hover_thrust_newtons,
                self.MAX_XY_TORQUE,
                self.MAX_XY_TORQUE,
                self.MAX_Z_TORQUE,
            ],
            dtype=np.float64,
        )

        self.MPC_ALLOCATION = allocation
        self.MPC_WORKSPACE = create_mpc_workspace(
            model=model,
            config=self.MPC_SIMULATION_CONFIG.controller,
            actuator_constraint_matrix=constraint_matrix,
            actuator_lower_bound=lower_bound,
            actuator_upper_bound=upper_bound,
            input_scales=input_scales,
        )

    def create_position_command(
        self,
        target_xyz: np.ndarray,
        target_yaw_radians: float,
    ) -> np.ndarray:
        """Create a physical [x, y, z, yaw] command accepted by the MPC environment."""
        target_position = np.asarray(target_xyz, dtype=np.float64).reshape(-1)
        if target_position.shape != (3,):
            raise ValueError("target_xyz must contain exactly three values.")

        command = np.array(
            [
                target_position[0],
                target_position[1],
                target_position[2],
                target_yaw_radians,
            ],
            dtype=np.float64,
        )
        lower, upper = _validate_target_bounds(self.MPC_SIMULATION_CONFIG)
        if np.any(command < lower) or np.any(command > upper):
            raise ValueError(f"MPC target {command} lies outside [{lower}, {upper}].")
        return command.reshape(1, 4)

    def _read_mpc_state(self) -> np.ndarray:
        angular_velocity_body = self._convertWorldVectorToBodyFrame(
            vector_world=self.ang_v[0],
            quaternion_xyzw=self.quat[0],
        )
        return create_state_vector(
            position_m=self.pos[0],
            linear_velocity_world_m_s=self.vel[0],
            rpy_radians=self.rpy[0],
            angular_velocity_body_rad_s=angular_velocity_body,
        )

    def _actionSpace(self) -> spaces.Box:
        lower, upper = _validate_target_bounds(self.MPC_SIMULATION_CONFIG)
        return spaces.Box(
            low=lower.reshape(1, 4).astype(np.float32),
            high=upper.reshape(1, 4).astype(np.float32),
            dtype=np.float32,
        )

    def _observationSpace(self) -> spaces.Box:
        lower = np.full((1, 16), -np.inf, dtype=np.float32)
        upper = np.full((1, 16), np.inf, dtype=np.float32)
        return spaces.Box(low=lower, high=upper, dtype=np.float32)

    def _computeObs(self) -> np.ndarray:
        return self._getDroneStateVector(0)[:16].reshape(1, 16).astype(np.float32)

    def _preprocessAction(self, action: np.ndarray) -> np.ndarray:
        command = np.asarray(action, dtype=np.float64).reshape(-1)
        if command.shape != (4,):
            raise ValueError("MPC action must contain [x, y, z, yaw].")
        lower, upper = _validate_target_bounds(self.MPC_SIMULATION_CONFIG)
        if np.any(command < lower) or np.any(command > upper):
            raise ValueError(f"MPC target {command} lies outside [{lower}, {upper}].")
        if self.MPC_WORKSPACE is None or self.MPC_ALLOCATION is None:
            raise RuntimeError("MPC workspace has not been initialized.")

        current_state = self._read_mpc_state()
        target_yaw = create_nearest_yaw_reference(
            current_yaw_radians=current_state[8],
            target_yaw_radians=command[3],
        )
        reference_state = create_reference_state(
            target_position_m=command[:3],
            target_yaw_radians=target_yaw,
        )
        solution = solve_mpc(
            workspace=self.MPC_WORKSPACE,
            current_state=current_state,
            reference_state=reference_state,
        )
        commanded_wrench = solution.delta_wrench.copy()
        commanded_wrench[0] += self.MPC_WORKSPACE.model.hover_thrust_newtons
        motor_rpm = allocate_wrench_to_rpm(
            wrench=commanded_wrench,
            allocation=self.MPC_ALLOCATION,
        )

        self.LAST_MPC_SOLUTION = solution
        self.LAST_COMMANDED_WRENCH = commanded_wrench
        return motor_rpm.reshape(1, 4)

    def _resetControllerState(self, options: dict | None = None) -> None:
        del options
        self.LAST_MPC_SOLUTION = None
        self.LAST_COMMANDED_WRENCH = np.zeros(4, dtype=np.float64)
        if self.MPC_WORKSPACE is not None:
            reset_mpc_workspace(self.MPC_WORKSPACE)

    def _computeReward(self) -> float:
        return 0.0

    def _computeTerminated(self) -> bool:
        return False

    def _computeTruncated(self) -> bool:
        return False

    def _computeInfo(self) -> dict:
        if self.LAST_MPC_SOLUTION is None:
            return {}
        return {
            "mpc_status": self.LAST_MPC_SOLUTION.status,
            "mpc_solve_time_seconds": self.LAST_MPC_SOLUTION.solve_time_seconds,
            "mpc_iterations": self.LAST_MPC_SOLUTION.iterations,
            "mpc_objective": self.LAST_MPC_SOLUTION.objective_value,
        }
