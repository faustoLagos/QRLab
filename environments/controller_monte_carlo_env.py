from __future__ import annotations

from typing import Any

import numpy as np
import pybullet as p

from environments.mpc_simulation_env import MPCSimulationEnv
from environments.pid_simulation_env import PIDSimulationEnv
from environments.utils.domain_randomization import DomainRandomizationMixin


DEFAULT_TARGET_POSITION = np.array([0.0, 0.0, 1.0], dtype=np.float64)
MPC_SOLVER_ERROR_PREFIX = "OSQP failed to solve the MPC problem: "


def extract_mpc_solver_failure_status(error: RuntimeError) -> str | None:
    """Extract an OSQP status from the MPC baseline's solver exception."""
    message = str(error).strip()
    if not message.startswith(MPC_SOLVER_ERROR_PREFIX):
        return None

    return message[len(MPC_SOLVER_ERROR_PREFIX):].rstrip(".").strip()


def create_mpc_solver_failure_reason(status: str) -> str:
    """Create a stable hard-failure label from an OSQP status string."""
    normalized_status = "_".join(status.lower().replace("-", " ").split())
    return f"mpc_solver_{normalized_status}"


class MonteCarloEvaluationMixin:
    """Shared Monte Carlo reset, diagnostics, and termination semantics."""

    def __init__(
        self,
        *args: Any,
        target_xyzs: np.ndarray = DEFAULT_TARGET_POSITION,
        target_yaw_radians: float = 0.0,
        episode_length_seconds: float = 20.0,
        randomize_initial_conditions: bool = True,
        min_altitude: float = 0.1,
        max_position_error: float = 3.0,
        max_roll_pitch_rad: float = np.deg2rad(15.0),
        max_speed: float = 2.0,
        max_angular_speed: float = 2.0,
        terminate_on_ground_contact: bool = True,
        hard_max_position_error: float | None = 20.0,
        **kwargs: Any,
    ) -> None:
        target_position = np.asarray(target_xyzs, dtype=np.float64).reshape(-1)
        if target_position.shape != (3,):
            raise ValueError("target_xyzs must contain exactly three values.")

        self.TARGET_POS = target_position
        self.TARGET_YAW_RADIANS = float(target_yaw_radians)
        self.TARGET_QUATERNION = np.asarray(
            p.getQuaternionFromEuler([0.0, 0.0, self.TARGET_YAW_RADIANS]),
            dtype=np.float64,
        )
        self.EPISODE_LENGTH_SECONDS = float(episode_length_seconds)
        self.RANDOMIZE_INITIAL_CONDITIONS = bool(randomize_initial_conditions)

        self.TRAINING_MIN_ALTITUDE = float(min_altitude)
        self.TRAINING_MAX_POSITION_ERROR = float(max_position_error)
        self.TRAINING_MAX_ROLL_PITCH_RAD = float(max_roll_pitch_rad)
        self.TRAINING_MAX_SPEED = float(max_speed)
        self.TRAINING_MAX_ANGULAR_SPEED = float(max_angular_speed)

        self.TERMINATE_ON_GROUND_CONTACT = bool(terminate_on_ground_contact)
        self.HARD_MAX_POSITION_ERROR = (
            None
            if hard_max_position_error is None
            else float(hard_max_position_error)
        )

        self._ic_rng = np.random.default_rng()
        self._trial_seed: int | None = None
        self._initial_conditions: dict[str, float] = {}
        self._reset_episode_diagnostics()

        super().__init__(*args, **kwargs)

    def _reset_episode_diagnostics(self) -> None:
        self._current_training_envelope_violations: list[str] = []
        self._episode_training_envelope_violations: set[str] = set()
        self._last_hard_failure_reasons: list[str] = []

    def _reset_trial_rngs(self, seed: int | None) -> None:
        seed_sequence = np.random.SeedSequence(seed)
        ic_seed, _, _ = seed_sequence.spawn(3)
        self._ic_rng = np.random.default_rng(ic_seed)
        self._trial_seed = seed

    def _initial_state_for_trial(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.RANDOMIZE_INITIAL_CONDITIONS:
            return (
                self.TARGET_POS.copy(),
                np.array([0.0, 0.0, self.TARGET_YAW_RADIANS], dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )

        position = np.array(
            [
                self._ic_rng.uniform(-2.0, 2.0),
                self._ic_rng.uniform(-2.0, 2.0),
                self._ic_rng.uniform(0.1, 2.0),
            ],
            dtype=np.float64,
        )
        rpy = np.array(
            [
                self._ic_rng.uniform(-0.2, 0.2),
                self._ic_rng.uniform(-0.2, 0.2),
                self._ic_rng.uniform(-np.pi, np.pi),
            ],
            dtype=np.float64,
        )
        linear_velocity = self._ic_rng.uniform(-1.0, 1.0, size=3)
        angular_velocity = self._ic_rng.uniform(-1.0, 1.0, size=3)
        return position, rpy, linear_velocity, angular_velocity

    @staticmethod
    def _initial_conditions_dictionary(
        position: np.ndarray,
        rpy: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> dict[str, float]:
        x0, y0, z0 = position
        roll0, pitch0, yaw0 = rpy
        vx0, vy0, vz0 = linear_velocity
        wx0, wy0, wz0 = angular_velocity
        return {
            "x0": float(x0),
            "y0": float(y0),
            "z0": float(z0),
            "roll0": float(roll0),
            "pitch0": float(pitch0),
            "yaw0": float(yaw0),
            "vx0": float(vx0),
            "vy0": float(vy0),
            "vz0": float(vz0),
            "wx0": float(wx0),
            "wy0": float(wy0),
            "wz0": float(wz0),
        }

    def _resetTaskState(self, options: dict | None = None) -> None:
        super()._resetTaskState(options=options)
        position, rpy, linear_velocity, angular_velocity = (
            self._initial_state_for_trial()
        )
        self.INIT_XYZS = position.reshape(1, 3)
        self.INIT_RPYS = rpy.reshape(1, 3)

        p.resetBasePositionAndOrientation(
            int(self.DRONE_IDS[0]),
            position,
            p.getQuaternionFromEuler(rpy),
            physicsClientId=self.CLIENT,
        )
        p.resetBaseVelocity(
            int(self.DRONE_IDS[0]),
            linear_velocity,
            angular_velocity,
            physicsClientId=self.CLIENT,
        )
        self._initial_conditions = self._initial_conditions_dictionary(
            position=position,
            rpy=rpy,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )

    def _training_envelope_violations(self, state: np.ndarray) -> list[str]:
        position_error = float(np.linalg.norm(state[0:3] - self.TARGET_POS))
        speed = float(np.linalg.norm(state[10:13]))
        angular_speed = float(np.linalg.norm(state[13:16]))

        conditions = (
            (state[2] < self.TRAINING_MIN_ALTITUDE, "altitude_below_training_limit"),
            (
                position_error > self.TRAINING_MAX_POSITION_ERROR,
                "position_error_training_limit",
            ),
            (
                abs(state[7]) > self.TRAINING_MAX_ROLL_PITCH_RAD,
                "roll_training_limit",
            ),
            (
                abs(state[8]) > self.TRAINING_MAX_ROLL_PITCH_RAD,
                "pitch_training_limit",
            ),
            (speed > self.TRAINING_MAX_SPEED, "linear_speed_training_limit"),
            (
                angular_speed > self.TRAINING_MAX_ANGULAR_SPEED,
                "angular_speed_training_limit",
            ),
        )
        return [reason for violated, reason in conditions if violated]

    def _hard_failure_reasons(self, state: np.ndarray) -> list[str]:
        if not np.all(np.isfinite(state)):
            return ["non_finite_state"]

        ground_contact = bool(
            self.TERMINATE_ON_GROUND_CONTACT
            and p.getContactPoints(
                bodyA=int(self.DRONE_IDS[0]),
                bodyB=int(self.PLANE_ID),
                physicsClientId=self.CLIENT,
            )
        )
        position_error = float(np.linalg.norm(state[0:3] - self.TARGET_POS))
        catastrophic_divergence = bool(
            self.HARD_MAX_POSITION_ERROR is not None
            and position_error > self.HARD_MAX_POSITION_ERROR
        )

        conditions = (
            (ground_contact, "ground_contact"),
            (catastrophic_divergence, "catastrophic_position_divergence"),
        )
        return [reason for failed, reason in conditions if failed]

    def _refresh_training_envelope_diagnostics(self) -> None:
        state = np.asarray(self._getDroneStateVector(0), dtype=np.float64)
        current = self._training_envelope_violations(state)
        self._current_training_envelope_violations = current
        self._episode_training_envelope_violations.update(current)

    def _computeTerminated(self) -> bool:
        state = np.asarray(self._getDroneStateVector(0), dtype=np.float64)
        current = self._training_envelope_violations(state)
        self._current_training_envelope_violations = current
        self._episode_training_envelope_violations.update(current)
        self._last_hard_failure_reasons = self._hard_failure_reasons(state)
        return bool(self._last_hard_failure_reasons)

    def _computeTruncated(self) -> bool:
        return bool(
            self.step_counter / self.PYB_FREQ >= self.EPISODE_LENGTH_SECONDS
        )

    def _computeInfo(self) -> dict:
        self._refresh_training_envelope_diagnostics()
        info = dict(super()._computeInfo())
        info.update(
            {
                "failure_reasons": list(self._last_hard_failure_reasons),
                "hard_failure_reasons": list(self._last_hard_failure_reasons),
                "current_training_envelope_violations": list(
                    self._current_training_envelope_violations
                ),
                "training_envelope_violations": sorted(
                    self._episode_training_envelope_violations
                ),
                "training_envelope_violated": bool(
                    self._episode_training_envelope_violations
                ),
                "simulation_time_s": float(self.step_counter)
                / float(self.PYB_FREQ),
            }
        )
        return info

    def _computeResetInfo(self) -> dict:
        info = dict(super()._computeResetInfo())
        info.update(
            {
                "initial_conditions": dict(self._initial_conditions),
                "trial_seed": self._trial_seed,
                "randomized_initial_conditions": self.RANDOMIZE_INITIAL_CONDITIONS,
            }
        )
        return info

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        self._reset_trial_rngs(seed)
        self._reset_episode_diagnostics()
        return super().reset(seed=seed, options=options)


class PIDMonteCarloEnv(
    MonteCarloEvaluationMixin,
    DomainRandomizationMixin,
    PIDSimulationEnv,
):
    """Firmware-like PID baseline with Monte Carlo evaluation semantics."""


class MPCMonteCarloEnv(
    MonteCarloEvaluationMixin,
    DomainRandomizationMixin,
    MPCSimulationEnv,
):
    """Nominal-model MPC baseline with randomized physical-plant semantics."""

    CONTROLLER_FAILURE_CRITERIA = {
        "mpc_solver_unsolved_terminates": True,
        "solver_failure_is_counted_as_hard_failure": True,
    }

    def _create_solver_failure_transition(
        self,
        status: str,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Convert an MPC solver failure into a Monte Carlo hard failure."""
        failure_reason = create_mpc_solver_failure_reason(status)
        self.LAST_MPC_SOLUTION = None
        self.LAST_COMMANDED_WRENCH = np.zeros(4, dtype=np.float64)
        self._last_hard_failure_reasons = [failure_reason]

        observation = self._computeObs()
        info = self._computeInfo()
        info.update(
            {
                "mpc_status": status,
                "mpc_solver_failed": True,
                "failure_reasons": [failure_reason],
                "hard_failure_reasons": [failure_reason],
            }
        )
        return observation, 0.0, True, False, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance MPC or record an unsolved QP as a robustness failure."""
        try:
            return super().step(action)
        except RuntimeError as error:
            solver_status = extract_mpc_solver_failure_status(error)
            if solver_status is None:
                raise
            return self._create_solver_failure_transition(solver_status)
