"""Monte Carlo evaluation environment for the Crazyflie RL controller.

Training termination limits are logged as transient-envelope violations but do
not terminate Monte Carlo rollouts. Monte Carlo termination is reserved for
hard physical/numerical failures. Recovery/stabilization is evaluated in
monte_carlo_core.py, not in the environment.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pybullet as p
from gymnasium import spaces

from environments.BaseRLAviary import BaseRLAviary
from environments.utils.domain_randomization import DomainRandomizationMixin
from environments.utils.enums import ActionType, DroneModel, ObservationType, Physics


class MonteCarloEnv(DomainRandomizationMixin, BaseRLAviary):
    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=np.array([[0.0, 0.0, 0.1]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        target_xyzs=np.array([0.0, 0.0, 1.0]),
        target_q_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        physics: Physics = Physics.PYB_GND,
        pyb_freq: int = 200,
        ctrl_freq: int = 100,
        gui: bool = False,
        record: bool = False,
        observation_space: ObservationType = ObservationType.KIN,
        action_space: ActionType = ActionType.RPM,
        episode_length_seconds: float = 10.0,
        randomize_initial_conditions: bool = True,
        # Original TRAINING envelope: diagnostics only in Monte Carlo.
        min_altitude: float = 0.1,
        max_position_error: float = 3.0,
        max_roll_pitch_rad: float = np.deg2rad(15.0),
        max_speed: float = 2.0,
        max_angular_speed: float = 2.0,
        # Genuine Monte Carlo hard-failure criteria.
        terminate_on_ground_contact: bool = True,
        hard_max_position_error: Optional[float] = 20.0,
        *args,
        **kwargs,
    ):
        self.INIT_XYZS = np.asarray(initial_xyzs, dtype=float)
        self.INIT_RPYS = np.asarray(initial_rpys, dtype=float)
        self.TARGET_POS = np.asarray(target_xyzs, dtype=float)
        self.TARGET_QUATERNION = np.asarray(target_q_xyzw, dtype=float)
        self.EPISODE_LENGTH_SECONDS = float(episode_length_seconds)
        self.RANDOMIZE_INITIAL_CONDITIONS = bool(randomize_initial_conditions)

        self.TRAINING_MIN_ALTITUDE = float(min_altitude)
        self.TRAINING_MAX_POSITION_ERROR = float(max_position_error)
        self.TRAINING_MAX_ROLL_PITCH_RAD = float(max_roll_pitch_rad)
        self.TRAINING_MAX_SPEED = float(max_speed)
        self.TRAINING_MAX_ANGULAR_SPEED = float(max_angular_speed)

        self.TERMINATE_ON_GROUND_CONTACT = bool(terminate_on_ground_contact)
        self.HARD_MAX_POSITION_ERROR = (
            None if hard_max_position_error is None else float(hard_max_position_error)
        )

        # Must exist before BaseAviary can call subclass observation/info methods.
        self._ic_rng = np.random.default_rng()
        self._noise_rng = np.random.default_rng()
        self._reset_episode_diagnostics()

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=self.INIT_XYZS,
            initial_rpys=self.INIT_RPYS,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=observation_space,
            act=action_space,
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Reward (kept unchanged in meaning; MC analysis does not use reward).
    # ------------------------------------------------------------------

    @staticmethod
    def _exponential_reward(b: float, current, target) -> float:
        return float(np.exp(-b * np.linalg.norm(current - target) ** 2))

    def _xy_error_reward(self, current, target):
        return 0.6 * self._exponential_reward(4.0, current, target) + 0.4 * self._exponential_reward(150.0, current, target)

    def _z_error_reward(self, current, target):
        return 0.6 * self._exponential_reward(4.0, current, target) + 0.4 * self._exponential_reward(150.0, current, target)

    def _linear_velocity_error_reward(self, current, target):
        return self._exponential_reward(1.5, current, target)

    @staticmethod
    def _orientation_error_reward(theta: float) -> float:
        return float(0.6 * np.exp(-0.5 * theta**2) + 0.4 * np.exp(-150.0 * theta**2))

    def quat_geodesic_angle_from_qerr_xyzw(self, q: np.ndarray) -> float:
        q_err = self._quat_error_xyzw(self.TARGET_QUATERNION, q, ensure_pos_w=True)
        q_err /= np.linalg.norm(q_err)
        return float(2.0 * np.arctan2(np.linalg.norm(q_err[:3]), np.clip(q_err[3], 0.0, 1.0)))

    def _delta_action_penalty(self, drone_id: int = 0, weight: float = 0.01) -> float:
        if len(self.action_buffer) < 2:
            return 0.0
        du = (
            np.asarray(self.action_buffer[-1][drone_id, :], dtype=np.float32)
            - np.asarray(self.action_buffer[-2][drone_id, :], dtype=np.float32)
        )
        return float(weight * np.dot(du, du))

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        theta = self.quat_geodesic_angle_from_qerr_xyzw(state[3:7])
        return float(
            0.01
            + 0.15 * self._linear_velocity_error_reward(state[10:13], np.zeros(3))
            + 0.20 * self._orientation_error_reward(theta)
            + 0.25 * self._xy_error_reward(state[0:2], self.TARGET_POS[0:2])
            + 0.25 * self._z_error_reward(state[2], self.TARGET_POS[2])
            - self._delta_action_penalty(0, 0.02)
        )

    # ------------------------------------------------------------------
    # Evaluation semantics.
    # ------------------------------------------------------------------

    def _reset_episode_diagnostics(self) -> None:
        self._current_training_envelope_violations: List[str] = []
        self._episode_training_envelope_violations = set()
        self._last_hard_failure_reasons: List[str] = []

    def _training_envelope_violations(self, state: np.ndarray) -> List[str]:
        """Training thresholds, retained only as logged diagnostics."""
        position_error = float(np.linalg.norm(state[0:3] - self.TARGET_POS))
        speed = float(np.linalg.norm(state[10:13]))
        angular_speed = float(np.linalg.norm(state[13:16]))

        reasons = []
        if state[2] < self.TRAINING_MIN_ALTITUDE:
            reasons.append("altitude_below_training_limit")
        if position_error > self.TRAINING_MAX_POSITION_ERROR:
            reasons.append("position_error_training_limit")
        if abs(state[7]) > self.TRAINING_MAX_ROLL_PITCH_RAD:
            reasons.append("roll_training_limit")
        if abs(state[8]) > self.TRAINING_MAX_ROLL_PITCH_RAD:
            reasons.append("pitch_training_limit")
        if speed > self.TRAINING_MAX_SPEED:
            reasons.append("linear_speed_training_limit")
        if angular_speed > self.TRAINING_MAX_ANGULAR_SPEED:
            reasons.append("angular_speed_training_limit")
        return reasons

    def _hard_failure_reasons(self, state: np.ndarray) -> List[str]:
        """Only events that should actually terminate an MC rollout."""
        if not np.all(np.isfinite(state)):
            return ["non_finite_state"]

        reasons = []
        if self.TERMINATE_ON_GROUND_CONTACT:
            contacts = p.getContactPoints(
                bodyA=int(self.DRONE_IDS[0]),
                bodyB=int(self.PLANE_ID),
                physicsClientId=self.CLIENT,
            )
            if contacts:
                reasons.append("ground_contact")

        if self.HARD_MAX_POSITION_ERROR is not None:
            position_error = float(np.linalg.norm(state[0:3] - self.TARGET_POS))
            if position_error > self.HARD_MAX_POSITION_ERROR:
                reasons.append("catastrophic_position_divergence")
        return reasons

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)

        current = self._training_envelope_violations(state)
        self._current_training_envelope_violations = current
        self._episode_training_envelope_violations.update(current)

        self._last_hard_failure_reasons = self._hard_failure_reasons(state)
        return bool(self._last_hard_failure_reasons)

    def _computeTruncated(self):
        return self.step_counter / self.PYB_FREQ >= self.EPISODE_LENGTH_SECONDS

    def _computeInfo(self):
        # 'failure_reasons' is kept for compatibility with the current MC core.
        return {
            "failure_reasons": list(self._last_hard_failure_reasons),
            "hard_failure_reasons": list(self._last_hard_failure_reasons),
            "current_training_envelope_violations": list(self._current_training_envelope_violations),
            "training_envelope_violations": sorted(self._episode_training_envelope_violations),
            "training_envelope_violated": bool(self._episode_training_envelope_violations),
            "simulation_time_s": float(self.step_counter) / float(self.PYB_FREQ),
        }

    # ------------------------------------------------------------------
    # Observation.
    # ------------------------------------------------------------------

    def _observationSpace(self):
        low = np.array([[-np.inf] * 13 + [-1.0] * 4], dtype=np.float32)
        high = np.array([[np.inf] * 13 + [1.0] * 4], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @staticmethod
    def _compute_raw_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        return current - target

    def _compute_noisy_error(self, current, target, noise_definition=(0.0, 0.001, 3)):
        return self._compute_raw_error(current, target) + self._noise_rng.normal(*noise_definition)

    @staticmethod
    def _quat_xyzw_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(4)
        norm = np.linalg.norm(q)
        if norm < eps:
            raise ValueError("Quaternion norm is near zero; cannot normalize")
        return q / norm

    @staticmethod
    def _quat_xyzw_conjugate(q: np.ndarray) -> np.ndarray:
        x, y, z, w = np.asarray(q, dtype=np.float64).reshape(4)
        return np.array([-x, -y, -z, w], dtype=np.float64)

    @staticmethod
    def _quat_xyzw_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64).reshape(4)
        x2, y2, z2, w2 = np.asarray(q2, dtype=np.float64).reshape(4)
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ], dtype=np.float64)

    @classmethod
    def _quat_error_xyzw(cls, q_target, q_current, ensure_pos_w=True):
        q_target = cls._quat_xyzw_normalize(q_target)
        q_current = cls._quat_xyzw_normalize(q_current)
        q_error = cls._quat_xyzw_multiply(cls._quat_xyzw_conjugate(q_target), q_current)
        if ensure_pos_w and q_error[3] < 0.0:
            q_error *= -1.0
        return cls._quat_xyzw_normalize(q_error)

    def _noisy_quaternion(self, q: np.ndarray, noise=(0.0, 0.002, 4)) -> np.ndarray:
        return self._quat_xyzw_normalize(q + self._noise_rng.normal(*noise))

    @staticmethod
    def _rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
        u, s = q[:3], q[3]
        uv = np.cross(u, v)
        return v + 2.0 * (s * uv + np.cross(u, uv))

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        q_current = state[3:7]
        q_current_inv = self._quat_xyzw_conjugate(q_current)

        world_pos_error = self._compute_noisy_error(state[0:3], self.TARGET_POS, (0.0, 0.001, 3))
        world_lin_vel = state[10:13] + self._noise_rng.normal(0.0, 0.001, 3)
        world_ang_vel = state[13:16] + self._noise_rng.normal(0.0, 0.002, 3)
        q_error = self._quat_error_xyzw(self.TARGET_QUATERNION, q_current, ensure_pos_w=True)

        obs = np.hstack([
            self._rotate_vector_by_quaternion(world_pos_error, q_current_inv),
            self._noisy_quaternion(q_error, (0.0, 0.002, 4)),
            self._rotate_vector_by_quaternion(world_lin_vel, q_current_inv),
            self._rotate_vector_by_quaternion(world_ang_vel, q_current_inv),
            np.asarray(self.action_buffer[-1][0, :], dtype=np.float32),
        ])
        return obs.reshape(1, 17).astype(np.float32)

    # ------------------------------------------------------------------
    # Reset.
    # ------------------------------------------------------------------

    def _reset_trial_rngs(self, seed: Optional[int]) -> None:
        """Create independent deterministic streams for IC, DR, and observation noise.

        Using separate streams makes the three factor-class studies pairable:
        trial i uses the same IC realization in IC-only and combined, and the
        same DR realization in DR-only and combined, without RNG-consumption
        order coupling the factors.
        """
        seed_sequence = np.random.SeedSequence(seed)
        ic_seed, dr_seed, noise_seed = seed_sequence.spawn(3)
        self._ic_rng = np.random.default_rng(ic_seed)
        self._noise_rng = np.random.default_rng(noise_seed)

        # DomainRandomizationMixin owns this generator. Seed it before
        # super().reset(), because its reset() applies DR upstream.
        if hasattr(self, "_dr_random_generator"):
            self._dr_random_generator = np.random.default_rng(dr_seed)

    def _reset_action_buffer(self) -> None:
        self.action_buffer.clear()
        zero_action = np.zeros((self.NUM_DRONES, 4), dtype=np.float32)
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(zero_action.copy())

    def _initial_state_for_trial(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.RANDOMIZE_INITIAL_CONDITIONS:
            return (
                self.TARGET_POS.copy(),
                np.asarray(p.getEulerFromQuaternion(self.TARGET_QUATERNION), dtype=float),
                np.zeros(3, dtype=float),
                np.zeros(3, dtype=float),
            )

        position = np.array([
            self._ic_rng.uniform(-2.0, 2.0),
            self._ic_rng.uniform(-2.0, 2.0),
            self._ic_rng.uniform(0.1, 2.0),
        ])
        rpy = np.array([
            self._ic_rng.uniform(-0.2, 0.2),
            self._ic_rng.uniform(-0.2, 0.2),
            self._ic_rng.uniform(-np.pi, np.pi),
        ])
        linear_velocity = self._ic_rng.uniform(-1.0, 1.0, size=3)
        angular_velocity = self._ic_rng.uniform(-1.0, 1.0, size=3)
        return position, rpy, linear_velocity, angular_velocity

    def reset(self, seed: int = None, options: dict = None):
        # Must precede super().reset(): BaseAviary.reset() calls _computeObs().
        self._reset_trial_rngs(seed)
        self._reset_episode_diagnostics()
        self._reset_action_buffer()

        _, info = super().reset(seed=seed, options=options)
        dr_params = dict(info.get("dr_params", {}) or {})

        position, rpy, linear_velocity, angular_velocity = self._initial_state_for_trial()
        self.INIT_XYZS = position.reshape(1, 3)
        self.INIT_RPYS = rpy.reshape(1, 3)

        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0], position, p.getQuaternionFromEuler(rpy), physicsClientId=self.CLIENT
        )
        p.resetBaseVelocity(
            self.DRONE_IDS[0], linear_velocity, angular_velocity, physicsClientId=self.CLIENT
        )
        self._updateAndStoreKinematicInformation()

        # Initial-condition violations are logged, not treated as termination.
        initial_state = self._getDroneStateVector(0)
        current = self._training_envelope_violations(initial_state)
        self._current_training_envelope_violations = current
        self._episode_training_envelope_violations.update(current)

        obs_new = self._computeObs()
        info_new = self._computeInfo()
        x0, y0, z0 = position
        roll0, pitch0, yaw0 = rpy
        vx0, vy0, vz0 = linear_velocity
        wx0, wy0, wz0 = angular_velocity

        info_new["initial_conditions"] = {
            "x0": float(x0), "y0": float(y0), "z0": float(z0),
            "roll0": float(roll0), "pitch0": float(pitch0), "yaw0": float(yaw0),
            "vx0": float(vx0), "vy0": float(vy0), "vz0": float(vz0),
            "wx0": float(wx0), "wy0": float(wy0), "wz0": float(wz0),
        }
        info_new["dr_params"] = dr_params
        info_new["trial_seed"] = seed
        info_new["randomized_initial_conditions"] = self.RANDOMIZE_INITIAL_CONDITIONS

        for key, value in info.items():
            if key not in info_new:
                info_new[key] = value

        return obs_new, info_new
