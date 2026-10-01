import numpy as np
from gymnasium import spaces
from environments.utils.domain_randomization import DomainRandomizationMixin
from environments.BaseRLAviary import BaseRLAviary
from environments.utils.enums import DroneModel, Physics, ActionType, ObservationType
import pybullet as p


class TestEnv(BaseRLAviary):
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=np.array([[0, 0, 0.1]]),
                 initial_rpys=np.array([[0, 0, 0]]),
                 target_xyzs=np.array([0, 0, 1]),
                 target_q_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                 physics: Physics = Physics.PYB_GND,
                 pyb_freq: int = 200,
                 ctrl_freq: int = 100,
                 gui=False,
                 record=False,
                 observation_space: ObservationType = ObservationType.KIN,
                 action_space: ActionType = ActionType.RPM,
                 *arg, **kwargs
                 ):
        self.INIT_XYZS = initial_xyzs
        self.TARGET_POS = target_xyzs
        self.TARGET_QUATERNION = target_q_xyzw
        self.EPISODE_LENGTH_SECONDS = 5
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         act=action_space,
                         *arg, **kwargs
                         )

    ################################################################################

    def _computeReward(self):
        return 1

    ################################################################################

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)

        current_position = state[0:3]
        position_error = np.linalg.norm(current_position - self.TARGET_POS)
        current_velocity = state[10:13]
        velocity_norm = np.linalg.norm(current_velocity)
        current_omega = state[13:16]
        omega_norm = np.linalg.norm(current_omega)

        failure = (
                (state[2] < 0.1) or
                (position_error > 3.0) or
                (np.abs(state[7]) > np.deg2rad(15)) or
                (np.abs(state[8]) > np.deg2rad(15)) or
                (velocity_norm > 2.0) or
                (omega_norm > 2.0)
        )

        if failure:
            return True

        return False

    ################################################################################

    def _computeTruncated(self):
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LENGTH_SECONDS:
            return True

        return False

    ################################################################################

    def _computeInfo(self):
        return {"answer": 42}  # Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def _observationSpace(self):
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([[lo, lo, lo, lo ,lo ,lo ,lo , lo, lo, lo, lo, lo, lo, -1, -1, -1, -1]])
        obs_upper_bound = np.array([[hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, 1, 1, 1, 1]])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    @staticmethod
    def _compute_raw_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        return current - target

    def _compute_noisy_error(
            self,
            current: np.ndarray,
            target: np.ndarray,
            noise_definition: tuple = (0.0, 0.001, 3)
    ) -> np.ndarray:

        return (self._compute_raw_error(current, target) +
                np.random.normal(noise_definition[0], noise_definition[1], noise_definition[2]))

    @staticmethod
    def _quat_xyzw_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(4)
        n = np.linalg.norm(q)
        if n < eps:
            raise ValueError("Quat norm is near zero, can't normalize")
        return q / n

    @staticmethod
    def _quat_xyzw_conjugate(q: np.ndarray) -> np.ndarray:
        x, y, z, w = np.asarray(q, dtype=np.float64).reshape(4)
        return  np.array([-x, -y, -z, w], dtype=np.float64)

    @staticmethod
    def _quat_xyzw_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64).reshape(4)
        x2, y2, z2, w2 = np.asarray(q2, dtype=np.float64).reshape(4)

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([x, y, z, w], dtype=np.float64)

    @classmethod
    def _quat_error_xyzw(cls, q_target: np.ndarray, q_current: np.ndarray, ensure_pos_w: bool = True) -> np.ndarray:
        q_target = cls._quat_xyzw_normalize(q_target)
        q_current = cls._quat_xyzw_normalize(q_current)
        q_target_inv = cls._quat_xyzw_conjugate(q_target)
        q_e = cls._quat_xyzw_multiply(q_target_inv, q_current)
        if ensure_pos_w and q_e[3] < 0.0:
            q_e *= -1

        return cls._quat_xyzw_normalize(q_e)

    def _noisy_quaternion(self, q: np.ndarray, noise: tuple = (0.0, 0.002, 4)) -> np.ndarray:
        noisy_q = q + np.random.normal(*noise)
        return self._quat_xyzw_normalize(noisy_q)

    @staticmethod
    def _rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
        u = q[0:3]
        s = q[3]

        uv = np.cross(u, v)
        uuv = np.cross(u, uv)

        return v + 2.0 * (s * uv + uuv)

    def _computeObs(self) -> np.ndarray:
        obs_17 = np.zeros((1, 17))
        obs = self._getDroneStateVector(0)
        world_pos_error = self._compute_noisy_error(obs[0:3], self.TARGET_POS, (0.0, 0.001, 3))
        world_lin_vel = obs[10:13] + self.np_random.normal(0.0, 0.001, 3)
        world_ang_vel = obs[13:16] + self.np_random.normal(0.0, 0.002, 3)

        q_current = obs[3:7]
        body_ang_vel = self._convertWorldVectorToBodyFrame(world_ang_vel, q_current)
        q_err = self._quat_error_xyzw(self.TARGET_QUATERNION, q_current, ensure_pos_w=True)

        obs_17[0, :] = np.hstack([
            world_pos_error,
            self._noisy_quaternion(q_err, (0.0, 0.002, 4)),
            world_lin_vel,
            body_ang_vel,
            np.asarray(self.action_buffer[-1][0, :], dtype=np.float32)
        ]).reshape(17, )

        ret = np.array([obs_17[0, :]]).astype('float32')
        return ret

    def reset(
            self,
            seed: int = None,
            options: dict = None,
    ):
        obs, info = super().reset(seed=seed, options=options)
        dr_params = info.get("dr_params", None)

        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            self.INIT_XYZS[0, :],
            p.getQuaternionFromEuler(self.INIT_RPYS[0]),
            physicsClientId=self.CLIENT
        )
        p.resetBaseVelocity(self.DRONE_IDS[0], [0, 0, 0], [0, 0, 0], physicsClientId=self.CLIENT)
        self._updateAndStoreKinematicInformation()
        obs_new = self._computeObs()
        info_new = self._computeInfo()

        if dr_params is not None:
            info_new["dr_params"] = dr_params

        for k, v in info.items():
            if k not in info_new:
                info_new[k] = v

        return obs_new, info_new
