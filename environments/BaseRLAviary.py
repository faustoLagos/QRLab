import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from environments.BaseAviary import BaseAviary
from environments.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType


RPM_ACTION_REFERENCE_GRAVITY = 9.82
FIRMWARE_NOMINAL_BATTERY_VOLTAGE = 4.2
FIRMWARE_PWM_MAX = 65535.0
FIRMWARE_PWM_SCALE = 65536.0
FIRMWARE_THRUST_SCALE_GRAMS = 60.0
FIRMWARE_THRUST_TO_VOLTAGE_QUADRATIC = -0.0006239
FIRMWARE_THRUST_TO_VOLTAGE_LINEAR = 0.088
FIRMWARE_RPM_TO_PWM_OFFSET = 4070.3
FIRMWARE_RPM_TO_PWM_SLOPE = 0.2685


def calculate_hover_rpm(gravity_acceleration: float, mass: float, thrust_coefficient: float) -> float:
    """Calculates the fixed hover-RPM reference used by normalized RPM actions."""
    return float(np.sqrt(gravity_acceleration * mass / (4.0 * thrust_coefficient)))


def convert_rpm_to_firmware_pwm(commanded_rpm: np.ndarray) -> np.ndarray:
    """Reproduces the OOT RPM-to-PWM conversion before battery compensation."""
    pwm = (np.asarray(commanded_rpm, dtype=np.float64) - FIRMWARE_RPM_TO_PWM_OFFSET) / FIRMWARE_RPM_TO_PWM_SLOPE
    return np.trunc(pwm)


def compensate_firmware_pwm_for_battery(
        requested_pwm: np.ndarray,
        supply_voltage: float,
) -> np.ndarray:
    """Reproduces the brushed-motor battery compensation used by the firmware."""
    if supply_voltage < 2.0:
        return np.asarray(requested_pwm, dtype=np.float64).copy()

    thrust_grams = (np.asarray(requested_pwm, dtype=np.float64) / FIRMWARE_PWM_SCALE) * FIRMWARE_THRUST_SCALE_GRAMS
    required_voltage = (
        FIRMWARE_THRUST_TO_VOLTAGE_QUADRATIC * np.square(thrust_grams)
        + FIRMWARE_THRUST_TO_VOLTAGE_LINEAR * thrust_grams
    )
    compensated_pwm = FIRMWARE_PWM_MAX * required_voltage / supply_voltage
    return np.trunc(compensated_pwm)


def cap_firmware_pwm_coupled(requested_pwm: np.ndarray, idle_pwm: float = 0.0) -> np.ndarray:
    """Reproduces the firmware's coupled upper saturation and lower idle cap."""
    pwm = np.asarray(requested_pwm, dtype=np.float64)
    reduction = max(0.0, float(np.max(pwm) - FIRMWARE_PWM_MAX))
    return np.maximum(pwm - reduction, idle_pwm)


def convert_firmware_pwm_to_effective_rpm(final_pwm: np.ndarray) -> np.ndarray:
    """Maps final firmware PWM back to an effective RPM for the existing plant model.

    This inverse linear calibration is a simulation closure. The real firmware
    stops at PWM; BaseAviary's plant currently consumes RPM.
    """
    pwm = np.asarray(final_pwm, dtype=np.float64)
    return FIRMWARE_RPM_TO_PWM_OFFSET + FIRMWARE_RPM_TO_PWM_SLOPE * pwm


def simulate_firmware_actuator_path(
        commanded_rpm: np.ndarray,
        supply_voltage: float,
) -> np.ndarray:
    """Applies the deterministic Stage-4 firmware actuator path."""
    requested_pwm = convert_rpm_to_firmware_pwm(commanded_rpm)
    compensated_pwm = compensate_firmware_pwm_for_battery(requested_pwm, supply_voltage)
    capped_pwm = cap_firmware_pwm_coupled(compensated_pwm)
    return convert_firmware_pwm_to_effective_rpm(capped_pwm)

class BaseRLAviary(BaseAviary):
    """Base single and multi-agent environment class for reinforcement learning."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 firmware_actuator: bool=False,
                 firmware_battery_voltage: float=FIRMWARE_NOMINAL_BATTERY_VOLTAGE,
                 ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The normalized motor action representation.

        """
        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(ctrl_freq//2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        ####
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.FIRMWARE_ACTUATOR = firmware_actuator
        self.FIRMWARE_BATTERY_VOLTAGE = float(firmware_battery_voltage)

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         )
        #### Fixed action normalization reference ##################
        self.ACTION_HOVER_RPM = calculate_hover_rpm(
            gravity_acceleration=RPM_ACTION_REFERENCE_GRAVITY,
            mass=float(self.M),
            thrust_coefficient=float(self.KF),
        )

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            p.loadURDF("block.urdf",
                       [1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("cube_small.urdf",
                       [0, 1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("duck_vhacd.urdf",
                       [-1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("teddy_vhacd.urdf",
                       [0, -1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
        else:
            pass

    ################################################################################

    def _actionSpace(self):
        """Returns the normalized motor action space."""

        action_dimensions = {
            ActionType.RPM: 4,
        }

        try:
            size = action_dimensions[self.ACT_TYPE]
        except KeyError as error:
            raise ValueError(
                f"Unsupported action type: {self.ACT_TYPE}"
            ) from error

        self.ACTION_DIMENSION = size
        self._resetActionBuffer()

        return spaces.Box(
            low=-np.ones((self.NUM_DRONES, size), dtype=np.float32),
            high=np.ones((self.NUM_DRONES, size), dtype=np.float32),
            dtype=np.float32
        )

    ################################################################################

    def _resetActionBuffer(self) -> None:
        """Resets the action history to zero-valued actions."""
        zero_action = np.zeros(
            (self.NUM_DRONES, self.ACTION_DIMENSION),
            dtype=np.float32,
        )
        self.action_buffer.clear()
        self.action_buffer.extend(
            zero_action.copy() for _ in range(self.ACTION_BUFFER_SIZE)
        )

    ################################################################################

    def _resetControllerState(self, options: dict | None = None) -> None:
        """Resets RL-controller state shared by all RL environments."""
        super()._resetControllerState(options=options)
        self._resetActionBuffer()

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Converts normalized motor actions into plant RPMs."""
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES,4))

        for drone_id, target in enumerate(action):
            if self.ACT_TYPE == ActionType.RPM:
                commanded_rpm = np.asarray(
                    self.ACTION_HOVER_RPM * (1.0 + 0.5 * target),
                    dtype=np.float64
                )
            else:
                raise ValueError(
                    f"Unsupported action type: {self.ACT_TYPE}"
                )

            rpm[drone_id] = (
                simulate_firmware_actuator_path(
                    commanded_rpm=commanded_rpm,
                    supply_voltage=self.FIRMWARE_BATTERY_VOLTAGE
                )
                if self.FIRMWARE_ACTUATOR
                else commanded_rpm
            )

        return rpm

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi] for i in range(self.NUM_DRONES)])
            #### Add action buffer to observation space ################
            act_lo = -1
            act_hi = +1

            action_history_size = (
                self.ACTION_BUFFER_SIZE * self.ACTION_DIMENSTION
            )

            action_lower_bound = np.full(
                (self.NUM_DRONES, action_history_size),
                -1.0
            )

            action_upper_bound = np.full(
                (self.NUM_DRONES, action_history_size),
                1.0
            )

            obs_lower_bound = np.hstack([
                obs_lower_bound,
                action_lower_bound
            ])

            obs_upper_bound = np.hstack([
                obs_upper_bound,
                action_upper_bound
            ])

            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            obs_12 = np.zeros((self.NUM_DRONES,12))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)
                obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            #### Add action buffer to observation #######################
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            return ret
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")
