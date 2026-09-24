import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from environments.BaseAviary import BaseAviary
from environments.utils.enums import DroneModel, Physics, ActionType


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
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         )
        #### Fixed action normalization reference ##################
        self.ACTION_HOVER_RPM = calculate_hover_rpm(
            gravity_acceleration=RPM_ACTION_REFERENCE_GRAVITY,
            mass=float(self.M),
            thrust_coefficient=float(self.KF),
        )

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
