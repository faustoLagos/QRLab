from dataclasses import dataclass
from math import copysign, sqrt

from controllers.crazyflie_attitude_pid import LegacyControlOutput


UINT16_MAX = 65535
BATTERY_FILTER_ALPHA = 0.01
BATTERY_INITIAL_VOLTAGE = 4.2
BATTERY_COMPENSATION_MIN_VOLTAGE = 2.0


@dataclass(frozen=True)
class CrazyflieActuatorParameters:
    thrust_min_newtons: float
    thrust_max_newtons: float
    motor_voltage_to_thrust_0: float
    motor_voltage_to_thrust_1: float
    motor_voltage_to_thrust_2: float
    motor_voltage_to_thrust_3: float
    idle_pwm: int = 0
    battery_filter_alpha: float = BATTERY_FILTER_ALPHA
    pwm_max: int = UINT16_MAX
    rpm_to_pwm_offset: float = 4070.3
    rpm_to_pwm_slope: float = 0.2685


@dataclass(frozen=True)
class BatteryCompensationState:
    supply_voltage: float = BATTERY_INITIAL_VOLTAGE


@dataclass(frozen=True)
class MotorActuatorOutput:
    uncapped_thrust_command: tuple[int, int, int, int]
    battery_compensated_uncapped_pwm: tuple[int, int, int, int]
    pwm: tuple[int, int, int, int]
    effective_rpm: tuple[float, float, float, float]
    filtered_supply_voltage: float
    was_capped: bool


def create_cf21_plus_actuator_parameters() -> CrazyflieActuatorParameters:
    """Return the current default CF2.1+ brushed-motor thrust-compensation parameters."""
    return CrazyflieActuatorParameters(
        thrust_min_newtons=0.012817578393224994,
        thrust_max_newtons=0.12,
        motor_voltage_to_thrust_0=-0.02476537915958403,
        motor_voltage_to_thrust_1=0.06523793527519485,
        motor_voltage_to_thrust_2=-0.026792504967750107,
        motor_voltage_to_thrust_3=0.006776789303971145,
    )


def mix_legacy_control(
    control: LegacyControlOutput,
) -> tuple[int, int, int, int]:
    """Apply the Crazyflie legacy X-quad power-distribution equations."""
    roll_part = int(control.roll / 2.0)
    pitch_part = int(control.pitch / 2.0)

    return (
        int(control.thrust - roll_part + pitch_part + control.yaw),
        int(control.thrust - roll_part - pitch_part - control.yaw),
        int(control.thrust + roll_part - pitch_part + control.yaw),
        int(control.thrust + roll_part + pitch_part - control.yaw),
    )


def update_battery_compensation_state(
    state: BatteryCompensationState,
    measured_battery_voltage: float,
    alpha: float = BATTERY_FILTER_ALPHA,
) -> BatteryCompensationState:
    """Apply the firmware first-order low-pass filter to battery voltage."""
    filtered_voltage = state.supply_voltage + alpha * (
        float(measured_battery_voltage) - state.supply_voltage
    )
    return BatteryCompensationState(supply_voltage=filtered_voltage)


def compensate_motor_thrust_for_battery(
    thrust_command: int,
    supply_voltage: float,
    parameters: CrazyflieActuatorParameters,
) -> float:
    """Invert the CF2.1+ cubic motor-voltage/thrust fit to obtain required PWM."""
    if supply_voltage < BATTERY_COMPENSATION_MIN_VOLTAGE:
        return 0.0

    thrust_newtons = (
        float(thrust_command) / float(parameters.pwm_max)
    ) * parameters.thrust_max_newtons

    if thrust_newtons < parameters.thrust_min_newtons:
        return 0.0

    cubic = parameters.motor_voltage_to_thrust_3
    quadratic = parameters.motor_voltage_to_thrust_2
    linear = parameters.motor_voltage_to_thrust_1
    constant = parameters.motor_voltage_to_thrust_0

    p = -quadratic / (3.0 * cubic)
    q = (
        p * p * p
        + (
            quadratic * linear
            - 3.0 * cubic * (constant - thrust_newtons)
        )
        / (6.0 * cubic * cubic)
    )
    r = linear / (3.0 * cubic)

    discriminant = q * q + (r - p * p) ** 3
    qrp = sqrt(max(discriminant, 0.0))

    motor_voltage = (
        _signed_cube_root(q + qrp)
        + _signed_cube_root(q - qrp)
        + p
    )
    duty_cycle = motor_voltage / supply_voltage
    return float(parameters.pwm_max) * duty_cycle


def compensate_motor_thrusts_for_battery(
    thrust_commands: tuple[int, int, int, int],
    supply_voltage: float,
    parameters: CrazyflieActuatorParameters,
) -> tuple[int, int, int, int]:
    """Apply battery compensation and reproduce int32 assignment truncation."""
    return tuple(
        int(
            compensate_motor_thrust_for_battery(
                thrust_command=thrust_command,
                supply_voltage=supply_voltage,
                parameters=parameters,
            )
        )
        for thrust_command in thrust_commands
    )


def cap_motor_pwm_coupled(
    uncapped_pwm: tuple[int, int, int, int],
    pwm_max: int = UINT16_MAX,
    idle_pwm: int = 0,
) -> tuple[tuple[int, int, int, int], bool]:
    """Apply the firmware's coupled upper saturation and lower idle cap."""
    highest_thrust = max(0, *uncapped_pwm)
    reduction = max(0, highest_thrust - pwm_max)

    capped_pwm = tuple(
        max(idle_pwm, motor_pwm - reduction)
        for motor_pwm in uncapped_pwm
    )
    return capped_pwm, reduction > 0


def convert_pwm_to_effective_rpm(
    pwm: tuple[int, int, int, int],
    parameters: CrazyflieActuatorParameters,
) -> tuple[float, float, float, float]:
    """Map final PWM to effective RPM for QRLab's current RPM-based plant model."""
    return tuple(
        0.0
        if motor_pwm <= 0
        else parameters.rpm_to_pwm_offset
        + parameters.rpm_to_pwm_slope * float(motor_pwm)
        for motor_pwm in pwm
    )


def run_crazyflie_actuator_path(
    control: LegacyControlOutput,
    measured_battery_voltage: float,
    battery_state: BatteryCompensationState,
    parameters: CrazyflieActuatorParameters,
) -> tuple[MotorActuatorOutput, BatteryCompensationState]:
    """Run legacy mixing, battery compensation, capping, and QRLab RPM closure."""
    next_battery_state = update_battery_compensation_state(
        state=battery_state,
        measured_battery_voltage=measured_battery_voltage,
        alpha=parameters.battery_filter_alpha,
    )

    uncapped_thrust_command = mix_legacy_control(control)
    battery_compensated_pwm = compensate_motor_thrusts_for_battery(
        thrust_commands=uncapped_thrust_command,
        supply_voltage=next_battery_state.supply_voltage,
        parameters=parameters,
    )
    pwm, was_capped = cap_motor_pwm_coupled(
        uncapped_pwm=battery_compensated_pwm,
        pwm_max=parameters.pwm_max,
        idle_pwm=parameters.idle_pwm,
    )
    effective_rpm = convert_pwm_to_effective_rpm(
        pwm=pwm,
        parameters=parameters,
    )

    return (
        MotorActuatorOutput(
            uncapped_thrust_command=uncapped_thrust_command,
            battery_compensated_uncapped_pwm=battery_compensated_pwm,
            pwm=pwm,
            effective_rpm=effective_rpm,
            filtered_supply_voltage=next_battery_state.supply_voltage,
            was_capped=was_capped,
        ),
        next_battery_state,
    )


def _signed_cube_root(value: float) -> float:
    """Return a real cube root while remaining compatible with Python 3.10."""
    if value == 0.0:
        return 0.0
    return copysign(abs(value) ** (1.0 / 3.0), value)
