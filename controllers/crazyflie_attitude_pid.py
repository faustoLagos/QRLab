from dataclasses import dataclass, replace

from controllers.crazyflie_pid_parameters import (
    ATTITUDE_DT,
    CrazyflieAttitudeRateParameters,
)
from controllers.pid import PidState, reset_pid, set_pid_desired, update_pid


@dataclass(frozen=True)
class AttitudeSetpoint:
    roll_degrees: float
    pitch_degrees: float
    yaw_degrees: float


@dataclass(frozen=True)
class AttitudeMeasurement:
    roll_degrees: float
    pitch_degrees: float
    yaw_degrees: float


@dataclass(frozen=True)
class BodyRateMeasurement:
    roll_degrees_per_second: float
    pitch_degrees_per_second: float
    yaw_degrees_per_second: float


@dataclass(frozen=True)
class DesiredBodyRates:
    roll_degrees_per_second: float
    pitch_degrees_per_second: float
    yaw_degrees_per_second: float


@dataclass(frozen=True)
class AttitudeRatePidState:
    attitude_roll: PidState = PidState()
    attitude_pitch: PidState = PidState()
    attitude_yaw: PidState = PidState()
    rate_roll: PidState = PidState()
    rate_pitch: PidState = PidState()
    rate_yaw: PidState = PidState()


@dataclass(frozen=True)
class RateActuatorOutput:
    roll: int
    pitch: int
    yaw: int


@dataclass(frozen=True)
class LegacyControlOutput:
    thrust: float
    roll: int
    pitch: int
    yaw: int
    desired_body_rates: DesiredBodyRates


def reset_attitude_rate_pid_state(
    state: AttitudeRatePidState,
    measured_attitude: AttitudeMeasurement,
) -> AttitudeRatePidState:
    """Reset the six inner-loop PIDs using the same values as the firmware."""
    return AttitudeRatePidState(
        attitude_roll=reset_pid(
            state.attitude_roll,
            measured_attitude.roll_degrees,
        ),
        attitude_pitch=reset_pid(
            state.attitude_pitch,
            measured_attitude.pitch_degrees,
        ),
        attitude_yaw=reset_pid(
            state.attitude_yaw,
            measured_attitude.yaw_degrees,
        ),
        rate_roll=reset_pid(state.rate_roll, 0.0),
        rate_pitch=reset_pid(state.rate_pitch, 0.0),
        rate_yaw=reset_pid(state.rate_yaw, 0.0),
    )


def run_attitude_controller(
    parameters: CrazyflieAttitudeRateParameters,
    controller_state: AttitudeRatePidState,
    setpoint: AttitudeSetpoint,
    measured_attitude: AttitudeMeasurement,
) -> tuple[DesiredBodyRates, AttitudeRatePidState]:
    """Convert desired Euler attitude into desired body rates at 500 Hz."""
    roll_state = set_pid_desired(
        controller_state.attitude_roll,
        setpoint.roll_degrees,
    )
    desired_roll_rate, roll_state = update_pid(
        config=parameters.attitude_roll,
        state=roll_state,
        measured=measured_attitude.roll_degrees,
    )

    pitch_state = set_pid_desired(
        controller_state.attitude_pitch,
        setpoint.pitch_degrees,
    )
    desired_pitch_rate, pitch_state = update_pid(
        config=parameters.attitude_pitch,
        state=pitch_state,
        measured=measured_attitude.pitch_degrees,
    )

    yaw_state = set_pid_desired(
        controller_state.attitude_yaw,
        cap_angle_degrees(setpoint.yaw_degrees),
    )
    desired_yaw_rate, yaw_state = update_pid(
        config=parameters.attitude_yaw,
        state=yaw_state,
        measured=cap_angle_degrees(measured_attitude.yaw_degrees),
        is_yaw_angle=True,
    )

    next_state = replace(
        controller_state,
        attitude_roll=roll_state,
        attitude_pitch=pitch_state,
        attitude_yaw=yaw_state,
    )

    return (
        DesiredBodyRates(
            roll_degrees_per_second=desired_roll_rate,
            pitch_degrees_per_second=desired_pitch_rate,
            yaw_degrees_per_second=desired_yaw_rate,
        ),
        next_state,
    )


def run_rate_controller(
    parameters: CrazyflieAttitudeRateParameters,
    controller_state: AttitudeRatePidState,
    desired_body_rates: DesiredBodyRates,
    measured_body_rates: BodyRateMeasurement,
) -> tuple[RateActuatorOutput, AttitudeRatePidState]:
    """Convert desired body rates into legacy roll/pitch/yaw actuator commands."""
    roll_state = set_pid_desired(
        controller_state.rate_roll,
        desired_body_rates.roll_degrees_per_second,
    )
    roll_output, roll_state = update_pid(
        config=parameters.rate_roll,
        state=roll_state,
        measured=measured_body_rates.roll_degrees_per_second,
    )

    pitch_state = set_pid_desired(
        controller_state.rate_pitch,
        desired_body_rates.pitch_degrees_per_second,
    )
    pitch_output, pitch_state = update_pid(
        config=parameters.rate_pitch,
        state=pitch_state,
        measured=measured_body_rates.pitch_degrees_per_second,
    )

    yaw_state = set_pid_desired(
        controller_state.rate_yaw,
        desired_body_rates.yaw_degrees_per_second,
    )
    yaw_output, yaw_state = update_pid(
        config=parameters.rate_yaw,
        state=yaw_state,
        measured=measured_body_rates.yaw_degrees_per_second,
    )

    next_state = replace(
        controller_state,
        rate_roll=roll_state,
        rate_pitch=pitch_state,
        rate_yaw=yaw_state,
    )

    return (
        RateActuatorOutput(
            roll=saturate_signed_int16(
                roll_output,
                parameters.actuator_output_limit,
            ),
            pitch=saturate_signed_int16(
                pitch_output,
                parameters.actuator_output_limit,
            ),
            yaw=saturate_signed_int16(
                yaw_output,
                parameters.actuator_output_limit,
            ),
        ),
        next_state,
    )


def run_attitude_rate_controller(
    parameters: CrazyflieAttitudeRateParameters,
    controller_state: AttitudeRatePidState,
    attitude_setpoint: AttitudeSetpoint,
    measured_attitude: AttitudeMeasurement,
    measured_body_rates: BodyRateMeasurement,
    thrust: float,
) -> tuple[LegacyControlOutput, AttitudeRatePidState]:
    """Run the complete 500 Hz attitude/rate cascade used by controller_pid."""
    desired_body_rates, state_after_attitude = run_attitude_controller(
        parameters=parameters,
        controller_state=controller_state,
        setpoint=attitude_setpoint,
        measured_attitude=measured_attitude,
    )

    actuator_output, next_state = run_rate_controller(
        parameters=parameters,
        controller_state=state_after_attitude,
        desired_body_rates=desired_body_rates,
        measured_body_rates=measured_body_rates,
    )

    return (
        LegacyControlOutput(
            thrust=float(thrust),
            roll=actuator_output.roll,
            pitch=actuator_output.pitch,
            yaw=-actuator_output.yaw,
            desired_body_rates=desired_body_rates,
        ),
        next_state,
    )


def advance_yaw_setpoint_from_rate(
    current_yaw_setpoint_degrees: float,
    yaw_rate_degrees_per_second: float,
    measured_yaw_degrees: float,
    yaw_max_delta_degrees: float = 0.0,
) -> float:
    """Advance a yaw-rate command using the firmware controller_pid semantics."""
    yaw_setpoint = cap_angle_degrees(
        current_yaw_setpoint_degrees
        + yaw_rate_degrees_per_second * ATTITUDE_DT
    )

    if yaw_max_delta_degrees != 0.0:
        delta = cap_angle_degrees(
            yaw_setpoint - measured_yaw_degrees
        )
        if delta > yaw_max_delta_degrees:
            yaw_setpoint = measured_yaw_degrees + yaw_max_delta_degrees
        elif delta < -yaw_max_delta_degrees:
            yaw_setpoint = measured_yaw_degrees - yaw_max_delta_degrees

    return cap_angle_degrees(yaw_setpoint)


def cap_angle_degrees(angle_degrees: float) -> float:
    """Map an angle to the firmware controller's [-180, 180] degree range."""
    result = float(angle_degrees)

    while result > 180.0:
        result -= 360.0

    while result < -180.0:
        result += 360.0

    return result


def saturate_signed_int16(value: float, limit: int = 32767) -> int:
    """Apply the firmware's signed-int16 saturation and truncation."""
    clipped_value = max(-float(limit), min(float(limit), float(value)))
    return int(clipped_value)
