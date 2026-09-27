from dataclasses import dataclass, replace
from math import cos, pi, sin

from controllers.crazyflie_pid_parameters import CrazyflieOuterLoopParameters
from controllers.pid import PidState, reset_pid, reset_pid_filter, set_pid_desired, update_pid


@dataclass(frozen=True)
class PositionSetpoint:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class VelocitySetpoint:
    x: float
    y: float
    z: float
    body_frame: bool = False


@dataclass(frozen=True)
class TranslationalState:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    yaw_degrees: float


@dataclass(frozen=True)
class PositionVelocityPidState:
    position_x: PidState = PidState()
    position_y: PidState = PidState()
    position_z: PidState = PidState()
    velocity_x: PidState = PidState()
    velocity_y: PidState = PidState()
    velocity_z: PidState = PidState()


@dataclass(frozen=True)
class OuterLoopOutput:
    thrust: float
    roll_degrees: float
    pitch_degrees: float
    velocity_setpoint_body_x: float
    velocity_setpoint_body_y: float
    velocity_setpoint_z: float


def reset_position_velocity_pid_state(
    state: PositionVelocityPidState,
    measured_state: TranslationalState,
) -> PositionVelocityPidState:
    """Reset the outer-loop PID state with the same measured-value choices as the firmware."""
    return PositionVelocityPidState(
        position_x=reset_pid_filter(reset_pid(state.position_x, measured_state.x)),
        position_y=reset_pid_filter(reset_pid(state.position_y, measured_state.y)),
        position_z=reset_pid_filter(reset_pid(state.position_z, measured_state.z)),
        velocity_x=reset_pid_filter(reset_pid(state.velocity_x, 0.0)),
        velocity_y=reset_pid_filter(reset_pid(state.velocity_y, 0.0)),
        velocity_z=reset_pid_filter(reset_pid(state.velocity_z, 0.0)),
    )


def run_position_controller(
    parameters: CrazyflieOuterLoopParameters,
    controller_state: PositionVelocityPidState,
    setpoint: PositionSetpoint,
    measured_state: TranslationalState,
) -> tuple[OuterLoopOutput, PositionVelocityPidState]:
    """Run the 100 Hz Crazyflie position loop followed by its velocity loop."""
    cosine_yaw, sine_yaw = _yaw_rotation_terms(measured_state.yaw_degrees)

    setpoint_body_x = setpoint.x * cosine_yaw + setpoint.y * sine_yaw
    setpoint_body_y = -setpoint.x * sine_yaw + setpoint.y * cosine_yaw
    state_body_x = measured_state.x * cosine_yaw + measured_state.y * sine_yaw
    state_body_y = -measured_state.x * sine_yaw + measured_state.y * cosine_yaw

    position_x_state = set_pid_desired(controller_state.position_x, setpoint_body_x)
    desired_velocity_x, position_x_state = update_pid(
        config=parameters.position_x,
        state=position_x_state,
        measured=state_body_x,
    )

    position_y_state = set_pid_desired(controller_state.position_y, setpoint_body_y)
    desired_velocity_y, position_y_state = update_pid(
        config=parameters.position_y,
        state=position_y_state,
        measured=state_body_y,
    )

    position_z_state = set_pid_desired(controller_state.position_z, setpoint.z)
    desired_velocity_z, position_z_state = update_pid(
        config=parameters.position_z,
        state=position_z_state,
        measured=measured_state.z,
    )

    state_after_position = replace(
        controller_state,
        position_x=position_x_state,
        position_y=position_y_state,
        position_z=position_z_state,
    )

    return _run_velocity_controller_body_frame(
        parameters=parameters,
        controller_state=state_after_position,
        velocity_body_x=desired_velocity_x,
        velocity_body_y=desired_velocity_y,
        velocity_z=desired_velocity_z,
        measured_state=measured_state,
    )


def run_velocity_controller(
    parameters: CrazyflieOuterLoopParameters,
    controller_state: PositionVelocityPidState,
    setpoint: VelocitySetpoint,
    measured_state: TranslationalState,
) -> tuple[OuterLoopOutput, PositionVelocityPidState]:
    """Run the 100 Hz Crazyflie velocity loop from a body- or world-frame command."""
    if setpoint.body_frame:
        velocity_body_x = setpoint.x
        velocity_body_y = setpoint.y
    else:
        cosine_yaw, sine_yaw = _yaw_rotation_terms(measured_state.yaw_degrees)
        velocity_body_x = setpoint.x * cosine_yaw + setpoint.y * sine_yaw
        velocity_body_y = setpoint.y * cosine_yaw - setpoint.x * sine_yaw

    return _run_velocity_controller_body_frame(
        parameters=parameters,
        controller_state=controller_state,
        velocity_body_x=velocity_body_x,
        velocity_body_y=velocity_body_y,
        velocity_z=setpoint.z,
        measured_state=measured_state,
    )


def _run_velocity_controller_body_frame(
    parameters: CrazyflieOuterLoopParameters,
    controller_state: PositionVelocityPidState,
    velocity_body_x: float,
    velocity_body_y: float,
    velocity_z: float,
    measured_state: TranslationalState,
) -> tuple[OuterLoopOutput, PositionVelocityPidState]:
    cosine_yaw, sine_yaw = _yaw_rotation_terms(measured_state.yaw_degrees)
    measured_body_vx = measured_state.vx * cosine_yaw + measured_state.vy * sine_yaw
    measured_body_vy = -measured_state.vx * sine_yaw + measured_state.vy * cosine_yaw

    velocity_x_state = set_pid_desired(controller_state.velocity_x, velocity_body_x)
    pitch_pid_output, velocity_x_state = update_pid(
        config=parameters.velocity_x,
        state=velocity_x_state,
        measured=measured_body_vx,
    )

    velocity_y_state = set_pid_desired(controller_state.velocity_y, velocity_body_y)
    roll_pid_output, velocity_y_state = update_pid(
        config=parameters.velocity_y,
        state=velocity_y_state,
        measured=measured_body_vy,
    )

    velocity_z_state = set_pid_desired(controller_state.velocity_z, velocity_z)
    thrust_raw, velocity_z_state = update_pid(
        config=parameters.velocity_z,
        state=velocity_z_state,
        measured=measured_state.vz,
    )

    pitch_degrees = _constrain(
        -pitch_pid_output,
        -parameters.pitch_limit_degrees,
        parameters.pitch_limit_degrees,
    )
    roll_degrees = _constrain(
        -roll_pid_output,
        -parameters.roll_limit_degrees,
        parameters.roll_limit_degrees,
    )

    thrust = thrust_raw * parameters.thrust_scale + parameters.thrust_base
    thrust = max(thrust, parameters.thrust_min)
    thrust = _constrain(thrust, 0.0, parameters.thrust_max)

    next_state = replace(
        controller_state,
        velocity_x=velocity_x_state,
        velocity_y=velocity_y_state,
        velocity_z=velocity_z_state,
    )

    return (
        OuterLoopOutput(
            thrust=thrust,
            roll_degrees=roll_degrees,
            pitch_degrees=pitch_degrees,
            velocity_setpoint_body_x=velocity_body_x,
            velocity_setpoint_body_y=velocity_body_y,
            velocity_setpoint_z=velocity_z,
        ),
        next_state,
    )


def _yaw_rotation_terms(yaw_degrees: float) -> tuple[float, float]:
    yaw_radians = yaw_degrees * pi / 180.0
    return cos(yaw_radians), sin(yaw_radians)


def _constrain(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
