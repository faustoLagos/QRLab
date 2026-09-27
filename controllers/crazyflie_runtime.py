from dataclasses import dataclass, replace

from controllers.crazyflie_actuator import (
    BatteryCompensationState,
    CrazyflieActuatorParameters,
    run_crazyflie_actuator_path,
)
from controllers.crazyflie_attitude_pid import (
    AttitudeMeasurement,
    AttitudeRatePidState,
    AttitudeSetpoint,
    BodyRateMeasurement,
    DesiredBodyRates,
    LegacyControlOutput,
    advance_yaw_setpoint_from_rate,
    reset_attitude_rate_pid_state,
    run_attitude_rate_controller,
)
from controllers.crazyflie_pid import (
    OuterLoopOutput,
    PositionSetpoint,
    PositionVelocityPidState,
    TranslationalState,
    VelocitySetpoint,
    reset_position_velocity_pid_state,
    run_position_controller,
    run_velocity_controller,
)
from controllers.crazyflie_pid_parameters import (
    CrazyflieAttitudeRateParameters,
    CrazyflieOuterLoopParameters,
)


STABILIZER_RATE_HZ = 1000
ATTITUDE_RATE_HZ = 500
POSITION_RATE_HZ = 100
ATTITUDE_STEPS = STABILIZER_RATE_HZ // ATTITUDE_RATE_HZ
POSITION_STEPS = STABILIZER_RATE_HZ // POSITION_RATE_HZ


@dataclass(frozen=True)
class CrazyfliePositionCommand:
    x: float
    y: float
    z: float
    yaw_degrees: float


@dataclass(frozen=True)
class CrazyflieVelocityCommand:
    vx: float
    vy: float
    vz: float
    yaw_rate_degrees_per_second: float
    body_frame: bool = False


@dataclass(frozen=True)
class CrazyflieControllerMeasurements:
    translation: TranslationalState
    attitude: AttitudeMeasurement
    body_rates: BodyRateMeasurement


@dataclass(frozen=True)
class CrazyflieRuntimeState:
    outer_pid: PositionVelocityPidState
    inner_pid: AttitudeRatePidState
    battery: BatteryCompensationState
    held_outer_output: OuterLoopOutput
    held_legacy_control: LegacyControlOutput
    yaw_setpoint_degrees: float


def validate_crazyflie_control_frequencies(
    pybullet_frequency_hz: int,
    environment_frequency_hz: int,
) -> None:
    """Require the firmware-equivalent 1 kHz/100 Hz execution schedule."""
    if pybullet_frequency_hz != STABILIZER_RATE_HZ:
        raise ValueError(
            f"Crazyflie PID control requires pyb_freq={STABILIZER_RATE_HZ}, "
            f"received {pybullet_frequency_hz}."
        )
    if environment_frequency_hz != POSITION_RATE_HZ:
        raise ValueError(
            f"Crazyflie high-level commands require ctrl_freq={POSITION_RATE_HZ}, "
            f"received {environment_frequency_hz}."
        )


def initialize_crazyflie_runtime_state(
    measurements: CrazyflieControllerMeasurements,
    outer_parameters: CrazyflieOuterLoopParameters,
) -> CrazyflieRuntimeState:
    """Initialize all firmware-style controller state from the measured vehicle state."""
    outer_pid = reset_position_velocity_pid_state(
        state=PositionVelocityPidState(),
        measured_state=measurements.translation,
    )
    inner_pid = reset_attitude_rate_pid_state(
        state=AttitudeRatePidState(),
        measured_attitude=measurements.attitude,
    )

    held_outer_output = OuterLoopOutput(
        thrust=outer_parameters.thrust_base,
        roll_degrees=0.0,
        pitch_degrees=0.0,
        velocity_setpoint_body_x=0.0,
        velocity_setpoint_body_y=0.0,
        velocity_setpoint_z=0.0,
    )
    held_legacy_control = LegacyControlOutput(
        thrust=outer_parameters.thrust_base,
        roll=0,
        pitch=0,
        yaw=0,
        desired_body_rates=DesiredBodyRates(
            roll_degrees_per_second=0.0,
            pitch_degrees_per_second=0.0,
            yaw_degrees_per_second=0.0,
        ),
    )

    return CrazyflieRuntimeState(
        outer_pid=outer_pid,
        inner_pid=inner_pid,
        battery=BatteryCompensationState(),
        held_outer_output=held_outer_output,
        held_legacy_control=held_legacy_control,
        yaw_setpoint_degrees=measurements.attitude.yaw_degrees,
    )


def run_position_control_substep(
    stabilizer_step_index: int,
    command: CrazyfliePositionCommand,
    measurements: CrazyflieControllerMeasurements,
    measured_battery_voltage: float,
    runtime_state: CrazyflieRuntimeState,
    outer_parameters: CrazyflieOuterLoopParameters,
    inner_parameters: CrazyflieAttitudeRateParameters,
    actuator_parameters: CrazyflieActuatorParameters,
) -> tuple[tuple[float, float, float, float], CrazyflieRuntimeState]:
    """Run one 1 kHz firmware step for an absolute-position command."""
    held_outer_output, outer_pid = _update_position_loop_if_due(
        stabilizer_step_index=stabilizer_step_index,
        command=command,
        measurements=measurements,
        runtime_state=runtime_state,
        outer_parameters=outer_parameters,
    )

    yaw_setpoint = runtime_state.yaw_setpoint_degrees
    held_legacy_control = runtime_state.held_legacy_control
    inner_pid = runtime_state.inner_pid

    if is_attitude_loop_due(stabilizer_step_index):
        yaw_setpoint = command.yaw_degrees
        held_legacy_control, inner_pid = run_attitude_rate_controller(
            parameters=inner_parameters,
            controller_state=inner_pid,
            attitude_setpoint=AttitudeSetpoint(
                roll_degrees=held_outer_output.roll_degrees,
                pitch_degrees=held_outer_output.pitch_degrees,
                yaw_degrees=yaw_setpoint,
            ),
            measured_attitude=measurements.attitude,
            measured_body_rates=measurements.body_rates,
            thrust=held_outer_output.thrust,
        )

    actuator_output, battery_state = run_crazyflie_actuator_path(
        control=held_legacy_control,
        measured_battery_voltage=measured_battery_voltage,
        battery_state=runtime_state.battery,
        parameters=actuator_parameters,
    )

    return (
        actuator_output.effective_rpm,
        CrazyflieRuntimeState(
            outer_pid=outer_pid,
            inner_pid=inner_pid,
            battery=battery_state,
            held_outer_output=held_outer_output,
            held_legacy_control=held_legacy_control,
            yaw_setpoint_degrees=yaw_setpoint,
        ),
    )


def run_velocity_control_substep(
    stabilizer_step_index: int,
    command: CrazyflieVelocityCommand,
    measurements: CrazyflieControllerMeasurements,
    measured_battery_voltage: float,
    runtime_state: CrazyflieRuntimeState,
    outer_parameters: CrazyflieOuterLoopParameters,
    inner_parameters: CrazyflieAttitudeRateParameters,
    actuator_parameters: CrazyflieActuatorParameters,
) -> tuple[tuple[float, float, float, float], CrazyflieRuntimeState]:
    """Run one 1 kHz firmware step for a velocity and yaw-rate command."""
    held_outer_output, outer_pid = _update_velocity_loop_if_due(
        stabilizer_step_index=stabilizer_step_index,
        command=command,
        measurements=measurements,
        runtime_state=runtime_state,
        outer_parameters=outer_parameters,
    )

    yaw_setpoint = runtime_state.yaw_setpoint_degrees
    held_legacy_control = runtime_state.held_legacy_control
    inner_pid = runtime_state.inner_pid

    if is_attitude_loop_due(stabilizer_step_index):
        yaw_setpoint = advance_yaw_setpoint_from_rate(
            current_yaw_setpoint_degrees=yaw_setpoint,
            yaw_rate_degrees_per_second=command.yaw_rate_degrees_per_second,
            measured_yaw_degrees=measurements.attitude.yaw_degrees,
            yaw_max_delta_degrees=inner_parameters.yaw_max_delta_degrees,
        )
        held_legacy_control, inner_pid = run_attitude_rate_controller(
            parameters=inner_parameters,
            controller_state=inner_pid,
            attitude_setpoint=AttitudeSetpoint(
                roll_degrees=held_outer_output.roll_degrees,
                pitch_degrees=held_outer_output.pitch_degrees,
                yaw_degrees=yaw_setpoint,
            ),
            measured_attitude=measurements.attitude,
            measured_body_rates=measurements.body_rates,
            thrust=held_outer_output.thrust,
        )

    actuator_output, battery_state = run_crazyflie_actuator_path(
        control=held_legacy_control,
        measured_battery_voltage=measured_battery_voltage,
        battery_state=runtime_state.battery,
        parameters=actuator_parameters,
    )

    return (
        actuator_output.effective_rpm,
        CrazyflieRuntimeState(
            outer_pid=outer_pid,
            inner_pid=inner_pid,
            battery=battery_state,
            held_outer_output=held_outer_output,
            held_legacy_control=held_legacy_control,
            yaw_setpoint_degrees=yaw_setpoint,
        ),
    )


def is_attitude_loop_due(stabilizer_step_index: int) -> bool:
    """Return whether the 500 Hz attitude/rate controller executes this 1 kHz step."""
    return stabilizer_step_index % ATTITUDE_STEPS == 0


def is_position_loop_due(stabilizer_step_index: int) -> bool:
    """Return whether the 100 Hz position/velocity controller executes this 1 kHz step."""
    return stabilizer_step_index % POSITION_STEPS == 0


def _update_position_loop_if_due(
    stabilizer_step_index: int,
    command: CrazyfliePositionCommand,
    measurements: CrazyflieControllerMeasurements,
    runtime_state: CrazyflieRuntimeState,
    outer_parameters: CrazyflieOuterLoopParameters,
) -> tuple[OuterLoopOutput, PositionVelocityPidState]:
    if not is_position_loop_due(stabilizer_step_index):
        return runtime_state.held_outer_output, runtime_state.outer_pid

    return run_position_controller(
        parameters=outer_parameters,
        controller_state=runtime_state.outer_pid,
        setpoint=PositionSetpoint(
            x=command.x,
            y=command.y,
            z=command.z,
        ),
        measured_state=measurements.translation,
    )


def _update_velocity_loop_if_due(
    stabilizer_step_index: int,
    command: CrazyflieVelocityCommand,
    measurements: CrazyflieControllerMeasurements,
    runtime_state: CrazyflieRuntimeState,
    outer_parameters: CrazyflieOuterLoopParameters,
) -> tuple[OuterLoopOutput, PositionVelocityPidState]:
    if not is_position_loop_due(stabilizer_step_index):
        return runtime_state.held_outer_output, runtime_state.outer_pid

    return run_velocity_controller(
        parameters=outer_parameters,
        controller_state=runtime_state.outer_pid,
        setpoint=VelocitySetpoint(
            x=command.vx,
            y=command.vy,
            z=command.vz,
            body_frame=command.body_frame,
        ),
        measured_state=measurements.translation,
    )
