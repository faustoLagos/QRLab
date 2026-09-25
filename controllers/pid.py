from dataclasses import dataclass, replace
from math import cos, isfinite, pi, tan


DEFAULT_PID_INTEGRATION_LIMIT = 5000.0
DEFAULT_PID_OUTPUT_LIMIT = 0.0


@dataclass(frozen=True)
class LowPass2pConfig:
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float


@dataclass(frozen=True)
class LowPass2pState:
    delay_element_1: float = 0.0
    delay_element_2: float = 0.0


@dataclass(frozen=True)
class PidConfig:
    kp: float
    ki: float
    kd: float
    kff: float
    dt: float
    integral_limit: float = DEFAULT_PID_INTEGRATION_LIMIT
    output_limit: float = DEFAULT_PID_OUTPUT_LIMIT
    derivative_filter: LowPass2pConfig | None = None
    filter_all_output: bool = False


@dataclass(frozen=True)
class PidState:
    desired: float = 0.0
    error: float = 0.0
    previous_measured: float = 0.0
    integral: float = 0.0
    derivative: float = 0.0
    out_p: float = 0.0
    out_i: float = 0.0
    out_d: float = 0.0
    out_ff: float = 0.0
    filter_state: LowPass2pState = LowPass2pState()


def create_low_pass_2p_config(
    sample_rate_hz: float,
    cutoff_frequency_hz: float,
) -> LowPass2pConfig:
    """Create the two-pole low-pass filter coefficients used by the firmware PID."""
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if cutoff_frequency_hz <= 0.0:
        raise ValueError("cutoff_frequency_hz must be positive.")

    frequency_ratio = sample_rate_hz / cutoff_frequency_hz
    omega = tan(pi / frequency_ratio)
    coefficient = 1.0 + 2.0 * cos(pi / 4.0) * omega + omega * omega

    b0 = omega * omega / coefficient
    return LowPass2pConfig(
        b0=b0,
        b1=2.0 * b0,
        b2=b0,
        a1=2.0 * (omega * omega - 1.0) / coefficient,
        a2=(1.0 - 2.0 * cos(pi / 4.0) * omega + omega * omega) / coefficient,
    )


def apply_low_pass_2p(
    config: LowPass2pConfig,
    state: LowPass2pState,
    sample: float,
) -> tuple[float, LowPass2pState]:
    """Apply one update of the Crazyflie-style two-pole low-pass filter."""
    delay_element_0 = (
        sample
        - state.delay_element_1 * config.a1
        - state.delay_element_2 * config.a2
    )
    if not isfinite(delay_element_0):
        delay_element_0 = sample

    output = (
        delay_element_0 * config.b0
        + state.delay_element_1 * config.b1
        + state.delay_element_2 * config.b2
    )

    next_state = LowPass2pState(
        delay_element_1=delay_element_0,
        delay_element_2=state.delay_element_1,
    )
    return output, next_state


def set_pid_desired(state: PidState, desired: float) -> PidState:
    """Return PID state with an updated setpoint."""
    return replace(state, desired=float(desired))


def reset_pid(state: PidState, measured: float) -> PidState:
    """Reset PID dynamic state using the current measured process value."""
    return replace(
        state,
        error=0.0,
        previous_measured=float(measured),
        integral=0.0,
        derivative=0.0,
        out_p=0.0,
        out_i=0.0,
        out_d=0.0,
        out_ff=0.0,
    )


def reset_pid_filter(state: PidState) -> PidState:
    """Reset the PID low-pass filter state."""
    return replace(state, filter_state=LowPass2pState())


def update_pid(
    config: PidConfig,
    state: PidState,
    measured: float,
    is_yaw_angle: bool = False,
) -> tuple[float, PidState]:
    """Update a PID using the control semantics of the Crazyflie firmware."""
    if config.dt <= 0.0:
        raise ValueError("PID dt must be positive.")

    measured_value = float(measured)
    error = state.desired - measured_value
    if is_yaw_angle:
        error = _wrap_degrees_once(error)

    out_p = config.kp * error

    measurement_delta = -(measured_value - state.previous_measured)
    if is_yaw_angle:
        measurement_delta = _wrap_degrees_once(measurement_delta)

    derivative = measurement_delta / config.dt
    filter_state = state.filter_state

    if config.derivative_filter is not None and not config.filter_all_output:
        derivative, filter_state = apply_low_pass_2p(
            config=config.derivative_filter,
            state=filter_state,
            sample=derivative,
        )

    if not isfinite(derivative):
        derivative = 0.0

    out_d = config.kd * derivative

    integral = state.integral + error * config.dt
    integral = _constrain_if_enabled(integral, config.integral_limit)
    out_i = config.ki * integral

    out_ff = config.kff * state.desired
    output = out_p + out_i + out_d + out_ff

    if config.derivative_filter is not None and config.filter_all_output:
        output, filter_state = apply_low_pass_2p(
            config=config.derivative_filter,
            state=filter_state,
            sample=output,
        )
        if not isfinite(output):
            output = 0.0

    output = _constrain_if_enabled(output, config.output_limit)

    next_state = replace(
        state,
        error=error,
        previous_measured=measured_value,
        integral=integral,
        derivative=derivative,
        out_p=out_p,
        out_i=out_i,
        out_d=out_d,
        out_ff=out_ff,
        filter_state=filter_state,
    )
    return output, next_state


def _constrain_if_enabled(value: float, limit: float) -> float:
    if limit == 0.0:
        return value
    absolute_limit = abs(limit)
    return max(-absolute_limit, min(absolute_limit, value))


def _wrap_degrees_once(angle_degrees: float) -> float:
    if angle_degrees > 180.0:
        return angle_degrees - 360.0
    if angle_degrees < -180.0:
        return angle_degrees + 360.0
    return angle_degrees
