from dataclasses import dataclass

from controllers.pid import PidConfig, create_low_pass_2p_config


POSITION_RATE_HZ = 100.0
POSITION_DT = 1.0 / POSITION_RATE_HZ
UINT16_MAX = 65535.0


@dataclass(frozen=True)
class CrazyflieOuterLoopParameters:
    position_x: PidConfig
    position_y: PidConfig
    position_z: PidConfig
    velocity_x: PidConfig
    velocity_y: PidConfig
    velocity_z: PidConfig
    roll_limit_degrees: float
    pitch_limit_degrees: float
    thrust_scale: float
    thrust_base: float
    thrust_min: float
    thrust_max: float


def create_cf2_outer_loop_parameters() -> CrazyflieOuterLoopParameters:
    """Return Crazyflie 2.x defaults for the 100 Hz position/velocity PID cascade."""
    position_xy_filter = create_low_pass_2p_config(
        sample_rate_hz=POSITION_RATE_HZ,
        cutoff_frequency_hz=20.0,
    )
    position_z_filter = create_low_pass_2p_config(
        sample_rate_hz=POSITION_RATE_HZ,
        cutoff_frequency_hz=20.0,
    )
    velocity_xy_filter = create_low_pass_2p_config(
        sample_rate_hz=POSITION_RATE_HZ,
        cutoff_frequency_hz=20.0,
    )
    velocity_z_filter = create_low_pass_2p_config(
        sample_rate_hz=POSITION_RATE_HZ,
        cutoff_frequency_hz=20.0,
    )

    velocity_limit_overhead = 1.10
    roll_pitch_limit_overhead = 1.10
    x_velocity_limit = 1.0
    y_velocity_limit = 1.0
    z_velocity_limit = 1.0
    roll_limit_degrees = 20.0
    pitch_limit_degrees = 20.0
    thrust_scale = 1000.0

    return CrazyflieOuterLoopParameters(
        position_x=PidConfig(
            kp=2.0,
            ki=0.0,
            kd=0.0,
            kff=0.0,
            dt=POSITION_DT,
            output_limit=x_velocity_limit * velocity_limit_overhead,
            derivative_filter=position_xy_filter,
        ),
        position_y=PidConfig(
            kp=2.0,
            ki=0.0,
            kd=0.0,
            kff=0.0,
            dt=POSITION_DT,
            output_limit=y_velocity_limit * velocity_limit_overhead,
            derivative_filter=position_xy_filter,
        ),
        position_z=PidConfig(
            kp=2.0,
            ki=0.5,
            kd=0.0,
            kff=0.0,
            dt=POSITION_DT,
            output_limit=max(z_velocity_limit, 0.5) * velocity_limit_overhead,
            derivative_filter=position_z_filter,
        ),
        velocity_x=PidConfig(
            kp=25.0,
            ki=1.0,
            kd=0.0,
            kff=0.0,
            dt=POSITION_DT,
            output_limit=pitch_limit_degrees * roll_pitch_limit_overhead,
            derivative_filter=velocity_xy_filter,
        ),
        velocity_y=PidConfig(
            kp=25.0,
            ki=1.0,
            kd=0.0,
            kff=0.0,
            dt=POSITION_DT,
            output_limit=roll_limit_degrees * roll_pitch_limit_overhead,
            derivative_filter=velocity_xy_filter,
        ),
        velocity_z=PidConfig(
            kp=25.0,
            ki=15.0,
            kd=0.0,
            kff=0.0,
            dt=POSITION_DT,
            output_limit=UINT16_MAX / 2.0 / thrust_scale,
            derivative_filter=velocity_z_filter,
        ),
        roll_limit_degrees=roll_limit_degrees,
        pitch_limit_degrees=pitch_limit_degrees,
        thrust_scale=thrust_scale,
        thrust_base=36000.0,
        thrust_min=20000.0,
        thrust_max=UINT16_MAX,
    )

ATTITUDE_RATE_HZ = 500.0
ATTITUDE_DT = 1.0 / ATTITUDE_RATE_HZ
SIGNED_INT16_MAX = 32767


@dataclass(frozen=True)
class CrazyflieAttitudeRateParameters:
    attitude_roll: PidConfig
    attitude_pitch: PidConfig
    attitude_yaw: PidConfig
    rate_roll: PidConfig
    rate_pitch: PidConfig
    rate_yaw: PidConfig
    yaw_max_delta_degrees: float
    actuator_output_limit: int


def create_cf2_attitude_rate_parameters(
    attitude_filter_enabled: bool = False,
    rate_filter_enabled: bool = False,
    filter_all_output: bool = False,
) -> CrazyflieAttitudeRateParameters:
    """Return Crazyflie 2.x defaults for the 500 Hz attitude/rate PID cascade."""
    attitude_filter = (
        create_low_pass_2p_config(
            sample_rate_hz=ATTITUDE_RATE_HZ,
            cutoff_frequency_hz=15.0,
        )
        if attitude_filter_enabled
        else None
    )

    roll_rate_filter = (
        create_low_pass_2p_config(
            sample_rate_hz=ATTITUDE_RATE_HZ,
            cutoff_frequency_hz=30.0,
        )
        if rate_filter_enabled
        else None
    )
    pitch_rate_filter = (
        create_low_pass_2p_config(
            sample_rate_hz=ATTITUDE_RATE_HZ,
            cutoff_frequency_hz=30.0,
        )
        if rate_filter_enabled
        else None
    )
    yaw_rate_filter = (
        create_low_pass_2p_config(
            sample_rate_hz=ATTITUDE_RATE_HZ,
            cutoff_frequency_hz=30.0,
        )
        if rate_filter_enabled
        else None
    )

    return CrazyflieAttitudeRateParameters(
        attitude_roll=PidConfig(
            kp=6.0,
            ki=3.0,
            kd=0.0,
            kff=0.0,
            dt=ATTITUDE_DT,
            integral_limit=20.0,
            derivative_filter=attitude_filter,
            filter_all_output=filter_all_output,
        ),
        attitude_pitch=PidConfig(
            kp=6.0,
            ki=3.0,
            kd=0.0,
            kff=0.0,
            dt=ATTITUDE_DT,
            integral_limit=20.0,
            derivative_filter=attitude_filter,
            filter_all_output=filter_all_output,
        ),
        attitude_yaw=PidConfig(
            kp=6.0,
            ki=1.0,
            kd=0.35,
            kff=0.0,
            dt=ATTITUDE_DT,
            integral_limit=360.0,
            derivative_filter=attitude_filter,
            filter_all_output=filter_all_output,
        ),
        rate_roll=PidConfig(
            kp=250.0,
            ki=500.0,
            kd=2.5,
            kff=0.0,
            dt=ATTITUDE_DT,
            integral_limit=33.3,
            derivative_filter=roll_rate_filter,
            filter_all_output=filter_all_output,
        ),
        rate_pitch=PidConfig(
            kp=250.0,
            ki=500.0,
            kd=2.5,
            kff=0.0,
            dt=ATTITUDE_DT,
            integral_limit=33.3,
            derivative_filter=pitch_rate_filter,
            filter_all_output=filter_all_output,
        ),
        rate_yaw=PidConfig(
            kp=120.0,
            ki=16.7,
            kd=0.0,
            kff=0.0,
            dt=ATTITUDE_DT,
            integral_limit=166.7,
            derivative_filter=yaw_rate_filter,
            filter_all_output=filter_all_output,
        ),
        yaw_max_delta_degrees=0.0,
        actuator_output_limit=SIGNED_INT16_MAX,
    )

