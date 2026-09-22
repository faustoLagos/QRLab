/**
 * ,---------,       ____  _ __
 * |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 * | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * | / ,--´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * Copyright (C) 2024 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received A copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * nn_controller.c - App layer application of an out of tree controller.
 */

#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"

#include "FreeRTOS.h"
#include "task.h"
#include "math3d.h"
#define DEBUG_MODULE "NNCONTROLLER"
#include "debug.h"

#include "nn_controller.h"
#include "log.h"
#include "param.h"
#include "usec_time.h"
#include "led.h"

static control_t_n control_n;
struct vec euler_angles;
static float state_array[17];
static uint32_t activationTime;

int PWM_NN_0, PWM_NN_1, PWM_NN_2, PWM_NN_3;


// We still need an appMain() function, but we will not really use it. Just let it quietly sleep.
void appMain() {
  DEBUG_PRINT("Waiting for activation ...\n");

  while(1) {
    vTaskDelay(M2T(2000));
  }
}

static void resetNNState(void) {
  control_n.out_0 = 0.0f;
  control_n.out_1 = 0.0f;
  control_n.out_2 = 0.0f;
  control_n.out_3 = 0.0f;
  control_n.rpm_0 = 0.0f;
  control_n.rpm_1 = 0.0f;
  control_n.rpm_2 = 0.0f;
  control_n.rpm_3 = 0.0f;
}

void controllerOutOfTreeInit() {
  resetNNState();
}

bool controllerOutOfTreeTest() {
  return true;
}

void controllerOutOfTree(
    control_t *control,
    const setpoint_t *setpoint,
    const sensorData_t *sensors,
    const state_t *state,
    const uint32_t tick
    ) {
  
  control->controlMode = controlModeNN;
  if (!RATE_DO_EXECUTE(RATE_500_HZ, tick)) {
    return;
  }

  struct quat q_current = mkquat(
      state->attitudeQuaternion.x,
      state->attitudeQuaternion.y,
      state->attitudeQuaternion.z,
      state->attitudeQuaternion.w
  );

  struct quat q_target = mkquat(
    0.0f,
    0.0f,
    0.0f,
    1.0f
  );

  struct quat q_err = quat_error_xyzw(q_target, q_current, true);

  struct vec world_pos_error;
  world_pos_error.x = state->position.x - setpoint->position.x;
  world_pos_error.y = state->position.y - setpoint->position.y;
  world_pos_error.z = state->position.z - setpoint->position.z;

  struct vec world_lin_vel;
  world_lin_vel.x = state->velocity.x;
  world_lin_vel.y = state->velocity.y;
  world_lin_vel.z = state->velocity.z;

  state_array[0] = world_pos_error.x;
  state_array[1] = world_pos_error.y;
  state_array[2] = world_pos_error.z;
  state_array[3] = q_err.x;
  state_array[4] = q_err.y;
  state_array[5] = q_err.z;
  state_array[6] = q_err.w;
  state_array[7] = world_lin_vel.x;
  state_array[8] = world_lin_vel.y;
  state_array[9] = world_lin_vel.z;
  state_array[10] = radians(sensors->gyro.x);
  state_array[11] = radians(sensors->gyro.y);
  state_array[12] = radians(sensors->gyro.z);
  state_array[13] = control_n.out_0;
  state_array[14] = control_n.out_1;
  state_array[15] = control_n.out_2;
  state_array[16] = control_n.out_3;

  if (setpoint->mode.z == modeDisable) {
    resetNNState();
    control->motorPwm[0] = 0;
    control->motorPwm[1] = 0;
    control->motorPwm[2] = 0;
    control->motorPwm[3] = 0;
    PWM_NN_0 = 0;
    PWM_NN_1 = 0;
    PWM_NN_2 = 0;
    PWM_NN_3 = 0;
    return;
  } else {
    uint64_t startTime = usecTimestamp();
    neuralNetworkComputation(&control_n, state_array);
    activationTime = (uint32_t)(usecTimestamp() - startTime);

    rpm2pwm(&control_n, &PWM_NN_0, &PWM_NN_1, &PWM_NN_2, &PWM_NN_3);

    control->motorPwm[0] = PWM_NN_0;
    control->motorPwm[1] = PWM_NN_1;
    control->motorPwm[2] = PWM_NN_2;
    control->motorPwm[3] = PWM_NN_3;
  }
}

void rpm2pwm(control_t_n *control_n, int *PWM_NN_0, int *PWM_NN_1, int *PWM_NN_2, int *PWM_NN_3) {
  *PWM_NN_0 = (control_n->rpm_0 - 4070.3f) / 0.2685f;
  *PWM_NN_1 = (control_n->rpm_1 - 4070.3f) / 0.2685f;
  *PWM_NN_2 = (control_n->rpm_2 - 4070.3f) / 0.2685f;
  *PWM_NN_3 = (control_n->rpm_3 - 4070.3f) / 0.2685f;
}

static inline float clampf_pm1(float v) {
  return (v > 1.0f) ? 1.0f : (v < -1.0f) ? -1.0f : v;
}

struct quat quat_normalize(struct quat q) {
  float n = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
  if (n < 1e-12f) return q;
  return mkquat(q.x/n, q.y/n, q.z/n, q.w/n);
}

struct quat quat_conjugate(struct quat q) {
  return mkquat(-q.x, -q.y, -q.z, q.w);
}

struct quat quat_multiply(struct quat q1, struct quat q2) {
  float x = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y;
  float y = q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x;
  float z = q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w;
  float w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z;

  return mkquat(x, y, z, w);
}

struct quat quat_error_xyzw(struct quat q_target, struct quat q_current, bool ensure_pos_w) {
  struct quat q_t_norm = quat_normalize(q_target);
  struct quat q_c_norm = quat_normalize(q_current);

  struct quat q_t_inv = quat_conjugate(q_t_norm);
  struct quat q_e = quat_multiply(q_t_inv, q_c_norm);

  if (ensure_pos_w && q_e.w < 0.0f) {
    q_e.x = -q_e.x;
    q_e.y = -q_e.y;
    q_e.z = -q_e.z;
    q_e.w = -q_e.w;
  }

  return quat_normalize(q_e);
}

struct vec vec_cross_product(struct vec v1, struct vec v2) {
  struct vec result;
  result.x = v1.y * v2.z - v1.z * v2.y;
  result.y = v1.z * v2.x - v1.x * v2.z;
  result.z = v1.x * v2.y - v1.y * v2.x;
  return result;
}

struct vec rotate_vector_by_quaternion(struct vec v, struct quat q) {
  struct vec u = {q.x, q.y, q.z};
  float s = q.w;

  struct vec uv = vec_cross_product(u, v);
  struct vec uuv = vec_cross_product(u, uv);

  struct vec rotated;
  rotated.x = v.x + 2.0f * (s * uv.x + uuv.x);
  rotated.y = v.y + 2.0f * (s * uv.y + uuv.y);
  rotated.z = v.z + 2.0f * (s * uv.z + uuv.z);

  return rotated;
}

LOG_GROUP_START(ctrlNN)
LOG_ADD(LOG_UINT32, activationTime, &activationTime)

LOG_ADD(LOG_FLOAT, ob_x, &state_array[0])
LOG_ADD(LOG_FLOAT, ob_y, &state_array[1])
LOG_ADD(LOG_FLOAT, ob_z, &state_array[2])

LOG_ADD(LOG_INT32, motor_pwm_0, &PWM_NN_0)
LOG_ADD(LOG_INT32, motor_pwm_1, &PWM_NN_1)
LOG_ADD(LOG_INT32, motor_pwm_2, &PWM_NN_2)
LOG_ADD(LOG_INT32, motor_pwm_3, &PWM_NN_3)

LOG_GROUP_STOP(ctrlNN)