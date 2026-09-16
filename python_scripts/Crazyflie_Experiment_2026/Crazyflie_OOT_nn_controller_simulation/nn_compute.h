//
// Created by fausto on 2024-12-10.
//

#ifndef NN_COMPUTE_H
#define NN_COMPUTE_H

#include <math.h>
#include <stdlib.h>

typedef struct control_t_n {
  float out_0;
  float out_1;
  float out_2;
  float out_3;
  float rpm_0;
  float rpm_1;
  float rpm_2;
  float rpm_3;
} control_t_n;

void neuralNetworkComputation(control_t_n *control_n, const float *state_array);

#endif //NN_COMPUTE_H
