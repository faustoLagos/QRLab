/* nn_controller_simulation.c
 * Test harness for simulating controllerOutOfTree using CSV data
 * Reads position, orientation (Euler), velocity, gyro and accelerometer CSVs,
 * runs controllerOutOfTree, and writes timestamped PWM outputs to CSV.
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <stdint.h>
 #include <string.h>
 #include <math.h>
 #include <stdarg.h>
 
 #include "stabilizer_types.h"
 #include "nn_controller.h"
 #include "math3d.h"
 
 extern void controllerOutOfTreeInit(void);
 extern void controllerOutOfTree(control_t *control,
                                 const setpoint_t *setpoint,
                                 const sensorData_t *sensors,
                                 const state_t *state,
                                 uint32_t tick);
 
 void consolePutchar(char c) { putchar(c); }
 int eprintf(const char *fmt, ...) {
     va_list args;
     va_start(args, fmt);
     int ret = vfprintf(stderr, fmt, args);
     va_end(args);
     return ret;
 }
 void vTaskDelay(uint32_t ms) { (void)ms; }
 
 static int loadCsv(const char *filename, double **t_out, double **v_out, int *n_out) {
     FILE *fp = fopen(filename, "r");
     if (!fp) { perror(filename); return -1; }
     int count = 0;
     char buf[256];
     while (fgets(buf, sizeof(buf), fp)) if (strchr(buf, ',')) count++;
     rewind(fp);
     double *t = malloc(count * sizeof(double));
     double *v = malloc(count * sizeof(double));
     int idx = 0;
     while (fgets(buf, sizeof(buf), fp)) {
         double ts, val;
         if (sscanf(buf, "%lf,%lf", &ts, &val) == 2) {
             t[idx] = ts;
             v[idx] = val;
             idx++;
         }
     }
     fclose(fp);
     *t_out = t;
     *v_out = v;
     *n_out = idx;
     return 0;
 }
 
 int main(void) {
     double *t_x,*x,*t_y,*y,*t_z,*z;
     double *t_qx,*qx,*t_qy,*qy,*t_qz,*qz,*t_qw,*qw;
     double *t_vx,*vx,*t_vy,*vy,*t_vz,*vz;
     double *t_wx,*wx,*t_wy,*wy,*t_wz,*wz;
     int n_x,n_y,n_z,n_qx,n_qy,n_qz,n_qw,n_vx,n_vy,n_vz,n_wx,n_wy,n_wz;
 
     if (loadCsv("x0.csv",&t_x,&x,&n_x) ||
         loadCsv("y0.csv",&t_y,&y,&n_y) ||
         loadCsv("z0.csv",&t_z,&z,&n_z) ||
         loadCsv("qx0.csv",&t_qx,&qx,&n_qx) ||
         loadCsv("qy0.csv",&t_qy,&qy,&n_qy) ||
         loadCsv("qz0.csv",&t_qz,&qz,&n_qz) ||
         loadCsv("qw0.csv",&t_qw,&qw,&n_qw) ||
         loadCsv("vx0.csv",&t_vx,&vx,&n_vx) ||
         loadCsv("vy0.csv",&t_vy,&vy,&n_vy) ||
         loadCsv("vz0.csv",&t_vz,&vz,&n_vz) ||
         loadCsv("wx0.csv",&t_wx,&wx,&n_wx) ||
         loadCsv("wy0.csv",&t_wy,&wy,&n_wy) ||
         loadCsv("wz0.csv",&t_wz,&wz,&n_wz)) {
         fprintf(stderr,"Failed to load CSV files\n");
         return 1;
     }
     int N = n_x;
 
     FILE *fp = fopen("pwm_log.csv","w");
     fprintf(fp,"time,PWM0,PWM1,PWM2,PWM3\n");
 
     controllerOutOfTreeInit();
 
     for (int i = 0; i < N; i++) {
         control_t control = {0};
         setpoint_t setpoint = {0};
         state_t state = {0};
         sensorData_t sensors = {0};

         setpoint.position.x = 0.0f;
         setpoint.position.y = 0.0f;
         setpoint.position.z = 1.0f;

         setpoint.attitudeQuaternion.x = 0.0f;
         setpoint.attitudeQuaternion.y = 0.0f;
         setpoint.attitudeQuaternion.z = 0.0f;
         setpoint.attitudeQuaternion.w = 1.0f;
 
         state.position.x = (float)x[i];
         state.position.y = (float)y[i];
         state.position.z = (float)z[i];
 
         state.attitudeQuaternion.x = (float)qx[i];
         state.attitudeQuaternion.y = (float)qy[i];
         state.attitudeQuaternion.z = (float)qz[i];
         state.attitudeQuaternion.w = (float)qw[i];
 
         state.velocity.x = (float)vx[i];
         state.velocity.y = (float)vy[i];
         state.velocity.z = (float)vz[i];
 
         sensors.gyro.x = degrees((float)wx[i]);
         sensors.gyro.y = degrees((float)wy[i]);
         sensors.gyro.z = degrees((float)wz[i]);
 
         controllerOutOfTree(&control, &setpoint, &sensors, &state, t_x[i]);
 
         fprintf(fp, "%f,%u,%u,%u,%u\n",
                 t_x[i],
                 control.motorPwm[0], control.motorPwm[1],
                 control.motorPwm[2], control.motorPwm[3]
                );
     }
 
     fclose(fp);
     return 0;
 }
 
