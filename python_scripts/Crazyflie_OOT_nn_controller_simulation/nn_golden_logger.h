#ifndef NN_GOLDEN_LOGGER_H
#define NN_GOLDEN_LOGGER_H

#ifdef __cplusplus
extern "C" {
#endif

// Fixed sizes for your network
#define NN_GOLDEN_OBS 12
#define NN_GOLDEN_HID 64
#define NN_GOLDEN_ACT 4

// Initialize logger.
//  - path: CSV path; if NULL, defaults to "nn_golden_matrix.csv"
//  - truncate: nonzero => start a fresh file (truncate), 0 => append (header written if file empty)
void nn_golden_logger_init(const char *path, int truncate);

// Write the golden matrix (exactly once if write_once != 0).
// Produces a 64-row CSV with columns: obs,h1,h2,act_raw,rpm_raw
void nn_golden_logger_write_matrix(const float obs[NN_GOLDEN_OBS],
                                   const float h1[NN_GOLDEN_HID],
                                   const float h2[NN_GOLDEN_HID],
                                   const float act_raw[NN_GOLDEN_ACT],
                                   const float act_clip[NN_GOLDEN_ACT],
                                   const float rpm_raw[NN_GOLDEN_ACT],
                                   const float rpm_clip[NN_GOLDEN_ACT],
                                   int write_once);

// Close the file (also auto-closed at process exit)
void nn_golden_logger_close(void);

#ifdef __cplusplus
}
#endif
#endif // NN_GOLDEN_LOGGER_H
