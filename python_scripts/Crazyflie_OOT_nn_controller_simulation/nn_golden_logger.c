#include "nn_golden_logger.h"
#include <stdio.h>
#include <stdlib.h>

#ifndef NN_GOLDEN_DEFAULT_PATH
#define NN_GOLDEN_DEFAULT_PATH "nn_gold_log.csv"
#endif

static FILE *g_fp = NULL;
static int   g_header_written = 0;
static int   g_written_once   = 0;

static void nn_golden_on_exit(void) { nn_golden_logger_close(); }

void nn_golden_logger_init(const char *path, int truncate) {
    if (g_fp) return;  // already open
    const char *p = path ? path : NN_GOLDEN_DEFAULT_PATH;
    const char *mode = truncate ? "wb" : "ab";
    g_fp = fopen(p, mode);
    if (!g_fp) return;
    atexit(nn_golden_on_exit);
    // header
    if (truncate) {
        fprintf(g_fp, "obs,h1,h2,act_raw,act_clip,rpm_raw,rpm_clip\n");
        fflush(g_fp);
        g_header_written = 1;
    } else {
        fseek(g_fp, 0, SEEK_END);
        long sz = ftell(g_fp);
        if (sz == 0) {
            fprintf(g_fp, "obs,h1,h2,act_raw,act_clip,rpm_raw,rpm_clip\n");
            fflush(g_fp);
        }
        g_header_written = 1;
    }
}

void nn_golden_logger_close(void) {
    if (g_fp) { fclose(g_fp); g_fp = NULL; }
}

// helper: print one numeric field or leave empty
static inline void print_field_or_empty(FILE *fp, int have, float v, int trailing_comma) {
    if (have) fprintf(fp, "%.8f", (double)v);
    if (trailing_comma) fputc(',', fp);
}

// Writes exactly 64 rows; see column fill rules in the header comment
void nn_golden_logger_write_matrix(const float obs[NN_GOLDEN_OBS],
                                   const float h1[NN_GOLDEN_HID],
                                   const float h2[NN_GOLDEN_HID],
                                   const float act_raw[NN_GOLDEN_ACT],
                                   const float act_clip[NN_GOLDEN_ACT],
                                   const float rpm_raw[NN_GOLDEN_ACT],
                                   const float rpm_clip[NN_GOLDEN_ACT],
                                   int write_once) {
    if (!g_fp) return;
    if (write_once && g_written_once) return;

    const int ROWS = NN_GOLDEN_HID;  // 64
    for (int r = 0; r < ROWS; ++r) {
        // col 1: obs (rows 0..11)
        print_field_or_empty(g_fp, r < NN_GOLDEN_OBS, (r < NN_GOLDEN_OBS) ? obs[r] : 0.0f, 1);

        // col 2: h1 (rows 0..63)
        print_field_or_empty(g_fp, 1, h1[r], 1);

        // col 3: h2 (rows 0..63)
        print_field_or_empty(g_fp, 1, h2[r], 1);

        // col 4: act_raw (rows 0..3)
        print_field_or_empty(g_fp, r < NN_GOLDEN_ACT, (r < NN_GOLDEN_ACT) ? act_raw[r] : 0.0f, 1);
        print_field_or_empty(g_fp, r < NN_GOLDEN_ACT, (r < NN_GOLDEN_ACT) ? act_clip[r] : 0.0f, 1);

        // col 5: rpm_raw (rows 0..3) — no trailing comma
        print_field_or_empty(g_fp, r < NN_GOLDEN_ACT, (r < NN_GOLDEN_ACT) ? rpm_raw[r] : 0.0f, 1);
        print_field_or_empty(g_fp, r < NN_GOLDEN_ACT, (r < NN_GOLDEN_ACT) ? rpm_clip[r] : 0.0f, 0);

        fputc('\n', g_fp);
    }
    fflush(g_fp);
    if (write_once) g_written_once = 1;
}
