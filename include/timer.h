//timer.h

#ifndef TIMER_H
#define TIMER_H

#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double start_ts;
    double end_ts;
} Timer;

void timer_start(Timer *t);

void timer_stop(Timer *t);

double timer_elapsed_ms(const Timer *t);

#ifdef __cplusplus
}
#endif

#endif