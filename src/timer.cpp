//timer.cpp

#include "timer.h"
#include <chrono>

using Clock = std::chrono::steady_clock;  // was high_resolution_clock

static double now_seconds() {   //returns the now time as double
    auto now = Clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}

void timer_start(Timer *t) {
    t->start_ts = now_seconds();
}

void timer_stop(Timer *t) {
    t->end_ts = now_seconds();
}

double timer_elapsed_ms(const Timer *t) {
    return (t->end_ts - t->start_ts) * 1000.0;
}