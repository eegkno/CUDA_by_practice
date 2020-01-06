#ifndef CPU_TIMER_H__
#define CPU_TIMER_H__

struct CpuTimer {
    // capture the start time
    clock_t start, stop;

    CpuTimer() {
        start = 0;
        stop = 0;
    }

    ~CpuTimer() {}

    void Start() { start = clock(); }

    void Stop() { stop = clock(); }

    float Elapsed() {
        float elapsedTime =
            (float)(stop - start) / (float)CLOCKS_PER_SEC * 1000.0f;
        return elapsedTime;
    }
};

#endif /* CPU_TIMER_H__ */
