#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__

struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer() {
        // create events
        // -:YOUR CODE HERE:-
    }

    ~GpuTimer() {
        // delete events
        // -:YOUR CODE HERE:-
    }

    void Start() {
        // start event
        // -:YOUR CODE HERE:-
    }

    void Stop() {
        // stop event
        // -:YOUR CODE HERE:-
    }

    float Elapsed() {
        // elapsed time
        // -:YOUR CODE HERE:-
        return elapsed;
    }
};

#endif /* GPU_TIMER_H__ */
