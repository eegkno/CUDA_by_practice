#ifndef __UTILITIES_H__
#define __UTILITIES_H__

#include <stdio.h>

void bandwidth(int n, float time) {
    cudaDeviceProp prop;

    float efbw, tbw, ram;

    cudaGetDeviceProperties(&prop, 0);

    tbw = ((prop.memoryClockRate * 10e2f) * (prop.memoryBusWidth / 8) * 2) /
          10e9f;
    efbw = ((n * n * 4 * 2) / 10e9) / (time * 10e-4f);
    ram = (efbw / tbw) * 100;

    printf("TBw = %.3f GB/s, EBw = %.3f GB/s, RAM Utilization = %.2f %%  \n",
           tbw, efbw, ram);
    // 40 - 60 % OK
    // 60 - 75 % Good
    // > 75 % excellent
}

#endif
