#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <assert.h>

//#include <CL/cl.h>

#include "oclcommon.h"

#define GROUP_SIZE 128

int main(int argc, char *argv[]){
    // check commandline parameters
    if (argc < 2) {
        fprintf(stderr, "Usage: %s [kernel] [length of vector]\n",
                argv[0]);
        exit(1);
    }
    
    cl_int errorCode;
    cl_device_type      deviceType = CL_DEVICE_TYPE_ALL;
    cl_device_id *      devices = NULL;
    cl_context          context = NULL;
    cl_command_queue    cmdQueue = NULL;
    cl_program          program = NULL;

    char *kernelfile = argv[1];
    int length = atoi(argv[2]);

    assert(initialization(
                deviceType,
                devices,
                &context,
                &cmdQueue,
                &program,
                kernelfile));

    float *X = (float*) malloc(sizeof(float)*length);
    float *Y = (float*) malloc(sizeof(float)*length);
    float *Z = (float*) malloc(sizeof(float)*length);

    for (int i = 0; i < length; i++) {
        X[i] = (float)i + 0.1;
        Y[i] = (float)i + 0.2;
        Z[i] = 0.0;
    } 

    cl_mem X_mem, Y_mem, Z_mem;
    ALLOCATE_GPU_READ(X_mem, X, sizeof(float)*length);
    ALLOCATE_GPU_READ(Y_mem, Y, sizeof(float)*length);
    ALLOCATE_GPU_READ_WRITE_INIT(Z_mem, Z, sizeof(float)*length); 
    
    size_t globalSize[1] = {length/4};
    size_t localSize[1] = {1};

    float alpha = 0.2;
    cl_kernel kernel = clCreateKernel(program, "saxpy_dim4", &errorCode); CHECKERROR;
    errorCode = clSetKernelArg(kernel, 0, sizeof(cl_mem), &X_mem); CHECKERROR;
    errorCode = clSetKernelArg(kernel, 1, sizeof(cl_mem), &Y_mem); CHECKERROR;
    errorCode = clSetKernelArg(kernel, 2, sizeof(cl_mem), &Z_mem); CHECKERROR;
    errorCode = clSetKernelArg(kernel, 3, sizeof(cl_float), &alpha); CHECKERROR;

    printf("Start to Run ...\n");
    cl_event runEvent;
    errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalSize, localSize, 0, NULL, &runEvent); CHECKERROR;
    errorCode = clFinish(cmdQueue);

    printf("Execution Time: %.2fns\n", executionTime(runEvent) / length * 1e9);

    printf("Start to Readback ...\n");
    errorCode = clEnqueueReadBuffer(cmdQueue, Z_mem, CL_TRUE, 0, sizeof(float)*length, Z, 0, NULL, NULL); CHECKERROR;
    
    printf("Checking Correctness ...\n");
    
    for (int i = 0; i < length; i++) {
        float res = X[i] * alpha + Y[i];
        float ans = Z[i];
        if (res - ans > 1E-4 || res - ans < -1E-4) {
            printf("%.10f %.10f %.10f\n", res, ans, res-ans);
            fprintf(stderr, "ERROR!");
            exit(1);
        }
    }   
    printf("OK\n");

    return 0;
}
