#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <OpenCL/cl.h>

#include "cl_utils.h"

#define DEVICE_ID 1
#define PLATFORM_NUM 1
#define KERLEN 8196
#define KERNEL_NAME "MatXMat.cl"

int main(int argc, char *argv[]) {

    char opt;
    
    int local_patch = 16;
    int mat_scale = 10;
    int times = 10;

    while((opt = getopt(argc, argv, "p:s:t:")) != -1) {
        switch(opt) {
            case 'p':
                local_patch = atoi(optarg);
                break;
            case 's':
                mat_scale = atoi(optarg);
                break;
            case 't':
                times = atoi(optarg);
                break;
            default:
                break;
        }
    }

    cl_int status;
    
    cl_platform_id platform;
    status = clGetPlatformIDs(PLATFORM_NUM, &platform, NULL);
    check_error(status, "clGetPlatformIDs");    

    cl_device_id device;
    status |= clGetDeviceIDs(
        platform, 
        CL_DEVICE_TYPE_GPU,
        DEVICE_ID,
        &device,
        NULL);
    check_error(status, "clGetDeviceIDs");

    get_cl_device_info(1, &device);

    cl_context_properties cps[3] = 
    { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    cl_context ctx = clCreateContext(
            cps, 
            PLATFORM_NUM,
            &device, 
            NULL,
            NULL,
            &status);
    check_error(status, "clCreateContext");

    cl_command_queue myqueue = clCreateCommandQueue(
            ctx,
            device, 
            0,
            &status);
    check_error(status, "clCreateCommandQueue");

    int scale = 1 << mat_scale;
    int w_A = scale * times, h_A = scale;
    int h_B = scale * times, w_B = scale;
    
    float *m_A = (float *) malloc(sizeof(float) * w_A * h_A);
    float *m_B = (float *) malloc(sizeof(float) * w_B * h_B);
    float *m_C = (float *) malloc(sizeof(float) * h_A * w_B);
    int i;
    for (i = 0; i < w_A * h_A; i++) m_A[i] = (float) i * 0.01;
    for (i = 0; i < w_B * h_B; i++) m_B[i] = (float) i * 0.02;

    cl_mem buf_A = clCreateBuffer(
            ctx,
            CL_MEM_READ_ONLY,
            w_A*h_A*sizeof(float),
            NULL,
            &status);
    status = clEnqueueWriteBuffer(
            myqueue,
            buf_A,
            CL_TRUE,
            0,
            w_A * h_A * sizeof(float),
            (void *) m_A,
            0,
            NULL,
            NULL);
    check_error(status, "clCreateBuffer | clEnqueueWriteBuffer (A)");

    cl_mem buf_B = clCreateBuffer(
            ctx,
            CL_MEM_READ_ONLY,
            w_B * h_B * sizeof(float),
            NULL,
            &status);
    status = clEnqueueWriteBuffer(
            myqueue, 
            buf_B,
            CL_TRUE,
            0,
            w_B * h_B * sizeof(float),
            (void *) m_B,
            0,
            NULL,
            NULL);
    check_error(status, "clCreateBuffer | clEnqueueWriteBuffer (B)");

    cl_mem buf_C = clCreateBuffer(
            ctx,
            CL_MEM_READ_ONLY,
            w_B * h_A * sizeof(float),
            NULL,
            &status);
    check_error(status, "clCreateBuffer (C)");

    FILE *fp = fopen(KERNEL_NAME, "r");
    char kerbuf[KERLEN];
    fread(kerbuf, 1, KERLEN, fp);
    fclose(fp);
    char *src[] = {kerbuf};

    cl_program prg = clCreateProgramWithSource(
                        ctx, 
                        1, 
                        (const char **) &src,
                        NULL,
                        &status);
    check_error(status, "clCreateProgramWithSource");

    status = clBuildProgram(prg, 0, NULL, NULL, NULL, NULL);
    get_cl_build_program_log(prg, device);
    check_error(status, "clBuildProgram");

    cl_kernel kernel = clCreateKernel(
            prg,
            "simpleMultiply",
            &status);
    check_error(status, "clCreateKernel");

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buf_C);
    clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&w_A);
    clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&w_B);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&buf_A);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&buf_B);

    size_t localws[2] = {16, 16};
    int w = get_divided_value(w_B, 16);
    int h = get_divided_value(h_A, 16);
    //printf("%d %d\n", w, h);
    size_t globalws[2] = {w, h};

    cl_event execute;
    status = clEnqueueNDRangeKernel(
            myqueue,
            kernel,
            2,
            NULL,
            globalws,
            localws,
            0,
            NULL,
            &execute);
    check_error(status, "clEnqueueNDRangeKernel");

    clFlush(myqueue);

    status = clEnqueueReadBuffer(
            myqueue,
            buf_C,
            CL_TRUE,
            0,
            w_B * h_A * sizeof(float),
            (void *) m_C,
            0,
            NULL,
            NULL);
    
    puts("==> Checking Correctness ...");
    puts("Skipped");
    /*
    for (int x = 0; x < h_A; x ++) {
        for (int y = 0; y < w_B; y ++) {
            
            float sum = 0.0f;
            for (int t = 0; t < w_A; t ++)
                sum += m_A[x*w_A+t] * m_B[t*w_B+y];

            if (abs(sum - m_C[x*w_B+y]) > 1E-6)
                printf("WRONG at (x, y)=(%3d, %3d): sum=%.6f m_C=%.7f\n", 
                        x, 
                        y, 
                        sum, 
                        m_C[x*w_B+y]);
        }
    }
    */

    return 0;
}
