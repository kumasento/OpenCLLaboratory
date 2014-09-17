#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <OpenCL/cl.h>

#include "cl_utils.h"

int main(int argc, char* argv[]) {

    int use_device_id = 0;
    char* kernel_name = NULL;
    char* kernel_func_name = NULL;

    char opt;
    while((opt = getopt(argc, argv, "f:d:u:")) != -1) {
        switch(opt) {
            case 'f':
                kernel_name = optarg;
                break;
            case 'd':
                use_device_id = atoi(optarg);
                break;
            case 'u':
                kernel_func_name = optarg;
                break;
            default:
                break;
        }
    }

    if (kernel_name == NULL) {
        fprintf(stderr, "Please use -f to add kernel file name. Exiting.");
        exit(1);
    }
    if (kernel_func_name == NULL) {
        fprintf(stderr, "Please use -u to add kernel file name. Exiting.");
        exit(1);
    }

    printf("Kernel File Name:\t%s\n", kernel_name);
    printf("Kernel Function Name:\t%s\n", kernel_func_name);
    printf("Device ID(using):\t%d\n", use_device_id);

    cl_int status;
    
    //1. Platforms
    cl_uint numPlatforms = 0;
    cl_platform_id* platforms = NULL;
    
    status = clGetPlatformIDs(0, NULL, &numPlatforms);

    platforms = 
        (cl_platform_id*) malloc(
                numPlatforms * sizeof(cl_platform_id));

    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    check_error(status, "clGetPlatformIDs");

    get_cl_platform_info(numPlatforms, platforms);

    //2. Devices
    cl_uint numDevices = 0;
    cl_device_id* devices = NULL;

    status |= clGetDeviceIDs(
                    platforms[0],
                    CL_DEVICE_TYPE_ALL,
                    0,
                    NULL,
                    &numDevices);

    devices = 
        (cl_device_id*) malloc(
                numDevices * sizeof(cl_device_id));

    status |= clGetDeviceIDs(
                    platforms[0],
                    CL_DEVICE_TYPE_ALL,
                    numDevices,
                    devices,
                    NULL);

    get_cl_device_info(numDevices, devices);

    //3. Context
    cl_context context = NULL;
    context = clCreateContext(
                NULL,
                numDevices,
                devices, 
                NULL,
                NULL,
                &status);
    check_error(status, "clCreateContext");

    //4. Command Queue
    cl_command_queue cmdQueue;
    cmdQueue = clCreateCommandQueue(
                    context,
                    devices[use_device_id],
                    0,
                    &status);
    check_error(status, "clCreateCommandQueue");

    ///////////////////vector add kernel
    // *1. data preparation
    int *A = NULL;
    int *B = NULL;
    int *C = NULL;

    const int elements = 2048;
    size_t datasize = sizeof(int) * elements;

    A = (int*) malloc(datasize);
    B = (int*) malloc(datasize);
    C = (int*) malloc(datasize);

    int i;
    for (i = 0; i < elements; i++) {
        A[i] = i;
        B[i] = i;
    }

    cl_mem bufferA;
    cl_mem bufferB;
    cl_mem bufferC;

    bufferA = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY,
                datasize,
                NULL,
                &status);
    bufferB = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY,
                datasize,
                NULL,
                &status);
    bufferC = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY,
                datasize,
                NULL,
                &status);

    check_error(status, "clCreateBuffer");
    
    // *2. interact with devices
    status |= clEnqueueWriteBuffer(
            cmdQueue,
            bufferA,
            CL_FALSE,
            0,
            datasize,
            A,
            0,
            NULL,
            NULL);
    status |= clEnqueueWriteBuffer(
            cmdQueue,
            bufferB,
            CL_FALSE,
            0,
            datasize,
            B,
            0,
            NULL,
            NULL);
    check_error(status, "clEnqueueWriteBuffer");

    // *3. create program
    FILE* fp = fopen(kernel_name, "r");
    if (fp == NULL) {
        fprintf(stderr, "Something wrong when opening file: %s\n", kernel_name);
        exit(1);
    }

    char kernel_buffer[FILE_LENGTH];
    fread(kernel_buffer, 1, FILE_LENGTH, fp);
    puts("file closing ...");

    fclose(fp);

    const char *srcptr[] = {kernel_buffer};

    cl_program program = clCreateProgramWithSource(
            context,
            1,
            srcptr,
            NULL,
            &status);
    check_error(status, "clCreateProgramWithSource");

    status |= clBuildProgram(
            program,
            numDevices,
            devices,
            NULL,
            NULL,
            NULL);

    get_cl_build_program_log(program, devices, use_device_id);

    check_error(status, "clBuildProgram");

    // *4. create kernel
    cl_kernel kernel = NULL;
    kernel = clCreateKernel(program, kernel_func_name, &status);
    check_error(status, "clCreateKernel");
    
    status |= clSetKernelArg(
            kernel,
            0,
            sizeof(cl_mem),
            &bufferA);
    status |= clSetKernelArg(
            kernel,
            1,
            sizeof(cl_mem),
            &bufferB);
    status |= clSetKernelArg(
            kernel,
            2,
            sizeof(cl_mem),
            &bufferC);
    check_error(status, "clSetKernelArgs");
    
    size_t globalWorkSize[1];
    globalWorkSize[0] = elements;

    // *5. enqueue kernel for execution
    status = clEnqueueNDRangeKernel(
            cmdQueue,
            kernel,
            1,
            NULL,
            globalWorkSize,
            NULL,
            0,
            NULL,
            NULL);
    status = clEnqueueReadBuffer(
            cmdQueue,
            bufferC,
            CL_TRUE,
            0,
            datasize,
            C,
            0,
            NULL,
            NULL);

    int result = 1;
    for (i = 0; i < elements; i++) 
        if (C[i] != i + i) {
            result = 0;
            break;
        }
    if (result) 
        puts("Output is correct!");
    else
        puts("Something wrong");

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseContext(context);

    free(A);
    free(B);
    free(C);
    free(platforms);
    free(devices);

    return 0;
}
