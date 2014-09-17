#include <stdio.h>
#include <stdlib.h>

#include <OpenCL/cl.h>

#include "cl_utils.h"

#define NUM_PLATFORM 1
#define NUM_DEVICE 2
#define PLATFORM_ID 1
#define DEVICE_ID 1

int main(int argc, char *argv[]) {
    
    cl_int status;
    cl_platform_id platform;

    status = clGetPlatformIDs(NUM_PLATFORM, &platform, NULL);
    check_error(status, "clGetPlatformIDs");

    cl_device_id device;
    status |= clGetDeviceIDs(
            platform,
            CL_DEVICE_TYPE_GPU,
            1,
            &device,
            NULL);
    check_error(status, "clGetDeviceIDs");

    get_cl_device_info(1, &device);

    cl_context_properties cps[3] = 
    { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    cl_context ctx = clCreateContext(
            cps,
            NUM_PLATFORM,
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


    /// This one needs image io and the codes are very similar to the matrices multiplication one.
    // Now Aborted

    return 0;
}
