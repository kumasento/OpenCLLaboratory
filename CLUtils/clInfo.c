#include <stdio.h>
#include <stdlib.h>

#include <OpenCL/cl.h>

#include "cl_utils.h"

int main(int argc, char* argv[]) {
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

    return 0;
}
