
#define MAXLEN 1000
#define FILE_LENGTH 1000

static void check_error(cl_int error, char *name) {
    if (error != CL_SUCCESS) {
        fprintf(stderr, "Non-successful return code %d for %s.   Exiting.\n", error, name);
        exit(1);
    }
    //else
    //    printf("[PASSED] %s...\n", name);
}

void get_cl_platform_info(cl_uint numPlatforms, cl_platform_id* platforms) {
    char name[128];
    char vendor[128];
    char version[128];

    puts("");
    printf("Number of platforms:\t%d\n", numPlatforms);

    int i;
    cl_int status = 0;
    for (i = 0; i < numPlatforms; i++) {
        status |= clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 128, vendor, NULL);
        status |= clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, name, NULL);
        status |= clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 128, version, NULL);
        check_error(status, "clGetPlatformInfo");
        
        printf("Platform Version:\t%s\n", version);
        printf("Platform Name:\t\t%s\n", name);
        printf("Platform Vendor:\t%s\n", vendor);
    }
}

void get_cl_device_info(cl_uint numDevices, cl_device_id* devices) {
    
    puts("");
    printf("Number of Devices:\t%d\n", numDevices);

    char name[MAXLEN];
    char vendor[MAXLEN];
    char version[MAXLEN];
    cl_uint maxunits = 0;
    cl_uint maxclockfreq = 0;
    cl_ulong globalmemsize = 0;

    int i;
    cl_int status = 0;
    for (i = 0; i < numDevices; i++) {
        status |= clGetDeviceInfo(devices[i], CL_DEVICE_NAME, MAXLEN, name, NULL);
        status |= clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, MAXLEN, vendor, NULL);
        status |= clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, MAXLEN, version, NULL);
        status |= clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxunits, NULL);
        status |= clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &maxclockfreq, NULL);
        status |= clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalmemsize, NULL);

        puts("");
        printf("Device ID:\t\t%d\n", i+1);
        printf("Device Name:\t\t%s\n", name);
        printf("Device Vendor:\t\t%s\n", vendor);
        printf("Device Version:\t\t%s\n", version);
        printf("Device Compute Units:\t%u\n", maxunits);
        printf("Device Clock Frequency:\t%u\n", maxclockfreq);
        printf("Device Mem Size:\t%lld\n", globalmemsize);
    }
}

void get_cl_build_program_log(cl_program program, cl_device_id* devices, int id) {
    puts("");
    puts("**Requesting Build Program Log ...**");
    
    size_t log_size;
    clGetProgramBuildInfo(program, devices[id], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    char *log = (char *) malloc(log_size);
    clGetProgramBuildInfo(program, devices[id], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);    
    printf("%s\n", log);
}
