/* Copyright 2015 The math21 Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdio>
#include <cstdlib>
#include "inner_c.h"
#include "test.h"

#ifndef MATH21_FLAG_USE_OPENCL

void math21_opencl_test() {

}

void math21_opencl_test2() {

}

void math21_opencl_test3() {

}

int math21_opencl_test4() {
    return 1;
}

#endif

#ifdef MATH21_FLAG_USE_OPENCL

void math21_opencl_test2() {
#define MEM_SIZE (128)

    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem memobj = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    char string[MEM_SIZE];
    const char *source_str = "__kernel void helloworldopenclkernel(__global char* string) {\n"
                       "    string[0] = 'H';\n"
                       "    string[1] = 'e';\n"
                       "    string[2] = 'l';\n"
                       "    string[3] = 'l';\n"
                       "    string[4] = 'o';\n"
                       "    string[5] = ',';\n"
                       "    string[6] = ' ';\n"
                       "    string[7] = 'W';\n"
                       "    string[8] = 'o';\n"
                       "    string[9] = 'r';\n"
                       "    string[10] = 'l';\n"
                       "    string[11] = 'd';\n"
                       "    string[12] = '!';\n"
                       "    string[13] = '\\0';\n"
                       "}";
    size_t source_size;
    source_size = strlen(source_str);

    /* Get Platform and Device Info */
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    /* Create OpenCL context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    /* Create Command Queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    /* Create Memory Buffer */
    memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(char), NULL, &ret);

    /* Create Kernel Program from the source */
    program = clCreateProgramWithSource(context, 1, &source_str,
                                        (const size_t *) &source_size, &ret);

    /* Build Kernel Program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    /* Create OpenCL Kernel */
    kernel = clCreateKernel(program, "helloworldopenclkernel", &ret);


    /* Set OpenCL Kernel Parameters */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &memobj);

    /* Execute OpenCL Kernel */
    ret = clEnqueueTask(command_queue, kernel, 0, NULL, NULL);

    /* Copy results from the memory buffer */
    ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0,
                              MEM_SIZE * sizeof(char), string, 0, NULL, NULL);

    /* Display Result */
    fprintf(stdout, "string: %s\n", string);

    /* Finalization */
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(memobj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
}

void math21_opencl_test() {
    cl_int status;
    cl_uint numPlatforms;

    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (CL_SUCCESS == status)
        printf("Detected OpenCL platforms: %d\n", numPlatforms);
    else
        printf("Error calling clGetPlatformIDs. Error code: %d\n", status);
}

const char *src_addKernel = "\n"
                        "__kernel void add(                  \n"
                        "   __global const uchar* a,         \n"
                        "   __global const uchar* b,         \n"
                        "   __global uchar* c,               \n"
                        "   const unsigned int count)        \n"
                        "{                                   \n"
                        "   uint idx = get_global_id(0);     \n"
                        "   if(idx < count)                  \n"
                        "       c[idx] = a[idx] + b[idx];    \n"
                        "}                                   \n"
                        "\n";


void cl_assert(cl_int error, char const *const message) {
    static char const *const codes[] = {
            "CL_SUCCESS",
            "CL_DEVICE_NOT_FOUND",
            "CL_DEVICE_NOT_AVAILABLE",
            "CL_COMPILER_NOT_AVAILABLE",
            "CL_MEM_OBJECT_ALLOCATION_FAILURE",
            "CL_OUT_OF_RESOURCES",
            "CL_OUT_OF_HOST_MEMORY",
            "CL_PROFILING_INFO_NOT_AVAILABLE",
            "CL_MEM_COPY_OVERLAP",
            "CL_IMAGE_FORMAT_MISMATCH",
            "CL_IMAGE_FORMAT_NOT_SUPPORTED",
            "CL_BUILD_PROGRAM_FAILURE",
            "CL_MAP_FAILURE",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "CL_INVALID_VALUE",
            "CL_INVALID_DEVICE_TYPE",
            "CL_INVALID_PLATFORM",
            "CL_INVALID_DEVICE",
            "CL_INVALID_CONTEXT",
            "CL_INVALID_QUEUE_PROPERTIES",
            "CL_INVALID_COMMAND_QUEUE",
            "CL_INVALID_HOST_PTR",
            "CL_INVALID_MEM_OBJECT",
            "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
            "CL_INVALID_IMAGE_SIZE",
            "CL_INVALID_SAMPLER",
            "CL_INVALID_BINARY",
            "CL_INVALID_BUILD_OPTIONS",
            "CL_INVALID_PROGRAM",
            "CL_INVALID_PROGRAM_EXECUTABLE",
            "CL_INVALID_KERNEL_NAME",
            "CL_INVALID_KERNEL_DEFINITION",
            "CL_INVALID_KERNEL",
            "CL_INVALID_ARG_INDEX",
            "CL_INVALID_ARG_VALUE",
            "CL_INVALID_ARG_SIZE",
            "CL_INVALID_KERNEL_ARGS",
            "CL_INVALID_WORK_DIMENSION",
            "CL_INVALID_WORK_GROUP_SIZE",
            "CL_INVALID_WORK_ITEM_SIZE",
            "CL_INVALID_GLOBAL_OFFSET",
            "CL_INVALID_EVENT_WAIT_LIST",
            "CL_INVALID_EVENT",
            "CL_INVALID_OPERATION",
            "CL_INVALID_GL_OBJECT",
            "CL_INVALID_BUFFER_SIZE",
            "CL_INVALID_MIP_LEVEL",
            "CL_INVALID_GLOBAL_WORK_SIZE",
    };

    if (error != CL_SUCCESS) {
        printf("%s (%s)\n", message, codes[-error]);
        exit(-1);
    }
}

void math21_opencl_test3() {
    char x_data[] = {70, 100, 100, 8, 100, 2, 80, 101, 100, 100, 50, 30, 0};
    char y_data[] = {2, 1, 8, 100, 11, 30, 7, 10, 14, 8, 50, 3, 0};
    size_t const count = sizeof(x_data);

    char *const a_data = (char *) malloc(count);
    char *const b_data = (char *) malloc(count);
    char *const c_data = (char *) malloc(count);

    memcpy(a_data, x_data, count);
    memcpy(b_data, y_data, count);
    memset(c_data, 0x00, count);

    cl_platform_id platforms[32];
    cl_uint num_platforms;
    char vendor[1024];
    cl_device_id devices[32];
    cl_uint num_devices;
    char deviceName[1024];
    cl_int err;

    err = clGetPlatformIDs(32, platforms, &num_platforms);

    cl_assert(err, "There was a problem getting the platforms");
    for (size_t p = 0; p < num_platforms; ++p) {
        cl_platform_id platform = platforms[p];
        clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
        printf("Platform Vendor:\t%s\n", vendor);

        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 32, devices, &num_devices);
        cl_assert(err, "There was a problem getting the device list");

        cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &err);
        cl_assert(err, "There was a problem creating the context.");

        cl_program program = clCreateProgramWithSource(context, 1, &src_addKernel, NULL, &err);
        cl_assert(err, "There was a problem creating the program.");

        err = clBuildProgram(program, num_devices, devices, NULL, NULL, NULL);
        for (size_t d = 0; d < num_devices; ++d) {
            cl_device_id device = devices[d];
            char buffer[2048];
            size_t length = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 2048, buffer, &length);
            if (length > 1)
                printf("%s\n", buffer);
            cl_assert(err, "There was a problem building the program.");
        }


        cl_kernel kernel = clCreateKernel(program, "add", &err);
        cl_assert(err, "There was a problem getting the kernel.");

        cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                         sizeof(char) * count, a_data, &err);
        cl_assert(err, "There was a problem creating the a_buffer.");
        cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                         sizeof(char) * count, b_data, &err);
        cl_assert(err, "There was a problem creating the b_buffer.");
        cl_mem c_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                                         sizeof(char) * count, c_data, &err);
        cl_assert(err, "There was a problem creating the c_buffer");


        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buffer);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buffer);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buffer);
        err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &count);
        cl_assert(err, "There was a problem setting the arguments.");

        for (size_t d = 0; d < num_devices; ++d) {
            cl_device_id device = devices[d];

            clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
            printf("  Device Name:\t%s\n", deviceName);

            cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
            cl_command_queue commands = clCreateCommandQueue(context, device, properties, &err);
            cl_assert(err, "There was a problem creating the command queue");

            size_t local[] = {count};
            size_t global[] = {count};
            cl_event event;
            err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, local, 0, NULL, &event);
            cl_assert(err, "There was a problem queueing the kernel.");

            err = clEnqueueReadBuffer(commands, c_buffer, CL_TRUE, 0, sizeof(char) * count,
                                      c_data, 0, NULL, NULL);
            cl_assert(err, "There was a problem reading the output buffer.");

            clFinish(commands);
            cl_ulong start, stop;
            err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
            err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(stop), &stop, NULL);
            cl_assert(err, "There was a problem getting profiling information.");
            printf("  Time: \t%lu ns.\n", stop - start);
            printf("  Output: \t%s\n", c_data);
            printf("\n");
            clReleaseCommandQueue(commands);
        }

        clReleaseMemObject(a_buffer);
        clReleaseMemObject(b_buffer);
        clReleaseMemObject(c_buffer);

        clReleaseProgram(program);
        clReleaseKernel(kernel);
        clReleaseContext(context);
    }

    free(a_data);
    free(b_data);
    free(c_data);
}

#include <stdlib.h>
#include <stdio.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

int math21_opencl_test4()
{
    const int GPU = 1;

    const char* src_variadic =
            "#define KERNEL(name, ...) kernel void name(__VA_ARGS__) \n"
            "                                                        \n"
            "KERNEL(test, global float* input, global float* output) \n"
            "{                                                       \n"
            "    int i = get_global_id(0);                           \n"
            "    output[i] = input[i];                               \n"
            "}                                                       \n"
            "                                                        \n"
    ;

    const char* src_not_variadic =
            "kernel void test(global float* input, global float* output) \n"
            "{                                                       \n"
            "    int i = get_global_id(0);                           \n"
            "    output[i] = input[i];                               \n"
            "}                                                       \n"
            "                                                        \n"
    ;

//    const char* src = src_variadic;
    const char* src = src_not_variadic;
    int err;

    cl_float input[16];
    cl_float output[16];

    size_t global = 16;
    size_t local = 16;

    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem input_buf;
    cl_mem output_buf;

    err = clGetPlatformIDs(1, &platform_id, NULL);
    if(err != CL_SUCCESS)
    {
        printf("error: clGetPlatformIDs\n");
        return EXIT_FAILURE;
    }

    err = clGetDeviceIDs(platform_id, GPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if(err != CL_SUCCESS)
    {
        printf("error: clGetDeviceIDs\n");
        return EXIT_FAILURE;
    }

    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if(err != CL_SUCCESS)
    {
        printf("error: clCreateContext\n");
        return EXIT_FAILURE;
    }

    command_queue = clCreateCommandQueue(context, device_id, 0, &err);
    if(err != CL_SUCCESS)
    {
        printf("error: clCreateCommandQueue\n");
        return EXIT_FAILURE;
    }

    program = clCreateProgramWithSource(context, 1, &src, NULL, &err);
    if(err != CL_SUCCESS)
    {
        printf("error: clCreateProgramWithSource\n");
        return EXIT_FAILURE;
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("error: clBuildProgram\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);

        return EXIT_FAILURE;
    }

    kernel = clCreateKernel(program, "test", &err);
    if(err != CL_SUCCESS)
    {
        printf("error: clCreateKernel\n");
        return EXIT_FAILURE;
    }

    input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 16*sizeof(cl_float), NULL, NULL);
    output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 16*sizeof(cl_float), NULL, NULL);
    if(!input_buf || !output_buf)
    {
        printf("error: clCreateBuffer\n");
        return EXIT_FAILURE;
    }

    err = clEnqueueWriteBuffer(command_queue, input_buf, CL_TRUE, 0, 16*sizeof(cl_float), input, 0, NULL, NULL);
    if(err != CL_SUCCESS)
    {
        printf("error: clEnqueueWriteBuffer\n");
        return EXIT_FAILURE;
    }

    err = 0;
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buf);
    if(err != CL_SUCCESS)
    {
        printf("error: clSetKernelArg\n");
        return EXIT_FAILURE;
    }

    err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if(err != CL_SUCCESS)
    {
        printf("error: clEnqueueNDRangeKernel\n");
        return EXIT_FAILURE;
    }

    clFinish(command_queue);

    err = clEnqueueReadBuffer(command_queue, output_buf, CL_TRUE, 0, 16*sizeof(cl_float), output, 0, NULL, NULL );
    if(err != CL_SUCCESS)
    {
        printf("error: clEnqueueReadBuffer\n");
        return EXIT_FAILURE;
    }

    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    printf("success\n");

    return EXIT_SUCCESS;
}

#endif