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

#include "inner.h"

#ifdef MATH21_FLAG_USE_OPENCL

#include <math21_opencl_kernel_source.h>
#include "opencl_c.h"
#include "opencl.h"
#include "clvector.h"
#include "cltool.h"

namespace math21 {
    int m21cltool::instance_count = 0;

    m21cltool::m21cltool(int gpu) {
        MATH21_ASSERT(0, "JUST CHECK");
        init(gpu);
        if (instance_count == 0) {
            instance_count++;
#ifdef MATH21_FLAG_USE_OPENCL_BLAS
            clblasSetup();
#endif
        }
    }

    m21cltool::m21cltool(cl_platform_id platform_id, cl_device_id device) {
        m21log(__FUNCTION__);
        commonConstructor(platform_id, device);
        if (instance_count == 0) {
            instance_count++;
#ifdef MATH21_FLAG_USE_OPENCL_BLAS
            clblasSetup();
#endif
        }
    }

    m21cltool::~m21cltool() {
        --instance_count;
        if (instance_count == 0) {
            // by ye
#ifdef MATH21_FLAG_USE_OPENCL_BLAS
            clblasTeardown();
#endif
        }
        clear();
    }

    void m21cltool::clear() {
        m21log("cltool clear");
        if (queue != 0) {
//            clFlush(*queue); // added by ye
//            clFinish(*queue);// added by ye
            error = clReleaseCommandQueue(*queue); // seems that it can't be called after main exits. so call math21_opencl_destroy() before main exits.
            if (error != CL_SUCCESS) {
                MATH21_ASSERT(0, "Error ReleaseCommandQueue: " + errorMessage(error));
            }
            delete queue;
            queue = 0;
        }
        if (context != 0) {
            clReleaseContext(*context);
            delete context;
            context = 0;
        }
    }

    void m21cltool::init_device(int gpuIndex, cl_platform_id &platform_id, cl_device_id &device) {
        cl_int error;
        cl_uint num_platforms;
        error = clGetPlatformIDs(1, &platform_id, &num_platforms);

        if (error != CL_SUCCESS) {
            MATH21_ASSERT(0, "Error getting OpenCL platforms ids: " + errorMessage(error));
        }
        if (num_platforms == 0) {
            MATH21_ASSERT(0, "Error: no OpenCL platforms available");
        }

        cl_uint num_devices;
        error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, 0, 0, &num_devices);
        if (error != CL_SUCCESS) {
            MATH21_ASSERT(0, "Error getting OpenCL device ids: " + errorMessage(error));
        }

        cl_device_id *device_ids = new cl_device_id[num_devices];
        error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, num_devices, device_ids,
                               &num_devices);
        if (error != CL_SUCCESS) {
            MATH21_ASSERT(0, "Error getting OpenCL device ids: " + errorMessage(error));
        }

        if (gpuIndex >= static_cast<int>(num_devices)) {
            MATH21_ASSERT(0,
                          "requested gpuindex " + toString(gpuIndex) + " goes beyond number of available device " +
                          toString(num_devices));
        }
        device = device_ids[gpuIndex];
        delete[] device_ids;
    }

    void m21cltool::init(int gpuIndex) {
        MATH21_ASSERT(0, "JUST CHECK");
        cl_platform_id platform_id;
        cl_device_id device;
        init_device(gpuIndex, platform_id, device);
        commonConstructor(platform_id, device);
    }

    void m21cltool::commonConstructor(cl_platform_id platform_id, cl_device_id device) {
        queue = 0;
        context = 0;
        error = 0;

        this->platform_id = platform_id;
        this->device = device;

        // Context
        context = new cl_context();
        *context = clCreateContext(0, 1, &device, NULL, NULL, &error);
        if (error != CL_SUCCESS) {
            MATH21_ASSERT(0, "Error creating OpenCL context, OpenCL errocode: " + errorMessage(error));
        }
        // Command-queue
        queue = new cl_command_queue;
        *queue = clCreateCommandQueue(*context, device, 0, &error);
//        *queue = clCreateCommandQueueWithProperties(*context, device, 0, &error);
        if (error != CL_SUCCESS) {
            MATH21_ASSERT(0, "Error creating OpenCL command queue, OpenCL errorcode: " + errorMessage(error));
        }
    }

    int m21cltool::roundUp(int quantization, int minimum) {
        MATH21_ASSERT(0, "JUST CHECK");
        return ((minimum + quantization - 1) / quantization * quantization);
    }

    int m21cltool::getPower2Upperbound(int value) {
        MATH21_ASSERT(0, "JUST CHECK");
        int upperbound = 1;
        while (upperbound < value) {
            upperbound <<= 1;
        }
        return upperbound;
    }

    std::shared_ptr<m21cltool> m21cltool::createForIndexedGpu(int gpu) {
        auto p = createForIndexedGpu_raw(gpu);
        auto sp =  std::shared_ptr<m21cltool>(p);
        return sp;
    }

    m21cltool *m21cltool::createForIndexedGpu_raw(int gpu) {
        cl_int error;
        int currentGpuIndex = 0;
        cl_platform_id platform_ids[10];
        cl_uint num_platforms;
        error = clGetPlatformIDs(10, platform_ids, &num_platforms);
        if (error != CL_SUCCESS) {
            MATH21_ASSERT(0, "Error getting OpenCL platforms ids, OpenCL errorcode: " + errorMessage(error));
        }
        if (num_platforms == 0) {
            MATH21_ASSERT(0, "Error: no OpenCL platforms available");
        }
        for (int platform = 0; platform < (int) num_platforms; platform++) {
            cl_platform_id platform_id = platform_ids[platform];

            cl_device_id device_ids[100];
            cl_uint num_devices;
            error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, 100, device_ids,
                                   &num_devices);
            if (error != CL_SUCCESS) {
                continue;
            }

            if ((gpu - currentGpuIndex) < (int) num_devices) {
                return new m21cltool(platform_id, device_ids[(gpu - currentGpuIndex)]);
            } else {
                currentGpuIndex += num_devices;
            }
        }
        if (gpu == 0) {
            MATH21_ASSERT(0, "No OpenCL-enabled GPUs found");
        } else {
            MATH21_ASSERT(0, "Not enough OpenCL-enabled GPUs found to satisfy gpu index: " + toString(gpu));
        }
        return 0;
    }

    std::shared_ptr<m21cltool> m21cltool::createForFirstGpu() {
        MATH21_ASSERT(0, "JUST CHECK");
        return createForIndexedGpu(0);
    }

    std::shared_ptr<m21cltool> m21cltool::createForPlatformDeviceIds(cl_platform_id platformId, cl_device_id deviceId) {
        MATH21_ASSERT(0, "JUST CHECK");
        return std::shared_ptr<m21cltool>(new m21cltool(platformId, deviceId));
    }

    std::shared_ptr<m21cltool> m21cltool::createForPlatformDeviceIndexes(int platformIndex, int deviceIndex) {
        MATH21_ASSERT(0, "JUST CHECK");
        cl_int error;
        cl_platform_id platform_ids[10];
        cl_uint num_platforms;
        error = clGetPlatformIDs(10, platform_ids, &num_platforms);
        if (error != CL_SUCCESS) {
            MATH21_ASSERT(0, "Error getting OpenCL platforms ids, OpenCL errorcode: " + errorMessage(error));
        }
        if (num_platforms == 0) {
            MATH21_ASSERT(0, "Error: no OpenCL platforms available");
        }
        if (platformIndex >= (int) num_platforms) {
            MATH21_ASSERT(0,
                          "Error: OpenCL platform index " + toString(platformIndex) +
                          " not available. There are only: " +
                          toString(num_platforms) + " platforms available");
        }
        cl_platform_id platform_id = platform_ids[platformIndex];
        cl_device_id device_ids[100];
        cl_uint num_devices;
        error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 100, device_ids, &num_devices);
        if (error != CL_SUCCESS) {
            MATH21_ASSERT(0, "Error getting OpenCL device ids for platform index " + toString(platformIndex) +
                             ": OpenCL errorcode: " + errorMessage(error));
        }
        if (num_devices == 0) {
            MATH21_ASSERT(0,
                          "Error: no OpenCL devices available for platform index " + toString(platformIndex));
        }
        if (deviceIndex >= (int) num_devices) {
            MATH21_ASSERT(0, "Error: OpenCL device index " + toString(deviceIndex) +
                             " goes beyond the available devices on platform index " + toString(platformIndex) +
                             ", which has " + toString(num_devices) + " devices");
        }
        return std::shared_ptr<m21cltool>(new m21cltool(platform_id, device_ids[deviceIndex]));
    }

    std::string m21cltool::errorMessage(cl_int error) {
        return math21::math21_string_to_string(error);
    }

    void m21cltool::gpu(int gpuIndex) {
        MATH21_ASSERT(0, "JUST CHECK");
        finish();
        clear();
        init(gpuIndex);
    }

    void m21cltool::finish() {
        error = clFinish(*queue);
        switch (error) {
            case CL_SUCCESS:
                break;
            case -36: MATH21_ASSERT(0,
                                    "Invalid command queue: often indicates out of bounds memory access within kernel");
            default:
                math21_opencl_checkError(error);
        }
    }

    // deprecated, use math21_opencl_build_program_from_file instead.
    std::shared_ptr<m21clprogram>
    m21cltool::buildProgramFromFiles(const std::vector<std::string> &sourcefileNames, const std::string &options) {
        return math21_opencl_buildProgramFromFiles_detail(sourcefileNames, options, device, context, source_map);
    }
}

#endif