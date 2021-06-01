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

#include "../../tool/files.h"
#include "inner.h"
#include "cltool.h"
#include "opencl_c.h"
#include "opencl.h"

#ifdef MATH21_FLAG_USE_OPENCL

using namespace math21;

int gpu_index = 0;

//std::shared_ptr<m21cltool> _cltool = m21cltool::createForIndexedGpu(0);
std::shared_ptr<m21cltool> _cltool = std::shared_ptr<m21cltool>(0); // by ye
Seqce<std::shared_ptr<m21clprogram>> _clprograms;

std::shared_ptr<m21cltool> math21_opencl_get_cltool() {
    if (_cltool.get() == 0) { // added by ye
        _cltool = m21cltool::createForIndexedGpu(0);
    }
    return _cltool;
}

void math21_opencl_add_program(const std::shared_ptr<m21clprogram> &program) {
//    m21log(__FUNCTION__);
    _clprograms.push(program);
}

void math21_opencl_destroy() {
#ifdef MATH21_WINDOWS
    // explicitly destroy
    if (_cltool.get() != 0) {
        _cltool.get()->clear();
    }
    if (!_clprograms.isEmpty()) {
        for (NumN i = 1; i <= _clprograms.size(); ++i) {
            _clprograms.at(i).get()->clear();
        }
        _clprograms.clear();
    }
#else
    // no need to destroy explicitly
#endif
}

void math21_opencl_set_device(int n) {
    if (gpu_index == -1) {
        gpu_index = n;
        _cltool = m21cltool::createForIndexedGpu(gpu_index);
    } else if (n != gpu_index) {
        MATH21_ASSERT(0, "don't support temporary\n");
    }
}

void math21_opencl_checkError(cl_int error) {
    if (error != CL_SUCCESS) {
        std::string message = math21_string_to_string(error);
        switch (error) {
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:
                message = "CL_MEM_OBJECT_ALLOCATION_FAILURE";
                break;
            case CL_BUILD_PROGRAM_FAILURE:
                message = "CL_BUILD_PROGRAM_FAILURE";
                break;
            case CL_INVALID_MEM_OBJECT:
                message = "CL_INVALID_MEM_OBJECT";
                break;
            case CL_INVALID_BUILD_OPTIONS:
                message = "CL_INVALID_BUILD_OPTIONS";
                break;
            case CL_INVALID_ARG_SIZE:
                message = "CL_INVALID_ARG_SIZE";
                break;
            case CL_INVALID_BUFFER_SIZE:
                message = "CL_INVALID_BUFFER_SIZE";
                break;
            default:
                break;
        }
        std::cout << "opencl execution error, code " << error << " " << message << std::endl;
        MATH21_ASSERT(0, std::string("OpenCL error, code: ") + message);
    }
}

void math21_opencl_printPlatformInfoString(const char *name, cl_platform_id platformId, cl_platform_info info) {
    char buffer[256];
    buffer[0] = 0;
    clGetPlatformInfo(platformId, info, 256, buffer, 0);
    if (name) {
        std::cout << name << ": ";
    }
    std::cout << buffer << std::endl;
}

void math21_opencl_printPlatformInfo(const char *name, cl_platform_id platformId, cl_platform_info info) {
    cl_ulong somelong = 0;
    clGetPlatformInfo(platformId, info, sizeof(cl_ulong), &somelong, 0);
    if (name) {
        std::cout << name << ": ";
    }
    std::cout << somelong << std::endl;
}

std::string math21_opencl_getPlatformInfoString(cl_platform_id platformId, cl_platform_info name) {
    char buffer[257];
    buffer[0] = 0;
    size_t namesize;
    cl_int error = clGetPlatformInfo(platformId, name, 256, buffer, &namesize);
    if (error != CL_SUCCESS) {
        if (error == CL_INVALID_PLATFORM) {
            MATH21_ASSERT(0,
                          "Failed to obtain platform info for platform id " + math21_string_to_string(platformId) +
                          ": invalid platform");
        } else if (error == CL_INVALID_VALUE) {
            MATH21_ASSERT(0,
                          "Failed to obtain platform info " + math21_string_to_string(name) + " for platform id " +
                          math21_string_to_string(platformId) + ": invalid value");
        } else {
            MATH21_ASSERT(0,
                          "Failed to obtain platform info " + math21_string_to_string(name) + " for platform id " +
                          math21_string_to_string(platformId) + ": unknown error code: " +
                          math21_string_to_string(error));
        }
    }
    return std::string(buffer);
}

std::shared_ptr<m21clprogram>
math21_opencl_build_program_from_file(const std::string &sourcefileName, const std::string &options) {
    auto cl = math21_opencl_get_cltool();
    std::vector<std::string> sourcefileNames(1);
    sourcefileNames[0] = sourcefileName;
    auto program = cl->buildProgramFromFiles(sourcefileNames, options);
    math21_opencl_add_program(program);
    return program;
}

std::shared_ptr<m21clprogram>
math21_opencl_build_program_from_two_files(const std::string &sourcefileName, const std::string &sourcefileName2,
                                           const std::string &options) {
    auto cl = math21_opencl_get_cltool();
    std::vector<std::string> sourcefileNames(2);
    sourcefileNames[0] = sourcefileName;
    sourcefileNames[1] = sourcefileName2;
    auto program = cl->buildProgramFromFiles(sourcefileNames, options);
    math21_opencl_add_program(program);
    return program;
}

std::shared_ptr<m21clprogram> math21_opencl_build_program_from_multiple_files(
        const std::vector<std::string> &sourcefileNames, const std::string &options) {
    auto cl = math21_opencl_get_cltool();
    auto program = cl->buildProgramFromFiles(sourcefileNames, options);
    math21_opencl_add_program(program);
    return program;
}

cl_command_queue math21_opencl_get_command_queue() {
    return math21_opencl_get_cltool()->get_command_queue();
}

cl_command_queue *math21_opencl_get_command_queue_pointer() {
    MATH21_ASSERT(0, "JUST CHECK");
    return math21_opencl_get_cltool()->get_command_queue_pointer();
}

cl_context math21_opencl_get_context() {
    return math21_opencl_get_cltool()->get_context();
}

cl_kernel math21_opencl_getKernel(std::shared_ptr<m21clprogram> &program, const std::string &kernelname) {
    return program->getKernel(kernelname);
}

std::shared_ptr<m21clprogram>
math21_opencl_buildProgramFromFiles_detail(const std::vector<std::string> &fileNames_src, const std::string &options,
                                           cl_device_id device, cl_context *context,
                                           std::map<std::string, std::string> &source_map) {
    cl_int error;
    NumN n = static_cast<NumN>(fileNames_src.size());
    MATH21_ASSERT(n >= 1);
    std::vector<std::string> sources(n);
    std::string whole_source = "";
    auto *src_sizes = new size_t[n];
    const char **source_char = new const char *[n];
    for (NumN i = 0; i < n; ++i) {
        auto &fileName = fileNames_src[i];
        sources[i] = math21_file_get_contents(fileName);
        if (sources[i].empty())//use the default buildin kernel source
        {
            if (source_map[fileName].empty()) {
                MATH21_ASSERT(0, "Failed to find the kernel source from file " << fileName);
            } else {
                sources[i] = source_map[fileName];
            }
        }
        source_char[i] = sources[i].c_str();
        src_sizes[i] = strlen(source_char[i]);

//        whole_source += sources[i];
    }

//    source_char[0] = whole_source.c_str();
//    src_sizes[0] = strlen(source_char[0]);

//    cl_program program = clCreateProgramWithSource(*context, 1, &source_char[0], &src_sizes[0], &error);
    cl_program program = clCreateProgramWithSource(*context, n, &source_char[0], &src_sizes[0], &error);
    delete[] src_sizes;
    delete[] source_char;

    math21_opencl_checkError(error);

    error = clBuildProgram(program, 1, &device, options.c_str(), NULL, NULL);
    if (error != CL_SUCCESS) {
        cl_int error_pre = error;

        char *build_log;
        size_t log_size;
        error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        math21_opencl_checkError(error);
        build_log = new char[log_size + 1];
        error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        math21_opencl_checkError(error);
        build_log[log_size] = '\0';
        std::string buildLogMessage = "";
        if (log_size > 2) {
            m21log(fileNames_src);
            buildLogMessage = math21_string_to_string(" build log: \n") + build_log;
            std::cout << buildLogMessage << std::endl;
        }
        delete[] build_log;
        math21_opencl_checkError(error);

        math21_opencl_checkError(error_pre);
    }
    std::shared_ptr<m21clprogram> res(new m21clprogram(program));
    return res;
}

int64_t math21_opencl_getDeviceInfoInt64(cl_device_id device, cl_device_info name) {
    cl_ulong value = 0;
    clGetDeviceInfo(device, name, sizeof(cl_ulong), &value, 0);
    return static_cast<int64_t>(value);
}

int math21_opencl_getComputeUnits(cl_device_id device) {
    return (int) math21_opencl_getDeviceInfoInt64(device, CL_DEVICE_MAX_COMPUTE_UNITS);
}

int math21_opencl_getLocalMemorySize(cl_device_id device) {
    return (int) math21_opencl_getDeviceInfoInt64(device, CL_DEVICE_LOCAL_MEM_SIZE);
}

int math21_opencl_getLocalMemorySizeKB(cl_device_id device) {
    return (int) (math21_opencl_getDeviceInfoInt64(device, CL_DEVICE_LOCAL_MEM_SIZE) / 1024);
}

int math21_opencl_getMaxWorkgroupSize(cl_device_id device) {
    return (int) math21_opencl_getDeviceInfoInt64(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
}

int math21_opencl_getMaxAllocSizeMB(cl_device_id device) {
    return (int) (math21_opencl_getDeviceInfoInt64(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE) / 1024 / 1024);
}

m21dim2 math21_opencl_gridsize(size_t n) {
    size_t k = (n - 1) / MATH21_OPENCL_BLOCK_SIZE + 1;
    size_t x = k;
    size_t y = 1;
    if (x > 65535) {
        x = ceil(sqrt(k));
        y = (n - 1) / (x * MATH21_OPENCL_BLOCK_SIZE) + 1;
    }
    m21dim2 d = {x, y};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

void math21_opencl_push_array(m21clvector x_gpu, const float *x, size_t n) {
    auto cl = math21_opencl_get_cltool();
    size_t size = sizeof(float) * n;
    cl_int status = clEnqueueWriteBuffer(math21_opencl_get_command_queue(), x_gpu.buffer, CL_TRUE, 0, size, x, 0, NULL,
                                         NULL);
    math21_opencl_checkError(status);
}

void math21_opencl_push_N8_array(m21clvector x_gpu, const NumN8 *x, size_t n) {
    auto cl = math21_opencl_get_cltool();
    size_t size = sizeof(NumN8) * n;
    cl_int status = clEnqueueWriteBuffer(math21_opencl_get_command_queue(), x_gpu.buffer, CL_TRUE, 0, size, x, 0, NULL,
                                         NULL);
    math21_opencl_checkError(status);
}

void math21_opencl_pull_array(m21clvector x_gpu, float *x, size_t n) {
    cl_event event = NULL;
    cl_int status;
    size_t size = sizeof(float) * n;
    status = clEnqueueReadBuffer(math21_opencl_get_command_queue(), x_gpu.buffer, CL_TRUE, 0, size, x, 0, NULL, &event);
    math21_opencl_checkError(status);
    cl_int err = clWaitForEvents(1, &event);
    clReleaseEvent(event);
    if (err != CL_SUCCESS) {
        MATH21_ASSERT(0, "wait for event on copytohost failed with " + math21_string_to_string(err));
    }
}

void math21_opencl_pull_N8_array(m21clvector x_gpu, NumN8 *x, size_t n) {
    auto cl = math21_opencl_get_cltool();
    cl_event event = NULL;
    cl_int status;
    size_t size = sizeof(NumN8) * n;
    status = clEnqueueReadBuffer(math21_opencl_get_command_queue(), x_gpu.buffer, CL_TRUE, 0, size, x, 0, NULL, &event);
    math21_opencl_checkError(status);
    cl_int err = clWaitForEvents(1, &event);
    clReleaseEvent(event);
    if (err != CL_SUCCESS) {
        MATH21_ASSERT(0, "wait for event on copytohost failed with " + math21_string_to_string(err));
    }
}

void math21_opencl_vector_log_pointer(m21clvector v) {
    math21_opencl_vector_log_pointer(std::cout, v);
//    printf("clvector: size = %ld, %p, "
//           "size_address = %ld, address = %p, offset = %d\n",
//           v.size, v.buffer, v.size_address, v.address, v.offset);
}

void math21_opencl_vector_log_pointer(std::ostream &out, m21clvector v) {
    out << "clvector: size = " << v.size << ", " << v.buffer
        << ", size_address = " << v.size_address << ", address = "
        << v.address << ", offset = " << v.offset << "\n";
}

std::ostream &operator<<(std::ostream &out, const m21clvector &m) {
    math21_opencl_vector_log_pointer(out, m);
    return out;
}

#endif