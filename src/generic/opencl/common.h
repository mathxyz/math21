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

#pragma once

#include "inner_cc.h"
#include "../../algebra/files.h"

#ifdef MATH21_FLAG_USE_OPENCL

using namespace math21;

template<typename T>
std::string math21_opencl_template_kernelName_1(const std::string &functionName) {
    std::string typeName = math21_type_name<T>();
    std::string kernelName = functionName + "_" + typeName;
    return kernelName;
}

template<typename T1, typename T2>
std::string math21_opencl_template_kernelName_3(const std::string &functionName) {
    std::string typeName1 = math21_type_name<T1>();
    std::string typeName2 = math21_type_name<T2>();
    std::string kernelName = functionName + "_" + typeName1 + "_" + typeName2;
    return kernelName;
}

template<typename T1, typename T2>
std::string math21_opencl_template_kernelNameSuffix(const std::string &d_functionName) {
    std::string typeName1 = math21_type_name<T1>();
    std::string typeName2 = math21_type_name<T2>();
    std::string kernelName_suf = typeName1 + "_" + typeName2;
    if (!d_functionName.empty()) {
        kernelName_suf += "_" + d_functionName;
    }
    return kernelName_suf;
}

std::string math21_opencl_template_kernelName_using_suffix(const std::string &kernelName, const std::string &suffix);

template<typename T>
std::string math21_opencl_template_kernelName_2(const std::string &functionName, const std::string &d_functionName) {
    std::string typeName = math21_type_name<T>();
    std::string kernelName = functionName + "_" + typeName + "_" + d_functionName;
    return kernelName;
}

std::string math21_opencl_options_f_like(std::string d_function_ptr, std::string d_function_name);

template<typename T>
std::string
math21_opencl_options_for_program(std::string d_function_ptr = "", std::string d_function_name = "") {
    MATH21_ASSERT(sizeof(T) == 4 || sizeof(T) == 8, "Only NumR32 and NumR64 supported currently!");
    std::string options = "";
    options += "-D NumReal=" + math21_type_name<T>() + " ";
    options += math21_opencl_options_f_like(d_function_ptr, d_function_name);
    options += "-I ";
    options += MATH21_INCLUDE_PATH;
    return options;
}

template<typename T1, typename T2>
std::string
math21_opencl_options_for_the_program_2(std::string d_function_ptr = "", std::string d_function_name = "") {
    MATH21_ASSERT(sizeof(T1) == 1 || sizeof(T1) == 4 || sizeof(T1) == 8,
                  "Only NumN8, NumR32 and NumR64 supported currently!");
    MATH21_ASSERT(sizeof(T2) == 1 || sizeof(T2) == 4 || sizeof(T2) == 8,
                  "Only NumN8, NumR32 and NumR64 supported currently!");
    std::string options = "";
    options += "-D NumType1=" + math21_type_name<T1>() + " ";
    options += "-D NumType2=" + math21_type_name<T2>() + " ";
    if (!d_function_ptr.empty()) {
        options += "-D " + d_function_ptr + "=" + d_function_name + " ";
    }
    options += "-I ";
    options += MATH21_INCLUDE_PATH;
    return options;
}

#define MATH21_TEMPLATE_BEFORE_FOR_TWO_TYPES(kernel_file_name, map, kernelNameSuffix, d_function_ptr, d_function_name) \
if (!map.has(kernelNameSuffix)) { \
auto p = math21_opencl_build_program_from_file(kernel_file_name, math21_opencl_options_for_the_program_2<T1, T2>(d_function_ptr, d_function_name)); \
map.add(kernelNameSuffix, p); \
}

#define MATH21_TEMPLATE_AFTER() \
m21dim2 dim = math21_opencl_gridsize(n); \
size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE}; \
 \
cl_event e; \
cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0, \
                                      NULL, &e); \
math21_opencl_checkError(error); \
math21_opencl_checkError(clWaitForEvents(1, &e)); \
clReleaseEvent(e);


#endif