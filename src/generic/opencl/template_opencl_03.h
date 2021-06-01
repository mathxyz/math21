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
#include "common.h"

#ifdef MATH21_FLAG_USE_OPENCL

const static std::string kernel_file = "generic_03.kl";
const static std::string kernel_file_transpose = "generic_03_transpose.kl";
static std::shared_ptr<m21clprogram> program = NULL;
_Map<std::string, std::shared_ptr<m21clprogram>> programs_trans;
// todo: reduce the number of programs.

#define MATH21_TEMPLATE_BEFORE() \
if (program == NULL) { \
program = math21_opencl_build_program_from_file(kernel_file, math21_opencl_options_for_program<T>()); \
}

// see math21_matrix_multiply_k1AB_add_k2C_similar
// C = k1*(A*B) + k2*C or similar
template<typename T>
void math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_opencl(
        NumB ta, NumB tb, NumN nr_C, NumN nc_C, NumN n_common, T k1,
        PointerVoidInputWrapper A, NumN stride_a,
        PointerVoidInputWrapper B, NumN stride_b,
        T k2, PointerVoidWrapper C, NumN stride_c) {
//    A -= 1;
//    B -= 1;
//    C -= 1;
    NumN n = nr_C * nc_C;
    MATH21_TEMPLATE_BEFORE();
    std::string kernelName;
    if (!ta && !tb) {
        kernelName = math21_opencl_template_kernelName_1<T>(
                "math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_nn_naive_opencl_kernel");
    } else if (ta && !tb) {
        kernelName = math21_opencl_template_kernelName_1<T>(
                "math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_tn_naive_opencl_kernel");
    } else if (!ta && tb) {
        kernelName = math21_opencl_template_kernelName_1<T>(
                "math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_nt_naive_opencl_kernel");
    } else {
        kernelName = math21_opencl_template_kernelName_1<T>(
                "math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_tt_naive_opencl_kernel");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(NumN), (const void *) &nr_C));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(NumN), (const void *) &nc_C));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(NumN), (const void *) &n_common));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(T), (const void *) &k1));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (const void *) &A.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(NumN), (const void *) &stride_a));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(cl_mem), (const void *) &B.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(NumN), (const void *) &stride_b));
    math21_opencl_checkError(clSetKernelArg(kernel, 9, sizeof(T), (const void *) &k2));
    math21_opencl_checkError(clSetKernelArg(kernel, 10, sizeof(cl_mem), (const void *) &C.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 11, sizeof(NumN), (const void *) &stride_c));

    MATH21_TEMPLATE_AFTER();
}

template<typename T1, typename T2>
void math21_template_matrix_transpose_opencl(
        NumN n, PointerVoidInputWrapper x, PointerVoidWrapper y, NumN nr_x, NumN nc_x) {
//    x -= 1;
//    y -= 1;
    std::string d_function_ptr = "";
    std::string d_function_name = "";
    std::string kernelNameSuffix = math21_opencl_template_kernelNameSuffix<T1, T2>(d_function_name);
    MATH21_TEMPLATE_BEFORE_FOR_TWO_TYPES(kernel_file_transpose, programs_trans, kernelNameSuffix, d_function_ptr, d_function_name);

    std::string kernelName = math21_opencl_template_kernelName_using_suffix(
            "math21_template_matrix_transpose_opencl_kernel", kernelNameSuffix);
    cl_kernel kernel = math21_opencl_getKernel(programs_trans.valueAt(kernelNameSuffix), kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(NumN), (const void *) &nr_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(NumN), (const void *) &nc_x));

    MATH21_TEMPLATE_AFTER();
}

template<typename T1, typename T2>
void math21_template_tensor_swap_axes_24_in_d5_opencl(
        PointerVoidInputWrapper x, PointerVoidWrapper y, NumN d1, NumN d2, NumN d3, NumN d4, NumN d5) {
//    x -= 1;
//    y -= 1;
    NumN n = d1 * d2 * d3 * d4 * d5;

    std::string d_function_ptr = "";
    std::string d_function_name = "";
    std::string kernelNameSuffix = math21_opencl_template_kernelNameSuffix<T1, T2>(d_function_name);
    MATH21_TEMPLATE_BEFORE_FOR_TWO_TYPES(kernel_file_transpose, programs_trans, kernelNameSuffix, d_function_ptr, d_function_name);

    std::string kernelName = math21_opencl_template_kernelName_using_suffix(
            "math21_template_tensor_swap_axes_24_in_d5_opencl_kernel", kernelNameSuffix);
    cl_kernel kernel = math21_opencl_getKernel(programs_trans.valueAt(kernelNameSuffix), kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(NumN), (const void *) &d1));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(NumN), (const void *) &d2));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(NumN), (const void *) &d3));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(NumN), (const void *) &d4));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(NumN), (const void *) &d5));

    MATH21_TEMPLATE_AFTER();
}

#endif