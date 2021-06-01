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

const static std::string kernel_file = "generic_01.kl";
const static std::string kernel_file_vector_set = "generic_01_vector_set.kl";
static std::shared_ptr<m21clprogram> program = NULL;
_Map<std::string, std::shared_ptr<m21clprogram>> programs_vector_set;
_Map<std::string, std::shared_ptr<m21clprogram>> programs_tensor_3d_f_set;

#define MATH21_TEMPLATE_BEFORE() \
if (program == NULL) { \
program = math21_opencl_build_program_from_file(kernel_file, math21_opencl_options_for_program<T>()); \
}

#define MATH21_TEMPLATE_BEFORE2(map, d_function_ptr, d_function_name) \
if (!map.has(d_function_name)) { \
auto program = math21_opencl_build_program_from_file(kernel_file, math21_opencl_options_for_program<T>(d_function_ptr, d_function_name)); \
map.add(d_function_name, program); \
}

// a special kind of sub, region sub.
// x is sub-tensor of y
template<typename T>
void math21_template_tensor_sub_set_or_get_opencl(NumN n, PointerVoidWrapper x, PointerVoidWrapper y, NumN dims,
                                                  PointerNumNInputWrapper dx, PointerNumNInputWrapper dy,
                                                  PointerNumNInputWrapper offset, NumB isGet) {
    math21_tool_assert(dims <= MATH21_KERNEL_ARRAY_MAX_LENGTH);
//    x -= 1;
//    y -= 1;
//    dx -= 1;
//    dy -= 1;
//    offset -= 1;
    MATH21_TEMPLATE_BEFORE();

    std::string kernelName = math21_opencl_template_kernelName_1<T>(
            "math21_template_tensor_sub_set_or_get_opencl_kernel");
    cl_kernel kernel = math21_opencl_getKernel(program, kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(NumN), (const void *) &dims));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (const void *) &dx.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (const void *) &dy.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(cl_mem), (const void *) &offset.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(NumN), (const void *) &isGet));

    MATH21_TEMPLATE_AFTER();
}

template<typename T>
void math21_template_vector_kx_opencl(NumN n, T k, PointerVoidWrapper x, NumN stride_x) {
    MATH21_TEMPLATE_BEFORE();

    std::string kernelName = math21_opencl_template_kernelName_1<T>("math21_template_vector_kx_opencl_kernel");
    cl_kernel kernel = math21_opencl_getKernel(program, kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(T), (const void *) &k));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(NumN), (const void *) &stride_x));

    MATH21_TEMPLATE_AFTER();
}

template<typename T>
void math21_template_vector_kx_add_y_opencl(
        NumN n, T k, PointerVoidInputWrapper x, NumN stride_x, PointerVoidWrapper y, NumN stride_y) {
//    x -= 1;
//    y -= 1;
    MATH21_TEMPLATE_BEFORE();
    std::string kernelName = math21_opencl_template_kernelName_1<T>("math21_template_vector_kx_add_y_opencl_kernel");
    cl_kernel kernel = math21_opencl_getKernel(program, kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(T), (const void *) &k));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(NumN), (const void *) &stride_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (const void *) &y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(NumN), (const void *) &stride_y));

    MATH21_TEMPLATE_AFTER();
}

// see math21_vector_assign_from_vector_byte_opencl
template<typename T1, typename T2>
void math21_template_vector_set_by_vector_opencl(NumN n, PointerVoidInputWrapper x, NumN stride_x, PointerVoidWrapper y,
                                                 NumN stride_y,
                                                 NumN offset_x, NumN offset_y) {
//    x += offset_x;
//    y += offset_y;
//    x -= 1;
//    y -= 1;


//    MATH21_TEMPLATE_BEFORE();
//    std::string kernelName = math21_opencl_template_kernelName_1<T>(
//            "math21_template_vector_set_by_vector_opencl_kernel");
//    cl_kernel kernel = math21_opencl_getKernel(program, kernelName);


    std::string d_function_ptr = "";
    std::string d_function_name = "";
    std::string kernelNameSuffix = math21_opencl_template_kernelNameSuffix<T1, T2>(d_function_name);
    MATH21_TEMPLATE_BEFORE_FOR_TWO_TYPES(kernel_file_vector_set, programs_vector_set, kernelNameSuffix, d_function_ptr,
                                         d_function_name);

    std::string kernelName = math21_opencl_template_kernelName_using_suffix(
            "math21_template_vector_set_by_vector_opencl_kernel", kernelNameSuffix);
    cl_kernel kernel = math21_opencl_getKernel(programs_vector_set.valueAt(kernelNameSuffix), kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(NumN), (const void *) &stride_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (const void *) &y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(NumN), (const void *) &stride_y));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(NumN), (const void *) &offset_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(NumN), (const void *) &offset_y));

    MATH21_TEMPLATE_AFTER();
}

template<typename T>
void math21_template_matrix_set_by_matrix_opencl(NumN d1, NumN d2,
                                                 PointerVoidInputWrapper x, NumN d1_x, NumN d2_x, NumN stride1_x,
                                                 NumN stride2_x,
                                                 PointerVoidWrapper y, NumN d1_y, NumN d2_y, NumN stride1_y,
                                                 NumN stride2_y,
                                                 NumN offset_x, NumN offset_y) {
    d2_x = stride1_x * d2_x; // stride absorbed into next dim, so stride will become 1.
    d2_y = stride1_y * d2_y;
//    x += offset_x;
//    y += offset_y;
//    x -= 1;
//    y -= 1;
    NumN n = d1 * d2;
    MATH21_TEMPLATE_BEFORE();
    std::string kernelName = math21_opencl_template_kernelName_1<T>(
            "math21_template_matrix_set_by_matrix_opencl_kernel");
    cl_kernel kernel = math21_opencl_getKernel(program, kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(NumN), (const void *) &d2));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(NumN), (const void *) &d2_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(NumN), (const void *) &stride2_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (const void *) &y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(NumN), (const void *) &d2_y));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(NumN), (const void *) &stride2_y));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(NumN), (const void *) &offset_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 9, sizeof(NumN), (const void *) &offset_y));

    MATH21_TEMPLATE_AFTER();
}

template<typename T>
void math21_template_tensor_3d_set_by_tensor_3d_opencl(NumN d1, NumN d2, NumN d3,
                                                       PointerVoidInputWrapper x, NumN d1_x, NumN d2_x, NumN d3_x,
                                                       NumN stride1_x, NumN stride2_x, NumN stride3_x,
                                                       PointerVoidWrapper y, NumN d1_y, NumN d2_y, NumN d3_y,
                                                       NumN stride1_y, NumN stride2_y, NumN stride3_y,
                                                       NumN offset_x, NumN offset_y) {
    d2_x = stride1_x * d2_x;
    d2_y = stride1_y * d2_y;
    d3_x = stride2_x * d3_x;
    d3_y = stride2_y * d3_y;
//    x += offset_x;
//    y += offset_y;
//    x -= 1;
//    y -= 1;
    NumN n = d1 * d2 * d3;
    MATH21_TEMPLATE_BEFORE();
    std::string kernelName = math21_opencl_template_kernelName_1<T>(
            "math21_template_tensor_3d_set_by_tensor_3d_opencl_kernel");
    cl_kernel kernel = math21_opencl_getKernel(program, kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(NumN), (const void *) &d2));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(NumN), (const void *) &d3));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(NumN), (const void *) &d2_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(NumN), (const void *) &d3_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(NumN), (const void *) &stride3_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(cl_mem), (const void *) &y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(NumN), (const void *) &d2_y));
    math21_opencl_checkError(clSetKernelArg(kernel, 9, sizeof(NumN), (const void *) &d3_y));
    math21_opencl_checkError(clSetKernelArg(kernel, 10, sizeof(NumN), (const void *) &stride3_y));
    math21_opencl_checkError(clSetKernelArg(kernel, 11, sizeof(NumN), (const void *) &offset_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 12, sizeof(NumN), (const void *) &offset_y));

    MATH21_TEMPLATE_AFTER();
}

template<typename T>
void math21_template_tensor_3d_f_set_by_tensor_3d_opencl(NumN fname, NumN d1, NumN d2, NumN d3,
                                                         PointerVoidInputWrapper x, NumN d1_x, NumN d2_x, NumN d3_x,
                                                         NumN stride1_x, NumN stride2_x, NumN stride3_x,
                                                         PointerVoidWrapper y, NumN d1_y, NumN d2_y, NumN d3_y,
                                                         NumN stride1_y, NumN stride2_y, NumN stride3_y,
                                                         NumN offset_x, NumN offset_y) {
    d2_x = stride1_x * d2_x;
    d2_y = stride1_y * d2_y;
    d3_x = stride2_x * d3_x;
    d3_y = stride2_y * d3_y;
//    x += offset_x;
//    y += offset_y;
//    x -= 1;
//    y -= 1;
    NumN n = d1 * d2 * d3;
    std::string d_function_ptr = "f_addto_like_ptr";
    std::string d_function_name;
    if (fname == m21_fname_addto) {
        d_function_name = "math21_device_f_addto";
    } else if (fname == m21_fname_multo) {
        d_function_name = "math21_device_f_multo";
    } else {
        MATH21_ASSERT(0, "not support calling function with name " << fname);
    }
    MATH21_TEMPLATE_BEFORE2(programs_tensor_3d_f_set, d_function_ptr, d_function_name);
    std::string kernelName = math21_opencl_template_kernelName_2<T>(
            "math21_template_tensor_3d_f_set_by_tensor_3d_opencl_kernel", d_function_name);

    cl_kernel kernel = math21_opencl_getKernel(programs_tensor_3d_f_set.valueAt(d_function_name), kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(NumN), (const void *) &d2));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(NumN), (const void *) &d3));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(NumN), (const void *) &d2_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(NumN), (const void *) &d3_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(NumN), (const void *) &stride3_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(cl_mem), (const void *) &y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(NumN), (const void *) &d2_y));
    math21_opencl_checkError(clSetKernelArg(kernel, 9, sizeof(NumN), (const void *) &d3_y));
    math21_opencl_checkError(clSetKernelArg(kernel, 10, sizeof(NumN), (const void *) &stride3_y));
    math21_opencl_checkError(clSetKernelArg(kernel, 11, sizeof(NumN), (const void *) &offset_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 12, sizeof(NumN), (const void *) &offset_y));
    MATH21_TEMPLATE_AFTER();
}

template<typename T>
void math21_template_vector_set_by_value_opencl(NumN n, T value, PointerVoidWrapper x, NumN stride_x) {
//    x -= 1;
    MATH21_TEMPLATE_BEFORE();
    std::string kernelName = math21_opencl_template_kernelName_1<T>(
            "math21_template_vector_set_by_value_opencl_kernel");
    cl_kernel kernel = math21_opencl_getKernel(program, kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(T), (const void *) &value));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(NumN), (const void *) &stride_x));

    MATH21_TEMPLATE_AFTER();
}

template<typename T>
void math21_template_vector_xy_opencl(NumN n, PointerVoidInputWrapper x, NumN stride_x, PointerVoidWrapper y,
                                      NumN stride_y) {
//    x -= 1;
//    y -= 1;
    MATH21_TEMPLATE_BEFORE();
    std::string kernelName = math21_opencl_template_kernelName_1<T>("math21_template_vector_xy_opencl_kernel");
    cl_kernel kernel = math21_opencl_getKernel(program, kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(NumN), (const void *) &stride_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (const void *) &y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(NumN), (const void *) &stride_y));

    MATH21_TEMPLATE_AFTER();
}

template<typename T>
void math21_template_vector_sin_opencl(NumN n, PointerVoidInputWrapper x, PointerVoidWrapper y) {
    MATH21_TEMPLATE_BEFORE();
    std::string kernelName = math21_opencl_template_kernelName_1<T>("math21_template_vector_sin_opencl_kernel");
    cl_kernel kernel = math21_opencl_getKernel(program, kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &y.buffer));

    MATH21_TEMPLATE_AFTER();
}

template<typename T>
void math21_template_vector_cos_opencl(NumN n, PointerVoidInputWrapper x, PointerVoidWrapper y) {
    MATH21_TEMPLATE_BEFORE();
    std::string kernelName = math21_opencl_template_kernelName_1<T>("math21_template_vector_cos_opencl_kernel");
    cl_kernel kernel = math21_opencl_getKernel(program, kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &y.buffer));

    MATH21_TEMPLATE_AFTER();
}

template<typename T>
void math21_template_vector_addToC_opencl(NumN n, PointerVoidInputWrapper A,
                                          PointerVoidInputWrapper B, PointerVoidWrapper C) {
    MATH21_TEMPLATE_BEFORE();
    std::string kernelName = math21_opencl_template_kernelName_1<T>("math21_template_vector_addToC_opencl_kernel");
    cl_kernel kernel = math21_opencl_getKernel(program, kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *) &A.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &B.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (const void *) &C.buffer));

    MATH21_TEMPLATE_AFTER();
}

template<typename T>
void math21_template_vector_mulToC_opencl(NumN n, PointerVoidInputWrapper A,
                                          PointerVoidInputWrapper B, PointerVoidWrapper C) {
    MATH21_TEMPLATE_BEFORE();
    std::string kernelName = math21_opencl_template_kernelName_1<T>("math21_template_vector_mulToC_opencl_kernel");
    cl_kernel kernel = math21_opencl_getKernel(program, kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *) &A.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &B.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (const void *) &C.buffer));

    MATH21_TEMPLATE_AFTER();
}

template<typename T>
void math21_template_vector_broadcast_in_dn_opencl(NumN n, PointerVoidInputWrapper x, PointerVoidWrapper y,
                                                   NumN dims_x, PointerNumNInputWrapper dx,
                                                   NumN dims_y, PointerNumNInputWrapper dy) {
    // todo: uncomment out using opencl svm when opencl>=2
//    x -= 1;
//    y -= 1;
//    dx -= 1;
//    dy -= 1;

    MATH21_TEMPLATE_BEFORE();
    std::string kernelName = math21_opencl_template_kernelName_1<T>(
            "math21_template_vector_broadcast_in_dn_opencl_kernel");
    cl_kernel kernel = math21_opencl_getKernel(program, kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(NumN), (const void *) &dims_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (const void *) &dx.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(NumN), (const void *) &dims_y));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(cl_mem), (const void *) &dy.buffer));

    MATH21_TEMPLATE_AFTER();
}

template<typename T>
void math21_template_optimization_adam_update_part_2_opencl(
        NumN n, PointerVoidWrapper x, PointerVoidInputWrapper m,
        PointerVoidInputWrapper v,
        T beta1, T beta2, T alpha, T eps, NumN t) {
//    x -= 1;
//    m -= 1;
//    v -= 1;

    MATH21_TEMPLATE_BEFORE();
    std::string kernelName = math21_opencl_template_kernelName_1<T>(
            "math21_template_optimization_adam_update_part_2_opencl_kernel");
    cl_kernel kernel = math21_opencl_getKernel(program, kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &m.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (const void *) &v.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(T), (const void *) &beta1));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(T), (const void *) &beta2));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(T), (const void *) &alpha));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(T), (const void *) &eps));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(NumN), (const void *) &t));

    MATH21_TEMPLATE_AFTER();
}

#endif