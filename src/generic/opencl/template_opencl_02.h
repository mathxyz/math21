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

const static std::string kernel_file = "generic_02.kl";
static std::shared_ptr<m21clprogram> program = NULL;
// todo: reduce the number of programs.
_Map<std::string, std::shared_ptr<m21clprogram>> programsForfshrink;
_Map<std::string, std::shared_ptr<m21clprogram>> programs_tensor_f_inner_product_like_shrink;
_Map<std::string, std::shared_ptr<m21clprogram>> programs_f_bc_add_like_ptr;
_Map<std::string, std::shared_ptr<m21clprogram>> programs_f_bc_sin_like_ptr;
_Map<std::string, std::shared_ptr<m21clprogram>> programs_f_kx_like_ptr;

#define MATH21_TEMPLATE_BEFORE() \
if (program == NULL) { \
program = math21_opencl_build_program_from_file(kernel_file, math21_opencl_options_for_program<T>()); \
}

#define MATH21_TEMPLATE_BEFORE2(map, d_function_ptr, d_function_name) \
if (!map.has(d_function_name)) { \
auto program = math21_opencl_build_program_from_file(kernel_file, math21_opencl_options_for_program<T>(d_function_ptr, d_function_name)); \
map.add(d_function_name, program); \
}

// todo: make program file small
// one kernel one program
template<typename T>
void math21_template_tensor_f_shrink_opencl(NumN fname, NumN n, PointerVoidInputWrapper x, PointerVoidWrapper y,
                                            NumN dims_x, PointerNumNInputWrapper dx, NumN dims_y,
                                            PointerNumNInputWrapper dy,
                                            NumN nb, PointerNumNInputWrapper b,
                                            NumN nv, NumN dims_v, PointerNumNInputWrapper dv) {
    std::string d_function_ptr = "f_shrink_min_like_ptr";
    std::string d_function_name;
    if (fname == m21_fname_sum) {
        d_function_name = "math21_device_f_sum";
    } else if (fname == m21_fname_mean) {
        d_function_name = "math21_device_f_mean";
    } else if (fname == m21_fname_max) {
        d_function_name = "math21_device_f_max";
    } else if (fname == m21_fname_min) {
        d_function_name = "math21_device_f_min";
    } else if (fname == m21_fname_argmax) {
        d_function_ptr = "f_shrink_argmin_like_ptr";
        d_function_name = "math21_device_f_argmax";
    } else if (fname == m21_fname_argmin) {
        d_function_ptr = "f_shrink_argmin_like_ptr";
        d_function_name = "math21_device_f_argmin";
    } else {
        MATH21_ASSERT(0, "not support calling function with name " << fname);
    }

    MATH21_TEMPLATE_BEFORE2(programsForfshrink, d_function_ptr, d_function_name);
    std::string kernelName = math21_opencl_template_kernelName_2<T>(
            "math21_template_tensor_f_shrink_opencl_kernel", d_function_name);

    cl_kernel kernel = math21_opencl_getKernel(programsForfshrink.valueAt(d_function_name), kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(NumN), (const void *) &dims_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (const void *) &dx.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(NumN), (const void *) &dims_y));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(cl_mem), (const void *) &dy.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(NumN), (const void *) &nb));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(cl_mem), (const void *) &b.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 9, sizeof(NumN), (const void *) &nv));
    math21_opencl_checkError(clSetKernelArg(kernel, 10, sizeof(NumN), (const void *) &dims_v));
    math21_opencl_checkError(clSetKernelArg(kernel, 11, sizeof(cl_mem), (const void *) &dv.buffer));

    MATH21_TEMPLATE_AFTER();

}

template<typename T>
void math21_template_tensor_f_inner_product_like_shrink_opencl(NumN fname, NumN n,
                                                               PointerVoidInputWrapper x1, PointerVoidInputWrapper x2,
                                                               PointerVoidWrapper y,
                                                               NumN dims_x, PointerNumNInputWrapper dx, NumN dims_y,
                                                               PointerNumNInputWrapper dy,
                                                               NumN nb, PointerNumNInputWrapper b,
                                                               NumN nv, NumN dims_v, PointerNumNInputWrapper dv) {
//    x1 -= 1;
//    x2 -= 1;
//    y -= 1;
//    dx -= 1;
//    dy -= 1;
//    b -= 1;
//    dv -= 1;
    std::string d_function_ptr = "f_inner_product_like_ptr";
    std::string f;
    if (fname == m21_fname_inner_product) {
        f = "math21_device_f_inner_product";
    } else if (fname == m21_fname_distance_1) {
        f = "math21_device_f_distance_1";
    } else if (fname == m21_fname_distance_2_square) {
        f = "math21_device_f_distance_2_square";
    } else {
        MATH21_ASSERT(0, "not support calling function with name " << fname);
    }

    MATH21_TEMPLATE_BEFORE2(programs_tensor_f_inner_product_like_shrink, d_function_ptr, f);
    std::string kernelName = math21_opencl_template_kernelName_2<T>(
            "math21_template_tensor_f_inner_product_like_shrink_opencl_kernel", f);

    cl_kernel kernel = math21_opencl_getKernel(programs_tensor_f_inner_product_like_shrink.valueAt(f), kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *) &x1.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &x2.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (const void *) &y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(NumN), (const void *) &dims_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (const void *) &dx.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(NumN), (const void *) &dims_y));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(cl_mem), (const void *) &dy.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(NumN), (const void *) &nb));
    math21_opencl_checkError(clSetKernelArg(kernel, 9, sizeof(cl_mem), (const void *) &b.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 10, sizeof(NumN), (const void *) &nv));
    math21_opencl_checkError(clSetKernelArg(kernel, 11, sizeof(NumN), (const void *) &dims_v));
    math21_opencl_checkError(clSetKernelArg(kernel, 12, sizeof(cl_mem), (const void *) &dv.buffer));

    MATH21_TEMPLATE_AFTER();
}

// todo: use index 1 for x, y
// a special kind of sub
// x is sub-tensor of y
template<typename T>
void math21_template_tensor_f_with_broadcast_in_dn_opencl(NumN fname, NumN n,
                                                          PointerVoidInputWrapper x1,
                                                          PointerVoidInputWrapper x2,
                                                          PointerVoidWrapper y,
                                                          NumN dims_x1, PointerNumNInputWrapper dx1,
                                                          NumN dims_x2, PointerNumNInputWrapper dx2,
                                                          NumN dims_y, PointerNumNInputWrapper dy) {
//    x1 -= 1;
//    x2 -= 1;
//    y -= 1;
//    dx1 -= 1;
//    dx2 -= 1;
//    dy -= 1;
    std::string d_function_ptr = "f_bc_add_like_ptr";
    std::string f;
    if (fname == m21_fname_add) {
        f = "math21_device_f_add";
    } else if (fname == m21_fname_subtract) {
        f = "math21_device_f_subtract";
    } else if (fname == m21_fname_multiply) {
        f = "math21_device_f_multiply";
    } else if (fname == m21_fname_divide) {
        f = "math21_device_f_divide";
    } else if (fname == m21_fname_ele_is_equal) {
        f = "math21_device_f_is_equal";
    } else if (fname == m21_fname_set_using_mask) {
    } else {
        MATH21_ASSERT(0, "not support calling function with name " << fname);
    }
    cl_kernel kernel;
    if (fname == m21_fname_set_using_mask) {
        MATH21_TEMPLATE_BEFORE();
            std::string kernelName = math21_opencl_template_kernelName_1<T>(
            "math21_template_tensor_set_using_mask_in_dn_opencl_kernel");
    kernel = math21_opencl_getKernel(program, kernelName);

    }else{
     MATH21_TEMPLATE_BEFORE2(programs_f_bc_add_like_ptr, d_function_ptr, f);
    std::string kernelName = math21_opencl_template_kernelName_2<T>(
            "math21_template_tensor_f_with_broadcast_in_dn_opencl_kernel", f);

    kernel = math21_opencl_getKernel(programs_f_bc_add_like_ptr.valueAt(f), kernelName);
    }

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *) &x1.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &x2.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (const void *) &y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(NumN), (const void *) &dims_x1));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (const void *) &dx1.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(NumN), (const void *) &dims_x2));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(cl_mem), (const void *) &dx2.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(NumN), (const void *) &dims_y));
    math21_opencl_checkError(clSetKernelArg(kernel, 9, sizeof(cl_mem), (const void *) &dy.buffer));

    MATH21_TEMPLATE_AFTER();
}

// todo: use index 1 for x, y
template<typename T>
void math21_template_vector_f_add_like_opencl(NumN fname, NumN n,
                                              PointerVoidInputWrapper x1,
                                              PointerVoidInputWrapper x2,
                                              PointerVoidWrapper y) {
//    x1 -= 1;
//    x2 -= 1;
//    y -= 1;

    std::string d_function_ptr = "f_bc_add_like_ptr";
    std::string f;
    if (fname == m21_fname_add) {
        f = "math21_device_f_add";
    } else if (fname == m21_fname_subtract) {
        f = "math21_device_f_subtract";
    } else if (fname == m21_fname_multiply) {
        f = "math21_device_f_multiply";
    } else if (fname == m21_fname_divide) {
        f = "math21_device_f_divide";
    } else if (fname == m21_fname_ele_is_equal) {
        f = "math21_device_f_is_equal";
    } else if (fname == m21_fname_set_using_mask) {
    } else {
        MATH21_ASSERT(0, "not support calling function with name " << fname);
    }
    cl_kernel kernel;
    if (fname == m21_fname_set_using_mask) {
        MATH21_TEMPLATE_BEFORE();
            std::string kernelName = math21_opencl_template_kernelName_1<T>(
            "math21_template_vector_set_using_mask_opencl_kernel");
    kernel = math21_opencl_getKernel(program, kernelName);
    }else{
     MATH21_TEMPLATE_BEFORE2(programs_f_bc_add_like_ptr, d_function_ptr, f);
    std::string kernelName = math21_opencl_template_kernelName_2<T>(
            "math21_template_vector_f_add_like_opencl_kernel", f);

    kernel = math21_opencl_getKernel(programs_f_bc_add_like_ptr.valueAt(f), kernelName);
    }

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *) &x1.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &x2.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (const void *) &y.buffer));

    MATH21_TEMPLATE_AFTER();
}

template<typename T>
void math21_template_vector_f_sin_like_opencl(NumN fname, NumN n,
                                              PointerVoidInputWrapper x, PointerVoidWrapper y) {
//    x -= 1;
//    y -= 1;

    std::string d_function_ptr = "f_bc_sin_like_ptr";
    std::string f;
    if (fname == m21_fname_sin) {
        f = "math21_device_f_sin";
    } else if (fname == m21_fname_cos) {
        f = "math21_device_f_cos";
    } else if (fname == m21_fname_tan) {
        f = "math21_device_f_tan";
    } else if (fname == m21_fname_exp) {
        f = "math21_device_f_exp";
    } else if (fname == m21_fname_log) {
        f = "math21_device_f_log";
    } else if (fname == m21_fname_abs) {
        f = "math21_device_f_abs";
    } else {
        MATH21_ASSERT(0, "not support calling function with name " << fname);
    }

    MATH21_TEMPLATE_BEFORE2(programs_f_bc_sin_like_ptr, d_function_ptr, f);
    std::string kernelName = math21_opencl_template_kernelName_2<T>(
            "math21_template_vector_f_sin_like_opencl_kernel", f);

    cl_kernel kernel = math21_opencl_getKernel(programs_f_bc_sin_like_ptr.valueAt(f), kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &y.buffer));

    MATH21_TEMPLATE_AFTER();
}

template<typename T>
void math21_template_vector_f_kx_like_opencl(NumN fname, NumN n, T k,
                                             PointerVoidInputWrapper x, PointerVoidWrapper y) {
//    x -= 1;
//    y -= 1;
    std::string d_function_ptr = "f_kx_like_ptr";
    std::string f;
    if (fname == m21_fname_kx_add) {
        f = MATH21_STRINGIFY(math21_device_f_add);
    } else if (fname == m21_fname_kx_subtract) {
        f = MATH21_STRINGIFY(math21_device_f_subtract);
    } else if (fname == m21_fname_xk_subtract) {
        f = MATH21_STRINGIFY(math21_device_f_xk_subtract);
    } else if (fname == m21_fname_kx_mul) {
        f = MATH21_STRINGIFY(math21_device_f_multiply);
    } else if (fname == m21_fname_kx_divide) {
        f = MATH21_STRINGIFY(math21_device_f_divide);
    } else if (fname == m21_fname_xk_divide) {
        f = MATH21_STRINGIFY(math21_device_f_xk_divide);
    } else if (fname == m21_fname_kx_pow) {
        f = MATH21_STRINGIFY(math21_device_f_kx_pow);
    } else if (fname == m21_fname_xk_pow) {
        f = MATH21_STRINGIFY(math21_device_f_xk_pow);
    } else {
        MATH21_ASSERT(0, "not support calling function with name " << fname);
    }
    MATH21_TEMPLATE_BEFORE2(programs_f_kx_like_ptr, d_function_ptr, f);
    std::string kernelName = math21_opencl_template_kernelName_2<T>(
            "math21_template_vector_f_kx_like_opencl_kernel", f);

    cl_kernel kernel = math21_opencl_getKernel(programs_f_kx_like_ptr.valueAt(f), kernelName);

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(NumN), (const void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(T), (const void *) &k));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (const void *) &y.buffer));

    MATH21_TEMPLATE_AFTER();
}
#endif