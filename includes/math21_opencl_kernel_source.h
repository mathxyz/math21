#pragma once
#include<map>
std::string generic_02 = "\n\
#include <math21_kernels.h>\n\
\n\
//#define MATH21_IS_FROM_CPU\n\
\n\
#if !defined(MATH21_IS_FROM_CPU)\n\
#if defined(MATH21_FLAG_USE_CUDA)\n\
#define MATH21_IS_FROM_CUDA\n\
#elif defined(MATH21_FLAG_USE_OPENCL)\n\
#define MATH21_IS_FROM_OPENCL\n\
#endif\n\
#endif\n\
\n\
#if defined(MATH21_IS_FROM_CPU)\n\
#define MATH21_MAKE_KERNEL_NAME_1(X) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_2(X) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_3(X) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_PAIR(X, Y) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_KERNEL_GET_ID()\n\
#define MATH21_KERNEL_TEMPLATE_HEADER(X) template<typename X>\n\
#define MATH21_KERNEL_EXPORT\n\
#define MATH21_KERNEL_GLOBAL\n\
#define MATH21_KERNEL_INPUT_ID , NumN id\n\
#define MATH21_DEVICE_F_MIN_LIKE_PTR math21_type_f_min_like f_shrink_min_like_ptr,\n\
#define MATH21_DEVICE_F_ARGMIN_LIKE_PTR math21_type_f_argmin_like f_shrink_argmin_like_ptr,\n\
#define MATH21_DEVICE_F_ADD_LIKE_PTR math21_type_f_add_like f_bc_add_like_ptr,\n\
#define MATH21_DEVICE_F_SIN_LIKE_PTR math21_type_f_sin_like f_bc_sin_like_ptr,\n\
#define MATH21_DEVICE_F_KX_LIKE_PTR math21_type_f_kx_like f_kx_like_ptr,\n\
#define MATH21_DEVICE_F_ADDTO_LIKE_PTR math21_type_f_addto_like f_addto_like_ptr,\n\
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y) X Y,\n\
\n\
#elif defined(MATH21_IS_FROM_CUDA)\n\
#define MATH21_MAKE_KERNEL_NAME_1(X) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_2(X) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_3(X) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_PAIR(X, Y) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_KERNEL_GET_ID() int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; id +=1;\n\
#define MATH21_KERNEL_TEMPLATE_HEADER(X) template<typename X>\n\
#define MATH21_KERNEL_EXPORT __global__\n\
#define MATH21_KERNEL_GLOBAL\n\
#define MATH21_KERNEL_INPUT_ID\n\
#define MATH21_DEVICE_F_MIN_LIKE_PTR math21_type_f_min_like f_shrink_min_like_ptr,\n\
#define MATH21_DEVICE_F_ARGMIN_LIKE_PTR math21_type_f_argmin_like f_shrink_argmin_like_ptr,\n\
#define MATH21_DEVICE_F_ADD_LIKE_PTR math21_type_f_add_like f_bc_add_like_ptr,\n\
#define MATH21_DEVICE_F_SIN_LIKE_PTR math21_type_f_sin_like f_bc_sin_like_ptr,\n\
#define MATH21_DEVICE_F_KX_LIKE_PTR math21_type_f_kx_like f_kx_like_ptr,\n\
#define MATH21_DEVICE_F_ADDTO_LIKE_PTR math21_type_f_addto_like f_addto_like_ptr,\n\
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y) X Y,\n\
\n\
#elif defined(MATH21_IS_FROM_OPENCL)\n\
#define MATH21_MAKE_KERNEL_NAME_1(X) MATH21_OPENCL_TEMPLATE_3(X, opencl_kernel, NumReal)\n\
#define MATH21_MAKE_KERNEL_NAME_2(X) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, NumReal, f_shrink_min_like_ptr)\n\
#define MATH21_MAKE_KERNEL_NAME_3(X) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, NumReal, f_shrink_argmin_like_ptr)\n\
#define MATH21_MAKE_KERNEL_NAME_PAIR(X, Y) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, NumReal, Y)\n\
#define MATH21_KERNEL_GET_ID() size_t global_x = get_global_id(0); size_t global_y = get_global_id(1); size_t global_z = get_global_id(2); size_t id = global_z * get_global_size(0) * get_global_size(1) + global_y * get_global_size(0) + global_x; id +=1;\n\
#define MATH21_KERNEL_TEMPLATE_HEADER(X)\n\
#define MATH21_KERNEL_EXPORT __kernel\n\
#define MATH21_KERNEL_GLOBAL __global\n\
#define MATH21_KERNEL_INPUT_ID\n\
#define MATH21_DEVICE_F_MIN_LIKE_PTR\n\
#define MATH21_DEVICE_F_ARGMIN_LIKE_PTR\n\
#define MATH21_DEVICE_F_ADD_LIKE_PTR\n\
#define MATH21_DEVICE_F_SIN_LIKE_PTR\n\
#define MATH21_DEVICE_F_KX_LIKE_PTR\n\
#define MATH21_DEVICE_F_ADDTO_LIKE_PTR\n\
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y)\n\
\n\
#else\n\
#error MATH21_IS_FROM_NONE\n\
#endif\n\
\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
#include <math21_opencl_device_code.h>\n\
#endif\n\
\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void\n\
MATH21_MAKE_KERNEL_NAME_2(math21_template_tensor_f_shrink)(MATH21_DEVICE_F_MIN_LIKE_PTR\n\
                                                           NumN n, MATH21_KERNEL_GLOBAL const NumReal *x,\n\
                                                           MATH21_KERNEL_GLOBAL NumReal *y,\n\
                                                           NumN dims_x, MATH21_KERNEL_GLOBAL const NumN *dx,\n\
                                                           NumN dims_y,\n\
                                                           MATH21_KERNEL_GLOBAL const NumN *dy,\n\
                                                           NumN nb, MATH21_KERNEL_GLOBAL const NumN *b,\n\
                                                           NumN nv, NumN dims_v, MATH21_KERNEL_GLOBAL\n\
                                                           const NumN *dv MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x -= 1;\n\
    y -= 1;\n\
    dx -= 1;\n\
    dy -= 1;\n\
    b -= 1;\n\
    dv -= 1;\n\
#endif\n\
\n\
    if (id > n) return;\n\
    NumN iy;\n\
    NumReal value = 0;\n\
    iy = id;\n\
    if (n == 1) {\n\
        NumN iv;\n\
        for (iv = 1; iv <= nv; ++iv) {\n\
            value = (f_shrink_min_like_ptr)(value, x[iv], iv);\n\
        }\n\
    } else {\n\
        NumN _indexx[MATH21_KERNEL_ARRAY_MAX_LENGTH], _indexy[MATH21_KERNEL_ARRAY_MAX_LENGTH],\n\
                _indexv[MATH21_KERNEL_ARRAY_MAX_LENGTH], _index0[MATH21_KERNEL_ARRAY_MAX_LENGTH],\n\
                ix;\n\
        NumN *indexx = math21_device_pointer_NumN_decrease_one(_indexx);\n\
        NumN *indexy = math21_device_pointer_NumN_decrease_one(_indexy);\n\
        NumN *indexv = math21_device_pointer_NumN_decrease_one(_indexv);\n\
        NumN *index0 = math21_device_pointer_NumN_decrease_one(_index0);\n\
\n\
        // 1->n, n->n, n->1\n\
        math21_device_index_1d_to_nd(indexy, iy, dy, dims_y);\n\
        math21_device_index_replace_inc_global_1(nb, b, index0, indexy, (NumN) 1);\n\
\n\
        NumN iv;\n\
        for (iv = 1; iv <= nv; ++iv) {\n\
            math21_device_index_1d_to_nd(indexv, iv, dv, dims_v);\n\
            math21_device_index_replace_inc(nb, index0, indexx, indexv, (NumN) 0);\n\
            math21_device_index_nd_to_1d(indexx, &ix, dx, dims_x);\n\
            value = (f_shrink_min_like_ptr)(value, x[ix], iv);\n\
        }\n\
    }\n\
    y[iy] = value;\n\
}\n\
\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void\n\
MATH21_MAKE_KERNEL_NAME_3(math21_template_tensor_f_shrink)(MATH21_DEVICE_F_ARGMIN_LIKE_PTR\n\
                                                           NumN n, MATH21_KERNEL_GLOBAL const NumReal *x,\n\
                                                           MATH21_KERNEL_GLOBAL NumReal *y,\n\
                                                           NumN dims_x, MATH21_KERNEL_GLOBAL const NumN *dx,\n\
                                                           NumN dims_y,\n\
                                                           MATH21_KERNEL_GLOBAL const NumN *dy,\n\
                                                           NumN nb, MATH21_KERNEL_GLOBAL const NumN *b,\n\
                                                           NumN nv, NumN dims_v, MATH21_KERNEL_GLOBAL\n\
                                                           const NumN *dv MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x -= 1;\n\
    y -= 1;\n\
    dx -= 1;\n\
    dy -= 1;\n\
    b -= 1;\n\
    dv -= 1;\n\
#endif\n\
\n\
    if (id > n) return;\n\
    NumN iy;\n\
    NumReal value = 0;\n\
    NumN i_value = 0;\n\
    iy = id;\n\
    if (n == 1) {\n\
        NumN iv;\n\
        for (iv = 1; iv <= nv; ++iv) {\n\
            value = (f_shrink_argmin_like_ptr)(value, x[iv], &i_value, iv, iv);\n\
        }\n\
    } else {\n\
        NumN _indexx[MATH21_KERNEL_ARRAY_MAX_LENGTH], _indexy[MATH21_KERNEL_ARRAY_MAX_LENGTH],\n\
                _indexv[MATH21_KERNEL_ARRAY_MAX_LENGTH], _index0[MATH21_KERNEL_ARRAY_MAX_LENGTH],\n\
                ix;\n\
        NumN *indexx = math21_device_pointer_NumN_decrease_one(_indexx);\n\
        NumN *indexy = math21_device_pointer_NumN_decrease_one(_indexy);\n\
        NumN *indexv = math21_device_pointer_NumN_decrease_one(_indexv);\n\
        NumN *index0 = math21_device_pointer_NumN_decrease_one(_index0);\n\
\n\
        // 1->n, n->n, n->1\n\
        math21_device_index_1d_to_nd(indexy, iy, dy, dims_y);\n\
        math21_device_index_replace_inc_global_1(nb, b, index0, indexy, (NumN) 1);\n\
\n\
        NumN iv;\n\
        for (iv = 1; iv <= nv; ++iv) {\n\
            math21_device_index_1d_to_nd(indexv, iv, dv, dims_v);\n\
            math21_device_index_replace_inc(nb, index0, indexx, indexv, (NumN) 0);\n\
            math21_device_index_nd_to_1d(indexx, &ix, dx, dims_x);\n\
            // globally\n\
//                value = (f_shrink_argmin_like_ptr)(value, x[ix], &i_value, ix, iv);\n\
            // locally\n\
            value = (f_shrink_argmin_like_ptr)(value, x[ix], &i_value, iv, iv);\n\
        }\n\
    }\n\
    y[iy] = i_value;\n\
}\n\
\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void\n\
MATH21_MAKE_KERNEL_NAME_PAIR(math21_template_tensor_f_inner_product_like_shrink, f_inner_product_like_ptr)(\n\
        MATH21_DEVICE_MAKE_F_LIKE_PTR(math21_type_f_inner_product_like, f_inner_product_like_ptr)\n\
        NumN n,\n\
        MATH21_KERNEL_GLOBAL const NumReal *x1,\n\
        MATH21_KERNEL_GLOBAL const NumReal *x2,\n\
        MATH21_KERNEL_GLOBAL NumReal *y,\n\
        NumN dims_x, MATH21_KERNEL_GLOBAL const NumN *dx,\n\
        NumN dims_y,\n\
        MATH21_KERNEL_GLOBAL const NumN *dy,\n\
        NumN nb, MATH21_KERNEL_GLOBAL const NumN *b,\n\
        NumN nv, NumN dims_v, MATH21_KERNEL_GLOBAL\n\
        const NumN *dv MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x1 -= 1;\n\
    x2 -= 1;\n\
    y -= 1;\n\
    dx -= 1;\n\
    dy -= 1;\n\
    b -= 1;\n\
    dv -= 1;\n\
#endif\n\
\n\
    if (id > n) return;\n\
    NumN iy;\n\
    NumReal value = 0;\n\
    iy = id;\n\
    if (n == 1) {\n\
        NumN iv;\n\
        for (iv = 1; iv <= nv; ++iv) {\n\
            value = (f_inner_product_like_ptr)(value, x1[iv], x2[iv], iv);\n\
        }\n\
    } else {\n\
        NumN _indexx[MATH21_KERNEL_ARRAY_MAX_LENGTH], _indexy[MATH21_KERNEL_ARRAY_MAX_LENGTH],\n\
                _indexv[MATH21_KERNEL_ARRAY_MAX_LENGTH], _index0[MATH21_KERNEL_ARRAY_MAX_LENGTH],\n\
                ix;\n\
        NumN *indexx = math21_device_pointer_NumN_decrease_one(_indexx);\n\
        NumN *indexy = math21_device_pointer_NumN_decrease_one(_indexy);\n\
        NumN *indexv = math21_device_pointer_NumN_decrease_one(_indexv);\n\
        NumN *index0 = math21_device_pointer_NumN_decrease_one(_index0);\n\
\n\
        // 1->n, n->n, n->1\n\
        math21_device_index_1d_to_nd(indexy, iy, dy, dims_y);\n\
        math21_device_index_replace_inc_global_1(nb, b, index0, indexy, (NumN) 1);\n\
\n\
        NumN iv;\n\
        for (iv = 1; iv <= nv; ++iv) {\n\
            math21_device_index_1d_to_nd(indexv, iv, dv, dims_v);\n\
            math21_device_index_replace_inc(nb, index0, indexx, indexv, (NumN) 0);\n\
            math21_device_index_nd_to_1d(indexx, &ix, dx, dims_x);\n\
            value = (f_inner_product_like_ptr)(value, x1[ix], x2[ix], iv);\n\
        }\n\
    }\n\
    y[iy] = value;\n\
}\n\
\n\
// y = x1 + x2\n\
// a special kind of sub\n\
// x1, x2 sub-tensor of y\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void\n\
MATH21_MAKE_KERNEL_NAME_PAIR(math21_template_tensor_f_with_broadcast_in_dn, f_bc_add_like_ptr)(\n\
        MATH21_DEVICE_F_ADD_LIKE_PTR NumN n,\n\
        MATH21_KERNEL_GLOBAL const NumReal *x1,\n\
        MATH21_KERNEL_GLOBAL const NumReal *x2,\n\
        MATH21_KERNEL_GLOBAL NumReal *y,\n\
        NumN dims_x1, MATH21_KERNEL_GLOBAL const NumN *dx1,\n\
        NumN dims_x2, MATH21_KERNEL_GLOBAL const NumN *dx2,\n\
        NumN dims_y, MATH21_KERNEL_GLOBAL const NumN *dy MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x1 -= 1;\n\
    x2 -= 1;\n\
    y -= 1;\n\
    dx1 -= 1;\n\
    dx2 -= 1;\n\
    dy -= 1;\n\
#endif\n\
\n\
    if (id > n) return;\n\
    NumN _indexx1[MATH21_KERNEL_ARRAY_MAX_LENGTH], _indexx2[MATH21_KERNEL_ARRAY_MAX_LENGTH],\n\
            _indexy[MATH21_KERNEL_ARRAY_MAX_LENGTH], ix1, ix2, iy;\n\
    NumN *indexx1 = math21_device_pointer_NumN_decrease_one(_indexx1);\n\
    NumN *indexx2 = math21_device_pointer_NumN_decrease_one(_indexx2);\n\
    NumN *indexy = math21_device_pointer_NumN_decrease_one(_indexy);\n\
\n\
    iy = id;\n\
    math21_device_index_1d_to_nd(indexy, iy, dy, dims_y);\n\
    math21_device_broadcast_index_to_original_brackets(indexy, dx1, indexx1, dims_x1);\n\
    math21_device_index_nd_to_1d(indexx1, &ix1, dx1, dims_x1);\n\
    math21_device_broadcast_index_to_original_brackets(indexy, dx2, indexx2, dims_x2);\n\
    math21_device_index_nd_to_1d(indexx2, &ix2, dx2, dims_x2);\n\
    y[iy] = (f_bc_add_like_ptr)(x1[ix1], x2[ix2]);\n\
}\n\
\n\
// y = x1 if x2 = 1\n\
// a special kind of sub\n\
// x1, x2 sub-tensor of y\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_tensor_set_using_mask_in_dn)(\n\
        NumN n,\n\
        MATH21_KERNEL_GLOBAL const NumReal *x1,\n\
        MATH21_KERNEL_GLOBAL const NumReal *x2,\n\
        MATH21_KERNEL_GLOBAL NumReal *y,\n\
        NumN dims_x1, MATH21_KERNEL_GLOBAL const NumN *dx1,\n\
        NumN dims_x2, MATH21_KERNEL_GLOBAL const NumN *dx2,\n\
        NumN dims_y, MATH21_KERNEL_GLOBAL const NumN *dy MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x1 -= 1;\n\
    x2 -= 1;\n\
    y -= 1;\n\
    dx1 -= 1;\n\
    dx2 -= 1;\n\
    dy -= 1;\n\
#endif\n\
\n\
    if (id > n) return;\n\
    NumN _indexx1[MATH21_KERNEL_ARRAY_MAX_LENGTH], _indexx2[MATH21_KERNEL_ARRAY_MAX_LENGTH],\n\
            _indexy[MATH21_KERNEL_ARRAY_MAX_LENGTH], ix1, ix2, iy;\n\
    NumN *indexx1 = math21_device_pointer_NumN_decrease_one(_indexx1);\n\
    NumN *indexx2 = math21_device_pointer_NumN_decrease_one(_indexx2);\n\
    NumN *indexy = math21_device_pointer_NumN_decrease_one(_indexy);\n\
\n\
    iy = id;\n\
    math21_device_index_1d_to_nd(indexy, iy, dy, dims_y);\n\
    math21_device_broadcast_index_to_original_brackets(indexy, dx1, indexx1, dims_x1);\n\
    math21_device_index_nd_to_1d(indexx1, &ix1, dx1, dims_x1);\n\
    math21_device_broadcast_index_to_original_brackets(indexy, dx2, indexx2, dims_x2);\n\
    math21_device_index_nd_to_1d(indexx2, &ix2, dx2, dims_x2);\n\
    if (x2[ix2] == 1) {\n\
        y[iy] = x1[ix1];\n\
    }\n\
}\n\
\n\
// y = x1 + x2\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_PAIR(math21_template_vector_f_add_like, f_bc_add_like_ptr)(\n\
        MATH21_DEVICE_F_ADD_LIKE_PTR NumN n,\n\
        MATH21_KERNEL_GLOBAL const NumReal *x1,\n\
        MATH21_KERNEL_GLOBAL const NumReal *x2,\n\
        MATH21_KERNEL_GLOBAL NumReal *y MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x1 -= 1;\n\
    x2 -= 1;\n\
    y -= 1;\n\
#endif\n\
    if (id > n) return;\n\
    NumN iy;\n\
    iy = id;\n\
    y[iy] = (f_bc_add_like_ptr)(x1[iy], x2[iy]);\n\
}\n\
\n\
// y = x1 when x2 = 1\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_vector_set_using_mask)(\n\
        NumN n,\n\
        MATH21_KERNEL_GLOBAL const NumReal *x1,\n\
        MATH21_KERNEL_GLOBAL const NumReal *x2,\n\
        MATH21_KERNEL_GLOBAL NumReal *y MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x1 -= 1;\n\
    x2 -= 1;\n\
    y -= 1;\n\
#endif\n\
    if (id > n) return;\n\
    NumN iy;\n\
    iy = id;\n\
    if (x2[iy] == 1) {\n\
        y[iy] = x1[iy];\n\
    }\n\
}\n\
\n\
// y = f(x)\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_PAIR(math21_template_vector_f_sin_like, f_bc_sin_like_ptr)(\n\
        MATH21_DEVICE_F_SIN_LIKE_PTR NumN n,\n\
        MATH21_KERNEL_GLOBAL const NumReal *x,\n\
        MATH21_KERNEL_GLOBAL NumReal *y MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x -= 1;\n\
    y -= 1;\n\
#endif\n\
    if (id > n) return;\n\
    NumN iy;\n\
    iy = id;\n\
    y[iy] = (f_bc_sin_like_ptr)(x[iy]);\n\
}\n\
\n\
// y = f(k, x)\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_PAIR(math21_template_vector_f_kx_like, f_kx_like_ptr)(\n\
        MATH21_DEVICE_F_KX_LIKE_PTR NumN n,\n\
        NumReal k,\n\
        MATH21_KERNEL_GLOBAL const NumReal *x,\n\
        MATH21_KERNEL_GLOBAL NumReal *y MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x -= 1;\n\
    y -= 1;\n\
#endif\n\
    if (id > n) return;\n\
    NumN iy;\n\
    iy = id;\n\
    y[iy] = (f_kx_like_ptr)(k, x[iy]);\n\
}\n\
"; 
std::string generic_03 = "\n\
#include <math21_kernels.h>\n\
\n\
//#define MATH21_IS_FROM_CPU\n\
\n\
#if !defined(MATH21_IS_FROM_CPU)\n\
#if defined(MATH21_FLAG_USE_CUDA)\n\
#define MATH21_IS_FROM_CUDA\n\
#elif defined(MATH21_FLAG_USE_OPENCL)\n\
#define MATH21_IS_FROM_OPENCL\n\
#endif\n\
#endif\n\
\n\
#if defined(MATH21_IS_FROM_CPU)\n\
#define MATH21_MAKE_KERNEL_NAME_1(X) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_2(X) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_3(X) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_PAIR(X, Y) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_KERNEL_GET_ID()\n\
#define MATH21_KERNEL_TEMPLATE_HEADER(X) template<typename X>\n\
#define MATH21_KERNEL_EXPORT\n\
#define MATH21_KERNEL_GLOBAL\n\
#define MATH21_KERNEL_INPUT_ID , NumN id\n\
#define MATH21_DEVICE_F_MIN_LIKE_PTR math21_type_f_min_like f_shrink_min_like_ptr,\n\
#define MATH21_DEVICE_F_ARGMIN_LIKE_PTR math21_type_f_argmin_like f_shrink_argmin_like_ptr,\n\
#define MATH21_DEVICE_F_ADD_LIKE_PTR math21_type_f_add_like f_bc_add_like_ptr,\n\
#define MATH21_DEVICE_F_SIN_LIKE_PTR math21_type_f_sin_like f_bc_sin_like_ptr,\n\
#define MATH21_DEVICE_F_KX_LIKE_PTR math21_type_f_kx_like f_kx_like_ptr,\n\
#define MATH21_DEVICE_F_ADDTO_LIKE_PTR math21_type_f_addto_like f_addto_like_ptr,\n\
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y) X Y,\n\
\n\
#elif defined(MATH21_IS_FROM_CUDA)\n\
#define MATH21_MAKE_KERNEL_NAME_1(X) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_2(X) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_3(X) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_PAIR(X, Y) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_KERNEL_GET_ID() int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; id +=1;\n\
#define MATH21_KERNEL_TEMPLATE_HEADER(X) template<typename X>\n\
#define MATH21_KERNEL_EXPORT __global__\n\
#define MATH21_KERNEL_GLOBAL\n\
#define MATH21_KERNEL_INPUT_ID\n\
#define MATH21_DEVICE_F_MIN_LIKE_PTR math21_type_f_min_like f_shrink_min_like_ptr,\n\
#define MATH21_DEVICE_F_ARGMIN_LIKE_PTR math21_type_f_argmin_like f_shrink_argmin_like_ptr,\n\
#define MATH21_DEVICE_F_ADD_LIKE_PTR math21_type_f_add_like f_bc_add_like_ptr,\n\
#define MATH21_DEVICE_F_SIN_LIKE_PTR math21_type_f_sin_like f_bc_sin_like_ptr,\n\
#define MATH21_DEVICE_F_KX_LIKE_PTR math21_type_f_kx_like f_kx_like_ptr,\n\
#define MATH21_DEVICE_F_ADDTO_LIKE_PTR math21_type_f_addto_like f_addto_like_ptr,\n\
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y) X Y,\n\
\n\
#elif defined(MATH21_IS_FROM_OPENCL)\n\
#define MATH21_MAKE_KERNEL_NAME_1(X) MATH21_OPENCL_TEMPLATE_3(X, opencl_kernel, NumReal)\n\
#define MATH21_MAKE_KERNEL_NAME_2(X) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, NumReal, f_shrink_min_like_ptr)\n\
#define MATH21_MAKE_KERNEL_NAME_3(X) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, NumReal, f_shrink_argmin_like_ptr)\n\
#define MATH21_MAKE_KERNEL_NAME_PAIR(X, Y) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, NumReal, Y)\n\
#define MATH21_KERNEL_GET_ID() size_t global_x = get_global_id(0); size_t global_y = get_global_id(1); size_t global_z = get_global_id(2); size_t id = global_z * get_global_size(0) * get_global_size(1) + global_y * get_global_size(0) + global_x; id +=1;\n\
#define MATH21_KERNEL_TEMPLATE_HEADER(X)\n\
#define MATH21_KERNEL_EXPORT __kernel\n\
#define MATH21_KERNEL_GLOBAL __global\n\
#define MATH21_KERNEL_INPUT_ID\n\
#define MATH21_DEVICE_F_MIN_LIKE_PTR\n\
#define MATH21_DEVICE_F_ARGMIN_LIKE_PTR\n\
#define MATH21_DEVICE_F_ADD_LIKE_PTR\n\
#define MATH21_DEVICE_F_SIN_LIKE_PTR\n\
#define MATH21_DEVICE_F_KX_LIKE_PTR\n\
#define MATH21_DEVICE_F_ADDTO_LIKE_PTR\n\
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y)\n\
\n\
#else\n\
#error MATH21_IS_FROM_NONE\n\
#endif\n\
\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
#include <math21_opencl_device_code.h>\n\
#endif\n\
\n\
// C = k1*A*B + k2*C\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_nn_naive)(\n\
        NumN size, NumN nr_C, NumN nc_C, NumN n_common, NumReal k1,\n\
        MATH21_KERNEL_GLOBAL const NumReal *A, NumN stride_a,\n\
        MATH21_KERNEL_GLOBAL const NumReal *B, NumN stride_b,\n\
        NumReal k2,\n\
        MATH21_KERNEL_GLOBAL NumReal *C, NumN stride_c MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    A -= 1;\n\
    B -= 1;\n\
    C -= 1;\n\
#endif\n\
    if (id > size) return;\n\
    NumN i, j, ia, ib, ic;\n\
    math21_device_index_1d_to_2d_fast(&i, &j, id, nc_C);\n\
    NumN k;\n\
    NumReal sum = 0;\n\
    for (k = 1; k <= n_common; ++k) {\n\
        math21_device_index_2d_to_1d_fast(i, k, &ia, stride_a);\n\
        math21_device_index_2d_to_1d_fast(k, j, &ib, stride_b);\n\
        sum += A[ia] * B[ib];\n\
    }\n\
    math21_device_index_2d_to_1d_fast(i, j, &ic, stride_c);\n\
    C[ic] = k1 * sum + k2 * C[ic];\n\
}\n\
\n\
// C = k1*A*B.t + k2*C\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_nt_naive)(\n\
        NumN size, NumN nr_C, NumN nc_C, NumN n_common, NumReal k1,\n\
        MATH21_KERNEL_GLOBAL const NumReal *A, NumN stride_a,\n\
        MATH21_KERNEL_GLOBAL const NumReal *B, NumN stride_b,\n\
        NumReal k2,\n\
        MATH21_KERNEL_GLOBAL NumReal *C, NumN stride_c MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    A -= 1;\n\
    B -= 1;\n\
    C -= 1;\n\
#endif\n\
    if (id > size) return;\n\
    NumN i, j, ia, ib, ic;\n\
    math21_device_index_1d_to_2d_fast(&i, &j, id, nc_C);\n\
    NumN k;\n\
    NumReal sum = 0;\n\
    for (k = 1; k <= n_common; ++k) {\n\
        math21_device_index_2d_to_1d_fast(i, k, &ia, stride_a);\n\
        math21_device_index_2d_to_1d_fast(j, k, &ib, stride_b);\n\
        sum += A[ia] * B[ib];\n\
    }\n\
    math21_device_index_2d_to_1d_fast(i, j, &ic, stride_c);\n\
    C[ic] = k1 * sum + k2 * C[ic];\n\
}\n\
\n\
// C = k1*A.t*B + k2*C\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_tn_naive)(\n\
        NumN size, NumN nr_C, NumN nc_C, NumN n_common, NumReal k1,\n\
        MATH21_KERNEL_GLOBAL const NumReal *A, NumN stride_a,\n\
        MATH21_KERNEL_GLOBAL const NumReal *B, NumN stride_b,\n\
        NumReal k2,\n\
        MATH21_KERNEL_GLOBAL NumReal *C, NumN stride_c MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    A -= 1;\n\
    B -= 1;\n\
    C -= 1;\n\
#endif\n\
    if (id > size) return;\n\
    NumN i, j, ia, ib, ic;\n\
    math21_device_index_1d_to_2d_fast(&i, &j, id, nc_C);\n\
    NumN k;\n\
    NumReal sum = 0;\n\
    for (k = 1; k <= n_common; ++k) {\n\
        math21_device_index_2d_to_1d_fast(k, i, &ia, stride_a);\n\
        math21_device_index_2d_to_1d_fast(k, j, &ib, stride_b);\n\
        sum += A[ia] * B[ib];\n\
    }\n\
    math21_device_index_2d_to_1d_fast(i, j, &ic, stride_c);\n\
    C[ic] = k1 * sum + k2 * C[ic];\n\
}\n\
\n\
// C = k1*A.t*B.t + k2*C\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_tt_naive)(\n\
        NumN size, NumN nr_C, NumN nc_C, NumN n_common, NumReal k1,\n\
        MATH21_KERNEL_GLOBAL const NumReal *A, NumN stride_a,\n\
        MATH21_KERNEL_GLOBAL const NumReal *B, NumN stride_b,\n\
        NumReal k2,\n\
        MATH21_KERNEL_GLOBAL NumReal *C, NumN stride_c MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    A -= 1;\n\
    B -= 1;\n\
    C -= 1;\n\
#endif\n\
    if (id > size) return;\n\
    NumN i, j, ia, ib, ic;\n\
    math21_device_index_1d_to_2d_fast(&i, &j, id, nc_C);\n\
    NumN k;\n\
    NumReal sum = 0;\n\
    for (k = 1; k <= n_common; ++k) {\n\
        math21_device_index_2d_to_1d_fast(k, i, &ia, stride_a);\n\
        math21_device_index_2d_to_1d_fast(j, k, &ib, stride_b);\n\
        sum += A[ia] * B[ib];\n\
    }\n\
    math21_device_index_2d_to_1d_fast(i, j, &ic, stride_c);\n\
    C[ic] = k1 * sum + k2 * C[ic];\n\
}\n\
"; 
std::string generic_03_transpose = "\n\
#include <math21_kernels.h>\n\
\n\
//#define MATH21_IS_FROM_CPU\n\
\n\
#if !defined(MATH21_IS_FROM_CPU)\n\
#if defined(MATH21_FLAG_USE_CUDA)\n\
#define MATH21_IS_FROM_CUDA\n\
#elif defined(MATH21_FLAG_USE_OPENCL)\n\
#define MATH21_IS_FROM_OPENCL\n\
#endif\n\
#endif\n\
\n\
#if defined(MATH21_IS_FROM_CPU)\n\
#define MATH21_MAKE_KERNEL_NAME_1(X) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_2(X) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_3(X) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_PAIR(X, Y) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_TWO(X, Y1, Y2) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_KERNEL_GET_ID()\n\
#define MATH21_KERNEL_TEMPLATE_HEADER(X) template<typename X>\n\
#define MATH21_KERNEL_TEMPLATE_THE_HEADER_2(X1, X2) template<typename X1, typename X2>\n\
#define MATH21_KERNEL_EXPORT\n\
#define MATH21_KERNEL_GLOBAL\n\
#define MATH21_KERNEL_INPUT_ID , NumN id\n\
#define MATH21_DEVICE_F_MIN_LIKE_PTR math21_type_f_min_like f_shrink_min_like_ptr,\n\
#define MATH21_DEVICE_F_ARGMIN_LIKE_PTR math21_type_f_argmin_like f_shrink_argmin_like_ptr,\n\
#define MATH21_DEVICE_F_ADD_LIKE_PTR math21_type_f_add_like f_bc_add_like_ptr,\n\
#define MATH21_DEVICE_F_SIN_LIKE_PTR math21_type_f_sin_like f_bc_sin_like_ptr,\n\
#define MATH21_DEVICE_F_KX_LIKE_PTR math21_type_f_kx_like f_kx_like_ptr,\n\
#define MATH21_DEVICE_F_ADDTO_LIKE_PTR math21_type_f_addto_like f_addto_like_ptr,\n\
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y) X Y,\n\
\n\
#elif defined(MATH21_IS_FROM_CUDA)\n\
#define MATH21_MAKE_KERNEL_NAME_1(X) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_2(X) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_3(X) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_PAIR(X, Y) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_TWO(X, Y1, Y2) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_KERNEL_GET_ID() int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; id +=1;\n\
#define MATH21_KERNEL_TEMPLATE_HEADER(X) template<typename X>\n\
#define MATH21_KERNEL_TEMPLATE_THE_HEADER_2(X1, X2) template<typename X1, typename X2>\n\
#define MATH21_KERNEL_EXPORT __global__\n\
#define MATH21_KERNEL_GLOBAL\n\
#define MATH21_KERNEL_INPUT_ID\n\
#define MATH21_DEVICE_F_MIN_LIKE_PTR math21_type_f_min_like f_shrink_min_like_ptr,\n\
#define MATH21_DEVICE_F_ARGMIN_LIKE_PTR math21_type_f_argmin_like f_shrink_argmin_like_ptr,\n\
#define MATH21_DEVICE_F_ADD_LIKE_PTR math21_type_f_add_like f_bc_add_like_ptr,\n\
#define MATH21_DEVICE_F_SIN_LIKE_PTR math21_type_f_sin_like f_bc_sin_like_ptr,\n\
#define MATH21_DEVICE_F_KX_LIKE_PTR math21_type_f_kx_like f_kx_like_ptr,\n\
#define MATH21_DEVICE_F_ADDTO_LIKE_PTR math21_type_f_addto_like f_addto_like_ptr,\n\
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y) X Y,\n\
\n\
#elif defined(MATH21_IS_FROM_OPENCL)\n\
#define MATH21_MAKE_KERNEL_NAME_1(X) MATH21_OPENCL_TEMPLATE_3(X, opencl_kernel, NumReal)\n\
#define MATH21_MAKE_KERNEL_NAME_2(X) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, NumReal, f_shrink_min_like_ptr)\n\
#define MATH21_MAKE_KERNEL_NAME_3(X) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, NumReal, f_shrink_argmin_like_ptr)\n\
#define MATH21_MAKE_KERNEL_NAME_PAIR(X, Y) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, NumReal, Y)\n\
#define MATH21_MAKE_KERNEL_NAME_TWO(X, Y1, Y2) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, Y1, Y2)\n\
#define MATH21_KERNEL_GET_ID() size_t global_x = get_global_id(0); size_t global_y = get_global_id(1); size_t global_z = get_global_id(2); size_t id = global_z * get_global_size(0) * get_global_size(1) + global_y * get_global_size(0) + global_x; id +=1;\n\
#define MATH21_KERNEL_TEMPLATE_HEADER(X)\n\
#define MATH21_KERNEL_TEMPLATE_THE_HEADER_2(X1, X2)\n\
#define MATH21_KERNEL_EXPORT __kernel\n\
#define MATH21_KERNEL_GLOBAL __global\n\
#define MATH21_KERNEL_INPUT_ID\n\
#define MATH21_DEVICE_F_MIN_LIKE_PTR\n\
#define MATH21_DEVICE_F_ARGMIN_LIKE_PTR\n\
#define MATH21_DEVICE_F_ADD_LIKE_PTR)\n\
#define MATH21_DEVICE_F_SIN_LIKE_PTR\n\
#define MATH21_DEVICE_F_KX_LIKE_PTR\n\
#define MATH21_DEVICE_F_ADDTO_LIKE_PTR\n\
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y)\n\
\n\
#else\n\
#error MATH21_IS_FROM_NONE\n\
#endif\n\
\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
#include <math21_opencl_device_code.h>\n\
#endif\n\
\n\
// y = x.t\n\
MATH21_KERNEL_TEMPLATE_THE_HEADER_2(NumType1, NumType2)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_TWO(math21_template_matrix_transpose, NumType1, NumType2)(\n\
        NumN n,\n\
        MATH21_KERNEL_GLOBAL const NumType1 *x,\n\
        MATH21_KERNEL_GLOBAL NumType2 *y,\n\
        NumN d1_x, NumN d2_x MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x -= 1;\n\
    y -= 1;\n\
#endif\n\
    if (id > n) return;\n\
    NumN i1, i2, ix, iy;\n\
    iy = id;\n\
    math21_device_index_1d_to_2d(&i2, &i1, iy, d2_x, d1_x);\n\
    math21_device_index_2d_to_1d(i1, i2, &ix, d1_x, d2_x);\n\
    y[iy] = x[ix];\n\
}\n\
\n\
// swap axes 2 and 4 in dim5 tensor\n\
// (d1, d2, d3, d4, d5) -> (d1, d4, d3, d2, d5)\n\
MATH21_KERNEL_TEMPLATE_THE_HEADER_2(NumType1, NumType2)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_TWO(math21_template_tensor_swap_axes_24_in_d5, NumType1, NumType2)(\n\
        NumN n,\n\
        MATH21_KERNEL_GLOBAL const NumType1 *x,\n\
        MATH21_KERNEL_GLOBAL NumType2 *y,\n\
        NumN d1, NumN d2, NumN d3, NumN d4, NumN d5 MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x -= 1;\n\
    y -= 1;\n\
#endif\n\
    if (id > n) return;\n\
    NumN i1, i2, i3, i4, i5, ix, iy;\n\
    iy = id;\n\
    math21_device_index_1d_to_5d(&i1, &i4, &i3, &i2, &i5, iy, d1, d4, d3, d2, d5);\n\
    math21_device_index_5d_to_1d(i1, i2, i3, i4, i5, &ix, d1, d2, d3, d4, d5);\n\
    y[iy] = x[ix];\n\
}\n\
"; 
std::string generic_01 = "\n\
#include <math21_kernels.h>\n\
\n\
//#define MATH21_IS_FROM_CPU\n\
\n\
#if !defined(MATH21_IS_FROM_CPU)\n\
#if defined(MATH21_FLAG_USE_CUDA)\n\
#define MATH21_IS_FROM_CUDA\n\
#elif defined(MATH21_FLAG_USE_OPENCL)\n\
#define MATH21_IS_FROM_OPENCL\n\
#endif\n\
#endif\n\
\n\
#if defined(MATH21_IS_FROM_CPU)\n\
#define MATH21_MAKE_KERNEL_NAME_1(X) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_2(X) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_3(X) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_PAIR(X, Y) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_KERNEL_GET_ID()\n\
#define MATH21_KERNEL_TEMPLATE_HEADER(X) template<typename X>\n\
#define MATH21_KERNEL_EXPORT\n\
#define MATH21_KERNEL_GLOBAL\n\
#define MATH21_KERNEL_INPUT_ID , NumN id\n\
#define MATH21_KERNEL_INPUT_OFFSETS_XY\n\
#define MATH21_DEVICE_F_MIN_LIKE_PTR math21_type_f_min_like f_shrink_min_like_ptr,\n\
#define MATH21_DEVICE_F_ARGMIN_LIKE_PTR math21_type_f_argmin_like f_shrink_argmin_like_ptr,\n\
#define MATH21_DEVICE_F_ADD_LIKE_PTR math21_type_f_add_like f_bc_add_like_ptr,\n\
#define MATH21_DEVICE_F_SIN_LIKE_PTR math21_type_f_sin_like f_bc_sin_like_ptr,\n\
#define MATH21_DEVICE_F_KX_LIKE_PTR math21_type_f_kx_like f_kx_like_ptr,\n\
#define MATH21_DEVICE_F_ADDTO_LIKE_PTR math21_type_f_addto_like f_addto_like_ptr,\n\
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y) X Y,\n\
\n\
#elif defined(MATH21_IS_FROM_CUDA)\n\
#define MATH21_MAKE_KERNEL_NAME_1(X) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_2(X) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_3(X) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_PAIR(X, Y) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_KERNEL_GET_ID() int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; id +=1;\n\
#define MATH21_KERNEL_TEMPLATE_HEADER(X) template<typename X>\n\
#define MATH21_KERNEL_EXPORT __global__\n\
#define MATH21_KERNEL_GLOBAL\n\
#define MATH21_KERNEL_INPUT_ID\n\
#define MATH21_KERNEL_INPUT_OFFSETS_XY\n\
#define MATH21_DEVICE_F_MIN_LIKE_PTR math21_type_f_min_like f_shrink_min_like_ptr,\n\
#define MATH21_DEVICE_F_ARGMIN_LIKE_PTR math21_type_f_argmin_like f_shrink_argmin_like_ptr,\n\
#define MATH21_DEVICE_F_ADD_LIKE_PTR math21_type_f_add_like f_bc_add_like_ptr,\n\
#define MATH21_DEVICE_F_SIN_LIKE_PTR math21_type_f_sin_like f_bc_sin_like_ptr,\n\
#define MATH21_DEVICE_F_KX_LIKE_PTR math21_type_f_kx_like f_kx_like_ptr,\n\
#define MATH21_DEVICE_F_ADDTO_LIKE_PTR math21_type_f_addto_like f_addto_like_ptr,\n\
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y) X Y,\n\
\n\
#elif defined(MATH21_IS_FROM_OPENCL)\n\
#define MATH21_MAKE_KERNEL_NAME_1(X) MATH21_OPENCL_TEMPLATE_3(X, opencl_kernel, NumReal)\n\
#define MATH21_MAKE_KERNEL_NAME_2(X) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, NumReal, f_shrink_min_like_ptr)\n\
#define MATH21_MAKE_KERNEL_NAME_3(X) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, NumReal, f_shrink_argmin_like_ptr)\n\
#define MATH21_MAKE_KERNEL_NAME_PAIR(X, Y) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, NumReal, Y)\n\
#define MATH21_KERNEL_GET_ID() size_t global_x = get_global_id(0); size_t global_y = get_global_id(1); size_t global_z = get_global_id(2); size_t id = global_z * get_global_size(0) * get_global_size(1) + global_y * get_global_size(0) + global_x; id +=1;\n\
#define MATH21_KERNEL_TEMPLATE_HEADER(X)\n\
#define MATH21_KERNEL_EXPORT __kernel\n\
#define MATH21_KERNEL_GLOBAL __global\n\
#define MATH21_KERNEL_INPUT_ID\n\
#define MATH21_KERNEL_INPUT_OFFSETS_XY , NumN offset_x, NumN offset_y\n\
#define MATH21_DEVICE_F_MIN_LIKE_PTR\n\
#define MATH21_DEVICE_F_ARGMIN_LIKE_PTR\n\
#define MATH21_DEVICE_F_ADD_LIKE_PTR\n\
#define MATH21_DEVICE_F_SIN_LIKE_PTR\n\
#define MATH21_DEVICE_F_KX_LIKE_PTR\n\
#define MATH21_DEVICE_F_ADDTO_LIKE_PTR\n\
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y)\n\
\n\
#else\n\
#error MATH21_IS_FROM_NONE\n\
#endif\n\
\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
#include <math21_opencl_device_code.h>\n\
#endif\n\
\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_tensor_sub_set_or_get)(\n\
        NumN n, MATH21_KERNEL_GLOBAL NumReal *x, MATH21_KERNEL_GLOBAL NumReal *y, NumN dims,\n\
        MATH21_KERNEL_GLOBAL const NumN *dx,\n\
        MATH21_KERNEL_GLOBAL const NumN *dy,\n\
        MATH21_KERNEL_GLOBAL const NumN *offset,\n\
        NumB isGet MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x -= 1;\n\
    y -= 1;\n\
    dx -= 1;\n\
    dy -= 1;\n\
    offset -= 1;\n\
#endif\n\
    if (id > n) return;\n\
    NumN _indexx[MATH21_KERNEL_ARRAY_MAX_LENGTH], _indexy[MATH21_KERNEL_ARRAY_MAX_LENGTH], ix, iy;\n\
    NumN *indexx = math21_device_pointer_NumN_decrease_one(_indexx);\n\
    NumN *indexy = math21_device_pointer_NumN_decrease_one(_indexy);\n\
\n\
    ix = id;\n\
    math21_device_index_1d_to_nd(indexx, ix, dx, dims);\n\
    math21_device_index_add_to_c_2(dims, indexx, offset, indexy);\n\
    math21_device_index_nd_to_1d(indexy, &iy, dy, dims);\n\
    if (!isGet) {\n\
        y[iy] = x[ix];\n\
    } else {\n\
        x[ix] = y[iy];\n\
    }\n\
}\n\
\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_vector_kx)(\n\
        NumN n, NumReal k, MATH21_KERNEL_GLOBAL NumReal *x, NumN stride_x MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
    if (id <= n) x[(id - 1) * stride_x] *= k;\n\
}\n\
\n\
// y = k*x + y\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_vector_kx_add_y)(\n\
        NumN n, NumReal k, MATH21_KERNEL_GLOBAL const NumReal *x, NumN stride_x, MATH21_KERNEL_GLOBAL NumReal *y,\n\
        NumN stride_y MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x -= 1;\n\
    y -= 1;\n\
#endif\n\
    if (id > n) return;\n\
    y[(id - 1) * stride_y + 1] += k * x[(id - 1) * stride_x + 1];\n\
}\n\
\n\
// 1, 2, 3 -> 1, 4, 7 when stride is 3.\n\
// d2_x = stride1_x * trailing_dimension\n\
// y = x\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_matrix_set_by_matrix)(\n\
        NumN n, NumN d2,\n\
        MATH21_KERNEL_GLOBAL const NumReal *x, NumN d2_x, NumN stride2_x,\n\
        MATH21_KERNEL_GLOBAL NumReal *y, NumN d2_y, NumN stride2_y\n\
        MATH21_KERNEL_INPUT_OFFSETS_XY MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x += offset_x;\n\
    y += offset_y;\n\
    x -= 1;\n\
    y -= 1;\n\
#endif\n\
    if (id > n) return;\n\
    NumN i1, i2, iy, ix;\n\
    math21_device_index_1d_to_2d_fast(&i1, &i2, id, d2);\n\
    math21_device_index_2d_to_1d_fast(i1, (i2 - 1) * stride2_x + 1, &ix, d2_x);\n\
    math21_device_index_2d_to_1d_fast(i1, (i2 - 1) * stride2_y + 1, &iy, d2_y);\n\
    y[iy] = x[ix];\n\
}\n\
\n\
// 1, 2, 3 -> 1, 4, 7 when stride is 3.\n\
// d2_x <- stride1_x * d2_x\n\
// y = x\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_tensor_3d_set_by_tensor_3d)(\n\
        NumN n, NumN d2, NumN d3,\n\
        MATH21_KERNEL_GLOBAL const NumReal *x, NumN d2_x, NumN d3_x, NumN stride3_x,\n\
        MATH21_KERNEL_GLOBAL NumReal *y, NumN d2_y, NumN d3_y, NumN stride3_y\n\
        MATH21_KERNEL_INPUT_OFFSETS_XY MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x += offset_x;\n\
    y += offset_y;\n\
    x -= 1;\n\
    y -= 1;\n\
#endif\n\
    if (id > n) return;\n\
    NumN i1, i2, i3, iy, ix;\n\
    math21_device_index_1d_to_3d_fast(&i1, &i2, &i3, id, d2, d3);\n\
    math21_device_index_3d_to_1d_fast(i1, i2, (i3 - 1) * stride3_x + 1, &ix, d2_x, d3_x);\n\
    math21_device_index_3d_to_1d_fast(i1, i2, (i3 - 1) * stride3_y + 1, &iy, d2_y, d3_y);\n\
    y[iy] = x[ix];\n\
}\n\
\n\
// 1, 2, 3 -> 1, 4, 7 when stride is 3.\n\
// d2_x <- stride1_x * d2_x\n\
// y = x\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_PAIR(math21_template_tensor_3d_f_set_by_tensor_3d, f_addto_like_ptr)(\n\
        MATH21_DEVICE_F_ADDTO_LIKE_PTR\n\
        NumN n, NumN d2, NumN d3,\n\
        MATH21_KERNEL_GLOBAL const NumReal *x, NumN d2_x, NumN d3_x, NumN stride3_x,\n\
        MATH21_KERNEL_GLOBAL NumReal *y, NumN d2_y, NumN d3_y, NumN stride3_y\n\
        MATH21_KERNEL_INPUT_OFFSETS_XY MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x += offset_x;\n\
    y += offset_y;\n\
    x -= 1;\n\
    y -= 1;\n\
#endif\n\
    if (id > n) return;\n\
    NumN i1, i2, i3, iy, ix;\n\
    math21_device_index_1d_to_3d_fast(&i1, &i2, &i3, id, d2, d3);\n\
    math21_device_index_3d_to_1d_fast(i1, i2, (i3 - 1) * stride3_x + 1, &ix, d2_x, d3_x);\n\
    math21_device_index_3d_to_1d_fast(i1, i2, (i3 - 1) * stride3_y + 1, &iy, d2_y, d3_y);\n\
    y[iy] = (f_addto_like_ptr)(y[iy], x[ix]);\n\
}\n\
\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_vector_set_by_value)(\n\
        NumN n, NumReal value, MATH21_KERNEL_GLOBAL NumReal *x, NumN stride_x MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x -= 1;\n\
#endif\n\
    if (id > n) return;\n\
    x[(id - 1) * stride_x + 1] = value;\n\
}\n\
\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_vector_xy)(\n\
        NumN n, MATH21_KERNEL_GLOBAL const NumReal *x, NumN stride_x, MATH21_KERNEL_GLOBAL NumReal *y,\n\
        NumN stride_y MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x -= 1;\n\
    y -= 1;\n\
#endif\n\
    if (id > n) return;\n\
    y[(id - 1) * stride_y + 1] *= x[(id - 1) * stride_x + 1];\n\
}\n\
\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_vector_sin)(\n\
        NumN n, MATH21_KERNEL_GLOBAL const NumReal *x, MATH21_KERNEL_GLOBAL NumReal *y MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
    if (id > n) return;\n\
    y[id - 1] = sin(x[id - 1]);\n\
}\n\
\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_vector_cos)(\n\
        NumN n, MATH21_KERNEL_GLOBAL const NumReal *x, MATH21_KERNEL_GLOBAL NumReal *y MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
    if (id <= n) y[id - 1] = cos(x[id - 1]);\n\
}\n\
\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_vector_addToC)(\n\
        NumN n, MATH21_KERNEL_GLOBAL const NumReal *A, MATH21_KERNEL_GLOBAL const NumReal *B,\n\
        MATH21_KERNEL_GLOBAL NumReal *C MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
    if (id <= n) C[id - 1] = A[id - 1] + B[id - 1];\n\
}\n\
\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_vector_mulToC)(\n\
        NumN n, MATH21_KERNEL_GLOBAL const NumReal *A, MATH21_KERNEL_GLOBAL const NumReal *B,\n\
        MATH21_KERNEL_GLOBAL NumReal *C MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
    if (id <= n) C[id - 1] = A[id - 1] * B[id - 1];\n\
}\n\
\n\
// a special kind of sub\n\
// x is sub-tensor of y\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_1(math21_template_vector_broadcast_in_dn)(\n\
        NumN n, MATH21_KERNEL_GLOBAL const NumReal *x, MATH21_KERNEL_GLOBAL NumReal *y,\n\
        NumN dims_x, MATH21_KERNEL_GLOBAL const NumN *dx,\n\
        NumN dims_y, MATH21_KERNEL_GLOBAL const NumN *dy MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x -= 1;\n\
    y -= 1;\n\
    dx -= 1;\n\
    dy -= 1;\n\
#endif\n\
\n\
    if (id > n) return;\n\
    NumN _indexx[MATH21_KERNEL_ARRAY_MAX_LENGTH], _indexy[MATH21_KERNEL_ARRAY_MAX_LENGTH], ix, iy;\n\
    NumN *indexx = math21_device_pointer_NumN_decrease_one(_indexx);\n\
    NumN *indexy = math21_device_pointer_NumN_decrease_one(_indexy);\n\
\n\
    iy = id;\n\
    math21_device_index_1d_to_nd(indexy, iy, dy, dims_y);\n\
    math21_device_broadcast_index_to_original_brackets(indexy, dx, indexx, dims_x);\n\
    math21_device_index_nd_to_1d(indexx, &ix, dx, dims_x);\n\
    y[iy] = x[ix];\n\
\n\
}\n\
\n\
// todo: optimize\n\
// alpha_t = alpha * sqrt(1 - beta2^t) / (1 - beta1^t),\n\
// eps_hat, see tensorflow/python/training/adam.py\n\
MATH21_KERNEL_TEMPLATE_HEADER(NumReal)\n\
MATH21_KERNEL_EXPORT void\n\
MATH21_MAKE_KERNEL_NAME_1(math21_template_optimization_adam_update_part_2)(NumN x_size, MATH21_KERNEL_GLOBAL NumReal *x,\n\
                                                                           MATH21_KERNEL_GLOBAL const NumReal *m,\n\
                                                                           MATH21_KERNEL_GLOBAL const NumReal *v,\n\
                                                                           NumReal beta1, NumReal beta2,\n\
                                                                           NumReal alpha, NumReal eps,\n\
                                                                           NumN t MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x -= 1;\n\
    m -= 1;\n\
    v -= 1;\n\
#endif\n\
\n\
    if (id > x_size) return;\n\
    // compute bias-corrected first moment estimate\n\
//    NumReal mhat = m[id] / (1.f - powf(beta1, t));\n\
    NumReal mhat = m[id] / (1.f - pow(beta1, t));\n\
    // compute bias-corrected second raw moment estimate\n\
    NumReal vhat = v[id] / (1.f - pow(beta2, t));\n\
\n\
    // update\n\
    // x = x - alpha * m / (sqrt(v) + eps)\n\
//    x[id] = x[id] + alpha * mhat / (sqrtf(vhat) + eps);\n\
    x[id] = x[id] + alpha * mhat / (sqrt(vhat) + eps);\n\
}\n\
"; 
std::string generic_01_vector_set = "\n\
#include <math21_kernels.h>\n\
\n\
//#define MATH21_IS_FROM_CPU\n\
\n\
#if !defined(MATH21_IS_FROM_CPU)\n\
#if defined(MATH21_FLAG_USE_CUDA)\n\
#define MATH21_IS_FROM_CUDA\n\
#elif defined(MATH21_FLAG_USE_OPENCL)\n\
#define MATH21_IS_FROM_OPENCL\n\
#endif\n\
#endif\n\
\n\
#if defined(MATH21_IS_FROM_CPU)\n\
#define MATH21_MAKE_KERNEL_NAME_1(X) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_2(X) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_3(X) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_PAIR(X, Y) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_TWO(X, Y1, Y2) MATH21_MACRO_CAT_2(X, cpu_kernel)\n\
#define MATH21_KERNEL_GET_ID()\n\
#define MATH21_KERNEL_TEMPLATE_HEADER(X) template<typename X>\n\
#define MATH21_KERNEL_TEMPLATE_THE_HEADER_2(X1, X2) template<typename X1, typename X2>\n\
#define MATH21_KERNEL_EXPORT\n\
#define MATH21_KERNEL_GLOBAL\n\
#define MATH21_KERNEL_INPUT_ID , NumN id\n\
#define MATH21_KERNEL_INPUT_OFFSETS_XY\n\
#define MATH21_DEVICE_F_MIN_LIKE_PTR math21_type_f_min_like f_shrink_min_like_ptr,\n\
#define MATH21_DEVICE_F_ARGMIN_LIKE_PTR math21_type_f_argmin_like f_shrink_argmin_like_ptr,\n\
#define MATH21_DEVICE_F_ADD_LIKE_PTR math21_type_f_add_like f_bc_add_like_ptr,\n\
#define MATH21_DEVICE_F_SIN_LIKE_PTR math21_type_f_sin_like f_bc_sin_like_ptr,\n\
#define MATH21_DEVICE_F_KX_LIKE_PTR math21_type_f_kx_like f_kx_like_ptr,\n\
#define MATH21_DEVICE_F_ADDTO_LIKE_PTR math21_type_f_addto_like f_addto_like_ptr,\n\
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y) X Y,\n\
\n\
#elif defined(MATH21_IS_FROM_CUDA)\n\
#define MATH21_MAKE_KERNEL_NAME_1(X) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_2(X) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_3(X) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_PAIR(X, Y) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_MAKE_KERNEL_NAME_TWO(X, Y1, Y2) MATH21_MACRO_CAT_2(X, cuda_kernel)\n\
#define MATH21_KERNEL_GET_ID() int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; id +=1;\n\
#define MATH21_KERNEL_TEMPLATE_HEADER(X) template<typename X>\n\
#define MATH21_KERNEL_TEMPLATE_THE_HEADER_2(X1, X2) template<typename X1, typename X2>\n\
#define MATH21_KERNEL_EXPORT __global__\n\
#define MATH21_KERNEL_GLOBAL\n\
#define MATH21_KERNEL_INPUT_ID\n\
#define MATH21_KERNEL_INPUT_OFFSETS_XY\n\
#define MATH21_DEVICE_F_MIN_LIKE_PTR math21_type_f_min_like f_shrink_min_like_ptr,\n\
#define MATH21_DEVICE_F_ARGMIN_LIKE_PTR math21_type_f_argmin_like f_shrink_argmin_like_ptr,\n\
#define MATH21_DEVICE_F_ADD_LIKE_PTR math21_type_f_add_like f_bc_add_like_ptr,\n\
#define MATH21_DEVICE_F_SIN_LIKE_PTR math21_type_f_sin_like f_bc_sin_like_ptr,\n\
#define MATH21_DEVICE_F_KX_LIKE_PTR math21_type_f_kx_like f_kx_like_ptr,\n\
#define MATH21_DEVICE_F_ADDTO_LIKE_PTR math21_type_f_addto_like f_addto_like_ptr,\n\
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y) X Y,\n\
\n\
#elif defined(MATH21_IS_FROM_OPENCL)\n\
#define MATH21_MAKE_KERNEL_NAME_1(X) MATH21_OPENCL_TEMPLATE_3(X, opencl_kernel, NumReal)\n\
#define MATH21_MAKE_KERNEL_NAME_2(X) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, NumReal, f_shrink_min_like_ptr)\n\
#define MATH21_MAKE_KERNEL_NAME_3(X) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, NumReal, f_shrink_argmin_like_ptr)\n\
#define MATH21_MAKE_KERNEL_NAME_PAIR(X, Y) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, NumReal, Y)\n\
#define MATH21_MAKE_KERNEL_NAME_TWO(X, Y1, Y2) MATH21_OPENCL_TEMPLATE_4(X, opencl_kernel, Y1, Y2)\n\
#define MATH21_KERNEL_GET_ID() size_t global_x = get_global_id(0); size_t global_y = get_global_id(1); size_t global_z = get_global_id(2); size_t id = global_z * get_global_size(0) * get_global_size(1) + global_y * get_global_size(0) + global_x; id +=1;\n\
#define MATH21_KERNEL_TEMPLATE_HEADER(X)\n\
#define MATH21_KERNEL_TEMPLATE_THE_HEADER_2(X1, X2)\n\
#define MATH21_KERNEL_EXPORT __kernel\n\
#define MATH21_KERNEL_GLOBAL __global\n\
#define MATH21_KERNEL_INPUT_ID\n\
#define MATH21_KERNEL_INPUT_OFFSETS_XY , NumN offset_x, NumN offset_y\n\
#define MATH21_DEVICE_F_MIN_LIKE_PTR\n\
#define MATH21_DEVICE_F_ARGMIN_LIKE_PTR\n\
#define MATH21_DEVICE_F_ADD_LIKE_PTR\n\
#define MATH21_DEVICE_F_SIN_LIKE_PTR\n\
#define MATH21_DEVICE_F_KX_LIKE_PTR\n\
#define MATH21_DEVICE_F_ADDTO_LIKE_PTR\n\
#define MATH21_DEVICE_MAKE_F_LIKE_PTR(X, Y)\n\
\n\
#else\n\
#error MATH21_IS_FROM_NONE\n\
#endif\n\
\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
#include <math21_opencl_device_code.h>\n\
#endif\n\
\n\
// y = x\n\
MATH21_KERNEL_TEMPLATE_THE_HEADER_2(NumType1, NumType2)\n\
MATH21_KERNEL_EXPORT void MATH21_MAKE_KERNEL_NAME_TWO(math21_template_vector_set_by_vector, NumType1, NumType2)(\n\
        NumN n, MATH21_KERNEL_GLOBAL const NumType1 *x, NumN stride_x, MATH21_KERNEL_GLOBAL NumType2 *y,\n\
        NumN stride_y MATH21_KERNEL_INPUT_OFFSETS_XY MATH21_KERNEL_INPUT_ID) {\n\
    MATH21_KERNEL_GET_ID();\n\
#if defined(MATH21_IS_FROM_OPENCL)\n\
    x += offset_x;\n\
    y += offset_y;\n\
    x -= 1;\n\
    y -= 1;\n\
#endif\n\
    if (id > n) return;\n\
    y[(id - 1) * stride_y + 1] = x[(id - 1) * stride_x + 1];\n\
}\n\
"; 
std::string matrix_opencl = "\n\
// Todo: use similar kernels in https://cnugteren.github.io/tutorial/pages/page4.html\n\
\n\
#define MATH21_OPENCL_BLOCK_SIZE 512\n\
\n\
// error\n\
// C = k1*A*B + k2*C\n\
__kernel void\n\
math21_matrix_multiply_k1AB_add_k2C_similar_nn_v2_opencl_kernel(int nr_C, int nc_C, int n_common, float k1,\n\
                                                                __global const float *A, int lda,\n\
                                                                __global const float *B, int ldb,\n\
                                                                float k2,\n\
                                                                __global float *C, int ldc) {\n\
    int size = nr_C * nc_C;\n\
    int id = get_global_id(0);\n\
    if (id >= size) return;\n\
    int j = id % nc_C;\n\
    id /= nc_C;\n\
    int i = id % nr_C;\n\
\n\
    __local float part[MATH21_OPENCL_BLOCK_SIZE];\n\
    int p = get_global_id(1);\n\
\n\
    int t;\n\
    float sum = 0.0f;\n\
    for (t = 0; t < n_common; t += MATH21_OPENCL_BLOCK_SIZE) {\n\
        int k = p + t;\n\
        sum += (p + t < n_common) ? A[i * lda + k] * B[k * ldb + j] : 0;\n\
    }\n\
\n\
    part[p] = sum;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
    if (p == 0) {\n\
        int index;\n\
        sum = 0;\n\
        for (index = 0; index < MATH21_OPENCL_BLOCK_SIZE; ++index) sum += part[index];\n\
    }\n\
\n\
    C[i * ldc + j] = k1 * sum + k2 * C[i * ldc + j];\n\
}\n\
\n\
// error\n\
// C = k1*A*B.t + k2*C\n\
__kernel void\n\
math21_matrix_multiply_k1AB_add_k2C_similar_nt_v2_opencl_kernel(int nr_C, int nc_C, int n_common, float k1,\n\
                                                                __global const float *A, int lda,\n\
                                                                __global const float *B, int ldb,\n\
                                                                float k2,\n\
                                                                __global float *C, int ldc) {\n\
    int size = nr_C * nc_C;\n\
    int id = get_global_id(0);\n\
    if (id >= size) return;\n\
    int j = id % nc_C;\n\
    id /= nc_C;\n\
    int i = id % nr_C;\n\
\n\
    __local float part[MATH21_OPENCL_BLOCK_SIZE];\n\
    int p = get_global_id(1);\n\
\n\
    int t;\n\
    float sum = 0.0f;\n\
    for (t = 0; t < n_common; t += MATH21_OPENCL_BLOCK_SIZE) {\n\
        int k = p + t;\n\
        sum += (p + t < n_common) ? A[i * lda + k] * B[j * ldb + k] : 0;\n\
    }\n\
\n\
    part[p] = sum;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
    if (p == 0) {\n\
        int index;\n\
        sum = 0;\n\
        for (index = 0; index < MATH21_OPENCL_BLOCK_SIZE; ++index) sum += part[index];\n\
    }\n\
\n\
    C[i * ldc + j] = k1 * sum + k2 * C[i * ldc + j];\n\
}\n\
\n\
// error\n\
// C = k1*A.t*B + k2*C\n\
__kernel void\n\
math21_matrix_multiply_k1AB_add_k2C_similar_tn_v2_opencl_kernel(int nr_C, int nc_C, int n_common, float k1,\n\
                                                                __global const float *A, int lda,\n\
                                                                __global const float *B, int ldb,\n\
                                                                float k2,\n\
                                                                __global float *C, int ldc) {\n\
    int size = nr_C * nc_C;\n\
    int id = get_global_id(0);\n\
    if (id >= size) return;\n\
    int j = id % nc_C;\n\
    id /= nc_C;\n\
    int i = id % nr_C;\n\
\n\
    __local float part[MATH21_OPENCL_BLOCK_SIZE];\n\
    int p = get_global_id(1);\n\
\n\
    int t;\n\
    float sum = 0.0f;\n\
    for (t = 0; t < n_common; t += MATH21_OPENCL_BLOCK_SIZE) {\n\
        int k = p + t;\n\
        sum += (p + t < n_common) ? A[k * lda + i] * B[k * ldb + j] : 0;\n\
    }\n\
\n\
    part[p] = sum;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
    if (p == 0) {\n\
        int index;\n\
        sum = 0;\n\
        for (index = 0; index < MATH21_OPENCL_BLOCK_SIZE; ++index) sum += part[index];\n\
    }\n\
\n\
    C[i * ldc + j] = k1 * sum + k2 * C[i * ldc + j];\n\
}\n\
\n\
// error\n\
// C = k1*A.t*B.t + k2*C\n\
__kernel void\n\
math21_matrix_multiply_k1AB_add_k2C_similar_tt_v2_opencl_kernel(int nr_C, int nc_C, int n_common, float k1,\n\
                                                                __global const float *A, int lda,\n\
                                                                __global const float *B, int ldb,\n\
                                                                float k2,\n\
                                                                __global float *C, int ldc) {\n\
    int size = nr_C * nc_C;\n\
    int id = get_global_id(0);\n\
    if (id >= size) return;\n\
    int j = id % nc_C;\n\
    id /= nc_C;\n\
    int i = id % nr_C;\n\
\n\
    __local float part[MATH21_OPENCL_BLOCK_SIZE];\n\
    int p = get_global_id(1);\n\
\n\
    int t;\n\
    float sum = 0.0f;\n\
    for (t = 0; t < n_common; t += MATH21_OPENCL_BLOCK_SIZE) {\n\
        int k = p + t;\n\
        sum += (p + t < n_common) ? A[k * lda + i] * B[j * ldb + k] : 0;\n\
    }\n\
\n\
    part[p] = sum;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
    if (p == 0) {\n\
        int index;\n\
        sum = 0;\n\
        for (index = 0; index < MATH21_OPENCL_BLOCK_SIZE; ++index) sum += part[index];\n\
    }\n\
\n\
    C[i * ldc + j] = k1 * sum + k2 * C[i * ldc + j];\n\
}\n\
\n\
// C = k1*A*B + k2*C\n\
__kernel void\n\
math21_matrix_multiply_k1AB_add_k2C_similar_nn_naive_opencl_kernel(int nr_C, int nc_C, int n_common, float k1,\n\
                                                                   __global const float *A, int lda,\n\
                                                                   __global const float *B, int ldb,\n\
                                                                   float k2,\n\
                                                                   __global float *C, int ldc) {\n\
    int size = nr_C * nc_C;\n\
    // z*dim(y) * dim(x) + y * dim(x) + x\n\
    // x = blockIdx.x * blockDim.x + threadIdx.x\n\
    // y = blockIdx.y * blockDim.y + threadIdx.y\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
             get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (id >= size) return;\n\
    int j = id % nc_C;\n\
    id /= nc_C;\n\
    int i = id % nr_C;\n\
\n\
    int k;\n\
    float sum = 0.0f;\n\
    for (k = 0; k < n_common; ++k) {\n\
        sum += A[i * lda + k] * B[k * ldb + j];\n\
    }\n\
    C[i * ldc + j] = k1 * sum + k2 * C[i * ldc + j];\n\
}\n\
\n\
// C = k1*A*B.t + k2*C\n\
__kernel void\n\
math21_matrix_multiply_k1AB_add_k2C_similar_nt_naive_opencl_kernel(int nr_C, int nc_C, int n_common, float k1,\n\
                                                                   __global const float *A, int lda,\n\
                                                                   __global const float *B, int ldb,\n\
                                                                   float k2,\n\
                                                                   __global float *C, int ldc) {\n\
    int size = nr_C * nc_C;\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
             get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (id >= size) return;\n\
    int j = id % nc_C;\n\
    id /= nc_C;\n\
    int i = id % nr_C;\n\
\n\
    int k;\n\
    float sum = 0.0f;\n\
    for (k = 0; k < n_common; ++k) {\n\
        sum += A[i * lda + k] * B[j * ldb + k];\n\
    }\n\
    C[i * ldc + j] = k1 * sum + k2 * C[i * ldc + j];\n\
}\n\
\n\
// C = k1*A.t*B + k2*C\n\
__kernel void\n\
math21_matrix_multiply_k1AB_add_k2C_similar_tn_naive_opencl_kernel(int nr_C, int nc_C, int n_common, float k1,\n\
                                                                   __global const float *A, int lda,\n\
                                                                   __global const float *B, int ldb,\n\
                                                                   float k2,\n\
                                                                   __global float *C, int ldc) {\n\
    int size = nr_C * nc_C;\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
             get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (id >= size) return;\n\
    int j = id % nc_C;\n\
    id /= nc_C;\n\
    int i = id % nr_C;\n\
\n\
    int k;\n\
    float sum = 0.0f;\n\
    for (k = 0; k < n_common; ++k) {\n\
        sum += A[k * lda + i] * B[k * ldb + j];\n\
    }\n\
    C[i * ldc + j] = k1 * sum + k2 * C[i * ldc + j];\n\
}\n\
\n\
// C = k1*A.t*B.t + k2*C\n\
__kernel void\n\
math21_matrix_multiply_k1AB_add_k2C_similar_tt_naive_opencl_kernel(int nr_C, int nc_C, int n_common, float k1,\n\
                                                                   __global const float *A, int lda,\n\
                                                                   __global const float *B, int ldb,\n\
                                                                   float k2,\n\
                                                                   __global float *C, int ldc) {\n\
    int size = nr_C * nc_C;\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
             get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (id >= size) return;\n\
    int j = id % nc_C;\n\
    id /= nc_C;\n\
    int i = id % nr_C;\n\
\n\
    int k;\n\
    float sum = 0.0f;\n\
    for (k = 0; k < n_common; ++k) {\n\
        sum += A[k * lda + i] * B[j * ldb + k];\n\
    }\n\
    C[i * ldc + j] = k1 * sum + k2 * C[i * ldc + j];\n\
}\n\
"; 
std::string vector_opencl_02 = "\n\
#define MATH21_OPENCL_BLOCK_SIZE 512\n\
\n\
__kernel void math21_vector_loss_l1_opencl_kernel(int n,\n\
                                                  __global const float *x, __global const float *t,\n\
                                                  __global float *dx,\n\
                                                  __global float *error) {\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i < n) {\n\
        float diff = t[i] - x[i];\n\
        error[i] = fabs(diff);\n\
        dx[i] = diff > 0 ? 1 : -1;\n\
    }\n\
}\n\
\n\
__kernel void\n\
math21_vector_loss_l2_opencl_kernel(int n, __global const float *x, __global const float *t, __global float *dx,\n\
                                    __global float *error) {\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i < n) {\n\
        float diff = t[i] - x[i];\n\
        error[i] = diff * diff;\n\
        dx[i] = diff;\n\
    }\n\
}\n\
\n\
__kernel void\n\
math21_vector_loss_smooth_l1_opencl_kernel(int n, __global const float *x, __global const float *t, __global float *dx,\n\
                                           __global float *error) {\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i < n) {\n\
        float diff = t[i] - x[i];\n\
        float abs_val = fabs(diff);\n\
        if (abs_val < 1) {\n\
            error[i] = diff * diff;\n\
            dx[i] = diff;\n\
        } else {\n\
            error[i] = 2 * abs_val - 1;\n\
            dx[i] = (diff > 0) ? 1 : -1;\n\
        }\n\
    }\n\
}\n\
\n\
__kernel void math21_vector_zero_by_thresh_opencl_kernel(int n, __global float *x, int stride_x, float thresh) {\n\
    size_t global_x = get_global_id(0);\n\
    size_t global_y = get_global_id(1);\n\
    size_t global_z = get_global_id(2);\n\
    size_t i = global_z * get_global_size(0) * get_global_size(1)\n\
               + global_y * get_global_size(0) + global_x;\n\
    if (i < n) {\n\
        if (fabs(x[i * stride_x]) < thresh) x[i * stride_x] = 0;\n\
    }\n\
}\n\
"; 
std::string vector_opencl_01 = "\n\
#define MATH21_OPENCL_BLOCK_SIZE 512\n\
\n\
__kernel void math21_vector_mean_fast_opencl_kernel(__global const float *x, int batch, int filters, int spatial,\n\
                                                    __global float *mean) {\n\
    __local float part[MATH21_OPENCL_BLOCK_SIZE];\n\
    int id = get_global_id(1);\n\
    int filter = get_global_id(0);\n\
    float sum = 0;\n\
    int i, j;\n\
    for (j = 0; j < batch; ++j) {\n\
        for (i = 0; i < spatial; i += MATH21_OPENCL_BLOCK_SIZE) {\n\
            int index = j * spatial * filters + filter * spatial + i + id;\n\
            sum += (i + id < spatial) ? x[index] : 0;\n\
        }\n\
    }\n\
    part[id] = sum;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
    if (id == 0) {\n\
        mean[filter] = 0;\n\
        for (i = 0; i < MATH21_OPENCL_BLOCK_SIZE; ++i) {\n\
            mean[filter] += part[i];\n\
        }\n\
        mean[filter] /= spatial * batch;\n\
    }\n\
}\n\
\n\
__kernel void\n\
math21_vector_mean_opencl_kernel(__global const float *x, int batch, int filters, int spatial, __global float *mean) {\n\
    float scale = 1.f / (batch * spatial);\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i >= filters) return;\n\
    int j, k;\n\
    mean[i] = 0;\n\
    for (j = 0; j < batch; ++j) {\n\
        for (k = 0; k < spatial; ++k) {\n\
            int index = j * filters * spatial + i * spatial + k;\n\
            mean[i] += x[index];\n\
        }\n\
    }\n\
    mean[i] *= scale;\n\
}\n\
\n\
__kernel void\n\
math21_vector_variance_fast_opencl_kernel(__global const float *x, __global float *mean, int batch, int filters,\n\
                                          int spatial, __global float *variance) {\n\
    __local float part[MATH21_OPENCL_BLOCK_SIZE];\n\
    int id = get_global_id(1);\n\
    int filter = get_global_id(0);\n\
    float sum = 0;\n\
    int i, j;\n\
    for (j = 0; j < batch; ++j) {\n\
        for (i = 0; i < spatial; i += MATH21_OPENCL_BLOCK_SIZE) {\n\
            int index = j * spatial * filters + filter * spatial + i + id;\n\
\n\
            sum += (i + id < spatial) ? pow((x[index] - mean[filter]), 2) : 0;\n\
        }\n\
    }\n\
    part[id] = sum;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
    if (id == 0) {\n\
        variance[filter] = 0;\n\
        for (i = 0; i < MATH21_OPENCL_BLOCK_SIZE; ++i) {\n\
            variance[filter] += part[i];\n\
        }\n\
        variance[filter] /= (spatial * batch - 1);\n\
    }\n\
}\n\
\n\
__kernel void\n\
math21_vector_variance_opencl_kernel(__global const float *x, __global float *mean, int batch, int filters, int spatial,\n\
                                     __global float *variance) {\n\
    float scale = 1.f / (batch * spatial - 1);\n\
    int j, k;\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i >= filters) return;\n\
    variance[i] = 0;\n\
    for (j = 0; j < batch; ++j) {\n\
        for (k = 0; k < spatial; ++k) {\n\
            int index = j * filters * spatial + i * spatial + k;\n\
            variance[i] += pow((x[index] - mean[i]), 2);\n\
        }\n\
    }\n\
    variance[i] *= scale;\n\
}\n\
\n\
__kernel void\n\
math21_vector_assign_from_vector_with_offset_opencl_kernel(int N, __global const float *X, int OFFX, int INCX,\n\
                                                           __global float *Y, int OFFY, int INCY) {\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i < N) Y[i * INCY + OFFY] = X[i * INCX + OFFX];\n\
}\n\
\n\
__kernel void\n\
math21_vector_assign_from_vector_N8_opencl_kernel(int N, __global const unsigned char *X, __global unsigned char *Y) {\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i < N) Y[i] = X[i];\n\
}\n\
\n\
__kernel void math21_vector_kx_opencl_kernel(int N, float ALPHA, __global float *X, int INCX) {\n\
    size_t global_x = get_global_id(0);\n\
    size_t global_y = get_global_id(1);\n\
    size_t global_z = get_global_id(2);\n\
    size_t i = global_z * get_global_size(0) * get_global_size(1)\n\
               + global_y * get_global_size(0) + global_x;\n\
    if (i < N) X[i * INCX] *= ALPHA;\n\
}\n\
\n\
__kernel void math21_vector_k_add_x_opencl_kernel(int N, float ALPHA, __global float *X, int INCX) {\n\
    size_t global_x = get_global_id(0);\n\
    size_t global_y = get_global_id(1);\n\
    size_t global_z = get_global_id(2);\n\
    size_t i = global_z * get_global_size(0) * get_global_size(1)\n\
               + global_y * get_global_size(0) + global_x;\n\
    if (i < N) X[i * INCX] += ALPHA;\n\
}\n\
\n\
__kernel void\n\
math21_vector_kx_add_y_with_offset_opencl_kernel(int N, float ALPHA, __global const float *X, int OFFX, int INCX,\n\
                                                 __global float *Y, int OFFY, int INCY) {\n\
    size_t global_x = get_global_id(0);\n\
    size_t global_y = get_global_id(1);\n\
    size_t global_z = get_global_id(2);\n\
    size_t i = global_z * get_global_size(0) * get_global_size(1)\n\
               + global_y * get_global_size(0) + global_x;\n\
    if (i < N) Y[OFFY + i * INCY] += ALPHA * X[OFFX + i * INCX];\n\
}\n\
\n\
__kernel void\n\
math21_vector_normalize_opencl_kernel(int N, __global float *x, __global float *mean, __global float *variance,\n\
                                      int batch, int filters, int spatial) {\n\
    int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
                get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (index >= N) return;\n\
    int f = (index / spatial) % filters;\n\
    x[index] = (x[index] - mean[f]) / (sqrt(variance[f] + .00001f));\n\
}\n\
\n\
__kernel void\n\
math21_vector_kx_with_in_class_opencl_kernel(__global float *output, __global float *biases, int n, int size) {\n\
    size_t global_x = get_global_id(0);\n\
    size_t global_y = get_global_id(1);\n\
    size_t global_z = get_global_id(2);\n\
    size_t filter = global_z - get_global_offset(2);\n\
    size_t x_dim_size = get_global_size(0);\n\
    size_t offset = x_dim_size * global_y + global_x;\n\
    if (offset < size) output[global_z * size + offset] *= biases[filter];\n\
}\n\
\n\
__kernel void\n\
math21_vector_x_add_b_with_in_class_opencl_kernel(__global float *output, __global const float *biases, int batch,\n\
                                                  int n, int size) {\n\
    int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
                get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (index >= n * size * batch) return;\n\
    int i = index % size;\n\
    index /= size;\n\
    int j = index % n;\n\
    index /= n;\n\
    int k = index;\n\
    output[(k * n + j) * size + i] += biases[j];\n\
}\n\
\n\
__kernel void\n\
math21_vector_sum_with_in_class_conn_opencl_kernel(__global float *bias_updates, __global float *delta, int batch,\n\
                                                   int n) {\n\
    int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
                get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (index >= n) return;\n\
    int b;\n\
    float sum = 0;\n\
    for (b = 0; b < batch; ++b) {\n\
        int i = b * n + index;\n\
        sum += delta[i];\n\
    }\n\
    bias_updates[index] += sum;\n\
}\n\
\n\
__kernel void\n\
math21_vector_sum_with_in_class_opencl_kernel(__global float *bias_updates, __global float *delta, int batch, int n,\n\
                                              int size) {\n\
    __local float part[MATH21_OPENCL_BLOCK_SIZE];\n\
    int i, b;\n\
    int filter = get_global_id(0);\n\
    int p = get_global_id(1);\n\
    float sum = 0;\n\
    for (b = 0; b < batch; ++b) {\n\
        for (i = 0; i < size; i += MATH21_OPENCL_BLOCK_SIZE) {\n\
            int index = p + i + size * (filter + n * b);\n\
            sum += (p + i < size) ? delta[index] : 0;\n\
        }\n\
    }\n\
    part[p] = sum;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
    if (p == 0) {\n\
        for (i = 0; i < MATH21_OPENCL_BLOCK_SIZE; ++i) bias_updates[filter] += part[i];\n\
    }\n\
}\n\
\n\
__kernel void\n\
math21_vector_sum_SchurProduct_with_in_class_opencl_kernel(__global float *x_norm, __global float *delta, int batch,\n\
                                                           int n, int size, __global float *scale_updates) {\n\
    __local float part[MATH21_OPENCL_BLOCK_SIZE];\n\
    int i, b;\n\
    int filter = get_global_id(0);\n\
    int p = get_global_id(1);\n\
    float sum = 0;\n\
    for (b = 0; b < batch; ++b) {\n\
        for (i = 0; i < size; i += MATH21_OPENCL_BLOCK_SIZE) {\n\
            int index = p + i + size * (filter + n * b);\n\
            sum += (p + i < size) ? delta[index] * x_norm[index] : 0;\n\
        }\n\
    }\n\
    part[p] = sum;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
    if (p == 0) {\n\
        for (i = 0; i < MATH21_OPENCL_BLOCK_SIZE; ++i) scale_updates[filter] += part[i];\n\
    }\n\
}\n\
\n\
__kernel void math21_vector_set_opencl_kernel(int N, float ALPHA, __global float *X, int INCX) {\n\
    size_t global_x = get_global_id(0);\n\
    size_t global_y = get_global_id(1);\n\
    size_t global_z = get_global_id(2);\n\
    size_t i = global_z * get_global_size(0) * get_global_size(1)\n\
               + global_y * get_global_size(0) + global_x;\n\
    if (i < N) X[i * INCX] = ALPHA;\n\
}\n\
\n\
__kernel void math21_vector_set_int_opencl_kernel(int N, int ALPHA, __global int *X, int INCX) {\n\
    size_t global_x = get_global_id(0);\n\
    size_t global_y = get_global_id(1);\n\
    size_t global_z = get_global_id(2);\n\
    size_t i = global_z * get_global_size(0) * get_global_size(1)\n\
               + global_y * get_global_size(0) + global_x;\n\
    if (i < N) X[i * INCX] = ALPHA;\n\
}\n\
\n\
__kernel void math21_vector_feature2d_add_2_opencl_kernel(int size, int mini_batch_size,\n\
                                                          int nch, int nr, int nc,\n\
                                                          float kx, __global const float *X, int nch_X, int nr_X,\n\
                                                          int nc_X,\n\
                                                          float stride_r_x, float stride_c_x,\n\
                                                          float ky, __global float *Y, int nch_Y, int nr_Y, int nc_Y,\n\
                                                          float stride_r_y, float stride_c_y) {\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
             get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (id >= size) return;\n\
    int ic = id % nc;\n\
    id /= nc;\n\
    int ir = id % nr;\n\
    id /= nr;\n\
    int ich = id % nch;\n\
    id /= nch;\n\
    int imb = id % mini_batch_size;\n\
\n\
    // X(imb, ich, ir*stride_r_x, ic*stride_c_x)\n\
    int index_X = ((imb * nch_X + ich) * nr_X + (int) (ir * stride_r_x)) * nc_X + (int) (ic * stride_c_x);\n\
    // Y(imb, ich, ir*stride_r_y, ic*stride_c_y)\n\
    int index_Y = ((imb * nch_Y + ich) * nr_Y + (int) (ir * stride_r_y)) * nc_Y + (int) (ic * stride_c_y);\n\
    Y[index_Y] = kx * X[index_X] + ky * Y[index_Y];\n\
}\n\
\n\
__kernel void math21_vector_feature2d_add_3_opencl_kernel(int size, int mini_batch_size,\n\
                                                          int nch, int nr, int nc,\n\
                                                          float kx, __global const float *X, int nch_X, int nr_X,\n\
                                                          int nc_X,\n\
                                                          float stride_r_x, float stride_c_x,\n\
                                                          float kx2, __global const float *X2, int nch_X2, int nr_X2,\n\
                                                          int nc_X2,\n\
                                                          float stride_r_x2, float stride_c_x2,\n\
                                                          float ky, __global float *Y, int nch_Y, int nr_Y, int nc_Y,\n\
                                                          float stride_r_y, float stride_c_y) {\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
             get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (id >= size) return;\n\
    int ic = id % nc;\n\
    id /= nc;\n\
    int ir = id % nr;\n\
    id /= nr;\n\
    int ich = id % nch;\n\
    id /= nch;\n\
    int imb = id % mini_batch_size;\n\
\n\
    // X(imb, ich, ir*stride_r_x, ic*stride_c_x)\n\
    int index_X = ((imb * nch_X + ich) * nr_X + (int) (ir * stride_r_x)) * nc_X + (int) (ic * stride_c_x);\n\
    // X2(imb, ich, ir*stride_r_x2, ic*stride_c_x2)\n\
    int index_X2 = ((imb * nch_X2 + ich) * nr_X2 + (int) (ir * stride_r_x2)) * nc_X2 + (int) (ic * stride_c_x2);\n\
    // Y(imb, ich, ir*stride_r_y, ic*stride_c_y)\n\
    int index_Y = ((imb * nch_Y + ich) * nr_Y + (int) (ir * stride_r_y)) * nc_Y + (int) (ic * stride_c_y);\n\
    Y[index_Y] = kx * X[index_X] + kx2 * X2[index_X2] + ky * Y[index_Y];\n\
}\n\
\n\
// X shape <= Y shape\n\
__kernel void math21_vector_feature2d_sumdownsample_opencl_kernel(int n, int mini_batch_size,\n\
                                                                  __global float *X, int nch_X, int nr_X, int nc_X,\n\
                                                                  int stride_X, float k, __global const float *Y) {\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i >= n) return;\n\
\n\
    int index_X = i;\n\
    int ic_X = i % nc_X;\n\
    i = i / nc_X;\n\
    int ir_X = i % nr_X;\n\
    i = i / nr_X;\n\
    int ich_X = i % nch_X;\n\
    i = i / nch_X;\n\
    int imb_X = i % mini_batch_size;\n\
\n\
    int ic_Y_abs = ic_X * stride_X;\n\
    int ir_Y_abs = ir_X * stride_X;\n\
    int ich_Y = ich_X;\n\
    int imb_Y = imb_X;\n\
\n\
    int nc_Y = nc_X * stride_X;\n\
    int nr_Y = nr_X * stride_X;\n\
    int nch_Y = nch_X;\n\
\n\
    int ksize = stride_X;\n\
    for (int ir_K = 0; ir_K < ksize; ++ir_K) {\n\
        for (int ic_K = 0; ic_K < ksize; ++ic_K) {\n\
            int ir_Y = ir_Y_abs + ir_K;\n\
            int ic_Y = ic_Y_abs + ic_K;\n\
            int index_Y = ((imb_Y * nch_Y + ich_Y) * nr_Y + ir_Y) * nc_Y + ic_Y;\n\
            X[index_X] += k * Y[index_Y];\n\
        }\n\
    }\n\
}\n\
\n\
// only upsample\n\
__kernel void math21_vector_feature2d_upsample_opencl_kernel(int n, int mini_batch_size,\n\
                                                             __global float *X, int nch_X, int nr_X, int nc_X,\n\
                                                             int stride_X, float k, __global float *Y) {\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i >= n) return;\n\
    int index_Y = i;\n\
    int ic_Y = i % (nc_X * stride_X);\n\
    i = i / (nc_X * stride_X);\n\
    int ir_Y = i % (nr_X * stride_X);\n\
    i = i / (nr_X * stride_X);\n\
    int ich_Y = i % nch_X;\n\
    i = i / nch_X;\n\
    int imb_Y = i % mini_batch_size;\n\
\n\
    int ic_X = ic_Y / stride_X;\n\
    int ir_X = ir_Y / stride_X;\n\
    int ich_X = ich_Y;\n\
\n\
    int index_X = imb_Y * nch_X * nr_X * nc_X + ich_X * nr_X * nc_X + ir_X * nc_X + ic_X;\n\
\n\
    Y[index_Y] += k * X[index_X];\n\
}\n\
\n\
__kernel void math21_vector_clip_opencl_kernel(int n, float k, __global float *x, int stride_x) {\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i < n) x[i * stride_x] = fmin(k, fmax(-k, x[i * stride_x]));\n\
//    if (i < n) x[i * stride_x] = fminf(k, fmaxf(-k, x[i * stride_x]));\n\
}\n\
\n\
__kernel void\n\
math21_vector_xy_opencl_kernel(int n, __global const float *x, int stride_x, __global float *y, int stride_y) {\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i < n) y[i * stride_y] *= x[i * stride_x];\n\
}\n\
\n\
__kernel void\n\
math21_vector_assign_by_mask_opencl_kernel(int n, __global float *x, float mask_num, __global const float *mask,\n\
                                           float val) {\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i < n && mask[i] == mask_num) x[i] = val;\n\
}\n\
\n\
__kernel void\n\
math21_vector_kx_by_mask_opencl_kernel(int n, float k, __global float *x, __global const float *mask, float mask_num) {\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i < n && mask[i] == mask_num) x[i] *= k;\n\
}\n\
"; 
std::string activation_opencl = "\n\
#define expf(X) exp(X)\n\
#define floorf(X) floor(X)\n\
\n\
typedef enum{\n\
    MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC, MATH21_FUNCTION_ACTIVATION_TYPE_RELU, MATH21_FUNCTION_ACTIVATION_TYPE_RELIE, MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR, MATH21_FUNCTION_ACTIVATION_TYPE_RAMP, MATH21_FUNCTION_ACTIVATION_TYPE_TANH, MATH21_FUNCTION_ACTIVATION_TYPE_PLSE, MATH21_FUNCTION_ACTIVATION_TYPE_LEAKY, MATH21_FUNCTION_ACTIVATION_TYPE_ELU, MATH21_FUNCTION_ACTIVATION_TYPE_LOGGY, MATH21_FUNCTION_ACTIVATION_TYPE_STAIR, MATH21_FUNCTION_ACTIVATION_TYPE_HARDTAN, MATH21_FUNCTION_ACTIVATION_TYPE_LHTAN, MATH21_FUNCTION_ACTIVATION_TYPE_SELU\n\
} MATH21_FUNCTION_ACTIVATION_TYPE;\n\
\n\
float lhtan_activate_kernel(float x)\n\
{\n\
    if(x < 0) return .001f*x;\n\
    if(x > 1) return .001f*(x-1.f) + 1.f;\n\
    return x;\n\
}\n\
float lhtan_gradient_kernel(float x)\n\
{\n\
    if(x > 0 && x < 1) return 1;\n\
    return .001;\n\
}\n\
\n\
float hardtan_activate_kernel(float x)\n\
{\n\
    if (x < -1) return -1;\n\
    if (x > 1) return 1;\n\
    return x;\n\
}\n\
float linear_activate_kernel(float x){return x;}\n\
float logistic_activate_kernel(float x){return 1.f/(1.f + expf(-x));}\n\
float loggy_activate_kernel(float x){return 2.f/(1.f + expf(-x)) - 1;}\n\
float relu_activate_kernel(float x){return x*(x>0);}\n\
float elu_activate_kernel(float x){return (x >= 0)*x + (x < 0)*(expf(x)-1);}\n\
float selu_activate_kernel(float x){return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(expf(x)-1);}\n\
float relie_activate_kernel(float x){return (x>0) ? x : .01f*x;}\n\
float ramp_activate_kernel(float x){return x*(x>0)+.1f*x;}\n\
float leaky_relu_activate_kernel(float x){return (x>0) ? x : .1f*x;}\n\
float tanh_activate_kernel(float x){return (2.f/(1 + expf(-2*x)) - 1);}\n\
float plse_activate_kernel(float x)\n\
{\n\
    if(x < -4) return .01f * (x + 4);\n\
    if(x > 4)  return .01f * (x - 4) + 1;\n\
    return .125f*x + .5f;\n\
}\n\
float stair_activate_kernel(float x)\n\
{\n\
    int n = floorf(x);\n\
    if (n%2 == 0) return floorf(x/2);\n\
    else return (x - n) + floorf(x/2);\n\
}\n\
\n\
\n\
float hardtan_gradient_kernel(float x)\n\
{\n\
    if (x > -1 && x < 1) return 1;\n\
    return 0;\n\
}\n\
float linear_gradient_kernel(float x){return 1;}\n\
float logistic_gradient_kernel(float x){return (1-x)*x;}\n\
float loggy_gradient_kernel(float x)\n\
{\n\
    float y = (x+1)/2;\n\
    return 2*(1-y)*y;\n\
}\n\
float relu_gradient_kernel(float x){return (x>0);}\n\
float elu_gradient_kernel(float x){return (x >= 0) + (x < 0)*(x + 1);}\n\
float selu_gradient_kernel(float x){return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}\n\
float relie_gradient_kernel(float x){return (x>0) ? 1 : .01f;}\n\
float ramp_gradient_kernel(float x){return (x>0)+.1f;}\n\
float leaky_gradient_kernel(float x){return (x>0) ? 1 : .1f;}\n\
float tanh_gradient_kernel(float x){return 1-x*x;}\n\
float plse_gradient_kernel(float x){return (x < 0 || x > 1) ? .01f : .125f;}\n\
float stair_gradient_kernel(float x)\n\
{\n\
    if (floorf(x) == x) return 0;\n\
    return 1;\n\
}\n\
\n\
float math21_function_activation_value_at_opencl_kernel(float x, int a)\n\
{\n\
    switch(a){\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR:\n\
            return linear_activate_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC:\n\
            return logistic_activate_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_LOGGY:\n\
            return loggy_activate_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_RELU:\n\
            return relu_activate_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_ELU:\n\
            return elu_activate_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_SELU:\n\
            return selu_activate_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_RELIE:\n\
            return relie_activate_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_RAMP:\n\
            return ramp_activate_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_LEAKY:\n\
            return leaky_relu_activate_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_TANH:\n\
            return tanh_activate_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_PLSE:\n\
            return plse_activate_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_STAIR:\n\
            return stair_activate_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_HARDTAN:\n\
            return hardtan_activate_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_LHTAN:\n\
            return lhtan_activate_kernel(x);\n\
    }\n\
    return 0;\n\
}\n\
\n\
float math21_function_activation_gradient_opencl_kernel(float x, int a)\n\
{\n\
    switch(a){\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR:\n\
            return linear_gradient_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC:\n\
            return logistic_gradient_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_LOGGY:\n\
            return loggy_gradient_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_RELU:\n\
            return relu_gradient_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_ELU:\n\
            return elu_gradient_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_SELU:\n\
            return selu_gradient_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_RELIE:\n\
            return relie_gradient_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_RAMP:\n\
            return ramp_gradient_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_LEAKY:\n\
            return leaky_gradient_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_TANH:\n\
            return tanh_gradient_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_PLSE:\n\
            return plse_gradient_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_STAIR:\n\
            return stair_gradient_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_HARDTAN:\n\
            return hardtan_gradient_kernel(x);\n\
        case MATH21_FUNCTION_ACTIVATION_TYPE_LHTAN:\n\
            return lhtan_gradient_kernel(x);\n\
    }\n\
    return 0;\n\
}\n\
\n\
__kernel void math21_function_activation_vector_opencl_kernel(__global float *x, int n, int a)\n\
{\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(i < n) x[i] = math21_function_activation_value_at_opencl_kernel(x[i], a);\n\
}\n\
\n\
__kernel void math21_function_activation_gradient_vector_opencl_kernel(__global float *x, int n, int a, __global float *delta)\n\
{\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(i < n) delta[i] *= math21_function_activation_gradient_opencl_kernel(x[i], a);\n\
}\n\
"; 
std::string update_opencl = "\n\
//#define powf(X) pow(X)\n\
//#define sqrtf(X) sqrt(X)\n\
__kernel void\n\
math21_optimization_adam_update_part_2_opencl_kernel(int x_size, __global float *x, __global float *m, __global float *v, float beta1, float beta2, float alpha, float eps, int t) {\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i >= x_size) return;\n\
\n\
    // compute bias-corrected first moment estimate\n\
    float mhat = m[i] / (1.f - pow(beta1, t));\n\
    // compute bias-corrected second raw moment estimate\n\
    float vhat = v[i] / (1.f - pow(beta2, t));\n\
\n\
    // update\n\
    // x = x - alpha * m / (sqrt(v) + eps)\n\
    x[i] = x[i] + alpha * mhat / (sqrt(vhat) + eps);\n\
}\n\
"; 
std::string average_pooling_opencl = "\n\
__kernel void math21_ml_function_average_pooling_forward_opencl_kernel(int n, int w, int h, int c, __global const float *input, __global float *output)\n\
{\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
             get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(id >= n) return;\n\
\n\
    int k = id % c;\n\
    id /= c;\n\
    int b = id;\n\
\n\
    int i;\n\
    int out_index = (k + c*b);\n\
    output[out_index] = 0;\n\
    for(i = 0; i < w*h; ++i){\n\
        int in_index = i + h*w*(k + b*c);\n\
        output[out_index] += input[in_index];\n\
    }\n\
    output[out_index] /= w*h;\n\
}\n\
__kernel void math21_ml_function_average_pooling_backward_opencl_kernel(int n, int w, int h, int c, __global float *in_delta, __global const float *out_delta)\n\
{\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
             get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(id >= n) return;\n\
\n\
    int k = id % c;\n\
    id /= c;\n\
    int b = id;\n\
\n\
    int i;\n\
    int out_index = (k + c*b);\n\
    for(i = 0; i < w*h; ++i){\n\
        int in_index = i + h*w*(k + b*c);\n\
        in_delta[in_index] += out_delta[out_index] / (w*h);\n\
    }\n\
}\n\
"; 
std::string dropout_opencl = "\n\
__kernel void math21_ml_function_dropout_forward_opencl_kernel(__global const float *x, __global float *y, int size, __global const float *rand, float prob, float scale)\n\
{\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
             get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (id < size) y[id] = (rand[id] < prob) ? 0 : scale * x[id];\n\
}\n\
\n\
__kernel void math21_ml_function_dropout_backward_opencl_kernel(__global const float *x, __global float *y, int size, __global const float *rand, float prob, float scale)\n\
{\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
             get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (id < size) {\n\
        if(rand[id] >= prob){\n\
            y[id] += scale * x[id];\n\
        }\n\
    }\n\
}\n\
"; 
std::string conv_opencl = "\n\
// int nr_X_prime = nch_X * ksize * ksize;\n\
// int num_kernels = nch_X * nc_X_prime_1 * nc_X_prime_2;\n\
// X_prime size: (nch_X * nr_K * nc_K ) * (nc_X_prime_1 * nc_X_prime_2)\n\
__kernel void math21_ml_function_conv_X_to_X_prime_opencl_kernel(int num_kernels, __global const float *X,\n\
                                                                 int nr_X, int nc_X,\n\
                                                                 int ksize, int pad, int stride,\n\
                                                                 int nc_X_prime_1, int nc_X_prime_2,\n\
                                                                 __global float *X_prime) {\n\
    // by ye\n\
    int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
                get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (index >= num_kernels) return;\n\
    int ic2 = index % nc_X_prime_2;\n\
    index = index / nc_X_prime_2;\n\
    int ic1 = index % nc_X_prime_1;\n\
    int ich_X = index / nc_X_prime_1;\n\
    int ir_X_abs = ic1 * stride - pad;\n\
    int ic_X_abs = ic2 * stride - pad;\n\
    X_prime += (ich_X * ksize * ksize * nc_X_prime_1 + ic1) * nc_X_prime_2 + ic2;\n\
    X += (ich_X * nr_X + ir_X_abs) * nc_X + ic_X_abs;\n\
    for (int ir_K = 0; ir_K < ksize; ++ir_K) {\n\
        for (int ic_K = 0; ic_K < ksize; ++ic_K) {\n\
            int ir_X = ir_X_abs + ir_K;\n\
            int ic_X = ic_X_abs + ic_K;\n\
\n\
            // int ir_X = ir_K + ic1 * stride - pad;\n\
            // int ic_X = ic_K + ic2 * stride - pad;\n\
            // X[(ich_X * nr_X + ir_X) * nc_X + ic_X]\n\
            // X[(ich_X * nr_X + ir_K + ic1 * stride - pad) * nc_X + ic_K + ic2 * stride - pad]\n\
            // X[(ich_X * nr_X + ir_K + ir_X_abs) * nc_X + ic_K + ic_X_abs]\n\
\n\
            // nr_X_prime = nch_X * ksize * ksize;\n\
            // ir = (ich_X, ir_K, ic_K)\n\
            // ir = ich_X * nr_K * nc_K + ir_K * nc_K + ic_K\n\
            // index_X_prime = (ir * nc_X_prime_1 + ic1) * nc_X_prime_2 + ic2\n\
            *X_prime = (ir_X >= 0 && ic_X >= 0 && ir_X < nr_X && ic_X < nc_X) ?\n\
                       X[ir_K * nc_X + ic_K] : 0;\n\
\n\
            X_prime += nc_X_prime_1 * nc_X_prime_2;\n\
        }\n\
    }\n\
}\n\
\n\
__kernel void math21_ml_function_conv_binarize_weights_opencl_kernel(__global float *weights, int n, int size, __global float *binary)\n\
{\n\
    int f = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (f >= n) return;\n\
    int i = 0;\n\
    float mean = 0;\n\
    for(i = 0; i < size; ++i){\n\
        mean += fabs(weights[f*size + i]);\n\
    }\n\
    mean = mean / size;\n\
    for(i = 0; i < size; ++i){\n\
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;\n\
        //binary[f*size + i] = weights[f*size + i];\n\
    }\n\
}\n\
__kernel void math21_ml_function_conv_binarize_input_opencl_kernel(__global float *x, int n, __global float *binary)\n\
{\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i >= n) return;\n\
    binary[i] = (x[i] >= 0) ? 1 : -1;\n\
}\n\
\n\
__kernel void math21_ml_function_conv_dX_prime_to_dX_opencl_kernel(int num_kernels, __global const float *dX_prime,\n\
                                                                   int nr_X, int nc_X,\n\
                                                                   int ksize, int pad, int stride,\n\
                                                                   int nc_X_prime_1, int nc_X_prime_2,\n\
                                                                   __global float *dX) {\n\
    int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
                get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
\n\
    if (index >= num_kernels) return;\n\
    float val = 0;\n\
    int w = index % nc_X + pad;\n\
    int h = (index / nc_X) % nr_X + pad;\n\
    int c = index / (nc_X * nr_X);\n\
    // compute the start and end of the output\n\
    int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;\n\
    int w_col_end = min(w / stride + 1, nc_X_prime_2);\n\
    int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;\n\
    int h_col_end = min(h / stride + 1, nc_X_prime_1);\n\
    // equivalent implementation\n\
    int offset =\n\
            (c * ksize * ksize + h * ksize + w) * nc_X_prime_1 * nc_X_prime_2;\n\
    int coeff_h_col = (1 - stride * ksize * nc_X_prime_1) * nc_X_prime_2;\n\
    int coeff_w_col = (1 - stride * nc_X_prime_1 * nc_X_prime_2);\n\
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {\n\
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {\n\
            val += dX_prime[offset + h_col * coeff_h_col + w_col * coeff_w_col];\n\
        }\n\
    }\n\
    dX[index] += val;\n\
}\n\
\n\
__kernel void math21_ml_function_conv_smooth_opencl_kernel(__global float *x, int n, int w, int h, int c, int size, float rate, __global float *delta)\n\
{\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
             get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(id >= n) return;\n\
\n\
    int j = id % w;\n\
    id /= w;\n\
    int i = id % h;\n\
    id /= h;\n\
    int k = id % c;\n\
    id /= c;\n\
    int b = id;\n\
\n\
    int w_offset = -(size/2.f);\n\
    int h_offset = -(size/2.f);\n\
\n\
    int out_index = j + w*(i + h*(k + c*b));\n\
    int l, m;\n\
    for(l = 0; l < size; ++l){\n\
        for(m = 0; m < size; ++m){\n\
            int cur_h = h_offset + i + l;\n\
            int cur_w = w_offset + j + m;\n\
            int index = cur_w + w*(cur_h + h*(k + b*c));\n\
            int valid = (cur_h >= 0 && cur_h < h &&\n\
                         cur_w >= 0 && cur_w < w);\n\
            delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;\n\
        }\n\
    }\n\
}\n\
"; 
std::string batch_normalization_opencl = "\n\
#define MATH21_OPENCL_BLOCK_SIZE 512\n\
__kernel void math21_ml_batchnormalization_backward_mu_fast_opencl_kernel(__global const float *delta, __global const float *variance, int batch, int filters, int spatial, __global float *mean_delta)\n\
{\n\
    __local float part[MATH21_OPENCL_BLOCK_SIZE];\n\
    int id = get_global_id(1);\n\
    int filter = get_global_id(0);\n\
    float sum = 0;\n\
    int i, j;\n\
    for (j = 0; j < batch; ++j) {\n\
        for (i = 0; i < spatial; i += MATH21_OPENCL_BLOCK_SIZE) {\n\
            int index = j * spatial*filters + filter * spatial + i + id;\n\
            sum += (i + id < spatial) ? delta[index] : 0;\n\
        }\n\
    }\n\
    part[id] = sum;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
    if (id == 0) {\n\
        mean_delta[filter] = 0;\n\
        for (i = 0; i < MATH21_OPENCL_BLOCK_SIZE; ++i) {\n\
            mean_delta[filter] += part[i];\n\
        }\n\
        mean_delta[filter] *= (-1.f / sqrt(variance[filter] + .00001f));\n\
    }\n\
}\n\
__kernel void math21_ml_batchnormalization_backward_mu_opencl_kernel(__global float *delta, __global float *variance, int batch, int filters, int spatial, __global float *mean_delta)\n\
{\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i >= filters) return;\n\
    int j,k;\n\
    mean_delta[i] = 0;\n\
    for (j = 0; j < batch; ++j) {\n\
        for (k = 0; k < spatial; ++k) {\n\
            int index = j*filters*spatial + i*spatial + k;\n\
            mean_delta[i] += delta[index];\n\
        }\n\
    }\n\
    mean_delta[i] *= (-1.f/sqrt(variance[i] + .00001f));\n\
}\n\
__kernel void  math21_ml_batchnormalization_backward_sigma_square_fast_opencl_kernel(__global float *x, __global float *delta, __global float *mean, __global float *variance, int batch, int filters, int spatial, __global float *variance_delta)\n\
{\n\
    __local float part[MATH21_OPENCL_BLOCK_SIZE];\n\
    int id = get_global_id(1);\n\
    int filter = get_global_id(0);\n\
    float sum = 0;\n\
    int i, j;\n\
    for(j = 0; j < batch; ++j){\n\
        for(i = 0; i < spatial; i += MATH21_OPENCL_BLOCK_SIZE){\n\
            int index = j*spatial*filters + filter*spatial + i + id;\n\
            sum += (i+id < spatial) ? delta[index]*(x[index] - mean[filter]) : 0;\n\
        }\n\
    }\n\
    part[id] = sum;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
    if(id == 0){\n\
        variance_delta[filter] = 0;\n\
        for(i = 0; i < MATH21_OPENCL_BLOCK_SIZE; ++i){\n\
            variance_delta[filter] += part[i];\n\
        }\n\
        variance_delta[filter] *= -.5f * pow(variance[filter] + .00001f, (float)(-3.f/2.f));\n\
    }\n\
}\n\
__kernel void math21_ml_batchnormalization_backward_input_opencl_kernel(int N, __global float *x, __global float *mean, __global float *variance, __global float *mean_delta, __global float *variance_delta, int batch, int filters, int spatial, __global float *delta)\n\
{\n\
    int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
                get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (index >= N) return;\n\
    int f = (index/spatial)%filters;\n\
    delta[index] = delta[index] * 1.f/(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2.f * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);\n\
}\n\
\n\
"; 
std::string softmax_opencl = "\n\
void math21_ml_function_softmax_opencl_device(__global float *input, int n, float temp, int stride, __global float *output) {\n\
    int i;\n\
    float sum = 0;\n\
    float largest = -INFINITY;\n\
    for (i = 0; i < n; ++i) {\n\
        int val = input[i * stride];\n\
        largest = (val > largest) ? val : largest;\n\
    }\n\
    for (i = 0; i < n; ++i) {\n\
        float e = exp(input[i * stride] / temp - largest / temp);\n\
        sum += e;\n\
        output[i * stride] = e;\n\
    }\n\
    for (i = 0; i < n; ++i) {\n\
        output[i * stride] /= sum;\n\
    }\n\
}\n\
\n\
__kernel void\n\
math21_ml_function_softmax_tree_opencl_kernel(__global float *input, int spatial, int batch, int stride, float temp, __global float *output,\n\
                                              int groups, __constant int *group_size, __constant int *group_offset) {\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
             get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (id >= spatial * batch * groups) return;\n\
    int s = id % spatial;\n\
    id = id / spatial;\n\
    int g = id % groups;\n\
    int b = id / groups;\n\
    int goff = group_offset[g] * spatial;\n\
    int boff = b * stride;\n\
    math21_ml_function_softmax_opencl_device(input + goff + boff + s, group_size[g], temp, spatial, output + goff + boff + s);\n\
}\n\
__kernel void\n\
math21_ml_function_softmax_opencl_kernel(__global float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride,\n\
                                         float temp, __global float *output) {\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
             get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (id >= batch * groups) return;\n\
    int b = id / groups;\n\
    int g = id % groups;\n\
    math21_ml_function_softmax_opencl_device(input + b * batch_offset + g * group_offset, n, temp, stride,\n\
                                             output + b * batch_offset + g * group_offset);\n\
}\n\
__kernel void\n\
math21_ml_function_softmax_x_ent_opencl_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error) {\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i < n) {\n\
        float t = truth[i];\n\
        float p = pred[i];\n\
        error[i] = (t) ? -log(p) : 0;\n\
        delta[i] = t - p;\n\
    }\n\
}\n\
"; 
std::string max_pooling_opencl = "\n\
__kernel void math21_ml_function_max_pooling_forward_opencl_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, __global float *input, __global float *output, __global int *indexes)\n\
{\n\
    int h = (in_h + pad - size)/stride + 1;\n\
    int w = (in_w + pad - size)/stride + 1;\n\
    int c = in_c;\n\
\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
             get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(id >= n) return;\n\
\n\
    int j = id % w;\n\
    id /= w;\n\
    int i = id % h;\n\
    id /= h;\n\
    int k = id % c;\n\
    id /= c;\n\
    int b = id;\n\
\n\
    int w_offset = -pad/2;\n\
    int h_offset = -pad/2;\n\
\n\
    int out_index = j + w*(i + h*(k + c*b));\n\
    float max = -INFINITY;\n\
    int max_i = -1;\n\
    int l, m;\n\
    for(l = 0; l < size; ++l){\n\
        for(m = 0; m < size; ++m){\n\
            int cur_h = h_offset + i*stride + l;\n\
            int cur_w = w_offset + j*stride + m;\n\
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));\n\
            int valid = (cur_h >= 0 && cur_h < in_h &&\n\
                         cur_w >= 0 && cur_w < in_w);\n\
            float val = (valid != 0) ? input[index] : -INFINITY;\n\
            max_i = (val > max) ? index : max_i;\n\
            max   = (val > max) ? val   : max;\n\
        }\n\
    }\n\
    output[out_index] = max;\n\
    indexes[out_index] = max_i;\n\
}\n\
__kernel void math21_ml_function_max_pooling_backward_opencl_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, __global float *delta, __global float *prev_delta, __global int *indexes)\n\
{\n\
    int h = (in_h + pad - size)/stride + 1;\n\
    int w = (in_w + pad - size)/stride + 1;\n\
    int c = in_c;\n\
    int area = (size-1)/stride;\n\
\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
             get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(id >= n) return;\n\
\n\
    int index = id;\n\
    int j = id % in_w;\n\
    id /= in_w;\n\
    int i = id % in_h;\n\
    id /= in_h;\n\
    int k = id % in_c;\n\
    id /= in_c;\n\
    int b = id;\n\
\n\
    int w_offset = -pad/2;\n\
    int h_offset = -pad/2;\n\
\n\
    float d = 0;\n\
    int l, m;\n\
    for(l = -area; l < area+1; ++l){\n\
        for(m = -area; m < area+1; ++m){\n\
            int out_w = (j-w_offset)/stride + m;\n\
            int out_h = (i-h_offset)/stride + l;\n\
            int out_index = out_w + w*(out_h + h*(k + c*b));\n\
            int valid = (out_w >= 0 && out_w < w &&\n\
                         out_h >= 0 && out_h < h);\n\
            d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;\n\
        }\n\
    }\n\
    prev_delta[index] += d;\n\
}\n\
"; 
std::map<std::string,std::string> source_map = {
std::make_pair("generic_02.kl",generic_02),
std::make_pair("generic_03.kl",generic_03),
std::make_pair("generic_03_transpose.kl",generic_03_transpose),
std::make_pair("generic_01.kl",generic_01),
std::make_pair("generic_01_vector_set.kl",generic_01_vector_set),
std::make_pair("matrix_opencl.cl",matrix_opencl),
std::make_pair("vector_opencl_02.cl",vector_opencl_02),
std::make_pair("vector_opencl_01.cl",vector_opencl_01),
std::make_pair("activation_opencl.cl",activation_opencl),
std::make_pair("update_opencl.cl",update_opencl),
std::make_pair("average_pooling_opencl.cl",average_pooling_opencl),
std::make_pair("dropout_opencl.cl",dropout_opencl),
std::make_pair("conv_opencl.cl",conv_opencl),
std::make_pair("batch_normalization_opencl.cl",batch_normalization_opencl),
std::make_pair("softmax_opencl.cl",softmax_opencl),
std::make_pair("max_pooling_opencl.cl",max_pooling_opencl),
};
