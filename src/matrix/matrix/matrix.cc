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

#include "../vector/files_c.h"
#include "../ten_ops.h"
#include "matrix_cc.h"
#include "matrix.h"

namespace math21 {

    void math21_tensor_log_cpu(const char *name, const NumR32 *data, const VecN &d) {
        if (d.isEmpty()) {
            return;
        }
        Tensor<NumR32> A;
        A.setSizeNoSpace(d);
        A.setSpace((void *) data, A.size() * sizeof(NumR32));
        A.log(name, 0, 1, 10);
    }

    void math21_tensor_log_wrapper(const char *name, PointerFloatInputWrapper data, const VecN &d) {
#if defined(MATH21_FLAG_USE_CPU)
        math21_tensor_log_cpu(name, data, d);
        return;
#else
        Tensor<NumR32> A;
        A.setSize(d);
        NumR32 *data_cpu = A.getDataAddress();
        math21_vector_pull_wrapper(data, data_cpu, A.size());
        A.log(name, 0, 1, 10);
#endif
    }

    NumB math21_rawtensor_convert_to_TenR_with_data_shared(const m21rawtensor &rawtensor, TenR &A) {
        if (rawtensor.type != m21_type_NumR) {
            m21warn("rawtensor.type not m21_type_NumR");
            m21warn("rawtensor.type", rawtensor.type);
            return 0;
        }
        VecN d(rawtensor.dims);
        math21_c_array_set(d.size(), rawtensor.d, d.getDataAddress());
        if (d.isEmpty()) {
            m21warn("d is empty");
            return 0;
        }
        A.setSizeNoSpace(d);
        A.setSpace(rawtensor.data, A.size() * sizeof(NumR));
        return 1;
    }

    NumB math21_tensor_convert_to_rawtensor_NumR_with_data_shared(const TenR &A, m21rawtensor &rawtensor) {
        rawtensor.type = m21_type_NumR;
        rawtensor.dims = A.dims();
        rawtensor.d = const_cast<NumN *>(A.getShapeDataAddress());
        rawtensor.data = (void *) A.getDataAddress();
        return 1;
    }
}

using namespace math21;

void math21_matrix_transpose(NumR32 *old, int rows, int cols) {
    auto *A_new = (NumR32 *) math21_vector_calloc_cpu(rows * cols, sizeof(NumR32));
    int x, y;
    for (x = 0; x < rows; ++x) {
        for (y = 0; y < cols; ++y) {
            A_new[y * rows + x] = old[x * cols + y];
        }
    }
    math21_vector_memcpy_cpu(old, A_new, rows * cols * sizeof(NumR32));
    math21_vector_free_cpu(A_new);
}

void math21_tensor_1d_float_log_cpu(const char *name, const NumR32 *data, NumN d1) {
    VecN d(1);
    d = d1;
    math21_tensor_log_cpu(name, data, d);
}

void math21_tensor_2d_float_log_cpu(const char *name, const NumR32 *data, NumN d1, NumN d2) {
    VecN d(2);
    d = d1, d2;
    math21_tensor_log_cpu(name, data, d);
}

void math21_tensor_3d_float_log_cpu(const char *name, const NumR32 *data, NumN d1, NumN d2, NumN d3) {
    VecN d(3);
    d = d1, d2, d3;
    math21_tensor_log_cpu(name, data, d);
}

void math21_tensor_4d_float_log_cpu(const char *name, const NumR32 *data, NumN d1, NumN d2, NumN d3, NumN d4) {
    VecN d(4);
    d = d1, d2, d3, d4;
    math21_tensor_log_cpu(name, data, d);
}

void
math21_tensor_5d_float_log_cpu(const char *name, const NumR32 *data, NumN d1, NumN d2, NumN d3, NumN d4, NumN d5) {
    VecN d(5);
    d = d1, d2, d3, d4, d5;
    math21_tensor_log_cpu(name, data, d);
}

void math21_tensor_1d_float_log_wrapper(const char *name, PointerFloatInputWrapper data, NumN d1) {
    VecN d(1);
    d = d1;
    math21_tensor_log_wrapper(name, data, d);
}

void math21_tensor_2d_float_log_wrapper(const char *name, PointerFloatInputWrapper data, NumN d1, NumN d2) {
    VecN d(2);
    d = d1, d2;
    math21_tensor_log_wrapper(name, data, d);
}

void math21_tensor_3d_float_log_wrapper(const char *name, PointerFloatInputWrapper data, NumN d1, NumN d2, NumN d3) {
    VecN d(3);
    d = d1, d2, d3;
    math21_tensor_log_wrapper(name, data, d);
}

void math21_tensor_4d_float_log_wrapper(const char *name, PointerFloatInputWrapper data, NumN d1, NumN d2, NumN d3,
                                        NumN d4) {
    VecN d(4);
    d = d1, d2, d3, d4;
    math21_tensor_log_wrapper(name, data, d);
}

void
math21_tensor_5d_float_log_wrapper(const char *name, PointerFloatInputWrapper data, NumN d1, NumN d2, NumN d3, NumN d4,
                                   NumN d5) {
    VecN d(5);
    d = d1, d2, d3, d4, d5;
    math21_tensor_log_wrapper(name, data, d);
}

void math21_rawtensor_log_cpu(const char *name, m21rawtensor rawtensor) {
    VecN d(rawtensor.dims);
    math21_c_array_set(d.size(), rawtensor.d, d.getDataAddress());
    MATH21_ASSERT(rawtensor.type == m21_type_NumR32, "not implement log")
    math21_tensor_log_cpu(name, (const NumR32 *) rawtensor.data, d);
}

void math21_rawtensor_log_wrapper(const char *name, m21rawtensor rawtensor) {
    MATH21_ASSERT(0)
}