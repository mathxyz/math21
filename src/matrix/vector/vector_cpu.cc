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

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "inner.h"
#include "../../probability/files_c.h"
#include "../../generic/files.h"
#include "vector_cpu.h"

using namespace math21;

// must call math21_vector_free_cpu
float *math21_vector_deserialize_c_cpu(FILE *f, size_t *n0) {
    if (!n0) {
        return 0;
    }
    size_t n;
    size_t read = fread(&n, sizeof(size_t), 1, f);
    if (read != 1) {
        return 0;
    }
    if (n == 0) {
        return 0;
    }
    float *v = math21_vector_create_with_default_value_cpu(n, 0);
    read = fread(v, sizeof(float), n, f);
    if (read != n) {
        math21_vector_free_cpu(v);
        return 0;
    }
    *n0 = (size_t) n;
    return v;
}

// must call math21_vector_free_cpu
NumN8 *math21_vector_deserialize_byte_c_cpu(FILE *f, size_t *n0) {
    if (!n0) {
        return 0;
    }
    size_t n;
    size_t read = fread(&n, sizeof(size_t), 1, f);
    if (read != 1) {
        return 0;
    }
    if (n == 0) {
        return 0;
    }
    NumN8 *v = math21_vector_create_with_default_value_byte_cpu(n, 0);
    read = fread(v, sizeof(NumN8), n, f);
    if (read != n) {
        math21_vector_free_cpu(v);
        return 0;
    }
    *n0 = (size_t) n;
    return v;
}

// must call math21_vector_free_cpu, no header.
NumN8 *math21_vector_raw_deserialize_byte_c_cpu(FILE *f, size_t n) {
    if (n <= 0) {
        return 0;
    }
    NumN8 *v = math21_vector_create_with_default_value_byte_cpu(n, 0);
    size_t read = fread(v, sizeof(NumN8), n, f);
    if (read != n) {
        return 0;
    }
    return v;
}

// must call math21_vector_free_cpu
float *math21_vector_deserialize_from_file_cpu(const char *name, size_t *n) {
    FILE *f = fopen(name, "rb");
    if (!f) {
        math21_file_error(name);
        return 0;
    }
    float *v = math21_vector_deserialize_c_cpu(f, n);
    fclose(f);
    return v;
}

// must call math21_vector_free_cpu
NumN8 *math21_vector_deserialize_from_file_byte_cpu(const char *name, size_t *n) {
    FILE *f = fopen(name, "rb");
    if (!f) {
        math21_file_error(name);
        return 0;
    }
    NumN8 *v = math21_vector_deserialize_byte_c_cpu(f, n);
    fclose(f);
    return v;
}

// must call math21_vector_free_cpu
NumN8 *math21_vector_raw_deserialize_from_file_byte_cpu(const char *name, size_t n) {
    FILE *f = fopen(name, "rb");
    if (!f) {
        math21_file_error(name);
        return 0;
    }
    NumN8 *v = math21_vector_raw_deserialize_byte_c_cpu(f, n);
    fclose(f);
    return v;
}

void math21_vector_serialize_c_cpu(FILE *f, const float *v, size_t n) {
    if (!v || n <= 0) {
        return;
    }
    fwrite(&n, sizeof(size_t), 1, f);
    fwrite(v, sizeof(float), n, f);
}

void math21_vector_serialize_byte_c_cpu(FILE *f, const NumN8 *v, size_t n) {
    if (!v || n <= 0) {
        return;
    }
    fwrite(&n, sizeof(size_t), 1, f);
    fwrite(v, sizeof(NumN8), n, f);
}

void math21_vector_raw_serialize_byte_c_cpu(FILE *f, const NumN8 *v, size_t n) {
    if (!v || n <= 0) {
        return;
    }
    fwrite(v, sizeof(NumN8), n, f);
}

void math21_vector_serialize_to_file_cpu(const char *name, const float *v, size_t n) {
    FILE *f = fopen(name, "wb");
    if (!f) {
        math21_file_error(name);
        return;
    }
    math21_vector_serialize_c_cpu(f, v, n);
    fclose(f);
}

void math21_vector_serialize_to_file_byte_cpu(const char *name, const NumN8 *v, size_t n) {
    FILE *f = fopen(name, "wb");
    if (!f) {
        math21_file_error(name);
        return;
    }
    math21_vector_serialize_byte_c_cpu(f, v, n);
    fclose(f);
}

void math21_vector_raw_serialize_to_file_byte_cpu(const char *name, const NumN8 *v, size_t n) {
    FILE *f = fopen(name, "wb");
    if (!f) {
        math21_file_error(name);
        return;
    }
    math21_vector_raw_serialize_byte_c_cpu(f, v, n);
    fclose(f);
}

// [from, to)
void math21_vector_save_cpu(const char *name, const float *v, size_t from, size_t to) {
    size_t n = to - from;
    if (!v || n <= 0) {
        return;
    }
    FILE *f = fopen(name, "a");
    if (!f) {
        math21_file_error(name);
        return;
    }
    v += from;
    int i;
    for (i = 0; i < n; ++i) fprintf(f, "%f ", v[i]);
    fprintf(f, "\n");
    fclose(f);
}

// [from, to)
void math21_vector_log_cpu(const char *name, const float *v, size_t from, size_t to) {
    if (name) {
        printf("%s\n", name);
    }
    size_t n = to - from;
    v += from;
    int i;
    for (i = 0; i < n; ++i) printf("%f ", v[i]);
    printf("\n");
}

float *math21_vector_create_with_default_value_cpu(size_t n, float value) {
    float *v;
    if (value == 0) {
        v = (float *) math21_vector_calloc_cpu(n, sizeof(float));
    } else {
        v = (float *) math21_vector_malloc_cpu(n * sizeof(float));
        math21_vector_set_cpu(n, value, v, 1);
    }
    return v;
}

NumN8 *math21_vector_create_with_default_value_byte_cpu(size_t n, NumN8 value) {
    NumN8 *v;
    if (value == 0) {
        v = (NumN8 *) math21_vector_calloc_cpu(n, sizeof(NumN8));
    } else {
        v = (NumN8 *) math21_vector_malloc_cpu(n * sizeof(NumN8));
        math21_vector_set_byte_cpu(n, value, v, 1);
    }
    return v;
}

void *math21_vector_create_buffer_cpu(size_t n, size_t elementSize) {
    return math21_vector_calloc_cpu(n, elementSize);
}

void *math21_vector_setSize_buffer_cpu(void *v, size_t n, size_t elementSize) {
    if (!v) {
        return math21_vector_create_buffer_cpu(n, elementSize);
    }
    return math21_vector_realloc_cpu(v, n * elementSize);
}

void *math21_vector_copy_buffer_cpu(void *dst, const void *src, size_t n, size_t elementSize) {
    return math21_vector_memcpy_cpu(dst, src, n * elementSize);
}

float *math21_vector_resize_with_default_value_cpu(float *v, size_t n, float value) {
    v = (float *) math21_vector_realloc_cpu(v, n * sizeof(float));
    math21_vector_set_cpu(n, value, v, 1);
    return v;
}

void *math21_vector_resize_buffer_cpu(void *v, size_t n, size_t elementSize) {
    v = math21_vector_realloc_cpu(v, n * elementSize);
    return v;
}

int *math21_vector_resize_with_default_value_int_cpu(int *v, size_t n, int value) {
    v = (int *) math21_vector_realloc_cpu(v, n * sizeof(int));
    math21_vector_set_int_cpu(n, value, v, 1);
    return v;
}

float *math21_vector_create_from_cpuvector_cpu(size_t n, const float *x, int stride_x) {
    float *v;
    if (!x) {
        v = (float *) math21_vector_calloc_cpu(n, sizeof(float));
    } else {
        v = (float *) math21_vector_malloc_cpu(n * sizeof(float));
        if (stride_x == 1) {
            math21_vector_memcpy_cpu(v, x, n * sizeof(float));
        } else {
            math21_vector_assign_from_vector_cpu(n, x, stride_x, v, 1);
        }
    }
    return v;
}

int *math21_vector_create_from_cpuvector_int_cpu(size_t n, const int *x, int stride_x) {
    int *v;
    if (!x) {
        v = (int *) math21_vector_calloc_cpu(n, sizeof(int));
    } else {
        v = (int *) math21_vector_malloc_cpu(n * sizeof(int));
        if (stride_x == 1) {
            math21_vector_memcpy_cpu(v, x, n * sizeof(int));
        } else {
            math21_vector_assign_from_vector_int_cpu(n, x, stride_x, v, 1);
        }
    }
    return v;
}

NumN8 *math21_vector_create_from_cpuvector_byte_cpu(NumSize n, const NumN8 *x, int stride_x) {
    NumN8 *v;
    if (!x) {
        v = (NumN8 *) math21_vector_calloc_cpu(n, sizeof(NumN8));
    } else {
        v = (NumN8 *) math21_vector_malloc_cpu(n * sizeof(NumN8));
        if (stride_x == 1) {
            math21_vector_memcpy_cpu(v, x, n * sizeof(NumN8));
        } else {
            math21_vector_assign_from_vector_byte_cpu(n, x, stride_x, v, 1);
        }
    }
    return v;
}

void math21_vector_free_cpu(void *x) {
    free(x);
}

// mu = E(X)
// X shape: mini_batch_size*features_size*in_class_size
// rnn: in_class_size=1
// cnn: in_class_size=nr_Y*nc_Y
void math21_vector_mean_cpu(const float *X, int mini_batch_size, int features_size, int in_class_size, float *mean) {
    float scale = 1.f / (mini_batch_size * in_class_size);
    int ifeature, imb, imember;
    for (ifeature = 0; ifeature < features_size; ++ifeature) {
        mean[ifeature] = 0;
        for (imb = 0; imb < mini_batch_size; ++imb) {
            for (imember = 0; imember < in_class_size; ++imember) {
                int index = imb * features_size * in_class_size + ifeature * in_class_size + imember;
                mean[ifeature] += X[index];
            }
        }
        mean[ifeature] *= scale;
    }
}

// sigma_square = Var(X) if in_class_size is 1.
void
math21_vector_variance_cpu(const float *X, const float *mean, int mini_batch_size, int features_size, int in_class_size,
                           float *variance) {
    float scale = 1.f / (mini_batch_size * in_class_size - 1);
    int ifeature, imb, imember;
    for (ifeature = 0; ifeature < features_size; ++ifeature) {
        variance[ifeature] = 0;
        for (imb = 0; imb < mini_batch_size; ++imb) {
            for (imember = 0; imember < in_class_size; ++imember) {
                int index = imb * features_size * in_class_size + ifeature * in_class_size + imember;
                variance[ifeature] += pow((X[index] - mean[ifeature]), 2);
            }
        }
        variance[ifeature] *= scale;
    }
}

// y = x, y(i) = x(i)
void math21_vector_assign_from_vector_N8_cpu(int n, const NumN8 *x, NumN8 *y) {
    int i;
    for (i = 0; i < n; ++i) y[i] = x[i];
}

// y = x, y(i) = x(i)
void math21_vector_assign_from_vector_cpu(int n, const float *x, int stride_x, float *y, int stride_y) {
    int i;
    for (i = 0; i < n; ++i) y[i * stride_y] = x[i * stride_x];
}

// y = x, y(i) = x(i)
void math21_vector_assign_from_vector_int_cpu(int n, const int *x, int stride_x, int *y, int stride_y) {
    int i;
    for (i = 0; i < n; ++i) y[i * stride_y] = x[i * stride_x];
}

// y = x, y(i) = x(i)
void math21_vector_assign_from_vector_byte_cpu(NumSize n, const NumN8 *x, int stride_x, NumN8 *y, int stride_y) {
    NumSize i;
    for (i = 0; i < n; ++i) y[i * stride_y] = x[i * stride_x];
}

// x = k*x
void math21_vector_kx_cpu(int n, float k, float *x, int stride_x) {
    int i;
    for (i = 0; i < n; ++i) x[i * stride_x] *= k;
}

// x(i) = k+x(i), for all i
void math21_vector_k_add_x_cpu(int n, float k, float *x, int stride_x) {
    int i;
    for (i = 0; i < n; ++i) x[i * stride_x] += k;
}

// y = k*x + y
void math21_vector_kx_add_y_cpu(int n, float k, const float *x, int stride_x, float *y, int stride_y) {
    int i;
    for (i = 0; i < n; ++i) y[i * stride_y] += k * x[i * stride_x];
}

void
math21_vector_normalize_cpu(float *x, const float *mean, const float *variance, int mini_batch_size, int features_size,
                            int in_class_size) {
    int b, f, i;
    for (b = 0; b < mini_batch_size; ++b) {
        for (f = 0; f < features_size; ++f) {
            for (i = 0; i < in_class_size; ++i) {
                int index = b * features_size * in_class_size + f * in_class_size + i;
                x[index] = (x[index] - mean[f]) / (sqrt(variance[f]) + .000001f);
            }
        }
    }
}

// y = k*x
void math21_vector_kx_with_in_class_cpu(float *x, const float *k, int mini_batch_size, int features_size,
                                        int in_class_size) {
    int imb, ifeature, imember;
    for (imb = 0; imb < mini_batch_size; ++imb) {
        for (ifeature = 0; ifeature < features_size; ++ifeature) {
            for (imember = 0; imember < in_class_size; ++imember) {
                x[(imb * features_size + ifeature) * in_class_size + imember] *= k[ifeature];
            }
        }
    }
}

// Y = X + b
// Y(imb, ifeature) = X(imb, ifeature) + b(ifeature)
// Y(imb, ifeature, imember) = X(imb, ifeature, imember) + b(ifeature)
void math21_vector_x_add_b_with_in_class_cpu(float *x, const float *b, int mini_batch_size, int features_size,
                                             int in_class_size) {
    int imb, ifeature, imember;
    for (imb = 0; imb < mini_batch_size; ++imb) {
        for (ifeature = 0; ifeature < features_size; ++ifeature) {
            for (imember = 0; imember < in_class_size; ++imember) {
                x[(imb * features_size + ifeature) * in_class_size + imember] += b[ifeature];
            }
        }
    }
}

float math21_vector_sum_cpu(const float *v, int n) {
    int i;
    float sum = 0;
    for (i = 0; i < n; ++i) sum += v[i];
    return sum;
}

// Y = W*X + b, dL/db += sum(dL/dY(i))
// db(ifeature) = sum(dY(ib, ifeature, imember))
void math21_vector_sum_with_in_class_cpu(float *db, const float *dY, int mini_batch_size, int features_size,
                                         int in_class_size) {
    int ifeature, imb;
    for (imb = 0; imb < mini_batch_size; ++imb) {
        for (ifeature = 0; ifeature < features_size; ++ifeature) {
            db[ifeature] += math21_vector_sum_cpu(dY + (imb * features_size + ifeature) * in_class_size, in_class_size);
        }
    }
}

// Y = k*X + b, dL/dk += sum(dL/dY(i) *.ele X(i))
void math21_vector_sum_SchurProduct_with_in_class_cpu(const float *X, const float *dY, int mini_batch_size,
                                                      int features_size,
                                                      int in_class_size, float *dk) {
    int imb, ifeature, imember;
    for (ifeature = 0; ifeature < features_size; ++ifeature) {
        float sum = 0;
        for (imb = 0; imb < mini_batch_size; ++imb) {
            for (imember = 0; imember < in_class_size; ++imember) {
                int index = (imb * features_size + ifeature) * in_class_size + imember;
                sum += dY[index] * X[index];
            }
        }
        dk[ifeature] += sum;
    }
}

void math21_vector_set_cpu(int n, float value, float *x, int stride) {
    math21_generic_vector_set_by_value_cpu(n, value, x, stride, m21_type_NumR32);
//    int i;
//    for (i = 0; i < n; ++i) x[i * stride] = value;
}

void math21_vector_set_int_cpu(int n, int value, int *x, int stride) {
    math21_generic_vector_set_by_value_cpu(n, value, x, stride, m21_type_NumN);
//    int i;
//    for (i = 0; i < n; ++i) x[i * stride] = value;
}

void math21_vector_set_byte_cpu(int n, NumN8 value, NumN8 *x, int stride) {
    math21_generic_vector_set_by_value_cpu(n, value, x, stride, m21_type_NumN8);
//    int i;
//    for (i = 0; i < n; ++i) x[i * stride] = value;
}

void math21_vector_set_random_cpu(int n, float *v) {
    int i;
    for (i = 0; i < n; ++i) {
        v[i] = (float) rand() / RAND_MAX;
    }
}

void math21_vector_log_byte_cpu(int n, NumN8 *x, int stride) {
    int i;
    for (i = 0; i < n; ++i) printf("%d ", x[i * stride]);

}

// x -> y, (d1, d2, d3) -> (d1, d2y, d3) with d2 >= d2y
// x <- y, (d1, d2, d3) <- (d1, d2y, d3) with d2 >= d2y
// offset2 >= 0
void math21_vector_assign_3d_d2_cpu(const float *data1, float *data2,
                                    int d1, int d2, int d3, int d2y, int offset2, int isToSmall) {
    NumSize size = d1 * d2y * d3;
    int id = 0;
    while (1) {
        if (id >= size) return;
        int i1, i2_y, ix, iy, i2, i3;
        iy = id;
        i3 = iy % d3;
        iy = iy / d3;
        i2_y = iy % d2y;
        i1 = iy / d2y;

        i2 = i2_y + offset2;
        ix = i1 * d2 * d3 + i2 * d3 + i3;
        iy = id;
        if (isToSmall) {
            data2[iy] = data1[ix];
        } else {
            data2[ix] = data1[iy];
        }
        ++id;
    }
}

// (d1, d2, d3, d4) -> (d1, d3, d2, d4)
void math21_vector_transpose_d1234_to_d1324_cpu(const float *x, float *y, int d1, int d2, int d3, int d4) {
    size_t size = d1 * d2 * d3 * d4;
    int id = 0;
    while (1) {
        if (id >= size) return;
        int i1, i2_y, i3_y, i4, ix, iy, d3_y, d2_y, i2, i3;
        iy = id;
        d3_y = d2;
        d2_y = d3;
        i4 = iy % d4;
        iy = iy / d4;
        i3_y = iy % d3_y;
        iy = iy / d3_y;
        i2_y = iy % d2_y;
        i1 = iy / d2_y;

        i2 = i3_y;
        i3 = i2_y;
        ix = i1 * d2 * d3 * d4 + i2 * d3 * d4 + i3 * d4 + i4;
        iy = id;
        y[iy] = x[ix];
        ++id;
    }
}

// Y <- kx*X + ky*Y
// Y(imb, ich, ir*stride_y, ic*stride_y) <- kx * X(imb, ich, ir*stride_x, ic*stride_x) + ky * Y(imb, ich, ir*stride_y, ic*stride_y)
void math21_vector_feature2d_add_2_cpu(
        int mini_batch_size,
        float kx, const float *X, int nch_X, int nr_X, int nc_X,
        float ky, float *Y, int nch_Y, int nr_Y, int nc_Y) {

    int nch = math21_number_min_2_int(nch_X, nch_Y);
    int nr = math21_number_min_2_int(nr_X, nr_Y);
    int nc = math21_number_min_2_int(nc_X, nc_Y);

    float stride_r_x = (float) nr_X / nr;
    float stride_r_y = (float) nr_Y / nr;
    float stride_c_x = (float) nc_X / nc;
    float stride_c_y = (float) nc_Y / nc;

    int ic, ir, ich, imb;
    for (imb = 0; imb < mini_batch_size; ++imb) {
        for (ich = 0; ich < nch; ++ich) {
            for (ir = 0; ir < nr; ++ir) {
                for (ic = 0; ic < nc; ++ic) {
                    // X(imb, ich, ir*stride_r_x, ic*stride_c_x)
                    int index_X = (int) (((imb * nch_X + ich) * nr_X + ir * stride_r_x) * nc_X + ic * stride_c_x);
                    // Y(imb, ich, ir*stride_r_y, ic*stride_c_y)
                    int index_Y = (int) (((imb * nch_Y + ich) * nr_Y + ir * stride_r_y) * nc_Y + ic * stride_c_y);
                    Y[index_Y] = kx * X[index_X] + ky * Y[index_Y];
                }
            }
        }
    }
}

// Y <- kx1*X1 + kx2*X2 + ky*Y
// Y(imb, ich, ir*stride_y, ic*stride_y) <- kx * X(imb, ich, ir*stride_x, ic*stride_x) + ky * Y(imb, ich, ir*stride_y, ic*stride_y)
void math21_vector_feature2d_add_3_cpu(
        int mini_batch_size,
        float kx, const float *X, int nch_X, int nr_X, int nc_X,
        float kx2, const float *X2, int nch_X2, int nr_X2, int nc_X2,
        float ky, float *Y, int nch_Y, int nr_Y, int nc_Y) {

    int nch = math21_number_min_3_int(nch_X, nch_X2, nch_Y);
    int nr = math21_number_min_3_int(nr_X, nr_X2, nr_Y);
    int nc = math21_number_min_3_int(nc_X, nc_X2, nc_Y);

    float stride_r_x = (float) nr_X / nr;
    float stride_r_x2 = (float) nr_X2 / nr;
    float stride_r_y = (float) nr_Y / nr;
    float stride_c_x = (float) nc_X / nc;
    float stride_c_x2 = (float) nc_X2 / nc;
    float stride_c_y = (float) nc_Y / nc;

    int ic, ir, ich, imb;
    for (imb = 0; imb < mini_batch_size; ++imb) {
        for (ich = 0; ich < nch; ++ich) {
            for (ir = 0; ir < nr; ++ir) {
                for (ic = 0; ic < nc; ++ic) {
                    // X(imb, ich, ir*stride_r_x, ic*stride_c_x)
                    int index_X = (int) (((imb * nch_X + ich) * nr_X + ir * stride_r_x) * nc_X + ic * stride_c_x);
                    // X2(imb, ich, ir*stride_r_x2, ic*stride_c_x2)
                    int index_X2 = (int) (((imb * nch_X2 + ich) * nr_X2 + ir * stride_r_x2) * nc_X2 + ic * stride_c_x2);
                    // Y(imb, ich, ir*stride_r_y, ic*stride_c_y)
                    int index_Y = (int) (((imb * nch_Y + ich) * nr_Y + ir * stride_r_y) * nc_Y + ic * stride_c_y);
                    Y[index_Y] = kx * X[index_X] + kx2 * X2[index_X2] + ky * Y[index_Y];
                }
            }
        }
    }
}

// Y <- k * upsample(X)
// X <- k * downsample(Y), downsample is sumdownsample
void math21_vector_feature2d_sample_cpu(
        int mini_batch_size,
        float *X, int nch_X, int nr_X, int nc_X, int stride_X, int is_upsample, float k, float *Y) {
    int ic, ir, ich, imb;
    for (imb = 0; imb < mini_batch_size; ++imb) {
        for (ich = 0; ich < nch_X; ++ich) {
            for (ir = 0; ir < nr_X * stride_X; ++ir) {
                for (ic = 0; ic < nc_X * stride_X; ++ic) {
                    int index_X =
                            imb * nch_X * nr_X * nc_X + ich * nr_X * nc_X + (ir / stride_X) * nc_X + ic / stride_X;
                    int index_Y =
                            imb * nch_X * nr_X * nc_X * stride_X * stride_X + ich * nc_X * nr_X * stride_X * stride_X +
                            ir * nc_X * stride_X +
                            ic;
                    // upsample
                    if (is_upsample) Y[index_Y] += k * X[index_X];
                        // sumdownsample
                    else X[index_X] += k * Y[index_Y];
                }
            }
        }
    }
}

// clip x, so that -k <= x <= k
void math21_vector_clip_cpu(int n, float k, float *x, int stride) {
    int i;
    for (i = 0; i < n; ++i) x[i * stride] = fmin(k, fmax(-k, x[i * stride]));
}

void math21_vector_xy_cpu(int n, const float *x, int stride_x, float *y, int stride_y) {
    int i;
    for (i = 0; i < n; ++i) y[i * stride_y] *= x[i * stride_x];
}

void math21_vector_pr_rand_uniform_01_cpu(float *v, int size) {
    int i;
    for (i = 0; i < size; ++i) {
        v[i] = math21_pr_rand_uniform(0, 1);
    }
}

// error = -y
// y = -|x-t|, dx = -1 when x>t, 1 otherwise
void math21_vector_loss_l1_cpu(int n, const float *x, const float *t, float *dx, float *error) {
    int i;
    for (i = 0; i < n; ++i) {
        float diff = t[i] - x[i];
        error[i] = fabs(diff);
        dx[i] = diff > 0 ? 1 : -1;
    }
}

// error = -2y
// y = -0.5*(x-t)^2, dx = -(x-t) = t-x
void math21_vector_loss_l2_cpu(int n, const float *x, const float *t, float *dx, float *error) {
    int i;
    for (i = 0; i < n; ++i) {
        float diff = t[i] - x[i];
        error[i] = diff * diff;
        dx[i] = diff;
    }
}

// error = -2y
// y = -0.5*(x-t)^2, when |x-t| <= delta
// y = -delta*(|x-t| - 0.5*delta), otherwise
// dx = -(x-t) = t-x, when |x-t| <= delta
// dx = -delta, when x-t>delta
// dx = delta, when x-t < -delta
// delta = 1
// Huber loss
void math21_vector_loss_smooth_l1_cpu(int n, const float *x, const float *t, float *dx, float *error) {
    int i;
    for (i = 0; i < n; ++i) {
        float diff = t[i] - x[i];
        float abs_val = fabs(diff);
        if (abs_val < 1) {
            error[i] = diff * diff;
            dx[i] = diff;
        } else {
            error[i] = 2 * abs_val - 1;
            dx[i] = (diff > 0) ? 1 : -1;
        }
    }
}
