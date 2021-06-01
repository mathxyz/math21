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
#include "array2d.h"

using namespace math21;

// as matrix
m21array2d math21_array2d_create(NumN nr, NumN nc, NumN type) {
    m21array2d m;
    m.type = type;
    m.nr = nr;
    m.nc = nc;
    m.data = static_cast<void **>(math21_vector_calloc_cpu(m.nr, sizeof(void *)));
    int i;
    NumN size;
    if (type == m21_type_NumR32) {
        size = 4;
    } else if (type == m21_type_NumR64) {
        size = 8;
    } else if (type == m21_type_NumR) {
        size = sizeof(NumR);
    } else {
        math21_tool_assert(0 && "data type not support!");
    }
    for (i = 0; i < m.nr; ++i) {
        m.data[i] = math21_vector_calloc_cpu(m.nc, size);
    }
    return m;
}

m21array2d math21_array2d_concat_vertically(m21array2d m1, m21array2d m2) {
    m21array2d m = {0};
    math21_tool_assert(m1.nc == m2.nc || m1.nc == 0 || m2.nc == 0);
    m.nc = xjmax(m1.nc, m2.nc);
    m.nr = m1.nr + m2.nr;
    if (m.nr == 0 || m.nc == 0)return m;
    m.data = static_cast<void **>(math21_vector_calloc_cpu(m.nr, sizeof(void *)));
    int i;
    for (i = 0; i < m1.nr; ++i) {
        m.data[i] = m1.data[i];
    }
    for (i = 0; i < m2.nr; ++i) {
        m.data[m1.nr + i] = m2.data[i];
    }
    return m;
}

void math21_array2d_free(m21array2d m) {
    int i;
    for (i = 0; i < m.nr; ++i) free(m.data[i]);
    free(m.data);
}

m21data math21_tool_data_concat_2(m21data d1, m21data d2) {
    m21data d = {0};
    d.shallow = 1;
    d.x = math21_array2d_concat_vertically(d1.x, d2.x);
    d.y = math21_array2d_concat_vertically(d1.y, d2.y);
    return d;
}

m21data math21_tool_data_concat_n(m21data *d, int n) {
    int i;
    m21data out = {0};
    for (i = 0; i < n; ++i) {
        m21data newdata = math21_tool_data_concat_2(d[i], out);
        math21_tool_data_free(out);
        out = newdata;
    }
    return out;
}

void math21_tool_data_free(m21data d) {
    if (!d.shallow) {
        math21_array2d_free(d.x);
        math21_array2d_free(d.y);
    } else {
        free(d.x.data);
        free(d.y.data);
    }
}

void math21_tool_data_get_next_mini_batch(m21data d, int n, int offset, float *x, float *y) {
    int j;
    for (j = 0; j < n; ++j) {
        int index = offset + j;
        memcpy(x + j * d.x.nc, d.x.data[index], d.x.nc * sizeof(float));
        if (y) memcpy(y + j * d.y.nc, d.y.data[index], d.y.nc * sizeof(float));
    }
}

m21data math21_tool_data_get_ith_part(m21data d, int i, int n) {
    m21data p = {0};
    p.shallow = 1;
    p.x.type = d.x.type;
    p.x.nr = d.x.nr * (i + 1) / n - d.x.nr * i / n;
    p.x.nc = d.x.nc;
    p.x.data = d.x.data + d.x.nr * i / n;

    p.y.type = d.y.type;
    p.y.nr = d.y.nr * (i + 1) / n - d.y.nr * i / n;
    p.y.nc = d.y.nc;
    p.y.data = d.y.data + d.y.nr * i / n;
    return p;
}