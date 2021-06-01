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

#include "../matrix/files.h"
#include "../ad/files.h"
#include "point_c.h"
#include "point_cc.h"

namespace math21 {
    std::string math21_point_type_name(const m21point &point) {
        return math21_type_name(point.type);
    }

    template<>
    m21point &math21_cast_to_T(m21point point) {
        MATH21_ASSERT(0, "m21point does't support cast")
        return *(m21point *) point.p;
    }

    template<>
    m21point math21_cast_to_point(const m21point &x) {
        return x;
    }

    void math21_point_log_cc(m21point point, const char *s) {
        math21_point_log_cc(std::cout, point, s);
    }

    void math21_point_log_cc(std::ostream &out, m21point point, const char *s) {
        NumN type = point.type;
        if (type == m21_type_TenN) {
            math21_cast_to_T<TenN>(point).log(out, s, 0, 0);
        } else if (type == m21_type_TenZ) {
            math21_cast_to_T<TenZ>(point).log(out, s, 0, 0);
        } else if (type == m21_type_TenR) {
            math21_cast_to_T<TenR>(point).log(out, s, 0, 0);
        } else if (type == m21_type_TenStr) {
            math21_cast_to_T<TenStr>(point).log(out, s, 0, 0);
        } else if (type == m21_type_NumN) {
            math21_print(out, math21_cast_to_T<NumN>(point), s);
        } else if (type == m21_type_NumZ) {
            math21_print(out, math21_cast_to_T<NumZ>(point), s);
        } else if (type == m21_type_NumR) {
            math21_print(out, math21_cast_to_T<NumR>(point), s);
        } else if (type == m21_type_string) {
            math21_print(out, math21_cast_to_T<std::string>(point), s);
        } else {
            m21warn("point log not implement! type", math21_point_type_name(point));
        }
    }

    void math21_io_serialize(std::ostream &out, const m21point &point, SerializeNumInterface &sn) {
        math21_io_serialize(out, point.type, sn);
        NumN type = point.type;
        if (type == m21_type_TenN) {
            math21_io_serialize(out, math21_cast_to_T<TenN>(point), sn);
        } else if (type == m21_type_TenZ) {
            math21_io_serialize(out, math21_cast_to_T<TenZ>(point), sn);
        } else if (type == m21_type_TenR) {
            math21_io_serialize(out, math21_cast_to_T<TenR>(point), sn);
        } else if (type == m21_type_ad_point) {
//            math21_io_serialize(out, math21_cast_to_T<ad::ad_point>(point), sn);
        } else {
            MATH21_ASSERT(0, "not implement! type = " << point.type << "\n");
        }
    }

    void math21_io_deserialize(std::istream &in, m21point &point, DeserializeNumInterface &sn) {
        NumN type;
        math21_io_deserialize(in, type, sn);
        point = math21_point_create_by_type(type);
        if (type == m21_type_TenN) {
            math21_io_deserialize(in, math21_cast_to_T<TenN>(point), sn);
        } else if (type == m21_type_TenZ) {
            math21_io_deserialize(in, math21_cast_to_T<TenZ>(point), sn);
        } else if (type == m21_type_TenR) {
            math21_io_deserialize(in, math21_cast_to_T<TenR>(point), sn);
        } else {
            MATH21_ASSERT(0, "not implement! type = " << point.type << "\n");
        }
    }
}

using namespace math21;

template<typename T>
void math21_operator_tensor_nd_create(Tensor<T> *&px, const VecN &d, void *data, NumB isDataShared) {
    px = new Tensor<T>();
    Tensor<T> &x = *px;
    if (isDataShared && data) {
        x.setSizeNoSpace(d);
        x.setSpace(data, x.size() * sizeof(T));
    } else {
        x.setSize(d);
        if (data) {
            math21_memory_memcpy(x.getDataAddress(), data, sizeof(T) * x.size());
        }
    }
}

m21point math21_tensor_1d_create(NumN type, NumN size, void *data, NumB isDataShared) {
    VecN d(1);
    d = size;
    m21point y = math21_tensor_nd_create(type, math21_cast_to_point(d), data, isDataShared);
    return y;
}

NumN math21_type_num_to_ten(NumN type) {
    if (type == m21_type_NumN) {
        return m21_type_TenN;
    } else if (type == m21_type_NumZ) {
        return m21_type_TenZ;
    } else if (type == m21_type_NumR) {
        return m21_type_TenR;
    }
    return type;
}

m21point math21_tensor_nd_create(NumN type0, m21point pd, void *data, NumB isDataShared) {
    NumN type = math21_type_num_to_ten(type0);
    const VecN &d = math21_cast_to_T<VecN>(pd);
    m21point point = {0};
    point.type = type;
    if (type == m21_type_TenN) {
        TenN *px;
        math21_operator_tensor_nd_create(px, d, data, isDataShared);
        point.p = px;
    } else if (type == m21_type_TenZ) {
        TenZ *px;
        math21_operator_tensor_nd_create(px, d, data, isDataShared);
        point.p = px;
    } else if (type == m21_type_TenR) {
        TenR *px;
        math21_operator_tensor_nd_create(px, d, data, isDataShared);
        point.p = px;
    } else {
        m21warn("type not tensor or num, type", type0);
        point.type = 0;
        point.p = 0;
    }
    return point;
}

m21point math21_point_create_by_type(NumN type) {
    m21point point = {0};
    point.type = type;
    if (type == m21_type_TenN) {
        point.p = new TenN();
    } else if (type == m21_type_TenZ) {
        point.p = new TenZ();
    } else if (type == m21_type_TenR) {
        point.p = new TenR();
    } else {
        MATH21_ASSERT(0, "not implement! type = " << point.type << "\n");
    }
    point.refCount = static_cast<NumN *>(math21_vector_malloc_cpu(sizeof(*point.refCount)));
    *point.refCount = 1;
    return point;
}

m21point math21_point_create_ad_point_const(m21point tenPoint) {
    auto *px = new ad::ad_point(math21_cast_to_T<TenR>(tenPoint), 0);
    return math21_cast_to_point(*px);
}

m21point math21_point_create_ad_point_input(m21point tenPoint) {
    auto *px = new ad::ad_point(math21_cast_to_T<TenR>(tenPoint), 1);
    return math21_cast_to_point(*px);
}

void math21_point_ad_point_set_value(m21point adPoint, m21point tenPoint) {
    auto &px = math21_cast_to_T<ad::ad_point>(adPoint);
    auto &value = math21_cast_to_T<TenR>(tenPoint);
    ad::ad_get_value(px) = value;
}

m21point math21_point_ad_point_get_value(m21point adPoint) {
    auto &px = math21_cast_to_T<ad::ad_point>(adPoint);
    return math21_cast_to_point(ad::ad_get_value(px));
}

void math21_tensor_destroy(m21point point) {
    NumN type = point.type;
    if (type == m21_type_TenN) {
        delete &math21_cast_to_T<TenN>(point);
    } else if (type == m21_type_TenZ) {
        delete &math21_cast_to_T<TenZ>(point);
    } else if (type == m21_type_TenR) {
        delete &math21_cast_to_T<TenR>(point);
    } else {
        MATH21_ASSERT(0, "type not tensor! type = " << point.type << "\n");
    }
}

// todo: check refCount
m21point math21_tensor_copy_shape(m21point point) {
    NumN type = point.type;
    if (type == m21_type_TenN) {
//        VecN *pd = new VecN();
//        VecN &d = *pd;
        VecN &d = *(new VecN());
        auto &y = math21_cast_to_T<TenN>(point);
        return math21_cast_to_point(y.shape(d));
    } else if (type == m21_type_TenZ) {
        VecN &d = *(new VecN());
        auto &y = math21_cast_to_T<TenZ>(point);
        return math21_cast_to_point(y.shape(d));
    } else if (type == m21_type_TenR) {
        VecN &d = *(new VecN());
        auto &y = math21_cast_to_T<TenR>(point);
        return math21_cast_to_point(y.shape(d));
    } else {
        MATH21_ASSERT(0, "type not tensor! type = " << point.type << "\n");
    }
    m21point point_empty = {0};
    return point_empty;
}

NumN math21_tensor_size(m21point point) {
    NumN type = point.type;
    if (type == m21_type_TenN) {
        auto &y = math21_cast_to_T<TenN>(point);
        return y.size();
    } else if (type == m21_type_TenZ) {
        auto &y = math21_cast_to_T<TenZ>(point);
        return y.size();
    } else if (type == m21_type_TenR) {
        auto &y = math21_cast_to_T<TenR>(point);
        return y.size();
    } else {
        MATH21_ASSERT(0, "type not tensor! type = " << point.type << "\n");
    }
    return 0;
}

void *math21_tensor_data(m21point point) {
    NumN type = point.type;
    if (type == m21_type_TenN) {
        auto &y = math21_cast_to_T<TenN>(point);
        return y.getDataAddress();
    } else if (type == m21_type_TenZ) {
        auto &y = math21_cast_to_T<TenZ>(point);
        return y.getDataAddress();
    } else if (type == m21_type_TenR) {
        auto &y = math21_cast_to_T<TenR>(point);
        return y.getDataAddress();
    } else {
        MATH21_ASSERT(0, "type not tensor! type = " << point.type << "\n");
    }
    return 0;
}

NumB math21_point_is_empty(m21point point) {
    if (point.p == 0) {
        return 1;
    } else {
        return 0;
    }
}

// can add other types.
void math21_point_deallocate(m21point point) {
    NumN type = point.type;
    if (type == m21_type_TenN) {
        delete &math21_cast_to_T<TenN>(point);
    } else if (type == m21_type_TenZ) {
        delete &math21_cast_to_T<TenZ>(point);
    } else if (type == m21_type_TenR) {
        delete &math21_cast_to_T<TenR>(point);
    } else if (type == m21_type_ad_point) {
        delete &math21_cast_to_T<ad::ad_point>(point);
    } else {
        m21warn("type not implement, type", type);
    }
    math21_memory_free_cpu(point.refCount);
}

void math21_point_addref(m21point point) {
    if (point.refCount) {
        (*point.refCount)++;
    }
}

m21point math21_point_share_assign(m21point point) {
    math21_point_addref(point);
    return point;
}

m21point math21_point_assign(m21point point) {
    return point;
}

m21point math21_point_init(m21point point) {
    point.type = 0;
    point.p = 0;
    point.refCount = 0;
    return point;
}

m21point math21_point_clear(m21point point) {
    if (!math21_point_is_empty(point)) {
        if (point.refCount) {
            (*point.refCount)--;
            if (*point.refCount == 0) {
                math21_point_deallocate(point);
            }
        }
        point.type = 0;
        point.p = 0;
        point.refCount = 0;
    }
    return point;
}

m21point math21_point_destroy(m21point point) {
    return math21_point_clear(point);
}

void math21_point_log(m21point point) {
    math21_point_log_cc(point);
}

NumB math21_point_save(m21point point, const char *path) {
    if (!math21_io_generic_type_write_to_file(point, path, 1)) {
        return 0;
    }
    return 1;
}

m21point math21_point_load(const char *path) {
    m21point point = {0};
    if (!math21_io_generic_type_read_from_file(point, path, 1)) {
        return point; // empty
    }
    return point;
}

NumB math21_point_isEqual(m21point x, m21point y, NumR epsilon = 0) {
    MATH21_ASSERT(0, "not implement!")
    return 0;
}

// release static variables.
// release some modules before main exits.
void math21_destroy() {
#ifdef MATH21_FLAG_USE_OPENCL
    math21_ad_destroy();
    math21_opencl_destroy();
    m21log("math21 destroyed!");
#endif
}