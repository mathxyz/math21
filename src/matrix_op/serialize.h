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

#include <fstream>
#include "inner.h"

namespace math21 {

    template<typename T>
    void math21_io_serialize(std::ostream &out, const Tensor <T> &m, SerializeNumInterface &sn) {
        math21_io_serialize(out, (NumN) m.isColumnMajor(), sn);
        math21_io_serialize(out, m.shape(), sn);
        if (sn.isBinary() && m.isContinuous() && m.isBasicType()) {
            if(m.is_cpu()){
                math21_io_serialize(out, (const NumN8 *) m.getDataAddress(), m.size() * sizeof(T), sn);
            }else{
                Tensor<T> m_c;
                m_c = m;
                math21_io_serialize(out, (const NumN8 *) m_c.getDataAddress(), m.size() * sizeof(T), sn);
            }
        } else {
            detail::math21_io_serialize_container(out, m, sn);
        }
    }

    template<typename T>
    void math21_io_deserialize(std::istream &in, Tensor <T> &m, DeserializeNumInterface &sn) {
        NumN is_column_major;
        math21_io_deserialize(in, is_column_major, sn);
        ArrayN d;
        math21_io_deserialize(in, d, sn);
        m.setColumnMajor((NumB) is_column_major);
        m.setSize(d);
        if (sn.isBinary() && m.isContinuous() && m.isBasicType()) {
            if(m.is_cpu()){
                math21_io_deserialize(in, (NumN8 *) m.getDataAddress(), m.size() * sizeof(T), sn);
            } else{
                MATH21_ASSERT(0)
//                Tensor<T> m_c;
//                m_c.setColumnMajor((NumB) is_column_major);
//                m_c.setSize(d);
//                math21_io_deserialize(in, (NumN8 *) m_c.getDataAddress(), m.size() * sizeof(T), sn);
//                m = m_c;
            }
        } else {
            detail::math21_io_deserialize_container(in, m, sn);
        }
    }
}