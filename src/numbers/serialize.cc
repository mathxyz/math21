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
#include "serialize.h"
#include "std_wrapper.h"

namespace math21 {

    NumN math21_type_get(const NumN &m) {
        return m21_type_NumN;
    }

    NumN math21_type_get(const NumZ &m) {
        return m21_type_NumZ;
    }

    NumN math21_type_get(const NumR &m) {
        return m21_type_NumR;
    }

    NumN math21_type_get(const NumSize &m) {
        return m21_type_NumSize;
    }

    template<>
    NumN math21_type_get<NumN>(const TenN &m) {
        return m21_type_TenN;
    }

    template<>
    NumN math21_type_get<NumN8>(const TenN8 &m) {
        MATH21_ASSERT(0, "template not needed!");
        return m21_type_TenN8;
    }

    template<>
    NumN math21_type_get<NumZ>(const TenZ &m) {
        return m21_type_TenZ;
    }

    template<>
    NumN math21_type_get<NumR>(const TenR &m) {
        return m21_type_TenR;
    }

    template<>
    NumN math21_type_get<NumN>() {
        return m21_type_NumN;
    }

    template<>
    NumN math21_type_get<NumN8>() {
        return m21_type_NumN8;
    }

    template<>
    NumN math21_type_get<NumZ>() {
        return m21_type_NumZ;
    }

    template<>
    NumN math21_type_get<NumR>() {
        return m21_type_NumR;
    }

    template<>
    NumN math21_type_get<NumSize>() {
        return m21_type_NumSize;
    }

#if defined(MATH21_USE_NUMR32)
    template<>
    NumN math21_type_get<NumR64>() {
        return m21_type_NumR64;
    }
#else

    template<>
    NumN math21_type_get<NumR32>() {
        return m21_type_NumR32;
    }

#endif

    template<>
    NumN math21_type_get<TenN>() {
        return m21_type_TenN;
    }

    template<>
    NumN math21_type_get<TenN8>() {
        return m21_type_TenN8;
    }

    template<>
    NumN math21_type_get<TenZ>() {
        return m21_type_TenZ;
    }

    template<>
    NumN math21_type_get<TenR>() {
        return m21_type_TenR;
    }

    template<>
    NumN math21_type_get<std::string>() {
        return m21_type_string;
    }

    template<>
    NumN math21_type_get<TenStr>() {
        return m21_type_TenStr;
    }

    template<>
    NumN math21_type_get<ad::ad_point>() {
        return m21_type_ad_point;
    }

    template<>
    std::string math21_type_name<NumN>() {
        return "NumN";
    }

    template<>
    std::string math21_type_name<NumZ>() {
        return "NumZ";
    }

    template<>
    std::string math21_type_name<NumR>() {
        return "NumR";
    }

#if defined(MATH21_USE_NUMR32)
    template<>
    std::string math21_type_name<NumR64>() {
        return "NumR64";
    }
#else

    template<>
    std::string math21_type_name<NumR32>() {
        return "NumR32";
    }

#endif

    template<>
    std::string math21_type_name<TenR>() {
        return "TenR";
    }

    std::string math21_type_name(NumN type) {
#define MATH21_LOCAL_F(a) case a: return MATH21_STRINGIFY(a);
        switch (type) {
            MATH21_LOCAL_F(m21_type_none)
            MATH21_LOCAL_F(m21_type_default)
            MATH21_LOCAL_F(m21_type_NumN)
            MATH21_LOCAL_F(m21_type_NumZ)
            MATH21_LOCAL_F(m21_type_NumR)
            MATH21_LOCAL_F(m21_type_Seqce)
            MATH21_LOCAL_F(m21_type_Tensor)
            MATH21_LOCAL_F(m21_type_Digraph)
            MATH21_LOCAL_F(m21_type_vector_float_c)
            MATH21_LOCAL_F(m21_type_vector_char_c)
            MATH21_LOCAL_F(m21_type_NumR32)
            MATH21_LOCAL_F(m21_type_NumR64)
            MATH21_LOCAL_F(m21_type_TenN)
            MATH21_LOCAL_F(m21_type_TenZ)
            MATH21_LOCAL_F(m21_type_TenR)
            MATH21_LOCAL_F(m21_type_ad_point)
            MATH21_LOCAL_F(m21_type_NumSize)
            default:
                return "UNKNOWN";
        }
#undef MATH21_LOCAL_F
    }

    void math21_io_serialize(std::ostream &out, const NumN &m, SerializeNumInterface &sn) {
        sn.serialize(out, m);
    }

    void math21_io_serialize(std::ostream &out, const NumZ &m, SerializeNumInterface &sn) {
        sn.serialize(out, m);
    }

    void math21_io_serialize(std::ostream &out, const NumR &m, SerializeNumInterface &sn) {
        sn.serialize(out, m);
    }

    void math21_io_serialize(std::ostream &out, const NumN8 *v, NumN n, SerializeNumInterface &sn) {
        sn.serialize(out, v, n);
    }

    void math21_io_serialize_header(std::ostream &out, SerializeNumInterface &sn) {
        NumN m;
        std::string s = "math21";
        for (NumN i = 1; i <= s.size(); ++i) {
            m = (NumN) s.c_str()[i - 1];
            sn.serialize(out, m);
        }
    }

    void math21_io_serialize(std::ostream &out, const std::string &m, SerializeNumInterface &sn) {
        NumN n = m.size();
        math21_io_serialize(out, n, sn);
        out.write(&m[0], m.size());
    }

    void math21_io_deserialize(std::istream &in, NumN &m, DeserializeNumInterface &sn) {
        sn.deserialize(in, m);
    }

    void math21_io_deserialize(std::istream &in, NumZ &m, DeserializeNumInterface &sn) {
        sn.deserialize(in, m);
    }

    void math21_io_deserialize(std::istream &in, NumR &m, DeserializeNumInterface &sn) {
        sn.deserialize(in, m);
    }

    void math21_io_deserialize(std::istream &in, NumN8 *v, NumN n, DeserializeNumInterface &sn) {
        sn.deserialize(in, v, n);
    }

#if defined(MATH21_USE_NUMR32)
    void math21_io_deserialize(std::istream &in, NumR64 &m, DeserializeNumInterface &sn) {
        MATH21_ASSERT(0, "not support NumR64!");
    }
#else

    void math21_io_deserialize(std::istream &in, NumR32 &m, DeserializeNumInterface &sn) {
        MATH21_ASSERT(0, "not support NumR32!");
    }

#endif

    NumB math21_io_deserialize_header(std::istream &in, DeserializeNumInterface &sn) {
        NumN m;
        NumN m_old;
        std::string s = "math21";
        for (NumN i = 1; i <= s.size(); ++i) {
            sn.deserialize(in, m);
            m_old = (NumN) s.c_str()[i - 1];
            if (m != m_old) {
                return 0;
            }
        }
        return 1;
    }

    // deprecated, use TenN8 instead of std::string
    void math21_io_deserialize(std::istream &in, std::string &m, DeserializeNumInterface &sn) {
        NumN n;
        math21_io_deserialize(in, n, sn);
        math21_tool_std_string_resize(m, n);

        in.read(&m[0], m.size());
    }

    NumN math21_io_read_file(const char *path, NumN8 *&data, size_t size) {
        FILE *fp = 0;
        if ((fp = fopen(path, "rb"))) {
        } else {
            printf("Failed to open file %s\n", path);
            return 0;
        }
        if (feof(fp)) {
            return 2;
        }
        size_t read = fread(data, size, 1, fp);
        if (read != 1) {
            return 0;
        }
        fclose(fp);
        return 1;
    }

    NumN math21_io_write_file(const char *path, const NumN8 *data, size_t size) {
        FILE *fp = 0;
        if ((fp = fopen(path, "wb"))) {
        } else {
            printf("Failed to open file %s\n", path);
            return 0;
        }
        if (feof(fp)) {
            return 2;
        }
        size_t write = fwrite(data, size, 1, fp);
        if (write != 1) {
            return 0;
        }
        fclose(fp);
        return 1;
    }

    void math21_io_serialize_type(std::ostream &out, SerializeNumInterface &sn,
                                  const NumN &type) {
        math21_io_serialize(out, type, sn);
    }

    void math21_io_deserialize_type(std::istream &in, DeserializeNumInterface &dsn, NumN &type) {
        math21_io_deserialize(in, type, dsn);
    }

    NumB math21_io_read_type_from_file(const char *path, NumN &type) {
        NumB flag = 1;
        std::ifstream in;
        in.open(path, std::ifstream::binary);
        if (in.is_open()) {
            DeserializeNumInterface_simple dsn;
            math21_io_deserialize_type(in, dsn, type);
        } else {
            printf("open %s fail!\n", path);
            flag = 0;
        }
        in.close();
        return flag;
    }

}