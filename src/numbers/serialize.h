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
#include "number.h"
#include "_assert.h"

namespace math21 {

    class SerializeNumInterface {
    public:
        SerializeNumInterface() {}

        virtual void serialize(std::ostream &out, const NumN &m) = 0;

        virtual void serialize(std::ostream &out, const NumZ &m) = 0;

        virtual void serialize(std::ostream &out, const NumR &m) = 0;

        virtual NumB isBinary() const = 0;

        // buffer
        // if binary, this must be implemented.
        virtual void serialize(std::ostream &out, const NumN8 *v, NumN n) {math21_tool_assert(0); };
    };

    class DeserializeNumInterface {
    public:
        DeserializeNumInterface() {}

        virtual void deserialize(std::istream &in, NumN &m) = 0;

        virtual void deserialize(std::istream &in, NumZ &m) = 0;

        virtual void deserialize(std::istream &in, NumR &m) = 0;

        virtual NumB isBinary() const = 0;

        // buffer
        virtual void deserialize(std::istream &in, NumN8 *v, NumN n) {math21_tool_assert(0); };
    };

    class SerializeNumInterface_simple : public SerializeNumInterface {
    public:
        SerializeNumInterface_simple() {}

        void serialize(std::ostream &out, const NumN &m) override {
            out.write((const char *) &m, sizeof(m));
        }

        void serialize(std::ostream &out, const NumZ &m) override {
            out.write((const char *) &m, sizeof(m));
        }

        void serialize(std::ostream &out, const NumR &m) override {
            out.write((const char *) &m, sizeof(m));
        }

        NumB isBinary() const override {
            return 1;
        }

        void serialize(std::ostream &out, const NumN8 *v, NumN n) override {
            out.write((const char *) v, n);
        }
    };

    class DeserializeNumInterface_simple : public DeserializeNumInterface {
    public:
        DeserializeNumInterface_simple() {}

        void deserialize(std::istream &in, NumN &m) override {
            in.read((char *) &m, sizeof(m));
        }

        void deserialize(std::istream &in, NumZ &m) override {
            in.read((char *) &m, sizeof(m));
        }

        void deserialize(std::istream &in, NumR &m) override {
            in.read((char *) &m, sizeof(m));
        }

        NumB isBinary() const override {
            return 1;
        }

        void deserialize(std::istream &in, NumN8 *v, NumN n) override {
            in.read((char *) v, n);
        }
    };

    class SerializeNumInterface_text : public SerializeNumInterface {
    public:
        SerializeNumInterface_text() {}

        void serialize(std::ostream &out, const NumN &m) override {
            out << m << std::endl;
        }

        void serialize(std::ostream &out, const NumZ &m) override {
            out << m << std::endl;
        }

        void serialize(std::ostream &out, const NumR &m) override {
            out << m << std::endl;
        }

        NumB isBinary() const override {
            return 0;
        }

        void serialize(std::ostream &out, const NumN8 *v, NumN n) override {math21_tool_assert(0); };
    };

    class DeserializeNumInterface_text : public DeserializeNumInterface {
    public:
        DeserializeNumInterface_text() {}

        void deserialize(std::istream &in, NumN &m) override {
            in >> m;
        }

        void deserialize(std::istream &in, NumZ &m) override {
            in >> m;
        }

        void deserialize(std::istream &in, NumR &m) override {
            in >> m;
        }

        NumB isBinary() const override {
            return 0;
        }

        void deserialize(std::istream &in, NumN8 *v, NumN n) override {math21_tool_assert(0); };
    };

    NumN math21_type_get(const NumN &m);

    NumN math21_type_get(const NumZ &m);

    NumN math21_type_get(const NumR &m);

    NumN math21_type_get(const NumSize &m);

    template<typename T>
    NumN math21_type_get(const Tensor<T> &m) {
        return m21_type_Tensor;
    }

    template<>
    NumN math21_type_get(const TenN &m);

    template<>
    NumN math21_type_get(const TenN8 &m);

    template<>
    NumN math21_type_get(const TenZ &m);

    template<>
    NumN math21_type_get(const TenR &m);

    template<typename T>
    NumN math21_type_get() {
        MATH21_ASSERT(0)
        return m21_type_default;
    }

    template<>
    NumN math21_type_get<NumN>();

    template<>
    NumN math21_type_get<NumN8>();

    template<>
    NumN math21_type_get<NumZ>();

    template<>
    NumN math21_type_get<NumR>();

    template<>
    NumN math21_type_get<NumSize>();

#if defined(MATH21_USE_NUMR32)

    template<>
    NumN math21_type_get<NumR64>();

#else

    template<>
    NumN math21_type_get<NumR32>();

#endif

    template<>
    NumN math21_type_get<TenN>();

    template<>
    NumN math21_type_get<TenN8>();

    template<>
    NumN math21_type_get<TenZ>();

    template<>
    NumN math21_type_get<TenR>();

    template<>
    NumN math21_type_get<std::string>();

    template<>
    NumN math21_type_get<TenStr>();

    template<>
    NumN math21_type_get<ad::ad_point>();

    template<typename T>
    std::string math21_type_name() {
        return "m21_type_default";
    }

    template<>
    std::string math21_type_name<NumN>();

    template<>
    std::string math21_type_name<NumZ>();

    template<>
    std::string math21_type_name<NumR>();

#if defined(MATH21_USE_NUMR32)

    template<>
    std::string math21_type_name<NumR64>();

#else

    template<>
    std::string math21_type_name<NumR32>();

#endif

    template<>
    std::string math21_type_name<TenR>();

    std::string math21_type_name(NumN type);

    void math21_io_serialize(std::ostream &out, const NumN &m, SerializeNumInterface &sn);

    void math21_io_serialize(std::ostream &out, const NumZ &m, SerializeNumInterface &sn);

    void math21_io_serialize(std::ostream &out, const NumR &m, SerializeNumInterface &sn);

    void math21_io_serialize(std::ostream &out, const NumN8 *v, NumN n, SerializeNumInterface &sn);

    void math21_io_serialize_header(std::ostream &out, SerializeNumInterface &sn);

    void math21_io_serialize(std::ostream &out, const std::string &m, SerializeNumInterface &sn);

    void math21_io_deserialize(std::istream &in, NumN &m, DeserializeNumInterface &sn);

    void math21_io_deserialize(std::istream &in, NumZ &m, DeserializeNumInterface &sn);

    void math21_io_deserialize(std::istream &in, NumR &m, DeserializeNumInterface &sn);

    void math21_io_deserialize(std::istream &in, NumN8 *v, NumN n, DeserializeNumInterface &sn);

#if defined(MATH21_USE_NUMR32)

    void math21_io_deserialize(std::istream &in, NumR64 &m, DeserializeNumInterface &sn);

#else

    void math21_io_deserialize(std::istream &in, NumR32 &m, DeserializeNumInterface &sn);

#endif

    NumB math21_io_deserialize_header(std::istream &in, DeserializeNumInterface &sn);

    void math21_io_deserialize(std::istream &in, std::string &m, DeserializeNumInterface &sn);

    NumN math21_io_read_file(const char *path, NumN8 *&data, size_t size);

    NumN math21_io_write_file(const char *path, const NumN8 *data, size_t size);

    void math21_io_serialize_type(std::ostream &out, SerializeNumInterface &sn,
                                  const NumN &type);

    void math21_io_deserialize_type(std::istream &in, DeserializeNumInterface &dsn, NumN &type);

    NumB math21_io_read_type_from_file(const char *path, NumN &type);

    template<typename T>
    NumB math21_io_generic_type_write_to_file(const T &A, const char *path, NumB isUseHeader = 1, NumB binary = 1) {
        NumB flag = 1;
        std::ofstream out;

        SerializeNumInterface *p_sn;
        SerializeNumInterface_text sn_text;
        SerializeNumInterface_simple sn_bin;
        if (!binary) {
            p_sn = &sn_text;
        } else {
            p_sn = &sn_bin;
        }
        SerializeNumInterface &sn = *p_sn;

        if (!binary) {
            out.open(path);
        } else {
            out.open(path, std::ofstream::binary);
        }

        if (out.is_open()) {
            if (isUseHeader) {
                math21_io_serialize_header(out, sn);
            }
//            NumN type;
//            type = math21_type_get(A);
//            math21_io_serialize_type(out, sn, type);
            math21_io_serialize(out, A, sn);
        } else {
            printf("open %s fail!\n", path);
            flag = 0;
        }
        out.close();
        return flag;
    }

    template<typename T>
    NumB math21_io_generic_type_read_from_file(T &A, const char *path, NumB isUseHeader = 1, NumB binary = 1) {
        NumB flag = 1;
        std::ifstream in;

        DeserializeNumInterface *p_sn;
        DeserializeNumInterface_text sn_text;
        DeserializeNumInterface_simple sn_bin;
        if (!binary) {
            p_sn = &sn_text;
        } else {
            p_sn = &sn_bin;
        }
        DeserializeNumInterface &dsn = *p_sn;

        if (!binary) {
            in.open(path);
        } else {
            in.open(path, std::ifstream::binary);
        }

        if (in.is_open()) {
            if (isUseHeader) {
//                NumN type;
                flag = math21_io_deserialize_header(in, dsn);
//                math21_io_deserialize_type(in, dsn, type);
                if (!flag) {
                    in.close();
                    printf("Not math21 file!\n");
                    return 0;
                }
//                if (type != math21_type_get(A)) {
//                    in.close();
//                    return 0;
//                }
            }
            math21_io_deserialize(in, A, dsn);
        } else {
            printf("open %s fail!\n", path);
            flag = 0;
        }
        in.close();
        return flag;
    }
}