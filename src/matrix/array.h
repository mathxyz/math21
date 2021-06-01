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

#include <iostream>
#include <stdio.h>
#include <vector>
#include "inner.h"
#include "_array.h"

/*
 * A 1d array without supporting push and pop.
 * See Seqce class.
 * Space related operations are only for basic types.
 * */
namespace math21 {

    // This class is only used for tensor shape.
    // we must assure that element access is thread-safe.
    template<typename T>
    class Array {
    private:
        const NumB basicType; // isBasicType

        // for basic types
        AutoBuffer autoBuffer;

        // for non-basic types
        BufferNew<T> bufferNew;

        NumN n; // n is not auxiliary.
        T *v; // v is auxiliary.
        void init() {
            n = 0;
            v = 0;
        }

        void updateAuxiliary() {
            if (isBasicType()) {
                if (autoBuffer.is_cpu()) {
                    v = (T *) autoBuffer.getObjCpu();
                } else {
                    v = 0;
                }
            } else {
                v = (T *) bufferNew.getObj();
            }
        }

        void destroy() {
            MATH21_ASSERT_CODE((is_cpu() && n > 0 && v != 0) || (n == 0 && v == 0) || (!is_cpu()),
                               "You've called setSizeNoSpace(), and should call setSpace() afterwards!");
            clear();
        }

        NumB _isBasicType() const {
            if (
                    typeid(T) == typeid(NumN) ||
                    typeid(T) == typeid(NumZ) ||
                    typeid(T) == typeid(NumR) ||
                    typeid(T) == typeid(NumN8) ||
                    typeid(T) == typeid(NumZ8) ||
                    typeid(T) == typeid(NumR32) ||
                    typeid(T) == typeid(NumR64) ||
                    typeid(T) == typeid(NumSize)
                    ) {
                return 1;
            } else {
                return 0;
            }
        }

    public:
        Array() : basicType(_isBasicType()) {
            init();
        }

        Array(NumN n) : basicType(_isBasicType()) {
            init();
            setSize(n);
        }

        // Copy constructor
        Array(const Array &B) : basicType(_isBasicType()) {
//            MATH21_ASSERT(0, "Copy constructor.");
            init();
            B.copyTo(*this);
        }

        // Copy constructor
        template<typename S>
        Array(const Array<S> &B):basicType(_isBasicType()) {
            init();
            MATH21_ASSERT(0, "Copy constructor.");
//            B.copyTo(*this);
        }

        virtual ~Array() {
            destroy();
        }

        void setDeviceType(NumN deviceType) {
            MATH21_ASSERT(isEmpty(), "it must be called when empty.");
            if (isBasicType()) {
                autoBuffer.setDeviceType(deviceType);
            } else {
                MATH21_ASSERT(0, "Set cpu mode not support when non-basic type.");
            }
        }

        NumN getDeviceType() const {
            if (isBasicType()) {
                return autoBuffer.getDeviceType();
            } else {
                return m21_device_type_default;
            }
        }

        NumB is_cpu() const {
            if (isBasicType()) {
                return autoBuffer.is_cpu();
            } else {
                return 1;
            }
        }

        NumB isBasicType() const {
            return basicType;
        }

        NumB isEmpty() const {
            if (n == 0) {
                MATH21_ASSERT_CODE(v == 0);
                return 1;
            }
            MATH21_ASSERT_CODE(!(is_cpu() && v == 0),
                               "You've called setSizeNoSpace(), and should call setSpace() afterwards!");
            return 0;
        }

        NumN size() const {
            return n;
        }

        // seems not used.
        //from left to right.
        NumN shape() const {
            return size();
        }

        // seems not used.
        // deprecate
        NumN nrows() const {
            return size();
        }

        // seems not used.
        NumN ncols() const {
            if (isEmpty()) {
                return 0;
            }
            return 1;
        }

        // seems not used.
        void ensureIndependence() {
            if (isBasicType()) {
                autoBuffer.ensureIndependence();
            } else {
                bufferNew.ensureIndependence();
            }
            updateAuxiliary();
        }

        // allow for any types
        void setSizeNoSpace(NumN n) {
            clear();
            if (n == 0) {
                return;
            }
//            autoBuffer.setSize(n * sizeof(T), paras);
            this->n = n;
            updateAuxiliary();
        }

        // keep size fixed, just set space.
        void setSpace(const SpaceParas *paras) {
            if (paras == 0) {
                return;
            }
            NumN n = size();
            setSize(n, paras);
        }

        // NOTE: data may not be zero
        void setSize(NumN n, const SpaceParas *paras = 0) {
            if (n == 0 && paras == 0) {
                clear();
                return;
            }
            if (isBasicType()) {
                if (paras) {
                    MATH21_ASSERT(paras->type == math21_type_get<T>(),
                                  "paras->type = " << math21_type_name(paras->type) << ", type = "
                                                   << math21_type_name<T>() << "\n");
                }
                autoBuffer.setSize(n * sizeof(T), math21_type_get<T>(), paras);
                this->n = n;
                updateAuxiliary();
            } else {
                bufferNew.setSize(n, paras);
                this->n = n;
                updateAuxiliary();
            }
        }

        NumB isIndependent() const {
            if (isBasicType()) {
                return autoBuffer.isIndependent();
            } else {
                return bufferNew.isIndependent();
            }
        }

        SpaceParas getSpace() const {
            if (isBasicType()) {
                return autoBuffer.getSpace();
            } else {
                math21_tool_assert(0 && "check");
                return bufferNew.getSpace();
            }
        }

        SpaceParas getSpace(NumN offset, NumN size, NumN unit = sizeof(char)) const {
            if (isBasicType()) {
                return autoBuffer.getSpace(offset, size, unit);
            } else {
                math21_tool_assert(0 && "check");
                return bufferNew.getSpace(offset, size, unit);
            }
        }

        const T &operator()(NumN n, NumN m) const {
            MATH21_ASSERT(is_cpu(), "non-cpu indexing not support.");
            MATH21_ASSERT_INDEX(n <= size() && m == 1);
            MATH21_ASSERT_CODE(v != 0,
                               "v is 0, maybe you forget call setSize, "
                               "or you've called setSizeNoSpace(), but didn't call setSpace() afterwards?")
            return v[n - 1];
        }

        const T &operator()(NumN n) const {
            MATH21_ASSERT(is_cpu(), "non-cpu indexing not support.");
            MATH21_ASSERT_INDEX(n <= size());
            MATH21_ASSERT_CODE(v != 0,
                               "v is 0, maybe you forget call setSize, "
                               "or you've called setSizeNoSpace(), but didn't call setSpace() afterwards?")
            return v[n - 1];
        }

        T &operator()(NumN n, NumN m) {
            MATH21_ASSERT(is_cpu(), "non-cpu indexing not support.");
            MATH21_ASSERT_INDEX(n <= size() && m == 1);
            MATH21_ASSERT_CODE(v != 0,
                               "v is 0, maybe you forget call setSize, "
                               "or you've called setSizeNoSpace(), but didn't call setSpace() afterwards?")
            return v[n - 1];
        }

        T &operator()(NumN n) {
            MATH21_ASSERT(is_cpu(), "non-cpu indexing not support.");
            MATH21_ASSERT_INDEX(n <= size());
            MATH21_ASSERT_CODE(v != 0,
                               "v is 0, maybe you forget call setSize, "
                               "or you've called setSizeNoSpace(), but didn't call setSpace() afterwards?")
            return v[n - 1];
        }

        T &at(NumN n) {
            MATH21_ASSERT(is_cpu(), "non-cpu indexing not support.");
            MATH21_ASSERT_INDEX(n <= size());
            MATH21_ASSERT_CODE(v != 0,
                               "v is 0, maybe you forget call setSize, "
                               "or you've called setSizeNoSpace(), but didn't call setSpace() afterwards?")
            return v[n - 1];
        }

        // cpu flag not cleared
        void clear() {
            n = 0;
            v = 0;
            if (isBasicType()) {
                autoBuffer.clear();
            } else {
                bufferNew.clear();
            }
        }

        NumB log(const char *s = 0, NumB isLogDetail = 1) const {
            return log(std::cout, s, isLogDetail);
        }

        NumB log(std::ostream &io, const char *s = 0, NumB isLogDetail = 1) const {
            if (!is_cpu()) {
                Array<T> A;
                copyTo(A);
                return A.log(io, s, isLogDetail);
            }
            if (s) {
                io << "Array " << s << " (size " << size() << "):\n";
            }
            if (isLogDetail) {
                if (isBasicType()) {
                    autoBuffer.log(io, s);
                } else {
                    bufferNew.log(io, s);
                }
            }
            if (size() == 0) {
                io << "{}\n";
                return 0;
            }
            for (NumN i = 1; i <= size(); ++i) {
                if (i == 1) {
                    io << "{";
                }
                if (i != size()) {
                    io << (*this)(i) << ", ";
                } else {
                    io << (*this)(size()) << "}\n";
                }
            }
            return 1;
        }

        // Continuous concept is meaningful only for all types.
        NumB isContinuous() const {
            return 1;
        }

        void copyTo(Array<T> &B) const {
            if (B.size() != size()) {
                B.setSize(size());
            }
            B.assign(*this);
        }

        //assignment
        Array<T> &operator=(const Array<T> &B) {
//            MATH21_ASSERT(0, "=");
            assign(B);
            return *this;
        }

        //assignment
        template<typename S>
        Array<T> &operator=(const Array<S> &B) {
            MATH21_ASSERT(0, "=");
//            assign(B);

//            return *this;
        }

        // used for dev.
        // use carefully
        const AutoBuffer &getAutoBuffer_dev() const {
            math21_tool_assert(isBasicType());
            return autoBuffer;
        }

        void logBuffer(std::ostream &io, const char *s) const {
            if (isBasicType()) {
                autoBuffer.log(io, s);
            } else {
                bufferNew.log(io, s);
            }
        }

        void assign(const Array<T> &B) {
            MATH21_ASSERT(B.size() == size(),
                          "vector size doesn't match in assign");
            if (B.isEmpty()) {
                return;
            }
            if (this != &B) {
                if (isContinuous() && B.isContinuous()) {
                    if (isBasicType()) {
                        autoBuffer.setDataDeep(B.autoBuffer);
                    } else {
                        bufferNew.setDataDeep(B.bufferNew);
                    }
                    updateAuxiliary();
                } else {
                    math21_tool_assert(0 && "check");
                    NumN i;
                    for (i = 1; i <= size(); i++) (*this).at(i) = B(i);
                }
            }
        }

        void assign(const T &a) {
            if (!is_cpu()) {
                Array<T> A(size());
                A.assign(a);
                assign(A);
                return;
            }
            NumN i;
            for (i = 1; i <= size(); i++) (*this).at(i) = a;
        }

        // todo: math21_vector_set_wrapper to speedup.
        template<typename S>
        void assign(const S &a) {
            if (!is_cpu()) {
                Array<T> A(size());
                A.assign(a);
                assign(A);
                return;
            }
            NumN i;
            for (i = 1; i <= size(); i++) (*this).at(i) = a;

        }

        template<typename S>
        Array &operator=(
                const literal_assign_helper_container<S, Array> &val
        ) {
            MATH21_ASSERT(0, "literal_assign_helper")
//            val.getArray()->copyTo(*this);
//            return *this;
        }

        template<typename S>
        const literal_assign_helper_container<T, Array> operator=(
                const S &val
        ) {
            // assign the given value to every spot in this matrix
            assign(val);
            // Now return the literal_assign_helper so that the user
            // can use the overloaded comma notation to initialize
            // the matrix if they want to.
            return literal_assign_helper_container<T, Array>(this);
        }

        void sort();

        template<typename Compare>
        void sort(const Compare &f);

        void swap(Array &B);
    };

    typedef Array<NumN> ArrayN;
    typedef Array<NumZ> ArrayZ;

    template<typename T>
    std::ostream &operator<<(std::ostream &out, const Array<T> &m) {
        m.log(out);
        return out;
    }

    template<typename T>
    void math21_io_serialize(std::ostream &out, const Array<T> &m, SerializeNumInterface &sn) {
        detail::math21_io_serialize_container(out, m, sn);
    }


    template<typename T>
    void math21_io_deserialize(std::istream &in, Array<T> &m, DeserializeNumInterface &sn) {
        detail::math21_io_deserialize_container(in, m, sn);
    }
}