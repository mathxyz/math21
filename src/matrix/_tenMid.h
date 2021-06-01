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

#include "array.h"
#include "vec_ops.h"
#include "_tenBefore.h"

namespace math21 {

    template<typename T>
    class TensorView;

    template<typename T>
    class TensorSub;

    // todo: remove setting zero in setSize when gpu to speed up.
    // convention: index from 1, not -1, 0 or others. Following it makes things simple.
    // If you want use tensor indexing from other than 1, use IndexFunctional instead.
    template<typename T>
    class Tensor {
    private:
        std::string name;
        Array<T> data;
        // todo: speed up using ragged tensor
        // Note: a is auxiliary, not necessary.
        Seqce<ArrayN> a; ////a is index to vector data. we can design a to be empty.
        ArrayN d; //
//        Tensor<NumN> d2;
        // Note: N is auxiliary, not necessary.
        NumN N; //N = d.size()
        NumB is_column_major; // can be inferred from a.

        // todo: deprecate
        // row_shape_size
        // if d = m1, then row_shape_size = 1
        // if d = m1 x n1, then row_shape_size = 1
        // if d = (m1 x n1), then row_shape_size = 2
        // if d = (m1) x (n1), then row_shape_size = 1
        // if d = (m1) x (n1, n2), then row_shape_size = 1
        // if d = (m1, m2) x (n1, n2), then row_shape_size = 2
        // if d = (m1, m2, m3) x (n1, n2), then row_shape_size = 3
        NumN row_shape_size; // for generalized matrix

        void _setSize_shape(const ArrayN &_d, NumB isClear = 1) {
            if (isClear) {
                if (!isEmpty()) {
                    clearSome();
                }
            }
//            if (_d.isEmpty() || math21_operator_container_min(_d) == 0) {
//                return;
//            }
            if (_d.isEmpty()) {
                return;
            }
            _d.copyTo(d);
//            d2.setSize(d.size());
            N = d.size();
//            MATH21_ASSERT(N > 0, "x-D tensor, and x must be positive integer");
        }

        // may deprecate
        NumB _setSize_check_data(NumN scale) {
            if (data.size() != scale) {
                return 0;
            }
            return 1;
        }

        //set size from left to right, from top to bottom.
        void _setSize_index(NumN &scale) {
            a.setSize(N);
            scale = 0;
            if (!isColumnMajor()) {
                for (NumN n = N; n >= 1; --n) {
                    a.at(n).setSize(dim(n));
                    if (n == N) {
                        for (NumN j = 1; j <= dim(n); j++) {
                            a.at(n).at(j) = j;
                        }
                        scale = dim(n);
                    } else {
                        for (NumN j = 1; j <= dim(n); j++) {
                            a.at(n).at(j) = (j - 1) * scale;
                        }
                        scale = scale * dim(n);
                    }
                }
            } else {
                for (NumN n = 1; n <= N; ++n) {
                    a.at(n).setSize(dim(n));
                    if (n == 1) {
                        for (NumN j = 1; j <= dim(n); j++) {
                            a.at(n).at(j) = j;
                        }
                        scale = dim(n);
                    } else {
                        for (NumN j = 1; j <= dim(n); j++) {
                            a.at(n).at(j) = (j - 1) * scale;
                        }
                        scale = scale * dim(n);
                    }
                }
            }
        }

        //set size from left to right, from top to bottom.
        void _setSize(const ArrayN &_d, const SpaceParas *paras = 0) {
            _setSize_shape(_d);
            if (isEmpty()) {
                return;
            }

            NumN scale = 0;
            _setSize_index(scale);
            data.setSize(scale, paras);
        }

        // space is null.
        void _setSizeNoSpace(const ArrayN &_d) {
            _setSize_shape(_d);
            if (isEmpty()) {
                return;
            }

            NumN scale = 0;
            _setSize_index(scale);
            data.setSizeNoSpace(scale);
        }

        void _setSizeDataFixed(const ArrayN &_d) {
            MATH21_ASSERT(volume() == math21_operator_container_multiply_all(_d),
                          "data size check failed!"
                                  << "\nvolume() = " << volume()
                                  << "\nmultiply_all(_d) = " << math21_operator_container_multiply_all(_d)
                                  << "\n_d = " << _d
            );
            _setSize_shape(_d, 0);
            if (isEmpty()) {
                return;
            }
            NumN scale = 0;
            _setSize_index(scale);
        }

        NumB isLogAutoBuffer() const {
            return 0;
        }

        void clearSome() {
            name = "";
            data.clear();
            a.clear();
            d.clear();
            N = 0;
            set_row_shape_size(0);
        }

        const Array<T> &getData() const {
            MATH21_ASSERT(isContinuous())
            return data;
        }

        // yes, don't use outside because the data may be stored
        // in arbitrary order, not just in row-major or column-major.
        Array<T> &getData() {
            MATH21_ASSERT(isContinuous())
            return data;
        }

        template<typename VecNType>
        NumN index_tensor_to_array(const VecNType &index) const {
            MATH21_ASSERT(dims() == index.size(), "not " << dims() << "-D tensor index");
            NumN sum = 0;
            for (NumN i = 1; i <= dims(); i++) {
                MATH21_ASSERT(index(i) >= 1 && index(i) <= dim(i),
                              "\tYou must give a valid index"
                                      << "\n\tcurrent index: " << index(i)
                                      << "\n\trequired [" << 1 << ", " << dim(i) << "]");
                sum = sum + a(i)(index(i));
            }
            return sum;
        }

        NumN index_tensor_to_array(NumN j1) const {
            return j1;
        }

        NumN index_tensor_to_array(NumN j1, NumN j2) const {
            if (dims() == 1) {
                MATH21_ASSERT(j2 == 1, "1-D tensor cols index not 1");
                return j1;
            }
            MATH21_ASSERT(dims() == 2, "not 2-D tensor");
            MATH21_ASSERT(j1 >= 1 && j1 <= dim(1) && j2 >= 1 && j2 <= dim(2),
                          "\tYou must give a valid index");
            return a(1)(j1) + a(2)(j2);
        }

        NumN index_tensor_to_array(NumN j1, NumN j2, NumN j3) const {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use");
            MATH21_ASSERT(dims() == 3, "not 3-D tensor");
            MATH21_ASSERT(j1 >= 1 && j1 <= dim(1) && j2 >= 1 && j2 <= dim(2) && j3 >= 1 && j3 <= dim(3),
                          "\tYou must give a valid index");
            return a(1)(j1) + a(2)(j2) + a(3)(j3);
        }

        NumN index_tensor_to_array(NumN j1, NumN j2, NumN j3, NumN j4) const {
            MATH21_ASSERT(dims() == 4, "not 4-D tensor");
            MATH21_ASSERT(
                    j1 >= 1 && j1 <= dim(1) && j2 >= 1 && j2 <= dim(2) && j3 >= 1 && j3 <= dim(3) && j4 >= 1 &&
                    j4 <= dim(4),
                    "\tYou must give a valid index");
            return a(1)(j1) + a(2)(j2) + a(3)(j3) + a(4)(j4);
        }

        NumN index_tensor_to_array(NumN j1, NumN j2, NumN j3, NumN j4, NumN j5) const {
            MATH21_ASSERT(dims() == 5, "not 5-D tensor");
            MATH21_ASSERT(
                    j1 >= 1 && j1 <= dim(1) && j2 >= 1 && j2 <= dim(2) && j3 >= 1 && j3 <= dim(3) && j4 >= 1 &&
                    j4 <= dim(4) && j5 >= 1 && j5 <= dim(5),
                    "\tYou must give a valid index");
            return a(1)(j1) + a(2)(j2) + a(3)(j3) + a(4)(j4) + a(5)(j5);
        }

        NumN index_tensor_to_array(NumN j1, NumN j2, NumN j3, NumN j4, NumN j5, NumN j6) const {
            MATH21_ASSERT(dims() == 6, "not 6-D tensor");
            MATH21_ASSERT(
                    j1 >= 1 && j1 <= dim(1) && j2 >= 1 && j2 <= dim(2) && j3 >= 1 && j3 <= dim(3) && j4 >= 1 &&
                    j4 <= dim(4) && j5 >= 1 && j5 <= dim(5) && j6 >= 1 && j6 <= dim(6),
                    "\tYou must give a valid index");
            return a(1)(j1) + a(2)(j2) + a(3)(j3) + a(4)(j4) + a(5)(j5) + a(6)(j6);
        }

    public:

        // data is kept when device type changed.
        void convertDeviceType(NumN deviceType) {
            if (getDeviceType() != deviceType) {
                Tensor<T> x;
                x.setDeviceType(deviceType);
                x = *this;
                x.swap(*this);
            }
        }

        // data is lost when device type changed.
        void setDeviceType(NumN deviceType) {
            if (getDeviceType() != deviceType) {
                clear();
                data.setDeviceType(deviceType);
            }
        }

        NumN getDeviceType() const {
            return data.getDeviceType();
        }

        NumB is_cpu() const {
            return data.is_cpu();
        }

        void clear() {
            clearSome();
            is_column_major = 0;
        }

        // not test
        void setColumnMajor(NumB isColumnMajor) {
            is_column_major = isColumnMajor;
            if (isEmpty()) {
                return;
            }
            NumN scale = 0;
            _setSize_index(scale);
        }

        NumB isColumnMajor() const {
            return is_column_major;
        }

        // memory related
        NumB isStandard() const {
            if (isContinuous() && !isColumnMajor()) {
                return 1;
            } else {
                return 0;
            }
        }

        void init() {
            clear();
        }

        //Construct empty matrix.
        Tensor() {
            init();
        }


        //copy constructor.
        // Now we allow copy constructor.
        Tensor(const Tensor<T> &B) {
//            MATH21_ASSERT(0 && "test")
            init();
            setColumnMajor(B.isColumnMajor());
            copyFrom(B);
        }

        // deprecated, use setSize instead.
        //construct like mlp, but from top to bottom.
        Tensor(ArrayN d) {
            MATH21_ASSERT(0, "deprecated, it's error-prone!")
            init();
            setSize(d);
        }

        //from left to right, d1 is lowest.
        Tensor(NumN d1) {
            init();
            setSize(d1);
        }

        //from left to right, d2 is lowest.
        Tensor(NumN d1, NumN d2) {
            init();
            setSize(d1, d2);
        }

        //from left to right, d3 is lowest.
        Tensor(NumN d1, NumN d2, NumN d3) {
            init();
            setSize(d1, d2, d3);
        }

        //from left to right, d4 is lowest.
        Tensor(NumN d1, NumN d2, NumN d3, NumN d4) {
            init();
            setSize(d1, d2, d3, d4);
        }

        //from left to right, d1 is lowest.
        void setSize(NumN d1) {
            if (isSameSize(d1)) {
                return;
            }
            ArrayN d(1);
            d = d1;
            _setSize(d);
        }

        //from left to right, d2 is lowest.
        void setSize(NumN d1, NumN d2) {
            if (isSameSize(d1, d2)) {
                return;
            }
            ArrayN d(2);
            d = d1, d2;
            _setSize(d);
        }

        //from left to right, d3 is lowest.
        void setSize(NumN d1, NumN d2, NumN d3) {
            if (isSameSize(d1, d2, d3)) {
                return;
            }
            ArrayN d(3);
            d = d1, d2, d3;
            _setSize(d);
        }

        //from left to right, d4 is lowest.
        void setSize(NumN d1, NumN d2, NumN d3, NumN d4) {
            if (isSameSize(d1, d2, d3, d4)) {
                return;
            }
            ArrayN d(4);
            d = d1, d2, d3, d4;
            _setSize(d);
        }

        void setSize(const ArrayN &_d) {
            if (isSameSize(_d)) {
                return;
            }
            _setSize(_d);
        }

        // reshape not sharing. See share reshape.
        //Tensor<NumN> is VecN actually.
        Tensor &reshape(const Tensor<NumN> &d) {
            MATH21_ASSERT_CODE(d.dims() == 1, "d must be type VecN.")
            ArrayN _d;
            _d.setSize(d.size());
            math21_operator_vec_2_array(d, _d);
            _setSizeDataFixed(_d);
            return *this;
        }

        Tensor &reshape(const ArrayN &d) {
            _setSizeDataFixed(d);
            return *this;
        }

        Tensor &toVector() {
            Tensor<NumN> d(1);
            d = volume();
            return reshape(d);
        }

        // deprecate, why?
        //Tensor<NumN> is VecN actually.
        void setSize(const Tensor<NumN> &d, const SpaceParas *paras = 0) {
            MATH21_ASSERT_CODE(d.dims() == 1, "d must be type VecN.")
            ArrayN _d;
            _d.setSize(d.size());
            math21_operator_vec_2_array(d, _d);
            if (isSameSize(_d) && paras == 0) {
                return;
            }
            _setSize(_d, paras);
        }

        // keep size fixed, just set space.
        void setSpace(const SpaceParas &paras) {
            Tensor<NumN> d;
            d.setSize(dims());
            math21_operator_array_2_vec(shape(), d);
            setSize(d, &paras);
        }

        // no ref
        void setSpace(const void *byte, NumN nBytes) {
            SpaceParas paras;
            paras.address = (char *) byte;
            paras.start = (char *) byte;
            paras.ref_count = 0;
            paras.size = nBytes;
            paras.unit = 1;
            paras.type = math21_type_get<T>();
            setSpace(paras);
        }

        void setSizeNoSpace(const Tensor<NumN> &d) {
            MATH21_ASSERT_CODE(d.dims() == 1, "d must be type VecN.")
            clearSome();
            ArrayN _d;
            _d.setSize(d.size());
            math21_operator_vec_2_array(d, _d);
            if (isSameSize(_d)) {
                return;
            }
            _setSizeNoSpace(_d);
        }

        void setSize(const Tensor<NumN> &dr, const Tensor<NumN> &dc) {
            Tensor<NumN> d;
            d.setSize(dr.size() + dc.size());
            math21_operator_container_set_partially(dr, d, 0, 0, dr.size());
            math21_operator_container_set_partially(dc, d, 0, dr.size(), dc.size());
            setSize(d);
            set_row_shape_size(dr.size());
        }

        // A tensor can still have shape even if it is empty.
        NumB isEmpty() const {
            return volume() == 0 ? (NumB) 1 : (NumB) 0;
        }

        virtual ~Tensor() {
            clear();
        }

        // deprecate
        virtual T &operator()(const ArrayN &index) {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use");
            return data.at(index_tensor_to_array(index));
        }

        // deprecate, use virtual const T &operator()(const Tensor<NumN> &index);
        virtual const T &operator()(const ArrayN &index) const {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use"
                    << "\n\tgetClassName(): " << getClassName());
            return data(index_tensor_to_array(index));
        }

        //index from high to low: index of 4322 is 4322, not 2234
        virtual T &operator()(const Tensor<NumN> &index) {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use");
            return data.at(index_tensor_to_array(index));
        }

        virtual const T &operator()(const Tensor<NumN> &index) const {
            MATH21_ASSERT(math21_string_is_equal(getClassName(), Tensor::getClassName()),
                          "You must overwrite to use"
                                  << "\n\tgetClassName(): " << getClassName());
            return data(index_tensor_to_array(index));
        }

        template<template<typename> class Container, typename NumNType>
        T &operator()(const Container<NumNType> &index_a, NumNType j1) {
            Tensor<NumN> index(index_a.size() + 1);
            math21_operator_container_set_partially(index_a, index, 0, 0, index_a.size());
            index(index.size()) = j1;
            return this->operator()(index);
        }

        template<template<typename> class Container, typename NumNType>
        const T &operator()(const Container<NumNType> &index_a, NumNType j1) const {
            Tensor<NumN> index(index_a.size() + 1);
            math21_operator_container_set_partially(index_a, index, 0, 0, index_a.size());
            index.at(index.size()) = j1;
            return this->operator()(index);
        }

        // see tensor as container.
        virtual T &at(NumN j1) {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use");
            return data.at(index_tensor_to_array(j1));
        }

        virtual T &operator()(NumN j1) {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use");
            return data.at(index_tensor_to_array(j1));
        }

        virtual const T &operator()(NumN j1) const {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use");
            return data(index_tensor_to_array(j1));
        }

        virtual T &operator()(NumN j1, NumN j2) {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use");
            return data.at(index_tensor_to_array(j1, j2));
        }

        virtual const T &operator()(NumN j1, NumN j2) const {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use");
            return data(index_tensor_to_array(j1, j2));
        }

        T &valueAt(NumN j1, NumN j2, NumN j3) {
            return this->operator()(j1, j2, j3);
        }

        virtual T &operator()(NumN j1, NumN j2, NumN j3) {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use");
            return data.at(index_tensor_to_array(j1, j2, j3));
        }

        virtual const T &operator()(NumN j1, NumN j2, NumN j3) const {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use");
            return data(index_tensor_to_array(j1, j2, j3));
        }

        virtual T &operator()(NumN j1, NumN j2, NumN j3, NumN j4) {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use");
            return data.at(index_tensor_to_array(j1, j2, j3, j4));
        }

        virtual const T &operator()(NumN j1, NumN j2, NumN j3, NumN j4) const {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use");
            return data(index_tensor_to_array(j1, j2, j3, j4));
        }

        virtual T &operator()(NumN j1, NumN j2, NumN j3, NumN j4, NumN j5) {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use");
            return data.at(index_tensor_to_array(j1, j2, j3, j4, j5));
        }

        virtual const T &operator()(NumN j1, NumN j2, NumN j3, NumN j4, NumN j5) const {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use");
            return data(index_tensor_to_array(j1, j2, j3, j4, j5));
        }

        virtual T &operator()(NumN j1, NumN j2, NumN j3, NumN j4, NumN j5, NumN j6) {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use");
            return data.at(index_tensor_to_array(j1, j2, j3, j4, j5, j6));
        }

        virtual const T &operator()(NumN j1, NumN j2, NumN j3, NumN j4, NumN j5, NumN j6) const {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use");
            return data(index_tensor_to_array(j1, j2, j3, j4, j5, j6));
        }

        // Emtpy tensor can have shape.
        // can be empty even when N > 0, added in version 3.0.0
        NumN dims() const {
            return N;
        }

        // for vector, matrix and tensor.
        NumN size() const {
            return volume();
        }

        NumB isScalarInMath() const {
            if (dims() == 1 && volume() == 1) {
                return 1;
            }
            return 0;
        }

        NumB isVectorInMath() const {
            if (0 < dims() && dims() <= 2) {
                if (nrows() == 1 || ncols() == 1) {
                    return 1;
                }
            }
            return 0;
        }

        NumB isRowVector() const {
            if (0 < dims() && dims() <= 2) {
                if (nrows() == 1) {
                    return 1;
                }
            }
            return 0;
        }

        NumB isColVector() const {
            if (0 < dims() && dims() <= 2) {
                if (ncols() == 1) {
                    return 1;
                }
            }
            return 0;
        }

        NumB isMatrixInMath() const {
            if (0 < dims() && dims() <= 2) {
                return 1;
            }
            return 0;
        }

        NumN nrows() const {
            if (isEmpty()) {
                return 0;
            }
            MATH21_ASSERT(0 < dims() && dims() <= 2, "not 2-D tensor");
            return dim(1);
        }

        NumN ncols() const {
            if (isEmpty()) {
                return 0;
            }
            MATH21_ASSERT(0 < dims() && dims() <= 2, "not 2-D tensor");
            if (dims() == 1) {
                return 1;
            }
            return dim(2);
        }

        NumN dim(NumN i) const {
            return d(i);
        }

        //from left to right.
        const ArrayN &shape() const {
            return d;
        }

        //from left to right.
        const Tensor<NumN> &shape(Tensor<NumN> &d2) const {
            if (!d2.is_cpu()) {
                Tensor<NumN> d_cpu;
                shape(d_cpu);
                d2 = d_cpu;
                return d2;
            }
            if (!d2.isSameSize(d.size())) {
                d2.setSize(d.size());
            }
            math21_operator_array_2_vec(shape(), d2);
            return d2;
        }

        // return how many numbers.
        NumN volume() const {
            return math21_operator_container_multiply_all(d);
        }

        virtual std::string getClassName() const {
            return "Tensor";
        }


        void setName(const char *_name) {
            name = _name;
        }

        const std::string &getName() const {
            return name;
        }

        NumB log(const char *name = 0, NumB isUsingCommas = 0, NumB isLogDetail = 1, NumN precision = 3) const;

        NumB log(std::ostream &io, const char *name = 0, NumB isUsingCommas = 0, NumB isLogDetail = 1,
                 NumN precision = 3) const;

        NumB logInfo(const char *name = "") const;

        NumB logInfo(std::ostream &io, const char *name = "") const;

        virtual NumB isWritable() const {
            return (NumB) 1;
        }

        NumB isSameSize(const NumN &n) const {
            if (dims() == 1 && dim(1) == n) {
                return 1;
            } else {
                return 0;
            }
        }

        NumB isSameSize(const NumN &n, const NumN &m) const {
            if (dims() == 2 && dim(1) == n && dim(2) == m) {
                return 1;
            } else {
                return 0;
            }
        }

        NumB isSameSize(NumN d1, NumN d2, NumN d3) const {
            if (dims() == 3 && dim(1) == d1 && dim(2) == d2 && dim(3) == d3) {
                return 1;
            } else {
                return 0;
            }
        }

        NumB isSameSize(NumN d1, NumN d2, NumN d3, NumN d4) const {
            if (dims() == 4 && dim(1) == d1 && dim(2) == d2 && dim(3) == d3 && dim(4) == d4) {
                return 1;
            } else {
                return 0;
            }
        }

        NumB isSameSize(const ArrayN &d) const {
            return math21_operator_container_isEqual(shape(), d);
        }

        NumB isSameSize(const Tensor<NumN> &d) const {
            MATH21_ASSERT_CODE(d.dims() == 1, "d must be type VecN.")
            ArrayN _d;
            _d.setSize(d.size());
            math21_operator_vec_2_array(d, _d);
            return isSameSize(_d);
        }

        NumB isSameSizeVirtually(const ArrayN &d) const {
            if (dims() == 1 && d.size() == 2) {
                if (d(1) == dim(1) && d(2) == 1) {
                    return 1;
                } else {
                    return 0;
                }
            } else if (dims() == 2 && d.size() == 1) {
                if (d(1) == dim(1) && dim(2) == 1) {
                    return 1;
                } else {
                    return 0;
                }
            }
            return isSameSize(d);
        }

        void assign(const Tensor<T> &B);

        template<typename S>
        void assign(const Tensor<S> &B);

        template<typename S>
        void assign(const S &a) {
            data.assign(a);
        }

        // This is just assignment in early version.
        // Now, we use copyFrom
        // So, this isn't just assignment, it also creates size.
        // If A.shape = B.shape, then it is equivalent to assignment.
        Tensor<T> &operator=(const Tensor<T> &B) {
            if (!isSameSize(B.shape())) {
                setSize(B.shape());
            }
            assign(B);
            return *this;
        }

        template<typename S>
        Tensor<T> &operator=(const Tensor<S> &B) {
            if (!isSameSize(B.shape())) {
                setSize(B.shape());
            }
            assign(B);
            return *this;
        }

        // copy is used in same-type case.
        void copyTo(Tensor<T> &B) const {
            B.setSize(shape());
            B.assign(*this);
        }

        void copyFrom(const Tensor<T> &B) {
            setSize(B.shape());
            assign(B);
        }

        virtual NumB isContinuous() const {
            return 1;
        }

        NumB isBasicType() const {
            return data.isBasicType();
        }

        T *getDataAddress() {
            MATH21_ASSERT(isContinuous() && is_cpu())
            SpaceParas paras = getSpace();
            return (T *) paras.start;
        }

        const T *getDataAddress() const {
            MATH21_ASSERT(isContinuous() && is_cpu())
            SpaceParas paras = getSpace();
            return (const T *) paras.start;
        }

        PointerN8Wrapper getDataAddressWrapper() {
            MATH21_ASSERT(isContinuous())
            SpaceParas paras = getSpace();
#if defined(MATH21_FLAG_USE_CPU)
            MATH21_ASSERT(is_cpu())
            return (PointerN8Wrapper) paras.start;
#else
            MATH21_ASSERT(!is_cpu())
            return (PointerN8Wrapper) paras.start_wrapper;
#endif
        }

        PointerN8InputWrapper getDataAddressWrapper() const {
            MATH21_ASSERT(isContinuous())
            SpaceParas paras = getSpace();
#if defined(MATH21_FLAG_USE_CPU)
            MATH21_ASSERT(is_cpu())
            return (PointerN8InputWrapper) paras.start;
#else
            MATH21_ASSERT(!is_cpu())
            return (PointerN8InputWrapper) paras.start_wrapper;
#endif
        }

        const NumN *getShapeDataAddress() const {
            SpaceParas paras = d.getSpace();
            return (const NumN *) paras.start;
        }

        // deprecated
        SpaceParas getSpace() const {
            return data.getSpace();
        }

        // deprecated
        SpaceParas getSpace(NumN offset, NumN size, NumN unit = sizeof(char)) const {
            return data.getSpace(offset, size, unit);
        }

//        const detail::literal_assign_helper_tensor<T> operator=(
//                const T &val
//        ) {
//            // assign the given value to every spot in this matrix
//            assign(val);
//            // Now return the literal_assign_helper so that the user
//            // can use the overloaded comma notation to initialize
//            // the matrix if they want to.
//            return detail::literal_assign_helper_tensor<T>(*this);
//        }
        template<typename S>
        const detail::literal_assign_helper_tensor<T> operator=(
                const S &val
        ) {
            // assign the given value to every spot in this matrix
//            assign((const T&) val);
            assign(val);
            // Now return the literal_assign_helper so that the user
            // can use the overloaded comma notation to initialize
            // the matrix if they want to.
            return detail::literal_assign_helper_tensor<T>(*this);
        }

        // from start_letter
        void letters(NumZ start_letter = 1) {
            if (!is_cpu()) {
                Tensor<T> A;
                A.setSize(shape());
                A.letters(start_letter);
                *this = A;
                return;
            }
            math21_operator_container_letters(data, start_letter);
        }

        //Tensor<NumN> is VecN
        TensorView<T> sliceView(const Seqce<Tensor<NumN> > &index) const;

        TensorSub<T> sliceSub(const Seqce<Tensor<NumN> > &index);

        //Tensor<NumN> is VecN
        TensorView<T> shrinkView(const Tensor<NumN> &index) const;

        //Tensor<NumN> is VecN
        TensorSub<T> shrinkSub(const Tensor<NumN> &index);

        void swap(Tensor &B) {
            MATH21_ASSERT(getClassName() == Tensor::getClassName(), "You must overwrite to use");
            data.swap(B.data);
            a.swap(B.a);
            d.swap(B.d);
            m21_swap(N, B.N);
            m21_swap(is_column_major, B.is_column_major);
        }

        void zeros() {
            if (!isEmpty()) {
                assign(0);
            }
        }

        void getIndex(NumN i, Tensor<NumN> &index) const {
            math21_operator_number_index_1d_to_tensor_nd(index, i, shape());
        }

        NumN get_row_shape_size() const {
            return row_shape_size;
        }

        void set_row_shape_size(NumN rowShapeSize) {
            row_shape_size = rowShapeSize;
        }

        void row_shape(Tensor<NumN> &dr) const {
            if (get_row_shape_size() == 0) {
                dr.setSize(1);
                dr = nrows();
            } else {
                dr.setSize(get_row_shape_size());
                math21_operator_container_set_partially(d, dr, 0, 0, get_row_shape_size());
            }
        }

        void col_shape(Tensor<NumN> &dc) const {
            if (get_row_shape_size() == 0) {
                dc.setSize(1);
                dc = ncols();
            } else {
                if (get_row_shape_size() == dims()) {
                    dc.setSize(1);
                    dc = 1;
                } else {
                    dc.setSize(dims() - get_row_shape_size());
                    math21_operator_container_set_partially(d, dc, get_row_shape_size(), 0,
                                                            dims() - get_row_shape_size());
                }
            }
        }

        // may merge with nrows()
        NumN nrows_generalized() const {
            if (isEmpty()) {
                return 0;
            }
            if (get_row_shape_size() == 0) {
                return nrows();
            } else {
                return math21_operator_container_multiply_some(d, get_row_shape_size());
            }
        }

        // may deprecate, and use vector space instead.
        NumN ncols_generalized() const {
            if (isEmpty()) {
                return 0;
            }
            if (get_row_shape_size() == 0) {
                return ncols();
            } else {
                if (get_row_shape_size() == dims()) {
                    return 1;
                } else {
                    return math21_operator_container_multiply_some(
                            d, dims() - get_row_shape_size(),
                            get_row_shape_size());
                }
            }
        }
    };

    template<typename T>
    NumB Tensor<T>::logInfo(const char *name) const {
        return logInfo(std::cout, name);
    }

    template<typename T>
    NumB Tensor<T>::logInfo(std::ostream &io, const char *name) const {
        if (!name) {
            name = "";
        }
        io << getClassName() << " " << (getName() + std::string(name)) << ", dims: " << dims()
           << ", dtype: " << math21_type_name<T>() << "\n";
        io << "architecture (from left to right): ";
        shape().log(io, 0, 0);
        io << "isColumnMajor: " << isColumnMajor() << ", ";
        io << "is cpu: " << is_cpu() << "\n";
//        shape().log("shape", 0);
        if (isLogAutoBuffer()) {
            data.logBuffer(io, name);
        }
        return 1;
    }

    template<typename T>
    std::ostream &operator<<(std::ostream &out, const Tensor<T> &m) {
        m.log(out);
        return out;
    }

    namespace detail {
        template<typename T>
        void math21_output(std::ostream &out, const Tensor<T> &A, NumB isUsingCommas);
    }

    template<typename T>
    NumB Tensor<T>::log(const char *name, NumB isUsingCommas, NumB isLogDetail, NumN precision) const {
        return log(std::cout, name, isUsingCommas, isLogDetail, precision);
    }

    template<typename T>
    NumB
    Tensor<T>::log(std::ostream &io, const char *name, NumB isUsingCommas, NumB isLogDetail, NumN precision) const {
        if (isLogDetail) {
            logInfo(io, name);
        } else {
            if (name) {
                io << name << "\n";
            }
        }
        if (isEmpty()) {
            return 0;
        }
        io << std::setprecision(precision);
        if (!is_cpu()) {
            Tensor<T> A;
            copyTo(A);
            detail::math21_output(io, A, isUsingCommas);
        } else {
            detail::math21_output(io, *this, isUsingCommas);
        }
        if (name || isLogDetail) {
            io << "\n";
        }
        return 1;
    }

    //////////////

    // assign B to A
    template<typename T, typename S>
    void math21_operator_tensor_assign_elementwise_recursive(Tensor<T> &A, const Tensor<S> &B) {
        if (B.isEmpty()) {
            return;
        }
        MATH21_ASSERT(A.isSameSize(B.shape()), "tensor size doesn't match in assign");

        if (A.dims() == 1) {
            for (NumN i = 1; i <= A.dim(1); i++) {
                A(i) = (T) B(i);
            }
        } else {
            for (NumN i = 1; i <= A.dim(1); i++) {
                VecN x(A.dims());
                x = 0;
                x(1) = i;
                TensorSub<T> ts = A.shrinkSub(x);
                TensorView<S> tv = B.shrinkView(x);
                math21_operator_tensor_assign_elementwise_recursive(ts, tv);
            }
        }
    }

    // assign B to A
    template<typename T, typename S>
    void math21_operator_tensor_assign_elementwise_no_recursive(Tensor<T> &A, const Tensor<S> &B) {
        if (B.isEmpty()) {
            return;
        }
        MATH21_ASSERT(A.isSameSize(B.shape()), "tensor size doesn't match in assign");
        VecN d;
        d.setSize(A.dims());
        d = 1;
        while (1) {
            A(d) = static_cast<T> (B(d));
            if (math21_operator_container_increaseNumFromRight(A.shape(), d) == 0) {
                break;
            }
        }
    }

    // assign B to A
    template<typename T>
    void math21_operator_tensor_assign_elementwise_no_recursive(Tensor<T> &A, const TenStr &B) {
        if (B.isEmpty()) {
            return;
        }
        MATH21_ASSERT(A.isSameSize(B.shape()), "tensor size doesn't match in assign");
        VecN d;
        d.setSize(A.dims());
        d = 1;
        while (1) {
            math21_string_to_type_generic(B(d), A(d));
            if (math21_operator_container_increaseNumFromRight(A.shape(), d) == 0) {
                break;
            }
        }
    }

    // for compatible type.
    // assign B to A
    template<typename T, typename S>
    void math21_operator_tensor_assign_elementwise(Tensor<T> &A, const Tensor<S> &B) {
//        math21_operator_tensor_assign_elementwise_recursive(A, B);
        math21_operator_tensor_assign_elementwise_no_recursive(A, B);
    }

    template<typename T>
    void Tensor<T>::assign(const Tensor<T> &B) {
        MATH21_ASSERT(isSameSize(B.shape()), "tensor size doesn't match in assign"
                << "\n\tdims(): " << dims()
                << "\n\tB.dims(): " << B.dims());
        if (isContinuous() && B.isContinuous() && (isColumnMajor() == B.isColumnMajor()) && isBasicType()) {
            data.assign(B.getData());
        } else {
            math21_operator_tensor_assign_elementwise(*this, B);
        }
    }

    // todo: optimize using math21_op_vector_set_by_vector
    template<typename T>
    template<typename S>
    void Tensor<T>::assign(const Tensor<S> &B) {
        MATH21_ASSERT(isSameSize(B.shape()), "tensor size doesn't match in assign"
                << "\n\tdims(): " << dims()
                << "\n\tB.dims(): " << B.dims());
        math21_operator_tensor_assign_elementwise(*this, B);
    }
}
