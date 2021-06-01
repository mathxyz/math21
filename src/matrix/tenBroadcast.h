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

#include "matrix.h"
#include "ten.h"
#include "ten_ops.h"

namespace math21 {

    template<template<typename> class Container>
    void math21_broadcast_index_to_original(const VecN &index, const Container<NumN> &d_ori, VecN &index_ori) {
        index_ori.setSize(d_ori.size());
        for (NumN i = 1; i <= d_ori.size(); ++i) {
            if (d_ori(i) != 1) {
                index_ori(i) = index(i);
            } else {
                index_ori(i) = 1;
            }
        }
    }

    NumB math21_broadcast_is_compatible_in_ele_op(const Seqce <VecN> &shapes, VecN &d);

    NumB math21_broadcast_is_compatible_in_stacked_matmul(
            const VecN &d_A, const VecN &d_B,
            VecN &d_A_new, VecN &d_B_new, VecN &d_C,
            VecN &d_A_standard, VecN &d_B_standard);

    // d is destination shape.
    template<typename VecType1, typename VecType2>
    NumB math21_operator_tensor_can_broadcast_to(const VecType1 &d_ori, const VecType2 &d, NumB &noNeedTo) {
        if (d_ori.size() > d.size()) {
            return 0;
        }
        if (d_ori.size() == 0) {
            return 0;
        }
        MATH21_ASSERT(math21_operator_container_multiply_all(d)!=0, "0 shape not supported in this version!");
        MATH21_ASSERT(math21_operator_container_multiply_all(d_ori)!=0, "0 shape not supported in this version!");
        NumN n = d_ori.size();
        noNeedTo = 1; // is same shape, so no need to broadcast
        for (NumN i = 1; i <= n; ++i) {
            if (d(i) != d_ori(i)) {
                noNeedTo = 0;
                if (d_ori(i) != 1) {
                    return 0;
                }
            }
        }
        if (noNeedTo) {
            NumN n_1 = math21_operator_container_number_of_equal(1, d, d_ori.size());
            if (n_1 != d.size() - d_ori.size()) {
                noNeedTo = 0;
            }
        }
        return 1;
    }

    // d is destination shape.
    template<typename VecType1, typename VecType2>
    NumB math21_broadcast_is_compatible_to(const VecType1 &d_ori, const VecType2 &_d) {
        NumB noNeedTo;
        if (_d.size() >= d_ori.size()) {
            return math21_operator_tensor_can_broadcast_to(d_ori, _d, noNeedTo);
        } else {
            VecN d;
            d.setSize(d_ori.size());
            d = 1;
            math21_operator_container_set_partially(_d, d, 0, 0, _d.size());
            return math21_operator_tensor_can_broadcast_to(d_ori, d, noNeedTo);
        }
    }

    template<typename VecType1, typename VecType2>
    NumB math21_operator_tensor_is_shape_nearly_same(const VecType1 &d1, const VecType2 &d2) {
        NumN n = xjmax(d1.size(), d2.size());
        VecN dx, dy;
        dx.setSize(n);
        dx = 1;
        dy.setSize(n);
        dy = 1;
        math21_operator_container_set_partially(d1, dx, 0, 0, d1.size());
        math21_operator_container_set_partially(d2, dy, 0, 0, d2.size());
        return math21_operator_container_isEqual(dx, dy);
    }


    // numpy broadcast order: from right to left
    // matlab broadcast order: from left to right. Every array in matlab has trailing dimensions of size 1.
    // math21 broadcast order: from left to right
    // https://numpy.org/doc/stable/user/basics.broadcasting.html
    // https://www.mathworks.com/help/matlab/matlab_prog/compatible-array-sizes-for-basic-operations.html
    // not making copies
    // There are cases where broadcasting is a bad idea because it leads to inefficient use of memory that slows computation.
    template<typename T>
    class TensorBroadcast : public Tensor<T> {
    private:
        // m may be TensorBroadcast or others.
        const Tensor <T> *_m;
        T dummy;
//        VecN _d; // todo: use this, and try to make Tensor more abstract?
    public:

        TensorBroadcast() : Tensor<T>(), _m(0) {
        }

        TensorBroadcast(const TensorBroadcast<T> &m) : Tensor<T>(), _m(&m.getTensor()) {
            MATH21_ASSERT(0, "can't call this in current version!");
        }

        // the d is destination shape
        TensorBroadcast(const Tensor <T> &m0, const VecN &d) : Tensor<T>() {
            set(m0, d);
        }

        // the d is destination shape
        void set(const Tensor <T> &m0, const VecN &d) {
            _m = &m0;
            NumB noNeedTo;
            NumB flag = math21_operator_tensor_can_broadcast_to(_m->shape(), d, noNeedTo);
            MATH21_ASSERT(flag == 1, "shape mismatched when broadcasting!\n"
                    << _m->shape().log("shape_ori") << d.log("shape") << math21_global_ad_log_data());
            if (noNeedTo) {
                if (math21_global_is_debug()) {
                    m21warn("broadcast nearly same shape!");
                    _m->shape().log("shape_ori");
                    d.log("shape");
                }
            }
            this->setSizeNoSpace(d);
        }

        // call clear to avoid MATH21_ASSERT no space error!
        virtual ~TensorBroadcast() {
            this->clear();
        }

        virtual NumB isWritable() const override {
            return (NumB) 0;
        }

        NumB isContinuous() const override {
            return 0;
        }

        const Tensor <T> &getTensor() const {
            MATH21_ASSERT(_m);
            return *_m;
        }

        virtual T &operator()(const VecN &index) override {
            MATH21_ASSERT_NOT_CALL(0, "dummy, can't call this!");
            return dummy;
        }

        virtual T &operator()(NumN j1) override {
            MATH21_ASSERT_NOT_CALL(0, "dummy, can't call this!");
            return dummy;
        }

        virtual T &operator()(NumN j1, NumN j2) override {
            MATH21_ASSERT_NOT_CALL(0, "dummy, can't call this!");
            return dummy;
        }

        virtual T &operator()(NumN j1, NumN j2, NumN j3) override {
            MATH21_ASSERT_NOT_CALL(0, "dummy, can't call this!");
            return dummy;
        }

        virtual T &operator()(NumN j1, NumN j2, NumN j3, NumN j4) override {
            MATH21_ASSERT_NOT_CALL(0, "dummy, can't call this!");
            return dummy;
        }

        virtual const T &operator()(const VecN &index) const override {
            MATH21_ASSERT(!this->isEmpty() && this->dims() == index.size(), "index not match tensor dim");
            for (NumN i = 1; i <= this->dims(); i++) {
                MATH21_ASSERT(index(i) >= 1 && index(i) <= this->dim(i),
                              "\tYou must give a valid index"
                                      << "\n\tcurrent index: " << index(i)
                                      << "\n\trequired [" << 1 << ", " << this->dim(i) << "]");
            }
            VecN index_ori;
            math21_broadcast_index_to_original(index, _m->shape(), index_ori);
            MATH21_ASSERT(_m);
            return _m->operator()(index_ori);
        }

        // as container.
        virtual const T &operator()(NumN j1) const override {
            MATH21_ASSERT(j1 <= this->volume());
            VecN index(this->dims());
            math21_operator_number_num_to_index_right(j1, index, this->shape());
            return operator()(index);
        }

        virtual const T &operator()(NumN j1, NumN j2) const override {
            VecN index(2);
            index = j1, j2;
            return operator()(index);
        }

        virtual const T &operator()(NumN j1, NumN j2, NumN j3) const override {
            VecN index(3);
            index = j1, j2, j3;
            return operator()(index);
        }

        virtual const T &operator()(NumN j1, NumN j2, NumN j3, NumN j4) const override {
            VecN index(4);
            index = j1, j2, j3, j4;
            return operator()(index);
        }

        virtual std::string getClassName() const override {
            return "TensorBroadcast";
        }

        // convert to continuous Tensor
        void toTensor(Tensor <T> &A) const {
            MATH21_ASSERT(A.getClassName() == "Tensor");
            A.setSize(this->shape());
            A.assign((const Tensor<T> &) *this);
        }
    };

    typedef TensorBroadcast<NumZ> TenBcZ;
    typedef TensorBroadcast<NumN> TenBcN;
    typedef TensorBroadcast<NumR> TenBcR;
}

