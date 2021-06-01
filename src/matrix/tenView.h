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

    /*
     * tensor view is like tensor sub, but is read only.
     * sub tensor is like sub matrix. It has rights of changing elements of matrix, but has no rights of changing shape of matrix.
     * TensorSub is more general than sub tensor in the following reasons.
     * 1. tensor sub can have less dim than tensor, but sub tensor can't.
     * 2. tensor sub can choose more than one time an element from the tensor in a given position, while sub tensor can select one element only one time.
     *
     * element access is slow!!!
     *
     * Todo: consider constructor safety!
     * */
    template<typename T>
    class TensorView : public Tensor<T> {
    private:
        // m may be TensorView or others.
        const Tensor<T> &m;
        T dummy;

        void init_tv() {
            MATH21_ASSERT(a.isEmpty());
            MATH21_ASSERT(b.isEmpty());
        }

    protected:
        VecN b;//// b is index to a.
        Seqce<VecN> a;////a is index to Tensor
    public:

        TensorView(const TensorView<T> &_m) : Tensor<T>(), m(_m.m) {
//            MATH21_ASSERT(0, "can't call this in current version!");
            // used when msvc
            init_tv();
            b = _m.b;
            _m.a.copyTo(a);
            VecN d;
            this->setSizeNoSpace(_m.shape(d));
        }

        //slice, we use this to discriminate move constructor or copy constructor which we don't handle.
        TensorView(const Tensor<T> &m, const Seqce<VecN> &index) : Tensor<T>(), m(m) {
            init_tv();
            MATH21_ASSERT(m.dims() > 0, "can't get sub tensor from emtpy tensor");
            MATH21_ASSERT(m.dims() == index.size());
            a.setSize(index.size());
            for (NumN i = 1; i <= a.size(); ++i) {
                //0 means use all.
                if (index(i).size() == 1 && index(i)(1) == 0) {
                    a.at(i).setSize(m.dim(i));
                    a.at(i).letters();
                } else {
                    a.at(i).setSize(index(i).size());
                    a.at(i).assign(index(i));
                }
            }
            VecN d(a.size());
            for (NumN i = 1; i <= d.size(); i++) {
                d(i) = a.at(i).size();
            }
            this->setSizeNoSpace(d);
        }

        //shrink, i.e., slice and remove. We use this to discriminate move constructor or copy constructor which we don't handle.
        TensorView(const Tensor<T> &_m, const VecN &index) : Tensor<T>(), m(_m) {
            init_tv();
            MATH21_ASSERT(m.dims() > 0, "can't get sub tensor from emtpy tensor");
            MATH21_ASSERT(m.dims() == index.size());
            b.setSize(index.size());
            b.assign(index);
            VecN d;
            math21_operator_tensor_shrink_shape(m, b, d);
            this->setSizeNoSpace(d);
        }

        // call clear to avoid MATH21_ASSERT no space error!
        virtual ~TensorView() {
            this->clear();
        }

        virtual NumB isWritable() const override {
            return (NumB) 0;
        }

        NumB isContinuous() const override {
            return 0;
        }

        const Tensor<T> &getTensor() const {
            return m;
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
            // slice
            if (b.isEmpty()) {
                VecN y(m.dims());
                for (NumN i = 1; i <= y.size(); i++) {
                    y(i) = a(i)(index(i));
                }
                return m(y);
            } else { // shrink
                VecN y(b.size());
                math21_operator_container_replace_inc(b, y, index, (NumN) 0);
                return m(y);
            }
        }

        // the index is from right so that it can be used together with other index-right containers.
        // as container.
        virtual const T &operator()(NumN j1) const override {
            MATH21_ASSERT(j1 <= this->volume());
            VecN index(this->dims());
            math21_operator_number_num_to_index_right(j1, index, this->shape());
            return operator()(index);
        }

        virtual const T &operator()(NumN j1, NumN j2) const override {
            MATH21_ASSERT(this->dims() == 2, "not 2-D tensor");
            MATH21_ASSERT(
                    j1 >= 1 && j1 <= this->dim(1) && j2 >= 1 && j2 <= this->dim(2),
                    "\tYou must give a valid index");
            if (b.isEmpty()) {
                return m(a(1)(j1), a(2)(j2));
            } else {
                VecN y(b.size());
                y.assign(b);
                NumN j = 1;
                for (NumN i = 1; i <= m.dims(); i++) {
                    if (y(i) == 0) {
                        if (j == 1) {
                            y(i) = j1;
                        } else if (j == 2) {
                            y(i) = j2;
                        }
                        j++;
                    }
                }
                MATH21_ASSERT(j - 1 == this->dims(), "j is " << j);
                return m(y);
            }
        }

        virtual const T &operator()(NumN j1, NumN j2, NumN j3) const override {
            MATH21_ASSERT(this->dims() == 3, "not 3-D tensor");
            MATH21_ASSERT(
                    j1 >= 1 && j1 <= this->dim(1) && j2 >= 1 && j2 <= this->dim(2) && j3 >= 1 && j3 <= this->dim(3),
                    "\tYou must give a valid index");
            if (b.isEmpty()) {
                return m(a(1)(j1), a(2)(j2), a(3)(j3));
            } else {
                VecN y(b.size());
                y.assign(b);
                NumN j = 1;
                for (NumN i = 1; i <= m.dims(); i++) {
                    if (y(i) == 0) {
                        if (j == 1) {
                            y(i) = j1;
                        } else if (j == 2) {
                            y(i) = j2;
                        } else if (j == 3) {
                            y(i) = j3;
                        }
                        j++;
                    }
                }
                MATH21_ASSERT(j - 1 == this->dims(), "j is " << j);
                return m(y);
            }
        }

        virtual const T &operator()(NumN j1, NumN j2, NumN j3, NumN j4) const override {
            MATH21_ASSERT(this->dims() == 4, "not 4-D tensor");
            MATH21_ASSERT(
                    j1 >= 1 && j1 <= this->dim(1) && j2 >= 1 && j2 <= this->dim(2) && j3 >= 1 &&
                    j3 <= this->dim(3) && j4 >= 1 && j4 <= this->dim(4),
                    "\tYou must give a valid index");
            if (b.isEmpty()) {
                return m(a(1)(j1), a(2)(j2), a(3)(j3), a(4)(j4));
            } else {
                VecN y(b.size());
                y.assign(b);
                NumN j = 1;
                for (NumN i = 1; i <= m.dims(); i++) {
                    if (y(i) == 0) {
                        if (j == 1) {
                            y(i) = j1;
                        } else if (j == 2) {
                            y(i) = j2;
                        } else if (j == 3) {
                            y(i) = j3;
                        } else if (j == 4) {
                            y(i) = j4;
                        }
                        j++;
                    }
                }
                MATH21_ASSERT(j - 1 == this->dims(), "j is " << j);
                return m(y);
            }
        }

        virtual std::string getClassName() const override {
            return "TensorView";
        }

        // convert to continuous Tensor
        void toTensor(Tensor<T> &A) const {
            MATH21_ASSERT(A.getClassName() == "Tensor");
            A.setSize(this->shape());
            A.assign((const Tensor<T> &) *this);
        }
    };

    typedef TensorView<NumZ> TenViewZ;
    typedef TensorView<NumN> TenViewN;
    typedef TensorView<NumR> TenViewR;

    //Tensor<NumN> is VecN
    template<typename T>
    TensorView<T> Tensor<T>::sliceView(const Seqce<Tensor<NumN> > &index) const {
        TensorView<T> tv(*this, index);
        return tv;
    }

    //Tensor<NumN> is VecN
    template<typename T>
    TensorView<T> Tensor<T>::shrinkView(const Tensor<NumN> &index) const {
        TensorView<T> tv(*this, index);
        return tv;
    }
}

