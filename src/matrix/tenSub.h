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
     * tensor sub is like but is more general than sub tensor.
     *
     * Todo: consider constructor safety!
     * */
    template<typename T>
    class TensorSub : public TensorView<T> {
    private:
        Tensor <T> &m;

        VecN y; // tmp

        void init_ts() {
        }

    public:

        TensorSub(Tensor <T> &m, const Seqce <VecN> &index) : TensorView<T>(m, index), m(m) {
            init_ts();
            MATH21_ASSERT(m.isWritable(),
                          "can't get tensor sub from tensor which is read only. Maybe you can use tensor view instead.");
            y.setSize(m.dims());
        }

        TensorSub(Tensor <T> &m, const VecN &index) : TensorView<T>(m, index), m(m) {
            init_ts();
            MATH21_ASSERT(m.isWritable(),
                          "can't get tensor sub from tensor which is read only. Maybe you can use tensor view instead.");
            y.setSize(m.dims());
        }

        virtual T &operator()(const VecN &index) override {
            MATH21_ASSERT(!this->isEmpty() && this->dims() == index.size(), "index not match tensor dim");
            for (NumN i = 1; i <= this->dims(); i++) {
                MATH21_ASSERT(index(i) >= 1 && index(i) <= this->dim(i),
                              "\tYou must give a valid index"
                                      << "\n\tcurrent index: " << index(i)
                                      << "\n\trequired [" << 1 << ", " << this->dim(i) << "]");
            }
            if (this->b.isEmpty()) {
                for (NumN i = 1; i <= this->dims(); i++) {
                    y(i) = this->a(i)(index(i));
                }
                return m(y);
            } else {
                y.assign(this->b);
                NumN j = 1;
                for (NumN i = 1; i <= m.dims(); i++) {
                    if (y(i) == 0) {
                        y(i) = index(j);
                        j++;
                    }
                }
                MATH21_ASSERT(j - 1 == this->dims(), "j is " << j);
                return m(y);
            }
        }

        // as container.
        virtual T &operator()(NumN j1) override {
            MATH21_ASSERT(j1 <= this->volume());
            VecN index(this->dims());
            math21_operator_number_num_to_index_right(j1, index, this->shape());
            return operator()(index);
        }

        virtual const T &operator()(NumN j1) const override {
            return TensorView<T>::operator()(j1);
        }

        virtual const T &operator()(NumN j1, NumN j2) const override {
            return TensorView<T>::operator()(j1, j2);
        }

        virtual const T &operator()(NumN j1, NumN j2, NumN j3) const override {
            return TensorView<T>::operator()(j1, j2, j3);
        }

        virtual T &operator()(NumN j1, NumN j2) override {
            MATH21_ASSERT(this->dims() == 2, "not 2-D tensor");
            MATH21_ASSERT(
                    j1 >= 1 && j1 <= this->dim(1) && j2 >= 1 && j2 <= this->dim(2),
                    "\tYou must give a valid index");
            if (this->b.isEmpty()) {
                return m(this->a(1)(j1), this->a(2)(j2));
            } else {
                y.assign(this->b);
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

        virtual T &operator()(NumN j1, NumN j2, NumN j3) override {
            MATH21_ASSERT(this->dims() == 3, "not 3-D tensor");
            MATH21_ASSERT(
                    j1 >= 1 && j1 <= this->dim(1) && j2 >= 1 && j2 <= this->dim(2) && j3 >= 1 && j3 <= this->dim(3),
                    "\tYou must give a valid index");
            if (this->b.isEmpty()) {
                return m(this->a(1)(j1), this->a(2)(j2), this->a(3)(j3));
            } else {
                y.assign(this->b);
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

        virtual T &operator()(NumN j1, NumN j2, NumN j3, NumN j4) override {
            MATH21_ASSERT(this->dims() == 4, "not 4-D tensor");
            MATH21_ASSERT(
                    j1 >= 1 && j1 <= this->dim(1) && j2 >= 1 && j2 <= this->dim(2) && j3 >= 1 &&
                    j3 <= this->dim(3) && j4 >= 1 && j4 <= this->dim(4),
                    "\tYou must give a valid index");
            if (this->b.isEmpty()) {
                return m(this->a(1)(j1), this->a(2)(j2), this->a(3)(j3), this->a(4)(j4));
            } else {
                y.assign(this->b);
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


        virtual NumB isWritable() const override {
            return (NumB) 1;
        }

        NumB isContinuous() const override {
            return 0;
        }

        virtual std::string getClassName() const override {
            return "TensorSub";
        }

        // convert to continuous Tensor
        void toTensor(Tensor <T> &A) const {
            MATH21_ASSERT(A.getClassName() == "Tensor");
            A.setSize(this->shape());
            A.assign((const Tensor<T> &) *this);
        }
    };

    typedef TensorSub<NumZ> TenSubZ;
    typedef TensorSub<NumN> TenSubN;
    typedef TensorSub<NumR> TenSubR;


    template<typename T>
    TensorSub<T> Tensor<T>::sliceSub(const Seqce <Tensor<NumN>> &index) {
        TensorSub<T> tv(*this, index);
        return tv;
    }

    //Tensor<NumN> is VecN
    template<typename T>
    TensorSub<T> Tensor<T>::shrinkSub(const Tensor <NumN> &index) {
        TensorSub<T> tv(*this, index);
        return tv;
    }

}

