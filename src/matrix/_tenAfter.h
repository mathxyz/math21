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

#include "_tenMid.h"

namespace math21 {
    template<typename T>
    class Tensor;

    namespace detail {

        template<typename T>
        void math21_output_figure_out_width(std::ostream &out, const T &m) {
            out << m;
        }

        template<typename T>
        void math21_output_number(std::ostream &out, const T &m) {
            out << m;
        }


        template<typename T>
        void _math21_output_matrix(std::ostream &out, const Tensor<T> &m, NumB isUsingCommas = 0) {
            std::string str;
            if (isUsingCommas) {
                str = ", ";
            } else {
                str = " ";
            }

            using namespace std;
            const streamsize old = out.width();

            // first figure out how wide we should make each field
            string::size_type w = 0;
            ostringstream sout;
            for (NumN r = 1; r <= m.nrows(); ++r) {
                for (NumN c = 1; c <= m.ncols(); ++c) {
                    math21_output_figure_out_width(sout, m(r, c));
//                sout << m(r, c);
                    w = xjmax(sout.str().size(), w);
                    sout.str("");
                }
            }

            // now actually print it
            std::stringstream ss;
            if (m.dims() == 1) {
                for (NumN r = 1; r <= m.nrows(); ++r) {
                    for (NumN c = 1; c <= m.ncols(); ++c) {
                        out.width(static_cast<streamsize>(w));
                        ss.str("");
                        ss << "(" << r << ")";
                        math21_output_number(out, m(r, c));
                        out << str;
                    }
                    if (r < m.nrows())out << "\n";
                }
            } else {
                for (NumN r = 1; r <= m.nrows(); ++r) {
                    for (NumN c = 1; c <= m.ncols(); ++c) {
                        out.width(static_cast<streamsize>(w));
                        ss.str("");
                        ss << "(" << r << "," << c << ")";
                        math21_output_number(out, m(r, c));
                        out << str;
                    }
                    if (r < m.nrows())out << "\n";
                }
            }
            if (!math21_global_tensor_is_log_no_last_new_line()) {
                out << "\n";
            }
            out.width(old);
        }

        template<typename T>
        void _math21_output(std::ostream &out, const Tensor<T> &A, Tensor<NumN> &v, NumN k) {
            if (k == A.dims()) {
                for (NumN i = 1; i <= A.dim(k); ++i) {
                    v(k) = i;
                    if (i == 1) {
                        out << "[";
                    }

                    if (math21_global_tensor_is_log_all_elements() || A.dim(k) <= 6) {
                        out << A(v);
                    } else {
                        if (i == 4) {
                            i = A.dim(k) - 3;
                            out << "...";
                        } else {
                            out << A(v);
                        }
                    }

                    if (i != A.dim(k)) {
                        out << ", ";
                    } else {
                        out << "]";
                    }
                }
            } else {
                for (NumN i = 1; i <= A.dim(k); ++i) {
                    v(k) = i;
                    if (i == 1) {
                        out << "[";
                    } else {
                        out << math21_string_replicate_n(k, " ");
                    }
                    if (math21_global_tensor_is_log_all_elements() || A.dim(k) <= 6) {
                        _math21_output(out, A, v, k + 1);
                    } else {
                        if (i == 4) {
                            i = A.dim(k) - 3;
                            out << "...";
                        } else {
                            _math21_output(out, A, v, k + 1);
                        }
                    }
                    if (i != A.dim(k)) {
                        out << "," << math21_string_replicate_n(A.dims() - k, "\n");
                    } else {
                        out << "]";
                    }
                }
            }
        }

        template<typename T>
        void math21_output(std::ostream &out, const Tensor<T> &A, NumB isUsingCommas) {
            MATH21_ASSERT(!A.isEmpty());
            if (math21_global_tensor_is_log_matlab_style() && (A.dims() == 1 || A.dims() == 2)) {
                _math21_output_matrix(out, A, isUsingCommas);
            } else {
                Tensor<NumN> v;
                v.setSize(A.dims());
                NumN k = 1;
                _math21_output(out, A, v, k);
            }
        }
    }

    ////////////// literal_assign_helper_tensor
    namespace detail {
        template<class T>
        literal_assign_helper_tensor<T>::literal_assign_helper_tensor(const literal_assign_helper_tensor<T> &item) : A(
                item.A), has_been_used(0) {
            item.d.copyTo(d);
        }

        template<class T>
        literal_assign_helper_tensor<T>::literal_assign_helper_tensor(Tensor<T> &m_) : A(m_), has_been_used(0) {
            isFull = 0;
            d.setSize(A.dims());
            d = 1;
            next();
        }

        template<class T>
        void literal_assign_helper_tensor<T>::clear() {
            MATH21_ASSERT(!has_been_used || isFull,
                          "You have used the tensor comma based assignment incorrectly by failing to\n"
                          "supply a full set of values for every element of a tensor object.\n");
        }

        template<class T>
        literal_assign_helper_tensor<T>::~literal_assign_helper_tensor() {
            clear();
        }

        template<class T>
        const literal_assign_helper_tensor<T> &literal_assign_helper_tensor<T>::operator,(
                const T &val
        ) const {
            MATH21_ASSERT(!isFull,
                          "You have used the tensor comma based assignment incorrectly by attempting to\n" <<
                                                                                                           "supply more values than there are elements in the tensor object being assigned to.\n\n"
                                                                                                           <<
                                                                                                           "Did you forget to call setSize()?"
                                                                                                           << "\n\t m->shape(): "
                                                                                                           << A.shape());
            A(d) = val;
            next();
            has_been_used = 1;
            return *this;
        }

        template<class T>
        void literal_assign_helper_tensor<T>::next(
        ) const {
            if (math21_operator_container_increaseNumFromRight(A.shape(), d) == 0) {
                isFull = 1;
            }
        }
    }
}