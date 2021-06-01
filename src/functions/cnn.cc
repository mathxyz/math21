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

#include <fstream>
#include "../opt/SteepestDescent.h"
#include "cnn.h"

namespace math21 {

//######################### common functions
    void math21_operator_ml_pooling_get_mk_ms(NumN m1, NumN n1, NumN m2, NumN n2,
                                              NumN &mk, NumN &nk, NumN &ms, NumN &ns) {
        ms = (m1 - m1 % m2) / m2;
        ns = (n1 - n1 % n2) / n2;
        mk = ms + m1 % m2;
        nk = ns + n1 % n2;
    }

    // we don't assert. So use carefully.
    void math21_operator_ml_pooling_valueAt(const TenR &x, TenR &xn_next, NumN cnn_type_pooling,
                                    NumN mk, NumN nk, NumN ms, NumN ns,
                                    Seqce <TenN> *p_xn_argmax,
                                    NumB isUsingDiff
    ) {
        NumN j1, j2, j3, i1, i2, i3;
        NumZ ii2, ii3; //absolute index w.r.t. x.
        NumZ ia, ic;
        NumR val;
        NumR y;
        NumR max;
        NumN ii2_max; // argmax index of x
        NumN ii3_max;

        if (cnn_type_pooling == cnn_type_pooling_average) {
            for (j1 = 1; j1 <= xn_next.dim(1); ++j1) {
                for (j2 = 1; j2 <= xn_next.dim(2); ++j2) {
                    for (j3 = 1; j3 <= xn_next.dim(3); ++j3) {
                        y = 0;
                        ia = (j2 - 1) * ms;
                        ic = (j3 - 1) * ns;
                        i1 = j1;
                        for (i2 = 1; i2 <= mk; ++i2) {
                            ii2 = ia + i2;
                            for (i3 = 1; i3 <= nk; ++i3) {
                                ii3 = ic + i3;
                                val = x(i1, ii2, ii3);
                                y = y + val;
                            }
                        }
                        y = y / (mk * nk);
                        xn_next(j1, j2, j3) = y;
                    }
                }
            }
        } else if (cnn_type_pooling == cnn_type_pooling_max) {
            for (j1 = 1; j1 <= xn_next.dim(1); ++j1) {
                for (j2 = 1; j2 <= xn_next.dim(2); ++j2) {
                    for (j3 = 1; j3 <= xn_next.dim(3); ++j3) {
                        y = 0;
                        ia = (j2 - 1) * ms;
                        ic = (j3 - 1) * ns;
                        i1 = j1;
                        for (i2 = 1; i2 <= mk; ++i2) {
                            ii2 = ia + i2;
                            for (i3 = 1; i3 <= nk; ++i3) {
                                ii3 = ic + i3;
                                val = x(i1, ii2, ii3);
                                if (i2 == 1 && i3 == 1) {
                                    max = val;
                                    ii2_max = ii2;
                                    ii3_max = ii3;
                                } else {
                                    if (val > max) {
                                        max = val;
                                        ii2_max = ii2;
                                        ii3_max = ii3;
                                    }
                                }
                            }
                        }
                        y = max;
                        if (isUsingDiff) {
                            Seqce <TenN> &xn_argmax = *p_xn_argmax;
                            xn_argmax(1)(j1, j2, j3) = ii2_max;
                            xn_argmax(2)(j1, j2, j3) = ii3_max;
                        }
                        xn_next(j1, j2, j3) = y;
                    }
                }
            }
        }
    }

//#########################
    cnn::cnn() {
        init();
    }

    void math21_serialize_model(std::ofstream &out, cnn &f, const VecR &theta) {
        SerializeNumInterface_simple sn;
        f.serialize(out, sn);
        math21_io_serialize(out, theta, sn);
    }

    void math21_deserialize_model(std::ifstream &in, cnn &f, VecR &theta) {
        DeserializeNumInterface_simple sn;
        f.deserialize(in, sn);
        math21_io_deserialize(in, theta, sn);
    }

    void OptimizationInterface_cnn::save_function(think::Optimization &opt) {
        SteepestDescent &sd = (SteepestDescent &) opt;
        cnn_cost_class &objectiveFunction = (cnn_cost_class &) sd.getFunctional();

        VecR &theta = (VecR &) sd.getMinimum();

        std::ofstream out;
        out.open(name.c_str(), std::ofstream::binary);
        math21_serialize_model(out, objectiveFunction.get_cnn(), theta);
        out.close();
    }

    void OptimizationInterface_cnn::onFinishOneInteration(think::Optimization &opt) {
        SteepestDescent &sd = (SteepestDescent &) opt;
        cnn_cost_class &mlpCostClass = (cnn_cost_class &) sd.getFunctional();
        mlpCostClass.updateParas();

        if (sd.getTime() % 10 == 1) {
            save_function(opt);
        }
    }

    void evaluate_cnn(cnn &f,
                      cnn_cost_class &J,
                      const Seqce<TenR> &X,
                      const Seqce<TenR> &Y, NumB isUsePr, NumB isUsingPenalty) {
        MATH21_ASSERT(!X.isEmpty(), "X is empty");
        MATH21_ASSERT(X.size() == Y.size(), "X and Y must contain same number of points");
        MATH21_ASSERT(X(1).dims() == 3 && Y(1).dims() == 3, "data point must be 3-D tensor");
        for (NumN i = 1; i <= X.size(); ++i) {
            MATH21_ASSERT(X(i).isSameSize(X(1).shape()) && Y(i).isSameSize(Y(1).shape()),
                          "data points must have same size currently.\n"
                          "You can use pooling as first layer to remove the restriction.");
        }
        TenR y_hat;
        if (X(1).volume() == 1 && Y(1).volume() == 1) {
            for (NumN i = 1; i <= X.size(); i++) {
                const TenR &a = f.valueAt(X(i));
                std::cout << "x: " << X(i)(1, 1, 1) << ", y: " << Y(i)(1, 1, 1) << ", f(x): ";
                if (isUsePr) {
                    if (y_hat.isEmpty()) {
                        y_hat.setSize(a.shape());
                    }
                    SoftargmaxOperator::valueAt(a, y_hat);
                    std::cout << y_hat(1, 1, 1);
                } else {
                    std::cout << a(1, 1, 1);
                }
                std::cout << ", loss: " << J.valueAt_cnn_y(f, a, Y(i), isUsingPenalty) << std::endl;
            }
        } else {
            for (NumN i = 1; i <= X.size(); i++) {
                const TenR &a = f.valueAt(X(i));
                std::cout << "x:\n";
                X(i).log(0, 0, 0);
                std::cout << "y:\n";
                Y(i).log(0, 0, 0);
                std::cout << "f(x):\n";
                if (isUsePr) {
                    if (y_hat.isEmpty()) {
                        y_hat.setSize(a.shape());
                    }
                    SoftargmaxOperator::valueAt(a, y_hat);
                    y_hat.log(0, 0, 0);
                } else {
                    a.log(0, 0, 0);
                }
                std::cout << "loss: " << J.valueAt_cnn_y(f, a, Y(i), isUsingPenalty) << std::endl;
                std::cout << "\n";
            }
        }

    }

    void evaluate_cnn_error_rate(cnn &f,
                                 cnn_cost_class &J,
                                 const Seqce<TenR> &X,
                                 const Seqce<TenR> &Y, NumB isUsePr, NumB isUsingPenalty) {
        MATH21_ASSERT(!X.isEmpty(), "X is empty");
        MATH21_ASSERT(X.size() == Y.size(), "X and Y must contain same number of points");
        MATH21_ASSERT(X(1).dims() == 3 && Y(1).dims() == 3, "data point must be 3-D tensor");
        for (NumN i = 1; i <= X.size(); ++i) {
            MATH21_ASSERT(X(i).isSameSize(X(1).shape()) && Y(i).isSameSize(Y(1).shape()),
                          "data points must have same size currently.\n"
                          "You can use pooling as first layer to remove the restriction.");
        }


        NumR error = 0;
        NumR cost = 0;

        NumN y_hat_max_index;
        NumN y_max_index;
        TenR y_hat;
        for (NumN i = 1; i <= X.size(); i++) {
            if (i % 10 == 1) {
                std::cout << "start evaluating " << i << "th point\n" << std::endl;
            }
            const TenR &a = f.valueAt(X(i));
            if (isUsePr) {
                if (y_hat.isEmpty()) {
                    y_hat.setSize(a.shape());
                }
                SoftargmaxOperator::valueAt(a, y_hat);
                y_hat_max_index = math21_operator_argmax(y_hat);
                y_max_index = math21_operator_argmax(Y(i));
                if (y_hat_max_index != y_max_index) {
                    error = 1;
                } else {
                    error = 0;
                }
            } else {
//                a.log(0, 0, 0);
                error = J.valueAt_cnn_y(f, a, Y(i), isUsingPenalty);
            }
//            std::cout << "error: " << error <<"\n"<< std::endl;
            cost = cost + error;
        }
        cost = cost / X.size();
        std::cout << "cost: " << cost << std::endl;
    }

}