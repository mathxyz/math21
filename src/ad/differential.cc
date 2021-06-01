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

#include "functions_01/files.h"
#include "differential.h"

namespace math21 {

    namespace ad {
        /*
1. DAG shape: ||, V, ^, |
   tape shape: |

1. dual space
[https://sites.math.northwestern.edu/~scanez/courses/334/notes/dual-spaces.pdf]
[Linear Algebra Done Right, 3rd ed. by Axler]
1. automatic differentiation
Chapter 4 of [Dougal's PhD thesis](https://dougalmaclaurin.com/phd-thesis.pdf)
[talk by Matt at the Deep Learning Summer School, Montreal 2017](http://videolectures.net/deeplearning2017_johnson_automatic_differentiation/).
[Autograd](https://github.com/HIPS/autograd)
[JAX](https://github.com/google/jax)

1 given a function f: R^m -> R^n, with Jacobian function J: R^m -> R^(n*m),
then
function JVP: (R^m, R^m) -> R^n defined as JVP(x, g) = J(x)*g,
function VJP: (R^m, R^n) -> R^m defined as VJP(x, g) = g.trans*J(x).
function JMP: (R^m, R^(m*l)) -> R^n defined as JMP(x, G) = J(x)*G,
function MJP: (R^m, R^(l*n)) -> R^n defined as MJP(x, G) = G*J(x),
1 function f can be defined as a class, and function JVP can be defined as another class, or just as a method of class f.
1 JVP and VJP can be used to compute higher order derivatives.
  JMP and MJP can be used to compute first order derivatives.
1 Any numerical types other than vector must map to a vector space R^D in order to use ad.
1 Directional derivative is a generalization of partial derivative.
1 function JVP(x, *) is the directional derivative of f at x.
let f: V -> W,
then
1) JVP: (V, V) -> W, JVP(x, a) = lim (f(x+alpha*a) - f(x))/alpha, alpha -> 0
here alpha in R, x, a, in V
2) VJP: (V, W^*) -> V^*, VJP(x, b^*)(a) := b^* JVP(x, a), where x, a in V, b^* in W^*
V^* is dual space of V

1. use dependent inputs not all inputs as dependency to speed up.
1. use global object rather than function???
1. use tapes rather than global graph to speed up.
1. use template class to make defining new function cr easier.
1. cr, jvp, vjp
1. make ad backend-independent.
1. write computation graph visualization, see https://github.com/hips/autograd/blob/master/examples/dot_graph.py
1. consider autograd.test_util.check_grads
         * */
        namespace ad_detail {
            class adtoposort {
            private:
                const VariableMap &data;
                Map child_counts;
                Seqce<NumN> childless_nodes;

                // time cost: O(E)
                void compute(NumN y) {
                    Seqce<NumN> stack;
                    stack.push(y);
                    while (!stack.isEmpty()) {
                        NumN x = stack.getLast();
                        stack.removeLast();
                        if (child_counts.has(x)) {
                            child_counts.valueAt(x) += 1;
                        } else {
                            child_counts.add(x, 1);
                            stack.push(data(x).getX());
                        }
                    }
                    childless_nodes.push(y);
                }

            public:
                adtoposort(NumN y, const VariableMap &data) : data(data) {
                    compute(y);
                }

                // time cost: O(E)
                NumN next() {
                    if (childless_nodes.isEmpty()) {
                        return 0;
                    }
                    NumN y = childless_nodes.getLast();
                    childless_nodes.removeLast();
                    const Set &X = data(y).getX();
                    // There must exist at least one x such that the child count of x is 1. (yii)
                    for (NumN i = 1; i <= X.size(); ++i) {
                        NumN x = X(i);
                        if (child_counts.valueAt(x) == 1) {
                            childless_nodes.push(x);
                        } else {
                            child_counts.valueAt(x) -= 1;
                        }
                    }
                    return y;
                }
            };

            void clear_cache_y(VariableMap &data) {
                NumN n = data.size();
                for (NumN i = 1; i <= n; ++i) {
                    data.at(i).clearCacheY();
                }
            }

            void add_cache_y_locally(const Set &X, NumN y, VariableMap &data) {
                for (NumN i = 1; i <= X.size(); ++i) {
                    data.at(X(i)).add_cache_y(y);
                }
            }

            // todo: may optimize by adding some stop conditions.
            // time cost: O(E)
            void add_cache_y_graph(NumN y, VariableMap &data) {
                clear_cache_y(data);
                Map child_counts;
                Seqce<NumN> stack;
                stack.push(y);
                while (!stack.isEmpty()) {
                    NumN x = stack.getLast();
                    stack.removeLast();
                    if (child_counts.has(x)) {
//                        child_counts.valueAt(x) += 1;
                    } else {
                        child_counts.add(x, 1);
                        stack.push(data(x).getX());
                        add_cache_y_locally(data(x).getX(), x, data);
                    }
                }
            }

            // flagMap can be set.
            // time cost: O(E)
            void compute_forward_flag(const Set &X, Map &flagMap, VariableMap &data) {
                Map child_counts;
                Seqce<NumN> stack;
                stack.push(X);
                while (!stack.isEmpty()) {
                    NumN x = stack.getLast();
                    stack.removeLast();
                    if (child_counts.has(x)) {
//                        child_counts.valueAt(x) += 1;
                    } else {
                        child_counts.add(x, 1);

                        if (!flagMap.has(x)) {
                            flagMap.add(x, 1);
                        }
                        // parent is just name
                        const auto &parents = data(x).getCacheY();
                        for (NumN i = 1; i <= parents.size(); ++i) {
                            NumN parent = parents(i);
                            if (data(parent).getType() == variable_type_constant ||
                                data(parent).getType() == variable_type_zero_derivative) {
                                continue;
                            }
                            stack.push(parent);
                        }
                    }
                }
            }

            class adtape {
            private:
                VariableMap &data;
                Map child_counts;
                Seqce<NumN> childless_nodes;
                Map flagMap;


                // time cost: O(E)
                void compute(NumN y) {
                    Seqce<NumN> stack;
                    stack.push(y);
                    while (!stack.isEmpty()) {
                        NumN x = stack.getLast();
                        stack.removeLast();
                        if (child_counts.has(x)) {
                            child_counts.valueAt(x) += 1;
                        } else {
                            child_counts.add(x, 1);
                            const auto &parents = data(x).getX();
                            for (NumN i = 1; i <= parents.size(); ++i) {
                                NumN parent = parents(i);
                                if (!flagMap.has(parent)) {
                                    continue;
                                }
                                stack.push(parent);
                            }
                        }
                    }
                    childless_nodes.push(y);
                }

            public:
                // get tape such that Set(tape) = {v |X->v->y}
                adtape(const Set &X, NumN y, VariableMap &data) : data(data) {
                    // add edges from x to others
                    add_cache_y_graph(y, data);
                    // get sub-graph ^
                    compute_forward_flag(X, flagMap, data);
                    // must clear cache because cache may be used by other tape when the tape is inside another tape.
                    clear_cache_y(data);
                    // get sub-graph V from ^ rather than ||, so the sub-graph is tape with shape |.
                    compute(y);
                }

                ~adtape() {}

                // time cost: O(E)
                NumN next() {
                    if (childless_nodes.isEmpty()) {
                        return 0;
                    }
                    NumN y = childless_nodes.getLast();
                    childless_nodes.removeLast();
                    const Set &X = data(y).getX();
                    // There must exist at least one x such that the child count of x is 1. (yii)
                    for (NumN i = 1; i <= X.size(); ++i) {
                        NumN x = X(i);
                        // x not in graph
                        if (!flagMap.has(x)) {
                            continue;
                        }
                        if (child_counts.valueAt(x) == 1) {
                            childless_nodes.push(x);
                        } else {
                            child_counts.valueAt(x) -= 1;
                        }
                    }
                    return y;
                }
            };

            const Set &math21_ad_diff_getX(const NumN &y, const VariableMap &data) {
                const Variable &vy = data(y);
                const Set &X1 = vy.getX();
                return X1;
            }

            void math21_ad_diff_getX(Set &X, const NumN &y, const Set &V, const VariableMap &data) {
                const Variable &vy = data(y);
                const Set &X1 = vy.getX();
                X1.intersect(V, X);
            }

            void math21_ad_diff_getY(const NumN &x, Set &Y, const Set &V0, const VariableMap &data) {
                const Variable &vx = data(x);
                const Set &Y1 = vx.getY();
                Y1.intersect(V0, Y);
            }

            // compute derivative, debug: n, level
            NumN math21_ad_diff_cd(const NumN &x, const Set &V, const Set &V0, Map &DT, NumN mode, VariableMap &data,
                                   NumN n = 1, NumN level = 1, NumN debugLevel = 0) {
                if (debugLevel) {
                    m21log("cd level", level);
                }
                {
                    NumN y;
                    if (DT.get(x, y)) {
                        return y;
                    }
                }

                Set Y;
                math21_ad_diff_getY(x, Y, V0, data);
                MATH21_ASSERT(!Y.isEmpty())

                Set S;
                Set output;
                for (NumN j = 1; j <= Y.size(); ++j) {
                    NumN y = Y(j);
                    NumN dy = math21_ad_diff_cd(y, V, V0, DT, mode, data, n, level + 1, debugLevel);
                    if (dy == 0) {
                        continue;
                    }
                    Set X;
                    math21_ad_diff_getX(X, y, V, data);
                    Function &f = data.at(y).getf();
                    if (mode == derivative_mode_dar) {
                        f.cr(X, x, y, dy, output, data);
                    } else if (mode == derivative_mode_dbr) {
                        f.backward(X, x, y, dy, output, data);
                    } else if (mode == derivative_mode_dbr_vjp) {
                        output.clear();
                        NumN dx = f.cr_vjp(X, x, y, dy, data);
                        if (dx != 0) {
                            output.set(dx);
                        }
                    } else {
                        MATH21_ASSERT(0)
                    }
                    if (!output.isEmpty()) {
                        NumN dxi = output(1);
                        S.add(dxi);
                    }
                }
                NumN dx;
                if (S.isEmpty()) {
                    dx = 0;
                } else if (S.size() == 1) {
                    dx = S(1);
                } else {
                    op_add _op_add;
                    Function &add = _op_add;
                    if (mode == derivative_mode_dar) {
                        add.f(S, output, data);
                    } else if (mode == derivative_mode_dbr) {
                        add.forward(S, output, data);
                    } else if (mode == derivative_mode_dbr_vjp) {
                        output.clear();
                        NumN y_add = add.evaluate(S, data);
                        if (y_add != 0) {
                            output.set(y_add);
                        }
                    } else {
                        math21_tool_assert(0);
                    }
                    if (!output.isEmpty()) {
                        dx = output(1);
                    } else {
                        math21_tool_assert(0);
                    }
                }
                DT.add(x, dx);
                return dx;
            }

            // todo: try to optimize
            // Map DT, differential table
            void math21_ad_diff_cd_inc(const Set &X0, NumN y0, Map &DT, NumN mode, VariableMap &data) {
                {
                    NumN dy0;
                    // todo: use another DT to speed up
                    MATH21_ASSERT(DT.get(y0, dy0));
                }

                // Set(tape) < Set(topsort)
//            adtoposort toposort(y0, data); // shape: V
                MATH21_PRINT_TIME_ELAPSED(adtape toposort(X0, y0, data)); // shape: |

                // debug time
                NumR timeSum = 0;
                timer time;
                while (1) {
                    time.start();
                    NumN y = toposort.next();
                    time.end();
                    timeSum += time.time();

                    if (y == 0) {
                        break;
                    }
                    if (data(y).getType() == variable_type_constant ||
                        data(y).getType() == variable_type_zero_derivative) {
                        continue;
                    }
                    time.start();
                    NumN dy = DT.valueAt(y);
                    time.end();
                    timeSum += time.time();
//                DT.remove(y); // add this when use tape
                    if (dy == 0) {
                        continue;
                    }
                    const Set &X = data(y).getX();
                    // getX(X, y, V);
                    for (NumN i = 1; i <= X.size(); ++i) {
                        NumN xi = X(i);
                        if (data(xi).getType() == variable_type_constant ||
                            data(xi).getType() == variable_type_zero_derivative) {
                            continue;
                        }
                        // vjp for x
                        Function &f = data.at(y).getf();
                        Set output;
                        if (mode == derivative_mode_dbr_vjp) {
                            output.clear();
                            NumN dxi_part = f.cr_vjp(X, xi, y, dy, data);
                            if (dxi_part != 0) {
                                output.set(dxi_part);
                            }
                        } else {
                            MATH21_ASSERT(0)
                        }
                        if (!output.isEmpty()) {
                            NumN dxi_part = output(1);
                            if (!DT.has(xi)) {
                                op_add add0;
                                Function &add = add0;
                                NumN dxi = add.evaluate(dxi_part, data);
                                DT.add(xi, dxi);
                                data.at(dxi).setName(math21_string_to_string("dx", i).c_str());
                            } else {

                                time.start();
                                NumN dxi = DT.valueAt(xi);
                                time.end();
                                timeSum += time.time();

                                Function &add = data.at(dxi).getf();
                                math21_tool_assert(add.getName() == op_add().getName());
                                auto &add0 = (op_add &) add;
                                Set input;
                                input.add(dxi_part);
                                add0.evaluate_inc(input, dxi, data);
                            }
                        }
                    }
                }
//            m21log("\ntimeSum time used", timeSum);
            }

            void math21_ad_diff_cds_inc(const Set &X, NumN y, Map &dX, Map &DT, NumN mode, NumN n,
                                        VariableMap &data) {
                data.backup();
                if (y == 0) {
                    m21log("Compute null function!");
                    return;
                }

                dX.clear();

                std::string y_name = data.at(y).getName();

                NumN dy;
                if (!DT.get(y, dy)) {
                    op_mat_eye mat_eye0;
                    Function &mat_eye = mat_eye0;
                    if (mode == derivative_mode_dar) {
                        mat_eye.f(y, dy, data);
                    } else if (mode == derivative_mode_dbr) {
                        mat_eye.forward(y, dy, data);
                    } else if (mode == derivative_mode_dbr_vjp) {
                        MATH21_ASSERT(data(y).getValue().isScalarInMath(),
                                      "Variable " << data(y).getName() << " must be num when using vjp mode.");
                        dy = data.createC("1");
                        Function::variable_set_device_type_using_variable(y, dy, data);
                        Function::variable_setSize_to_same_vspace_using_variable(y, dy, data);
                        data.at(dy).getValue() = 1;
                    } else {
                        MATH21_ASSERT(0);
                    }
                    data.at(dy).setName(math21_string_concatenate("d(", y_name, ")/d(self)").c_str());
                    DT.add(y, dy);
                }

                NumN level = 1;
//            Set X_connected;
//            X.intersect(V0, X_connected);
                math21_ad_diff_cd_inc(X, y, DT, mode, data);
//            DT.restrictTo(X_connected, dX);
                DT.restrictTo(X, dX);

                Set X0;
                Set X1;
                dX.getX(X1);
                X.difference(X1, X0);
                if (!X0.isEmpty()) {
                    for (NumN i = 1; i <= X0.size(); ++i) {
                        dX.add(X0(i), 0);
                    }
                }
            }

            // very very slow, many redundancies
            template<typename GraphType>
            NumB restrictSet_pass(const Set &X, Set &V0, const Set &Y, const GraphType &f, NumN &debug_count) {
                ++debug_count;
                if (X.size() == 0) {
                    return 0;
                }

                NumB flag = 0;
                NumN X_size = X.size();
                for (NumN i = 1; i <= X_size; ++i) {
                    NumN x = X(i);
                    if (Y.contains(x)) {
                        V0.add(x);
                        if (flag == 0) {
                            flag = 1;
                        }
                    } else {
                        const Set &Y1 = f.getY(x);
                        if (restrictSet_pass(Y1, V0, Y, f, debug_count)) {
                            V0.add(x);
                            if (flag == 0) {
                                flag = 1;
                            }
                        }
                    }
                }
                return flag;
            }

            struct GraphType_forward {
                const VariableMap &data;

                GraphType_forward(const VariableMap &data) : data(data) {
                }

                const Set &getY(NumN x) const {
                    math21_tool_assert(0);
                    return data(x).getY();
                }
            };

            struct GraphType_backward {
                const VariableMap &data;

                GraphType_backward(const VariableMap &data) : data(data) {
                }

                const Set &getY(NumN x) const {
                    return data(x).getX();
                }
            };

            // pass_forward <=> pass_backward
            void math21_ad_diff_pass_forward(const Set &X, Set &V0, const Set &Y, const VariableMap &data) {
                GraphType_forward forward(data);
                NumN debug_count = 0;
                restrictSet_pass(X, V0, Y, forward, debug_count);
                m21log("restrictSet_pass", debug_count);
            }

            // pass_forward <=> pass_backward
            void math21_ad_diff_pass_backward(const Set &X, Set &V0, const Set &Y, const VariableMap &data) {
                GraphType_backward backward(data);
                NumN debug_count = 0;
                restrictSet_pass(Y, V0, X, backward, debug_count);
                m21log("restrictSet_pass", debug_count);
            }

            // V0 = {V |X->v->Y}
            void
            math21_ad_diff_restrictSet(const Set &V, Set &V0, const Set &X, const Set &Y, const VariableMap &data) {
                V0.clear();
                math21_ad_diff_pass_backward(X, V0, Y, data);
                if (0) {
                    Set Vf;
                    math21_ad_diff_pass_forward(X, Vf, Y, data);
                    math21_tool_assert(V0.isEqual(Vf));
                }
            }

            void math21_ad_diff_cds(const Set &X, NumN y, const Set &V, Map &dX, Map &DT, NumN mode, VariableMap &data,
                                    NumN n, NumN debugLevel) {
                if (y == 0) {
                    m21log("Compute null function!");
                    return;
                }

                dX.clear();

                std::string y_name = data.at(y).getName();
                Set V0;
                Set Y;
                Y.add(y);
                math21_ad_diff_restrictSet(V, V0, X, Y, data);

                NumN dy;
                if (!DT.get(y, dy)) {
                    op_mat_eye mat_eye0;
                    Function &mat_eye = mat_eye0;
                    if (mode == derivative_mode_dar) {
                        mat_eye.f(y, dy, data);
                    } else if (mode == derivative_mode_dbr) {
                        mat_eye.forward(y, dy, data);
                    } else if (mode == derivative_mode_dbr_vjp) {
                        MATH21_ASSERT(data(y).getValue().size() == 1,
                                      "Variable " << data(y).getName() << " must be num when using vjp mode.");
                        dy = ad_global_get_constant_1();
                    } else {
                        MATH21_ASSERT(0)
                    }
                    data.at(dy).setName(math21_string_concatenate("d(", y_name, ")/d(self)").c_str());
                    DT.add(y, dy);
                }

                NumN level = 1;
                Set X_connected;
                X.intersect(V0, X_connected);
                for (NumN i = 1; i <= X_connected.size(); ++i) {
                    math21_ad_diff_cd(X_connected(i), V, V0, DT, mode, data, n, level, debugLevel);
                }
                DT.restrictTo(X_connected, dX);

                Set X0;
                Set X1;
                dX.getX(X1);
//            math21_convert_container_to_set(dX.getX(), X1);
                X.difference(X1, X0);
                if (!X0.isEmpty()) {
                    for (NumN i = 1; i <= X0.size(); ++i) {
                        dX.add(X0(i), 0);
                    }
                }
            }
        }

        using namespace ad_detail;

        Derivative::Derivative(VariableMap &data) : data(data) {
            debugLevel = 0;
            _is_cd_inc = 1;
//            _is_cd_inc = 0;
        }

        Derivative::~Derivative() {
        }

        void Derivative::removeLastRecord() {
            MATH21_ASSERT(_is_cd_inc == 1);
            data.restore();
        }

        NumN Derivative::cd(NumN x, NumN y) {
            Map dX;
            Set X;
            X.add(x);
            const Set &V = data.getV();
            cds(X, y, V, dX, 1);
            return dX.valueAt(x);
        }

        NumN Derivative::backward(NumN x, NumN y) {
            Map dX;
            Set X;
            X.add(x);
            const Set &V = data.getV();
            Map DT;
            _cds(X, y, V, dX, DT, derivative_mode_dbr, 1);
            return dX.valueAt(x);
        }

        NumN Derivative::grad_with_mode(NumN x, NumN y, NumN mode) {
            Map dX;
            Set X;
            X.add(x);
            const Set &V = data.getV();
            Map DT;
            if (_is_cd_inc) {
                _cds_inc(X, y, V, dX, DT, mode, 1);
            } else {
                _cds(X, y, V, dX, DT, mode, 1);
            }
            return dX.valueAt(x);
        }

        NumN Derivative::grad_jvp(NumN x, NumN y) {
            return grad_with_mode(x, y, derivative_mode_dbr_jvp);
        }

        NumN Derivative::grad_vjp(NumN x, NumN y) {
            NumN dx = grad_with_mode(x, y, derivative_mode_dbr_vjp);
            if (dx != 0) {
            } else {
                m21warn("grad_vjp returns 0");
//                auto d = data(x).getValue().shape();
//                dx = data.createC(0, "dx");
//                data.at(dx).getValue().setSize(d);
//                data.at(dx).getValue() = 0;
            }
            return dx;
        }

        NumN Derivative::cd(NumN x, NumN y, Map &DT) {
            Map dX;
            Set X;
            X.add(x);
            const Set &V = data.getV();
            _cds(X, y, V, dX, DT, derivative_mode_dar, 1);
            return dX.valueAt(x);
        }

        // If define-by-run, then compute must be called before cds. And this makes sure graph(compute) contains graph(cds).
        // Every function will check it by calling Variable.isComputed().
        // compute derivative. Set value to constant. n is used for debug.
        // V is modified in data.
        void Derivative::cds(const Set &X, NumN y, const Set &V, Map &dX, NumN n) {
            Map DT;
            _cds(X, y, V, dX, DT, derivative_mode_dar, n);
        }

        void Derivative::_cds(const Set &X, NumN y, const Set &V, Map &dX, Map &DT, NumN mode, NumN n) {
            math21_ad_diff_cds(X, y, V, dX, DT, mode, data, n, debugLevel);
        }

        void Derivative::_cds_inc(const Set &X, NumN y, const Set &V, Map &dX, Map &DT, NumN mode, NumN n) {
            math21_ad_diff_cds_inc(X, y, dX, DT, mode, n, data);
        }

        // X: input
        void Derivative::fv(const Set &X, NumN y, NumN mode, NumN level) {
            if (debugLevel) {
                m21log("fv level", level);
            }
            if (y == 0) {
                m21log("Compute function value with f=0!");
                return;
            }
            if (data(y).isComputed()) {
                return;
            }
            // Element in X is considered as input even through its type may not be the input type.
            if (X.contains(y)) {
                return;
            }
            NumN type = data.at(y).getType();
            if (type == variable_type_constant || type == variable_type_input) {
                return;
            }
            const Set &X0 = math21_ad_diff_getX(y, data);
            for (NumN i = 1; i <= X0.size(); ++i) {
                NumN x = X0(i);
                fv(X, x, mode, level + 1);
            }
            Function &f = data.at(y).getf();
            Set Y;
            Y.add(y);
            if (mode == derivative_mode_dar) {
                f.fv(data(y).getX(), Y, data);
            } else if (mode == derivative_mode_dar_dbr) {
                f.compute(data(y).getX(), Y, data, *this);
            } else {
                MATH21_ASSERT(0)
            }
            // todo: use the following and remove setComputed(1) in every specific function
            if (!data(y).isComputed()) {
                data.at(y).setComputed(1);
            }
        }

        // compute function values where no variable size in program level will be considered.
        void Derivative::fvs(const Set &X, const Set &Y) {
            for (NumN i = 1; i <= data.size(); ++i) {
                data.at(i).setComputed(0);
            }
            for (NumN i = 1; i <= Y.size(); ++i) {
                fv(X, Y(i), derivative_mode_dar);
            }
        }

        void Derivative::compute(NumN y) {
            Set X;
            fv(X, y, derivative_mode_dar_dbr);
        }

        void Derivative::compute(const Set &X, NumN y) {
            fv(X, y, derivative_mode_dar_dbr);
        }

        // compute function values where no variable size in program level will be considered.
        void Derivative::compute(const Set &X, const Set &Y, const Set &V, NumB isReset) {
            if (isReset) {
                for (NumN i = 1; i <= data.size(); ++i) {
                    data.at(i).setComputed(0);
                }
            }
            for (NumN i = 1; i <= Y.size(); ++i) {
                fv(X, Y(i), derivative_mode_dar_dbr);
            }
        }

        void Derivative::setDebugLevel(NumN debugLevel0) {
            debugLevel = debugLevel0;
        }

        VariableMap &Derivative::getData() {
            return data;
        }

        void math21_ad_diff_cd_inc_with_mode_dbr_vjp(
                const Set &X, NumN y, Map &DT, NumN mode, VariableMap &data) {
            math21_ad_diff_cd_inc(X, y, DT, derivative_mode_dbr_vjp, data);
        }
    }
}