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

#include "../ad/functions_01/op_get_shape.h"
#include "../ad/functions_01/op_share_reshape.h"
#include "../ad/functions_02/op_broadcast_tensor.h"
#include "../algebra/set.h"
#include "../ad/differential.h"
#include "function.h"

namespace math21 {
    namespace ad {
        AdVar::AdVar() {
            id = 0;
            data = 0;
        }

        AdVar::AdVar(NumN id, VariableMap *data) : id(id), data(data) {
        }

        AdVar::AdVar(const AdVar &v) {
            *this = v;
        }

        AdVar::~AdVar() {}

        AdVar &AdVar::operator=(const AdVar &v) {
            id = v.id;
            data = v.data;
            return *this;
        }

        //        const NumB Function::isSetSizeFlag = 0;
//        const NumB Function::isSetSizeFlag = 1;
        NumB Function::isSetSizeFlag = 0;
//        NumB Function::isElementWiseTestFlag = 1;
        NumB Function::isElementWiseTestFlag = 0;

        Function::Function() {
            isElementWiseFlag = 0;
            isGlobalFlag = 0;
        }

        Function::~Function() {
        }

        void Function::cr_jvp(const Set &X, NumN x, NumN y, NumN dy, Set &Y, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        NumN Function::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            MATH21_ASSERT(0)
            return 0;
        }

        NumN Function::cr_vjp(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
//            static NumN count = 0;
//            m21log(getName(), ++count);
            // todo: remove this
//            op_share_reshape a;
            if (variable_reshape_to_same_vspace_using_variable(y, dy, data) == 1) {
                MATH21_ASSERT(0, ">>>>>this shouldn't be called!");
            }

            NumN dx = cr_vjp_inner(X, x, y, dy, data);
            // dxi_part can't be global contant, 'cause it will be used by others.
            if (data(dx).getType() == variable_type_constant) {
                if (data(dx).getValue().size() == 1) {
                    MATH21_ASSERT(0, "check here")
                }
            }

            // put here so as to avoid putting everywhere.
            op_get_shape _get_shape;
            Function &f_get_shape = _get_shape;
            NumN d_x = f_get_shape.evaluate(x, data);
            op_share_reshape _op_share_reshape;
            Function &f_op_share_reshape = _op_share_reshape;
            dx = f_op_share_reshape.evaluate(dx, d_x, data);
            return dx;
        }

        void Function::cr_jmp(const Set &X, NumN x, NumN y, NumN dy, Set &Y, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        void Function::cr_mjp(const Set &X, NumN x, NumN y, NumN dy, Set &Y, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        NumN Function::evaluate(NumN x, VariableMap &data) {
            Set X;
            X.add(x);
            return evaluate(X, data);
        }

        NumN Function::evaluate(NumN x1, NumN x2, VariableMap &data) {
            Set X;
            X.add(x1);
            X.add(x2);
            return evaluate(X, data);
        }

        NumN Function::evaluate(NumN x1, NumN x2, NumN x3, VariableMap &data) {
            Set X;
            X.add(x1);
            X.add(x2);
            X.add(x3);
            return evaluate(X, data);
        }

        NumN Function::evaluate(const Set &X, VariableMap &data) {
            NumN y = data.createV(math21_string_concatenate(getName(), "(x)").c_str());
            variable_set_device_type_using_variable(X(1), y, data);
            data.at(y).setf(this);
            data.at(y).setX(X);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            Set Y;
            Y.add(y);
            fv(X, Y, data);
            return y;
        }

        void Function::df(const Set &X, NumN x, NumN y, NumN &dydx, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        void Function::df_dbr(const Set &X, NumN x, NumN y, NumN &dydx, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        void Function::cr(const Set &X, NumN x, NumN y, NumN dy, Set &Y, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        void Function::backward(const Set &X, NumN x, NumN y, NumN dy, Set &Y, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        void Function::f(const Set &X, Set &Y, VariableMap &data) {
            MATH21_ASSERT(0)
        }

        void Function::fv(const Set &X, const Set &Y, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        void Function::f(NumN x, NumN &y, VariableMap &data) {
            Set X;
            X.add(x);
            Set Y;
            f(X, Y, data);
            y = Y(1);
        }

        void Function::f(NumN x1, NumN x2, NumN &y, VariableMap &data) {
            Set X;
            X.add(x1);
            X.add(x2);
            Set Y;
            f(X, Y, data);
            y = Y(1);
        }

        void Function::f(NumN x1, NumN x2, NumN x3, NumN &y, VariableMap &data) {
            Set X;
            X.add(x1);
            X.add(x2);
            X.add(x3);
            Set Y;
            f(X, Y, data);
            y = Y(1);
        }

        void Function::compute(NumN x, NumN y, VariableMap &data, Derivative &derivative) {
            Set X;
            X.add(x);
            Set Y;
            Y.add(y);
            compute(X, Y, data, derivative);
        }

        void Function::compute(NumN x1, NumN x2, NumN y, VariableMap &data, Derivative &derivative) {
            Set X;
            X.add(x1);
            X.add(x2);
            Set Y;
            Y.add(y);
            compute(X, Y, data, derivative);
        }

        void Function::compute(NumN x1, NumN x2, NumN x3, NumN y, VariableMap &data, Derivative &derivative) {
            Set X;
            X.add(x1);
            X.add(x2);
            X.add(x3);
            Set Y;
            Y.add(y);
            compute(X, Y, data, derivative);
        }

        void Function::compute(const Set &X, const Set &Y, VariableMap &data, Derivative &derivative) {
            fv(X, Y, data);
        }

        void Function::forward(NumN x, NumN &y, VariableMap &data) {
            Set X;
            X.add(x);
            Set Y;
            forward(X, Y, data);
            y = Y(1);
        }

        void Function::forward(NumN x1, NumN x2, NumN &y, VariableMap &data) {
            Set X;
            X.add(x1);
            X.add(x2);
            Set Y;
            forward(X, Y, data);
            y = Y(1);
        }

        void Function::forward(NumN x1, NumN x2, NumN x3, NumN &y, VariableMap &data) {
            Set X;
            X.add(x1);
            X.add(x2);
            X.add(x3);
            Set Y;
            forward(X, Y, data);
            y = Y(1);
        }

        void Function::forward(const Set &X, Set &Y, VariableMap &data) {
            f(X, Y, data);
            Derivative derivative(data);
            compute(X, Y, data, derivative);
        }

        NumB Function::isSetSize() {
            return isSetSizeFlag;
        }

        void Function::setSetSizeFlag(NumB flag) {
            isSetSizeFlag = flag;
        }

        NumB Function::isElementWiseTest() {
            return isElementWiseTestFlag;
        }

        NumB Function::isElementWise() const {
            return isElementWiseFlag;
        }

        void Function::setElementWiseFlag(NumB flag) {
            isElementWiseFlag = flag;
        }

        NumB Function::isGlobal() const {
            return isGlobalFlag;
        }

        void Function::setGlobalFlag(NumB flag) {
            isGlobalFlag = flag;
        }

        // see TensorBroadcast error?
        // see np.broadcast_arrays
        void Function::broadcast_tensors(const Set &X, Set &Y, VariableMap &data) {
            math21_tool_assert(X.size() > 0);
            Seqce<VecN> shapes(X.size());
            for (NumN i = 1; i <= X.size(); ++i) {
                NumN x = X(i);
                const auto &x_value = data(x).getValue();
                VecN d;
                shapes.at(i) = x_value.shape(d);
            }
            VecN d;
            NumB flag = math21_broadcast_is_compatible_in_ele_op(shapes, d);
            MATH21_ASSERT(flag, "shape not compatible when broadcasting\n"
                    << X.log("X") << shapes.log("shapes") << data.log("data"));

            NumN k = data.createC("shape");
            data.setValue(k, 1);
            data.at(k).getValue() = d;
            Y.clear();
            for (NumN i = 1; i <= X.size(); ++i) {
                NumN x = X(i);
                NumN x_new = x;
                const auto &x_value = data(x).getValue();
                // todo: not broadcast same shape tensors
                if (!x_value.isSameSize(d)) {
                    op_broadcast_tensor bc;
                    Function &function = bc;
                    x_new = function.evaluate(x, k, data);
                }
                Y.add(x_new);
            }
        }

        // todo: remove this, and use broadcast_tensors instead.
        void Function::broadcast_num_to_vec(const Set &X, Set &Y, VariableMap &data) {
            broadcast_tensors(X, Y, data);
        }

        void Function::variable_set_device_type_using_variable(NumN x, NumN y, VariableMap &data) {
            math21_tool_assert(y != 0);
            const auto &x_value = data(x).getValue();
            auto &y_value = data.at(y).getValue();
            y_value.setDeviceType(x_value.getDeviceType());
        }

        void Function::variable_set_device_type_gpu(NumN y, VariableMap &data) {
            math21_tool_assert(y != 0);
            auto &y_value = data.at(y).getValue();
            y_value.setDeviceType(m21_device_type_gpu);
        }

        NumN Function::variable_get_device_type(NumN x, VariableMap &data) {
            return data(x).getValue().getDeviceType();
        }

        NumB Function::variable_is_cpu(NumN x, VariableMap &data) {
            return data(x).getValue().is_cpu();
        }

        NumB Function::variable_setSize_to_same_vspace_using_variable(NumN x, NumN y, VariableMap &data) {
            if (y == 0) {
                return 0;
            }
            const auto &x_value = data.at(x).getValue();
            auto &y_value = data.at(y).getValue();
            return math21_operator_tensor_setSize_to_same_vspace_using_value(x_value, y_value);
        }

        NumB Function::variable_setSize_to_same_vspace_using_shape(const VecN &d, NumN y, VariableMap &data) {
            if (y == 0) {
                return 0;
            }
            auto &y_value = data.at(y).getValue();
            return math21_operator_tensor_setSize_to_same_vspace_using_shape(d, y_value);
        }

        NumB Function::variable_reshape_to_same_vspace_using_variable(NumN x, NumN y, VariableMap &data) {
            if (y == 0) {
                return 0;
            }
            const auto &x_value = data.at(x).getValue();
            auto &y_value = data.at(y).getValue();
            return math21_operator_tensor_reshape_to_same_vspace_using_value(x_value, y_value);
        }

        NumB Function::variable_reshape_to_same_vspace_using_shape(const VecN &d, NumN y, VariableMap &data) {
            if (y == 0) {
                return 0;
            }
            auto &y_value = data.at(y).getValue();
            return math21_operator_tensor_reshape_to_same_vspace_using_shape(d, y_value);
        }
    }
}