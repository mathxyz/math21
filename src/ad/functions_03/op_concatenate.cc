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

#include "inner_cc.h"
#include "op_concatenate.h"

namespace math21 {
    namespace ad {
        NumN op_concatenate::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            NumN pos = math21_operator_container_arg(X, x);
            MATH21_ASSERT(pos < X.size())
            auto &axis_value = data(X(X.size())).getValue();
            MATH21_ASSERT(axis_value.isScalarInMath());
            NumZ _axis = (NumZ) axis_value(1);
            NumN axis = math21_number_container_pos_check(data(X(1)).getValue().dims(), _axis);
            VecN dis(pos - 1);
            for (NumN i = 1; i <= dis.size(); ++i) {
                dis.at(i) = data(X(i)).getValue().dim(axis);
            }
            NumN offset = static_cast<NumN>(math21_operator_container_sum(dis, 1));
            NumN di = data(X(pos)).getValue().dim(axis);

            auto pdy = ad_point(dy, 0);
            auto paxis = ad_point(X(X.size()), 0);
            return ad_axis_i_sub_get(pdy, offset, di, paxis).id;
        }

        NumN op_concatenate::evaluate(const Set &X, VariableMap &data) {
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

        // X = {x1, x2, ..., xn, axis}
        void op_concatenate::fv(const Set &X, const Set &Y, VariableMap &data) const {
            math21_tool_assert(X.size() >= 3);
            NumN y = Y(1);
            auto &y_value = data.at(y).getValue();
            Seqce<const TenR *> xs(X.size() - 1);
            for (NumN i = 1; i < X.size(); ++i) {
                NumN x = X(i);
                xs.at(i) = &data(x).getValue();
            }
            auto &axis_value = data(X(X.size())).getValue();
            MATH21_ASSERT(axis_value.isScalarInMath());
            NumZ axis = (NumZ) axis_value(1);
            math21_op_tensor_concatenate(xs, y_value, axis);
        }

        NumN op_axis_i_sub_get::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            NumN pos = math21_operator_container_arg(X, x);
            MATH21_ASSERT(pos == 1)
            auto pdy = ad_point(dy, 0);
            auto px = ad_point(x, 0);
            auto d_x = ad_get_shape(px);
            auto poffset = ad_point(X(2), 0);
            auto paxis = ad_point(X(4), 0);
            return ad_axis_i_sub_set(pdy, poffset, d_x, paxis).id;
        }

        NumN op_axis_i_sub_get::evaluate(const Set &X, VariableMap &data) {
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

        // X = {x, offset, di, axis}
        void op_axis_i_sub_get::fv(const Set &X, const Set &Y, VariableMap &data) const {
            math21_tool_assert(X.size() == 4);
            NumN y = Y(1);
            auto &y_value = data.at(y).getValue();
            auto &x_value = data(X(1)).getValue();
            auto &offset_value = data(X(2)).getValue();
            MATH21_ASSERT(offset_value.isScalarInMath());
            NumN offset = (NumN) offset_value(1);
            auto &di_value = data(X(3)).getValue();
            MATH21_ASSERT(di_value.isScalarInMath());
            NumN di = (NumN) di_value(1);
            auto &axis_value = data(X(4)).getValue();
            MATH21_ASSERT(axis_value.isScalarInMath());
            NumZ axis = (NumZ) axis_value(1);
            math21_op_tensor_sub_axis_i_get(y_value, x_value, offset, di, axis);
        }

        NumN op_axis_i_sub_set::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            NumN pos = math21_operator_container_arg(X, x);
            MATH21_ASSERT(pos == 1)
            MATH21_ASSERT(0, "not implement")
            return 0;
        }

        NumN op_axis_i_sub_set::evaluate(const Set &X, VariableMap &data) {
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

        // X = {x, offset, d_y, axis}
        void op_axis_i_sub_set::fv(const Set &X, const Set &Y, VariableMap &data) const {
            math21_tool_assert(X.size() == 4);
            NumN y = Y(1);
            auto &y_value = data.at(y).getValue();
            auto &x_value = data(X(1)).getValue();
            auto &offset_value = data(X(2)).getValue();
            MATH21_ASSERT(offset_value.isScalarInMath());
            NumN offset = (NumN) offset_value(1);

            VecN d_y;
            d_y = data(X(3)).getValue();
            y_value.setSize(d_y);
            y_value = 0;

            auto &axis_value = data(X(4)).getValue();
            MATH21_ASSERT(axis_value.isScalarInMath());
            NumZ axis = (NumZ) axis_value(1);
            math21_op_tensor_sub_axis_i_set(x_value, y_value, offset, axis);
        }
    }
}