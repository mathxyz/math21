///* Copyright 2015 The math21 Authors. All Rights Reserved.
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.
//==============================================================================*/
//
//#include "sin.h"
//
//namespace math21 {
//    namespace ad {
//        // sin(x): x, sin(x), dy=1, dx
//        void test_sin() {
//            Set X;
//            Set V;
//            NumN y;
//            VariableMap data(V);
//            NumN x = data.createV("x");
//            Variable &vx = data.at(x);
//            vx.setType(variable_type_input);
//            vx.getValue().setSize(2);
//            vx.getValue() = 2, 3.14;
////            vx.getValue().setSize(1);
////            vx.getValue() = 3.14;
//            Set input;
//            input.add(x);
//            X.add(input);
//
//            Set output;
//            op_mat_sin sin;
//            sin.f(input, output, data);
//
//            y = output(1);
//            Variable &vy = data.at(y);
//            vy.setType(variable_type_output);
//
//            Map dX;
//            Derivative d(data);
//            d.cds(X, y, V, dX);
//
//            Set Y;
//            Y.add(y);
//            Y.add(dX.getY());
//            d.fvs(X, Y, V);
//
//            X.log("X");
//            Y.log("Y");
//            V.log("V");
//            dX.log("dX");
//            data.log("data");
//        }
//
//
//        // 2*sin(x): x, sin(x), dy=1, dx
//        void test_2_sin() {
//            Set X;
//            Set V;
//            NumN y;
//            VariableMap data(V);
//            NumN x = data.createV("x");
//            Variable &vx = data.at(x);
//            vx.setType(variable_type_input);
//            vx.getValue().setSize(2);
//            vx.getValue() = 2, 3.14;
////            vx.getValue().setSize(1);
////            vx.getValue() = 3.14;
//            Set input;
//            input.add(x);
//            X.add(input);
//
//            Set output;
//            op_mat_sin sin;
//            sin.f(input, output, data);
//
//            output.copyTo(input);
//            op_vec_multiply multiply;
//            NumN k = data.createC(2, "2");
//            if (Function::isSetSize()) {
//                setSizeyByx(input(1), k, data);
//            }
//            input.add(k);
//            multiply.f(input, output, data);
//
//            y = output(1);
//            Variable &vy = data.at(y);
//            vy.setType(variable_type_output);
//
//            Map dX;
//            Derivative d(data);
//            d.cds(X, y, V, dX);
//
//            Set Y;
//            Y.add(y);
//            Y.add(dX.getY());
//            d.fvs(X, Y, V);
//
//            X.log("X");
//            Y.log("Y");
//            V.log("V");
//            dX.log("dX");
//            data.log("data");
//        }
//
//        // sin(cos(x))
//        void test_sin_cos_right(NumR x) {
//            NumR dx = xjcos(xjcos(x)) * (-xjsin(x));
//            m21_log(std::cout, "dx", dx);
//        }
//
//        // sin(cos(x)): x, sin(x), dy=1, dx
//        void test_sin_cos() {
//            Set X;
//            Set V;
//            NumN y;
//            VariableMap data(V);
//            NumN x = data.createV();
//            Variable &vx = data.at(x);
//            vx.getValue().setSize(2);
//            vx.getValue() = 2, 3.14;
//            Set input;
//            input.add(x);
//            X.add(input);
//
//            Set output;
//            op_mat_cos cos;
//            cos.f(input, output, data);
//            output.copyTo(input);
//
//            op_mat_sin sin;
//            sin.f(input, output, data);
//            y = output(1);
//
//            Map dX;
//            Derivative d(data);
//            d.cds(X, y, V, dX);
//
//            Set Y;
//            Y.add(y);
//            Y.add(dX.getY());
//            d.fvs(X, Y, V);
//
//            X.log("X");
//            Y.log("Y");
//            V.log("V");
//            dX.log("dX");
//            data.log("data");
//
//            test_sin_cos_right(vx.getValue()(1));
//            test_sin_cos_right(vx.getValue()(2));
//        }
//
//        // 2*cos(sin(cos(x)))
//        void test_2_cos_sin_cos_right(NumR x) {
//            NumR dx = 2 * (-sin(sin(cos(x)))) * xjcos(xjcos(x)) * (-xjsin(x));
//            m21_log(std::cout, "dx", dx);
//        }
//
//
//        // 2*cos(sin(cos(x))): x, sin(x), dy=1, dx
//        void test_2_cos_sin_cos() {
//            Set X;
//            Set V;
//            NumN y;
//            VariableMap data(V);
//            NumN x = data.createV();
//            Variable &vx = data.at(x);
////            vx.getValue().setSize(1);
////            vx.getValue() = 3.14;
//            vx.getValue().setSize(2);
//            vx.getValue() = 2, 3.14;
//            Set input;
//            input.add(x);
//            X.add(input);
//
//            Set output;
//            op_mat_cos cos;
//            cos.f(input, output, data);
//            output.copyTo(input);
//
//            op_mat_sin sin;
//            sin.f(input, output, data);
//            output.copyTo(input);
//
//            cos.f(input, output, data);
//            output.copyTo(input);
//
//            op_vec_multiply multiply;
//            NumN k = data.createC(2, "2");
//            if (Function::isSetSize()) {
//                setSizeyByx(input(1), k, data);
//            }
//            input.add(k);
//            multiply.f(input, output, data);
//            y = output(1);
//
//            Map dX;
//            Derivative d(data);
//            d.cds(X, y, V, dX);
//
//            Set Y;
//            Y.add(dX.getY());
//            Y.add(y);
//            d.fvs(X, Y, V);
//
//            X.log("X");
//            Y.log("Y");
//            V.log("V");
//            dX.log("dX");
//            data.log("data");
//
//            test_2_cos_sin_cos_right(vx.getValue()(1));
//            test_2_cos_sin_cos_right(vx.getValue()(2));
//        }
//
//        void test_sin_all() {
////            test_sin();
////            test_2_sin();
//            test_sin_cos();
////            test_2_cos_sin_cos();
//        }
//    }
//}