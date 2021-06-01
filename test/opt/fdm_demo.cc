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
//#include <fstream>
//#include "files.h"
//#include "inner.h"
//
//using namespace math21;
//
//NumR f_ex_x3_x2_x(NumR x) {
//    return -0.1 * pow(x, 4) - 0.15 * pow(x, 3) - 0.5 * pow(x, 2) - 0.25 * x + 1.2;
//}
//
//NumR f_ex_x3_x2_x_derivative(NumR x) {
//    return -0.4 * pow(x, 3) - 0.45 * pow(x, 2) - x - 0.25;
//}
//
//// h is step size
//// df(xi) = (f(xi1) -f(xi))/h, xi1 = xi + h
//NumR f_ex_x3_x2_x_derivative_fdm_forward(NumR (*f)(NumR x), NumR xi, NumR h = 0.25) {
//    NumR xi1 = xi + h;
//    return (f(xi1) - f(xi)) / h;
//}
//
//// h is step size
//// df(xi) = (f(xi) -f(xi_1))/h, xi_1 = xi - h
//NumR f_ex_x3_x2_x_derivative_fdm_backward(NumR (*f)(NumR x), NumR xi, NumR h = 0.25) {
//    NumR xi_1 = xi - h;
//    return (f(xi) - f(xi_1)) / h;
//}
//
//// h is step size
//// df(xi) = (f(xi1) - f(xi_1))/(2h), xi1 = xi + h, xi_1 = xi - h
//NumR f_ex_x3_x2_x_derivative_fdm_central(NumR (*f)(NumR x), NumR xi, NumR h = 0.25) {
//    NumR xi1 = xi + h;
//    NumR xi_1 = xi - h;
//    return (f(xi1) - f(xi_1)) / (2 * h);
//}
//
//void f_ex_x3_x2_x_fdm_test() {
//    NumR x = 0.5;
//    NumR dx = f_ex_x3_x2_x_derivative(x);
//    m21log("x", x);
//    m21log("df", dx);
//    m21log("df_fdm_forward", f_ex_x3_x2_x_derivative_fdm_forward(f_ex_x3_x2_x, x));
//    m21log("df_fdm_backward", f_ex_x3_x2_x_derivative_fdm_backward(f_ex_x3_x2_x, x));
//    m21log("df_fdm_central", f_ex_x3_x2_x_derivative_fdm_central(f_ex_x3_x2_x, x));
//}
