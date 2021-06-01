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
#include "files.h"
#include "inner.h"

namespace math21 {
    void test_num_NumN_and_NumZ() {
        math21_tool_log_title(__FUNCTION__);
        NumN a = 20;
        NumN b = 11;
        NumZ c = b - a;
        MATH21_PASS(a - b == 9)
        MATH21_PASS(c == -9, "" << c)
        MATH21_PASS(b - a == -9, "" << b - a)
        MATH21_PASS(-a == -20, "" << -a)
    }

    void test_array() {
        math21_tool_log_title(__FUNCTION__);
        ArrayZ v(40);
        ArrayZ u(50);
        u.log("u");
        v = 1;
        v.copyTo(u);
        u.log("u");
        v.log("v");
        v(2) = 5;
        u.log("u");
        v.log("v");
        math21_tool_log_title();
    }

    void test_array_sort() {
        ArrayZ v(4);
        v = 1, 3, 2, 4;
        v.log("v");
        v.sort();
        v.log("v");
    }

    void test_sort_li() {
        Seqce<NumZ> s;
        s.setSize(8);
        s = 55, 66, 33, 44, 77, 88, 11, 22;

        VecN idx;
        math21_algorithm_sort_indexes(s, idx);
        s.log("s1", 1);
        idx.log("idx");

        NumN x_min = math21_operator_argmin_li(s);
        NumN x_max = math21_operator_argmax_li(s);
        m21log("x_min", x_min);
        m21log("x_max", x_max);

        s.sort();
        s.log("s2", 1);


    }

    void test_sort_tensor_li() {
        VecZ s;
        s.setSize(2, 2, 2);
        s = 55, 66, 33, 44, 77, 88, 11, 22;
        math21_operator_sort(s);
        s.log("s2");
    }

    void test_sort_vector_li() {
        VecZ s;
        s.setSize(8);
        s = 55, 66, 33, 44, 77, 88, 11, 22;

        VecN idx;
        math21_algorithm_sort_indexes(s, idx);
        s.log("s1");
        idx.log("idx");

        NumN x_min = math21_operator_argmin_li(s);
        NumN x_max = math21_operator_argmax_li(s);
        m21log("x_min", x_min);
        m21log("x_max", x_max);

        math21_operator_sort(s);
        s.log("s2");
    }

    void test_shuffle_sort() {
        VecZ s;
        s.setSize(8);
        s = 55, 66, 33, 44, 77, 88, 11, 22;

        SeqceN x(s.size());
        SeqceN y(s.size());
        x.letters();
        y.letters();

        DefaultRandomEngine engine(21);
        math21_algorithm_shuffle(x, x.size(), engine);
        detail_li::Compare_index_index<NumZ, Tensor> comp(s, x);
        math21_algorithm_sort(y, comp);

        for (NumN i = 1; i <= s.size(); ++i) {
            m21log(s(x(y(i))));
        }

        NumN x_min = math21_operator_container_argmin_random(s, engine);
        NumN x_max = math21_operator_container_argmax_random(s, engine);
        m21log("x_min", x_min);
        m21log("x_max", x_max);
    }

    void test_array_memory_01() {
        AutoBuffer autoBuffer;
        autoBuffer.setSize(10 * sizeof(NumZ), m21_type_default);
        autoBuffer.log();

        ArrayZ v;
        SpaceParas paras = autoBuffer.getSpace();
        v.setSize(4, &paras);
        v = 1, 2, 3, 4;

        ArrayZ u;
        u.setSize(2, &paras);
//        u.ensureIndependence();

        v(2) = 5;
        u.log("u");
        v.log("v");
        autoBuffer.log();

        ArrayZ u_0(u.size());
        ArrayZ v_0(v.size());
        u_0 = 1, 5;
        v_0 = 1, 5, 3, 4;

        MATH21_PASS(math21_operator_container_isEqual(u, u_0, 0));
        MATH21_PASS(math21_operator_container_isEqual(v, v_0, 0));
    }

    void test_array_memory_02() {
        ArrayZ v;
        v.setSize(4);
        v = 1, 2, 3, 4;
        const AutoBuffer &autoBuffer = v.getAutoBuffer_dev();
        SpaceParas paras;

        ArrayZ u;
        paras = autoBuffer.getSpace(1, 2, sizeof(NumZ));
        u.setSize(2, &paras);
//        u.ensureIndependence();
        u.log("u");
        v.log("v");

        v(2) = 5;
        u.log("u");
        v.log("v");
        u.getAutoBuffer_dev().log("u");
        v.getAutoBuffer_dev().log("v");
    }

    void test_array_memory_03() {
        ArrayZ v;
        v.setSize(4);
        v = 1, 2, 3, 4;
        const AutoBuffer &autoBuffer = v.getAutoBuffer_dev();
        SpaceParas paras = autoBuffer.getSpace(0, 4, sizeof(NumZ));
        TenZ u;
        VecN d(2);
        d = 2, 2;
        u.setSize(d, &paras);
//        u.ensureIndependence();
        v.log("v");
        u.log("u");

        v(2) = 5;
        v.log("v");
        u.log("u");
        v.getAutoBuffer_dev().log("v");
        u.log("u");
    }

    void test_tensor_memory_01() {
        TenR v;
        v.setSize(1, 2, 3);
        v = 1, 2, 3, 4, 5, 6;
        NumR *data = math21_memory_tensor_data_address(v);
        MATH21_PASS(math21_operator_container_isEqual_c_array(v, data));
    }

    void test_no_memory_01() {
        ArrayZ v;
        v.setSize(4);
        v = 1, 2, 3, 4;
        v.log("v");
        SpaceParas paras;
        paras = v.getSpace(1, 2, sizeof(NumZ));

        ArrayZ u;
//        u.setSize(2);
//        u.log("u");
        u.setSizeNoSpace(2);
        u.setSpace(&paras);
//        u.ensureIndependence();

        v(2) = 5;
        v.log("v");
        u.log("u");

        // external data, so no ref_count
        SpaceParas paras_w;
        paras_w.address = paras.address;
        paras_w.start = paras.address;
        paras_w.ref_count = 0;
        paras_w.size = 2 * sizeof(NumZ);
        paras_w.unit = 1;
        ArrayZ w;
        w.setSizeNoSpace(2);
        w.setSpace(&paras_w);
        w.log("w");
    }

    void test_no_memory_02() {
        TenZ v;
        v.setSize(2, 2, 2);
        v = 1, 2, 3, 4, 5, 6, 7, 8;
        v.log("v");

        NumN offset = 0;
        SpaceParas paras = v.getSpace(offset, 4, sizeof(NumZ));
        TenZ u, w;
        VecN d(2);
        d = 2, 2;
        u.setSize(d);
        u = 0;
//        u.setSize(d, &paras);
//        u.setSizeNoSpace(d);
        u.setSpace(paras);
        u.log("u");

        offset = offset + u.volume();
        paras = v.getSpace(offset, 4, sizeof(NumZ));
        w.setSize(d, &paras);
//        u.ensureIndependence();
        w.log("w");

        v(1, 1, 2) = 5;

        TenZ x;
        x.setSizeNoSpace(d);
        x.setSpace(paras.start, paras.size);

        v.log("v");
        u.log("u");
        w.log("w");
        w.getSpace().log("w");
        x.log("x");
        x.getSpace().log("x");


    }


    void test_vector() {
        VecZ v;
        v.setSize(4);
        VecZ u;
        v = 1, 2, 3, 4;
        v.copyTo(u);
        u.log("u");
        v.log("v");
        v(2) = 5;
        u.log("u");
        v.log("v");
    }

    void test_vector_02() {
        VecZ v;
        v.setSize(4);
        v = 1, 2, 3, 4;
        v.log("v");
        VecZ &u = v;
        u.log("u");
    }

    void test_matrix() {
        MatZ C;
        C.setSize(2, 2);
        C = 3, 4, 5, 6;
        C.log();
        C(1, 1) = 2;
        C.log();
    }

    void test_matrix_operation() {
        MatR A;
        A.setSize(2, 2);
        A = 1, 2, 3, 4;
        A.log("A");
        VecR x;
        x.setSize(2);
        x = 2, 2;
        x.log("x");
        MatR output;

        math21_operator_multiply(1, A, x, output);
        output.log();

        math21_operator_trans_multiply(1, x, A, output);
        output.log();

        math21_operator_trans_multiply(1, x, x, output);
        output.log();
    }

    void test_tensor_01() {
        TenR A;
        A.setSize(2, 3);
        A = 1, 2, 3, 4, 5, 6;
        std::cout << A;

        TenR B;
        A.copyTo(B);
        B.log("B");
    }

    void test_tensor_02() {
        TenR A;
        A.setSize(2, 3);
        A = 1, 2, 3, 4, 5, 6;
        TenR B;
        A.copyTo(B);
        NumN i1 = 2;
        NumZ i2 = 2;
        B.operator()(i1, i2) = 9;
        B.log("B");
    }

    template<typename T>
    void test_tensor_copyTo(Tensor<T> &B, TenR &C) {
        TenR A;
        A.setSize(2, 3);
        A = 1, 2, 3, 4, 5, 6;
        A.copyTo(B);

        ArrayN d;
        d.setSize(A.dims());
        d = 1;
        C.setSize(A.shape());
        B(d) = 9;
        C(d) = 9;
    }

    void test_tensor_03() {
        TenR B, C;
        test_tensor_copyTo(B, C);
        B.log("B");
        C.log("C");
    }

    void test_serialize() {
        MatR m;
        m.setSize(3, 3);
        VecN v;
        v.setSize(3);
        v.letters();
        m.letters();
        m.log("m");
        v.log("v");

        std::ofstream out;
        out.open("z_matrix", std::ofstream::binary);
        SerializeNumInterface_simple sn;
        math21_io_serialize(out, m, sn);
        math21_io_serialize(out, v, sn);
        out.close();

        MatR m_2;
        VecN v_2;
        std::ifstream in;
        in.open("z_matrix", std::ifstream::binary);
        DeserializeNumInterface_simple dsn;
        math21_io_deserialize(in, m_2, dsn);
        math21_io_deserialize(in, v_2, dsn);
        in.close();
        m_2.log("m_2");
        v_2.log("v_2");

        MATH21_PASS(math21_op_isEqual(m, m_2));
        MATH21_PASS(math21_op_isEqual(v, v_2));
    }

    void test_serialize_seqce() {
        SeqceN v;
        v.setSize(3);
        v.letters();
        v.log("v");

        std::ofstream out;
        out.open("z_tmp", std::ofstream::binary);
        SerializeNumInterface_simple sn;
        math21_io_serialize(out, v, sn);
        out.close();

        SeqceN y;
        std::ifstream in;
        in.open("z_tmp", std::ifstream::binary);
        DeserializeNumInterface_simple dsn;
        math21_io_deserialize(in, y, dsn);
        in.close();
        y.log("y");
    }


    void test_solve_linear() {
        MatR A;
        A.setSize(3, 3);
        MatR B;
        B.setSize(3, 2);
        A =
                1, 2, 3,
                0, 5, 0,
                3, 9, 5;
        B =
                1, 2,
                1, 2,
                1, 2;
        MatR A_inv;
        A_inv.setSize(A.shape());
        A_inv.assign(A);
        MatR X;
        X.setSize(B.shape());
        X.assign(B);

        numerical_recipes::GaussJordanElimination gje;
        if (!gje.solve(A_inv, X)) {
            return;
        }

        A.log("A");
        A_inv.log("A_inv");
        B.log("B");
        X.log("X");

        MatR R;
        MatR tmp;
        math21_operator_multiply(1, A, X, tmp);
        math21_operator_linear(1, tmp, -1, B, R);
//        R = A * X - B;
        R.log("R");
        MatR I;
        math21_operator_multiply(1, A, A_inv, I);
//        I = A * A_inv;
        I.log("I");
    }

    void test_matrix_inversion() {
        MatR A;
        A.setSize(3, 3);
        A =
                1, 2, 3,
                0, 5, 0,
                3, 9, 5;

        MatR A_inv;
        if (!math21_operator_inverse(A, A_inv)) {
            return;
        }

        A.log("A");
        A_inv.log("A_inv");

        MatR I;
        math21_operator_multiply(1, A, A_inv, I);
//        I = A * A_inv;
        I.log("I");
    }

    void test_random_uniform() {
        DefaultRandomEngine engine(21);
        RanUniform ran(engine);

        MatR M(5, 5);
        ran.set(-1, 1);
        math21_random_draw(M, ran);
        M.log("M");

        MatR D(5, 5);
        ran.set(-1, -1);
        math21_random_draw(D, ran);
        D.log("D");

        MatZ B(5, 5);
        ran.set(-8, 8);
        math21_random_draw(B, ran);
        B.log("B");

        MatN C(5, 5);
        ran.set(-8, 8);
        math21_random_draw(C, ran);
        C.log("C");
    }

    void test_random_normal() {
        DefaultRandomEngine engine(21);
        RanNormal ran(engine);

        MatR m(5, 5);
        ran.set(0, 1);
        math21_random_draw(m, ran);
        m.log();

        MatR D(5, 5);
        ran.set(1, 2);
        math21_random_draw(D, ran);
        D.log();
    }

    void test_tensor_slice() {
        math21_tool_log_title(__FUNCTION__);
        TenR A(3, 3, 2);
        A.letters();
        A.log();
        Seqce<VecN> X;
        X.setSize(A.dims());
        for (NumN i = 1; i <= X.size(); i++) {
            X(i).setSize(2);
            X(i) = 1, 2;
        }
//    X.log("X", 0, 0);
        TenViewR tv = A.sliceView(X);
//    tv (1,1,1) = 5;
        tv.log("tv");

        TenSubR ts = A.sliceSub(X);
        NumZ i1 = 1;
        NumZ i2 = 1;
        NumN i3 = 1;
        ts.operator()(i1, i2, i3) = 5;
        ts.log("ts");
        A.log();
    }

    void test_tensor_slice_tensorsub() {
        math21_tool_log_title(__FUNCTION__);
        TenR A(2, 2, 2);
        A.letters();
        A.log();
        Seqce<VecN> X;
        X.setSize(A.dims());
        for (NumN i = 1; i <= X.size(); i++) {
            X(i).setSize(2);
            X(i) = 1, 2;
        }

        TenSubR ts = A.sliceSub(X);
        ts(1, 1, 1) = 5;
        ts.log("ts");
        A.log();

        const TenSubR &ts2 = ts;
        NumR num = ts2(1, 1, 1);
        m21log("num", num);
        ts2.log("ts2");

    }

    void test_tensor_shrink() {
        math21_tool_log_title(__FUNCTION__);
        TenR A(2, 3, 3, 2);
        A.letters();
//        A.log("A");
        VecN X;
        X.setSize(A.dims());
        X = 2, 0, 2, 0;

//        X.log("X", 0, 0);
        TenViewR tv = A.shrinkView(X);
//        tv.log("tv");

        TenR A1(3, 2);
        A1 =
                21, 22,
                27, 28,
                33, 34;
        TenR A2;
        tv.toTensor(A2);
        MATH21_PASS(math21_op_isEqual(A2, A1));

        TenSubR ts = A.shrinkSub(X);
        ts(1, 1) = 5;
//        ts.log("ts");

        A1 =
                5, 22,
                27, 28,
                33, 34;
        ts.toTensor(A2);
        MATH21_PASS(math21_op_isEqual(A2, A1));
        A.log("A", 1, 0);
    }

    void test_tensor_assign() {
        TenR A(3, 3, 2);
        A.letters();
        A.log("A");

        TenR B;
        B.setSize(A.shape());
        B.assign(A);
        B.log("B");

        MatR C(3);
        C.letters();
        C.log("C");

        TenR D(3);
        D.assign(C);
        D.log("D");
    }

    void test_tensor_reshape() {
        TenR A(3, 3, 2);
        A.letters();
        A.log("A");
        VecN d(2);
        d = 6, 3;
        A.reshape(d);
        A.log("A");

        A.toVector();
        A.log("A");
    }

    void test_array_swap() {
        ArrayN A(3);
        A = 1, 3, 5;
        ArrayN B(4);
        B = 2, 4, 6, 8;
        A.log("A");
        B.log("B");
        A.swap(B);
        A.log("A");
        B.log("B");
    }

    void test_seqce_swap() {

        Seqce<TenR> A(2);
        A.at(1).setSize(3);
        A.at(1) = 1, 3, 5;
        A.at(2).setSize(2, 2);
        A.at(2) = 2, 4, 6, 8;
        A.log("A");

        Seqce<TenR> B(1);
        B.at(1).setSize(3);
        B.at(1).letters();
        B.log("B");

        A.swap(B);
        A.log("A");
        B.log("B");
    }

    void test_tensor_swap() {
        TenR A(3);
        A = 1, 3, 5;
        TenR B(2, 2);
        B = 2, 4, 6, 8;
        A.log("A");
        B.log("B");
        A.swap(B);
        A.log("A");
        B.log("B");
    }

    void test_tensor_submatrix() {
        MatR A(3, 2);
        A.letters();
        MatR B;
        VecN row_ind(2);
        VecN col_ind(1);
        row_ind = 1, 3;
        col_ind = 0;
        math21_operator_matrix_submatrix(A, B, row_ind, col_ind);
        B.log("B");
    }

    void test_tensor_f_shrink() {
        TenR A;
        TenR B;
//        A.setDeviceType(m21_device_type_gpu);
//        B.setDeviceType(m21_device_type_gpu);
        A.setSize(3, 2, 2);
        A.letters();
        VecN index(A.dims());
        index = 0, 0, 1;
        TensorFunction_sum f_sum;
//        math21_operator_tensor_f_shrink(A, B, index, f_sum);
        math21_op_tensor_f_shrink(A, B, index, m21_fname_sum);
        A.log("A");
        B.log("B");

        TenR outcome(3, 2);
        outcome =
                3, 7,
                11, 15,
                19, 23;

        MATH21_PASS(math21_op_isEqual(B, outcome, 0))
//        exit(0);

        TensorFunction_argmin f_argmin;
//        math21_operator_tensor_f_shrink(A, B, index, f_argmin);
        math21_op_tensor_f_shrink(A, B, index, m21_fname_argmin);
        A.log("A");
        B.log("B");

        outcome =
                1, 1,
                1, 1,
                1, 1;

        MATH21_PASS(math21_op_isEqual(B, outcome, 0))

        TensorFunction_mean f_mean;
//        math21_operator_tensor_f_shrink(A, B, index, f_mean);
        math21_op_tensor_f_shrink(A, B, index, m21_fname_mean);
        A.log("A");
        B.log("B");

        outcome =
                1.5, 3.5,
                5.5, 7.5,
                9.5, 11.5;

        MATH21_PASS(math21_op_isEqual(B, outcome, 0))
    }

    void test_seqce_string() {
        Seqce<std::string> A(2);
        A.at(1) = "hello";
        A.at(2) = "math21";
        A.push("!");
        A.log("A");
    }

    void test_tensor_omp() {
        math21_omp_info();

        // base 2
        NumN bit = 8;
        TenR A(static_cast<NumN>(0x1 << bit), static_cast<NumN>(0x1 << bit));
        TenR B(static_cast<NumN>(0x1 << bit), static_cast<NumN>(0x1 << bit));
        TenR C, C_omp;

        DefaultRandomEngine engine(21);
        RanNormal ran(engine);
        ran.set(0, 1);
        math21_random_draw(A, ran);
        math21_random_draw(B, ran);

        NumR t = math21_time_getticks();
        math21_operator_multiply_no_parallel(1, A, B, C);
        t = math21_time_getticks() - t;
        m21log("time", t);

        NumR t_omp = math21_time_getticks();
        math21_operator_multiply(1, A, B, C_omp);
        t_omp = math21_time_getticks() - t_omp;
        m21log("time omp", t_omp);

        NumR d = math21_operator_container_distance(C, C_omp, 1);
        MATH21_PASS(math21_operator_container_isEqual(C, C_omp, 1e-12), "distance/size = " << d / C.size());
        m21log("Speed up", t / t_omp);
        C.logInfo("C");
    }

    void math21_numerical_recipes_test() {
        math21_tool_log_title(__FUNCTION__);
        math21_numerical_recipes_Fitmrq_test();
    }

    void math21_test_tensor_string() {
        math21_tool_log_title(__FUNCTION__);
        VecStr s(2);
        VecStr y;
        s = "math";
        s(2) = "21";
        s.log("");
        y = s;
        y.log("y");
    }

    void math21_test_tensor_log() {
        MatZ x;
        VecN d(4);
        d = 20, 1, 20, 2;
        x.setSize(d);
        x.letters();
        x.log("x");
    }

    void math21_test_operator_eigen_real_sys() {
        TenR A(3, 3);
        A =
                1, 2, 3,
                2, 4, 5,
                3, 5, 6;
        A.log();
        MatR L, X;
        math21_operator_eigen_real_sys_descending(A, L, X);
        L.log("Lambda", 0, 0, 7);
        X.log("X", 0, 0, 7);
        MatR L0, X0;
        L0.setSize(3);
        L0 = 11.34481428, 0.17091519, -0.51572947;
        X0.setSize(3, 3);
        X0 =
                0.32798528, 0.59100905, -0.73697623,
                0.59100905, -0.73697623, -0.32798528,
                0.73697623, 0.32798528, 0.59100905;
        MATH21_PASS(math21_op_isEqual(L, L0, 1e-6))
        MATH21_PASS(math21_op_isEqual(X, X0, 1e-6))
    }

    void test_raw_tensor_transpose() {
        Tensor<NumR32> A(2, 4, 3);
        Tensor<NumR32> B(4, 2, 3);
        A.letters();
        math21_vector_transpose_d1234_to_d1324_cpu(A.getDataAddress(),
                                                   B.getDataAddress(),
                                                   1, 2, 4, 3);

        Tensor<NumR32> A_new;
        A_new.setSize(A.shape());
        math21_vector_transpose_d1234_to_d1324_cpu(B.getDataAddress(),
                                                   A_new.getDataAddress(),
                                                   1, 4, 2, 3);

        MATH21_PASS(math21_operator_container_isEqual(A, A_new, 0));
    }

    void test_raw_tensor_set() {
        Tensor<NumR32> A(2, 4, 3);
        Tensor<NumR32> B(2, 2, 3);
        A.letters();
        B.zeros();
        math21_vector_assign_3d_d2_cpu(A.getDataAddress(),
                                       B.getDataAddress(),
                                       A.dim(1), A.dim(2), A.dim(3), B.dim(2), 1, 1);

        A.log("A");
        B.log("B");
    }

    void test_tensor_text_save() {
        const char *name = "./test.txt";
        std::string content;
        content = "math\n"
                  "21\n"
                  "\n";
        math21_file_text_string_save(name, content);
    }

    void test_tensor_text_read() {
        test_tensor_text_save();

        const char *name = "./test.txt";
        Seqce<TenN8> lines;
        math21_file_text_read_lines(name, lines);
        lines.log("lines");
    }

    void test_tensor_to_string() {
        TenN8 A(6);
        A = 'm', 'a', 't', 'h', '2', '1';
        std::string s;
        math21_operator_tensor_to_string(A, s);
        MATH21_PASS(s == "math21");
    }

    void test_string_replace() {
        TenN8 A(6);
        A = 'm', 'a', 't', 'h', '2', '1';
        math21_string_replace(A, '2', ' ');
        math21_string_replace(A, '1', ' ');
    }

    void test_tensor() {
//        test_num_NumN_and_NumZ();
//        math21_tool_log_num_type();
//        test_array();
//        test_vector();
//        test_vector_02();
//        test_matrix();
//        test_matrix_operation();
//        test_tensor_01();
//        test_tensor_02();
//        test_tensor_03();
//        test_serialize();
//        test_serialize_seqce();
//
//        test_solve_linear();
//        test_matrix_inversion();
//
//        test_array_memory_01();
//        test_array_memory_02();
//        test_array_memory_03();
//        test_tensor_memory_01();
//
//        test_no_memory_01();
//        test_no_memory_02();
//
//        test_random_uniform();
//        test_random_normal();
//
//        test_tensor_assign();
//        test_tensor_slice();
//        test_tensor_slice_tensorsub();
//        test_tensor_shrink();
//        test_tensor_reshape();
//
//        test_array_swap();
//        test_seqce_swap();
//        test_tensor_swap();
//
//        test_tensor_submatrix();
//        test_tensor_f_shrink();
//        test_seqce_string();
//        test_array_sort();
//        test_sort_li();
//        test_sort_vector_li();
//        test_sort_tensor_li();
//
//        test_shuffle_sort();
//
//        test_tensor_omp();
//        math21_cuda_test();
//        math21_cuda_test_02();
//        math21_cuda_atomicAdd_test();
//        math21_numerical_recipes_test();
//        math21_opencl_test();
//        math21_opencl_test2();
//        math21_opencl_test3();
//        math21_test_tensor_string();
//
//        math21_test_tensor_log();

//        math21_test_operator_eigen_real_sys();

//        test_raw_tensor_transpose();
//        test_raw_tensor_set();
//        test_tensor_text_save();
//        test_tensor_text_read();
//        test_tensor_to_string();
        test_string_replace();
    }
}