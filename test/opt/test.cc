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
//    using namespace opt;

    void test_steepest_decent() {

//    sine f;
//        polynomial f;
        f_example_2 f;
        OptimizationInterface_dummy oi;
        sd_update_rule_normal update_rule(f);
        SteepestDescent opt(update_rule, oi);
        opt.solve();
        ((VecR &) opt.getMinimum()).log("minima");
    }

    void test_ConjugateGradient() {
//        sine f;
//    polynomial f;
        f_example_2 f;
        ConjugateGradient opt(f, f.getX0());
        opt.solve();
    }

    void test_cnn() {

        ////////////////// data
        Seqce<TenR> X;
        X.setSize(8);
        Seqce<TenR> Y;
        Y.setSize(8);
        VecN d1;
        d1.setSize(3);
        VecN d2;
        d2.setSize(3);
        d1 = 1, 1, 3;
        d2 = 4, 1, 1;
        for (NumN i = 1; i <= X.size(); i++) {
            X(i).setSize(d1);
            Y(i).setSize(d2);
        }
        X(1) = 0, -1, 1;
        X(2) = 1, 0, 1;
        X(3) = 0, 1, 1;
        X(4) = -1, 0, 1;
        X(5) = 0, -1, -1;
        X(6) = 1, 0, -1;
        X(7) = 0, 1, -1;
        X(8) = -1, 0, -1;
        Y(1) = 1, 0, 0, 0;
        Y(2) = 1, 0, 0, 0;
        Y(3) = 0, 1, 0, 0;
        Y(4) = 0, 1, 0, 0;
        Y(5) = 0, 0, 1, 0;
        Y(6) = 0, 0, 1, 0;
        Y(7) = 0, 0, 0, 1;
        Y(8) = 0, 0, 0, 1;

//        X.log("X");
//        Y.log("Y");

        //////////////////
        cnn f;
        VecR theta;
        //////////////////
        const char *model_file_name = "model_cnn_c.bin";
        std::ifstream in;
//    in.open(model_file_name, std::ifstream::binary);
        if (in.is_open()) {
            m21log("deserialize cnn");
            math21_deserialize_model(in, f, theta);
            in.close();
        }

        ////////////////// create cnn
        if (f.isEmpty()) {
            m21log("create cnn");
            Seqce<cnn_config_fn *> config_fns;
            config_fns.setSize(2);
            VecN d;
            d.setSize(3);
//        d = 8, 1, 3;
//        d = 2, 2, 2;
            d = 1, 1, 1;
            config_fns(1) = new cnn_config_fn_fully(d, cnn_type_hn_ReLU);
            d.assign(d2);
            config_fns(2) = new cnn_config_fn_fully(d, cnn_type_hn_linear);
//    d = 1;
//    config_fns(4) = new cnn_config_fn_fully(d, cnn_type_hn_tanh);
//        d = 2;
            NumB isUsingDiff = 1;
            f.setSize(d1, config_fns, isUsingDiff);

            for (NumN i = 1; i <= config_fns.size(); i++) {
                delete config_fns(i);
            }
        }
        ////////////////////

        CostFunctional_nll_CrossEntroy_softmax_class L;
        cnn_cost_class J(f, L, X, Y, 5, 2);
//    J.getParas().lambda = 0.2;

        OptimizationInterface_cnn oi;
        oi.setName(model_file_name);
//        sd_update_rule_normal update_rule(J);
//    update_rule.tao = 80000;
//    sd_update_rule_momentum update_rule(J);
//    sd_update_rule_Nesterove_momentum update_rule(J);
//    sd_update_rule_AdaGrad update_rule(J);
//    sd_update_rule_RMSProp update_rule(J);
        sd_update_rule_RMSProp_Nesterov_momentum update_rule(J);
//        sd_update_rule_Adam update_rule(J);
        if (!theta.isEmpty()) {
            update_rule.setInit(theta);
        }
        SteepestDescent opt(update_rule, oi);
        update_rule.time_max = 1000;
//    update_rule.time_max = 50;
//    update_rule.x = a.getTheta();
        opt.solve();
        opt.getMinimum().log("minima");

        f.setTheta(opt.getMinimum());
        evaluate_cnn(f, J, X, Y, 1);
        evaluate_cnn_error_rate(f, J, X, Y, 1);

    }

    void test_lstm_text() {
        m21lstm_text_type_config config;
//        config.predict = 1;
        config.predict = 0;
        config.deviceType = m21_device_type_default;
//        config.deviceType = m21_device_type_gpu;
//        config.printTextInOpt = 1;
        config.debug = 1;
        config.max_batches = 60;
        config.num_iters = 30;
        const char *name = "./lstm_text.bin";
        config.functionParasPath = name;
        config.functionParasPath_save = name;
        config.n_lines = 1;
        math21_ml_lstm_text(config);
    }

    void test_opt() {
        test_lstm_text();
        math21_ml_lstm_sin01(m21lstm_sin01_type_config());

//        test_steepest_decent();
//        test_ConjugateGradient();

//        test_cnn();
    }
}