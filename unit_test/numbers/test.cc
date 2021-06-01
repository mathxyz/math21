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

#include <math21.h>

using namespace math21;
using namespace numerical_recipes;
using namespace ad;


// x shape: n_time_step * mbs * x_size, x_size = 128 or 256, and x in x_size is one-hot.
void math21_data_text_get_one_hot(MatR &x_mat, const unsigned char *data, VecSize &_mb_offsets, NumN x_size,
                                  NumSize n_data_x, NumN mbs, NumN n_time_step) {
    x_mat.setSize(n_time_step, mbs, x_size);
    auto x = x_mat.getDataAddress();
    NumSize *mb_offsets = _mb_offsets.getDataAddress();
    NumN i, j;
    for (j = 0; j < n_time_step; ++j) {
        for (i = 0; i < mbs; ++i) {
            unsigned char curr = data[(mb_offsets[i]) % n_data_x];
            if (curr == 0 || curr >= x_size) {
                math21_error("Bad char");
            }
            x[(j * mbs + i) * x_size + curr] = 1;
            mb_offsets[i] = (mb_offsets[i] + 1) % n_data_x;
        }
    }
}

void math21_data_rnn_read_text() {
    std::string filename = math21_string_to_string(MATH21_WORKING_PATH) + "/../y/autograd-master/examples/rnn.py";
    auto text = math21_string_read_file(filename.c_str());

    NumSize n_data_1d;
    NumN rnn_batch_size = 45;
    NumN n_time_step = 5;
    NumN x_size = 128;

    n_data_1d = strlen((const char *) text);
    VecSize rnn_batch_offsets;
    math21_pr_rand_VecSize(rnn_batch_offsets, rnn_batch_size);

    MatR x_mat;
    math21_data_text_get_one_hot(x_mat, text, rnn_batch_offsets, x_size,
                                 n_data_1d, rnn_batch_size, n_time_step);

    math21_vector_free_cpu(text);

    x_mat.log();
    std::string bin_file = math21_string_to_string(MATH21_WORKING_PATH) + "/../y/z2.bin";
    math21_io_save(bin_file.c_str(), x_mat);
}

void test_std() {
    m21log(__FUNCTION__);
    std::vector<int> v;
    NumN size = 10;
    std::string s;
    math21_tool_std_string_resize(s, size);
    m21log(s.size());
    math21_tool_std_vector_resize(v, size);
    m21log(v.size());
}

void test2() {
    TenR x(10);
    x.letters();
    x.log("xxxxx");

    m21point px = math21_cast_to_point(x);
    math21_point_log(px);
    auto py = math21_test_ad_logsumexp_like(px);
    math21_point_log(py);
    exit(0);
}

void test_ad_sinusoid() {

    ad_clear_graph();

    VecR x_value;
    x_value.setSize(2, 3);
    x_value.letters(4);

    ad_point x(x_value, 1);
//    ad_point x(x_value, 1, m21_device_type_gpu);
    x.log("x", 15);

    auto y = tanh(x);
    y.log("y", 15);

    NumN n = 1;
    y = ad_jacobian_one_graph(x, y);
    if (y.id == 0) {
        m21log("f=0, so stop computing f'");
        return;
    }
//    y.log("dy", 15);

    ad_get_value(x).letters();
    ad_fv(y);
//    ad_get_data().log("data");
//    y.log("dy_fv", 15);
    VecR dy_actual(2, 3, 2, 3);
    dy_actual =
            0.419974341614026, 0, 0,
            0, 0, 0,

            0, 0.0706508248531645, 0,
            0, 0, 0,

            0, 0, 0.00986603716544019,
            0, 0, 0,


            0, 0, 0,
            0.0013409506830259, 0, 0,

            0, 0, 0,
            0, 0.000181583230943807, 0,

            0, 0, 0,
            0, 0, 2.45765474053327e-05;
    MATH21_PASS(math21_operator_isEqual(ad_get_value(y), dy_actual, MATH21_EPS));
}

void test_ad_fv() {

    ad_clear_graph();

//    VecR x_value(2, 2);
    VecR x_value(2, 3);
    x_value.letters(4);

    ad_point x(x_value, 1);
//    x.log("x", 15);

    VecN d(1);
    d = 1;
//    auto y = ad_sum(x, d, 0);
    auto y = tanh(x);
    y.log("0", 15);

    y = ad_jacobian_one_graph(x, y);
    if (y.id == 0) {
        m21log("f=0, so stop computing f'");
        return;
    }
    y.log("dy", 15);

    ad_get_value(x).setSize(2, 3);
    ad_get_value(x).letters();
    ad_fv(y);
    y.log("dy_fv", 15);
    VecR dy_actual(2, 3, 2, 3);
    dy_actual =
            0.419974341614026, 0, 0,
            0, 0, 0,

            0, 0.0706508248531645, 0,
            0, 0, 0,

            0, 0, 0.00986603716544019,
            0, 0, 0,


            0, 0, 0,
            0.0013409506830259, 0, 0,

            0, 0, 0,
            0, 0.000181583230943807, 0,

            0, 0, 0,
            0, 0, 2.45765474053327e-05;
    MATH21_PASS(math21_operator_isEqual(ad_get_value(y), dy_actual, MATH21_EPS));
}

void test_ad_tanh_01() {

    ad_clear_graph();

    VecR x_value(2, 3);
    x_value.letters();

    ad_point x(x_value, 1);
//    x.log("x", 15);

    auto y = tanh(x);
//    y.log("0", 15);
    VecR y_actual(2, 3);
    y_actual =
            0.761594155955765, 0.964027580075817, 0.995054753686731,
            0.999329299739067, 0.999909204262595, 0.999987711650796;
    MATH21_PASS(math21_operator_isEqual(ad_get_value(y), y_actual, MATH21_EPS));

    NumN n = 1;
//    NumN n = 2;
//    NumN n = 6;
//            NumN n = 20;
    for (NumN i = 1; i <= n; ++i) {
//        m21log("size", ad_get_data().size());
//        y = grad(x, y);
//        y = grad(mean, y);
//        y = grad(cov, y);
        y = ad_jacobian(x, y);
//        y = ad_jacobian(mean, y);
//        y = ad_jacobian(cov, y);
//        y = ad_hessian(cov, y);
        if (y.id == 0) {
            m21log("f=0, so stop computing f'");
            m21log("i", i);
            break;
        }
//        y.log(math21_string_to_string(i).c_str(), 15);
        VecR dy_actual(2, 3, 2, 3);

        dy_actual =
                0.419974341614026, 0, 0,
                0, 0, 0,

                0, 0.0706508248531645, 0,
                0, 0, 0,

                0, 0, 0.00986603716544019,
                0, 0, 0,


                0, 0, 0,
                0.0013409506830259, 0, 0,

                0, 0, 0,
                0, 0.000181583230943807, 0,

                0, 0, 0,
                0, 0, 2.45765474053327e-05;
        MATH21_PASS(math21_operator_isEqual(ad_get_value(y), dy_actual, MATH21_EPS));
    }
}

/*

def make_pinwheel(radial_std, tangential_std, num_classes, num_per_class, rate,
                  rs=npr.RandomState(0)):
    """Based on code by Ryan P. Adams."""
    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

    features = rs.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:, 0] += 1
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return np.einsum('ti,tij->tj', features, rotations)

 * */
void test_ad_gmm_log_likelihood() {
    ad_clear_graph();

    NumN n_component, n_feature, n_data;
    n_component = 3;
    n_feature = 2;
    n_data = 12;

    NumN params_size = n_component + n_component * n_feature + n_component * n_feature * n_feature;
    VecR params_value(params_size);

    params_value =
            0.17640523, 0.04001572, 0.0978738,
            0.22408932, 0.1867558, -0.09772779,
            0.09500884, -0.01513572, -0.01032189,
            1., 0., 0., 1., 1., 0.,
            0., 1., 1., 0., 0., 1.;

    ad_point params(params_value, 1);

    VecR data(n_data, n_feature);
    data =
            -0.39603642, -1.47717844,
            0.25644726, -1.2728885,
            -0.55655118, -1.45844877,
            0.15256218, -1.27596051,
            -0.96915379, -0.01378327,
            -1.04556437, 0.01938928,
            -1.16825597, 0.37942548,
            -1.11491919, 0.20318167,
            1.34219476, 0.54403161,
            0.71343337, 0.83036026,
            -0.03691002, 0.23347363,
            0.99859678, 0.76817688;

    auto y = ad_gmm_log_likelihood(params, data,
                                   n_component, n_feature);
//    y.log("0", 15);

    VecR y_actual(1);
    y_actual = -31.3881680176618;
    MATH21_PASS(math21_operator_isEqual(ad_get_value(y), y_actual, MATH21_EPS));

    auto dy = grad(params, y);

    ad_fv(dy);
    VecR dy_actual(params_size);
    dy_actual = -0.271368888234273, 0.0960625754099658, 0.17530631282431, -0.923683342267052, -1.17641302810979, -0.574083889327581, -1.23686179541006, -0.789096975307208, -1.17954148617747, -0.831609586337767, 1.20934805077558, 1.20934805077558, -0.737816083936304, -1.25042399189853, 0.58228959522879, 0.58228959522879, -0.718005339205992, -1.37452427750125, 0.698962326840662, 0.698962326840662, -0.696034507809694;
    MATH21_PASS(math21_operator_isEqual(ad_get_value(dy), dy_actual, MATH21_EPS));
}

void test_ad_m21point_gmm_log_likelihood() {
    ad_clear_graph();

    NumN n_component, n_feature, n_data;
    n_component = 3;
    n_feature = 2;
    n_data = 12;

    NumN params_size = n_component + n_component * n_feature + n_component * n_feature * n_feature;
    VecR params_value(params_size);

    params_value =
            0.17640523, 0.04001572, 0.0978738,
            0.22408932, 0.1867558, -0.09772779,
            0.09500884, -0.01513572, -0.01032189,
            1., 0., 0., 1., 1., 0.,
            0., 1., 1., 0., 0., 1.;

    ad_point params(params_value, 1);
    ad_point v(params_value, 1);

    ad_get_value(params).at(1) = 0;

    VecR data(n_data, n_feature);
    data =
            -0.39603642, -1.47717844,
            0.25644726, -1.2728885,
            -0.55655118, -1.45844877,
            0.15256218, -1.27596051,
            -0.96915379, -0.01378327,
            -1.04556437, 0.01938928,
            -1.16825597, 0.37942548,
            -1.11491919, 0.20318167,
            1.34219476, 0.54403161,
            0.71343337, 0.83036026,
            -0.03691002, 0.23347363,
            0.99859678, 0.76817688;

    auto y = math21_test_ad_get_f_gmm_log_likelihood_with_order(
            math21_cast_to_point(params),
            math21_cast_to_point(data),
            n_component, n_feature, 1
    );
    auto dy = math21_point_ad_grad(math21_cast_to_point(params), y);

//    dy = math21_point_ad_hessian_vector_product(math21_cast_to_point(params), y,
//                                                math21_cast_to_point(v));

    ad_get_value(params).at(1) = 0.17640523;

    math21_point_ad_fv(y);
    math21_point_ad_fv(dy);

    VecR y_actual(1);
    y_actual = -31.3881680176618;
    MATH21_PASS(math21_operator_isEqual(ad_get_value(math21_cast_to_T<ad_point>(y)), y_actual, MATH21_EPS));

    VecR dy_actual(params_size);
    dy_actual = -0.271368888234273, 0.0960625754099658, 0.17530631282431, -0.923683342267052, -1.17641302810979, -0.574083889327581, -1.23686179541006, -0.789096975307208, -1.17954148617747, -0.831609586337767, 1.20934805077558, 1.20934805077558, -0.737816083936304, -1.25042399189853, 0.58228959522879, 0.58228959522879, -0.718005339205992, -1.37452427750125, 0.698962326840662, 0.698962326840662, -0.696034507809694;
    MATH21_PASS(math21_operator_isEqual(ad_get_value(math21_cast_to_T<ad_point>(dy)), dy_actual, MATH21_EPS));

    math21_point_destroy(y);
    math21_point_destroy(dy);
}

//
// y = sum(x1 + sin(x2))
void test_ad_arithmetic_gpu_01() {
    ad_clear_graph();
    TenR _x1(3);
//    TenR _x1(3, 4, 4);
    _x1.letters();
    TenR _x2(3);
    _x2.letters(4);
//    ad_point x1(_x1, 1);
    ad_point x1(_x1, 1, m21_device_type_gpu);
    x1.log("x1");
//    ad_point x2(_x2, 1);
    ad_point x2(_x2, 1, m21_device_type_gpu);
    x2 = ad_sin(x2);
    x2.log("x2", 35);
    VecR x2_actual(3);
    x2_actual = -0.756802499294281, -0.958924293518066, -0.279415488243103;
    MATH21_PASS(math21_op_isEqual(ad_get_value(x2), x2_actual, MATH21_EPS));

    auto y = ad_add(x1, x2);
    y.log("y", 15);
    VecR y_actual(3);
    y_actual = 0.243197500705719, 1.04107570648193, 2.72058439254761;
    MATH21_PASS(math21_op_isEqual(ad_get_value(y), y_actual, MATH21_EPS));

    y = ad_sum(y);
    y.log("y", 15);

    auto dy = ad_grad(x1, y);
    dy.log("dy", 15);
    VecR dy_actual(3);
    dy_actual = 1;
    MATH21_PASS(math21_op_isEqual(ad_get_value(dy), dy_actual, MATH21_EPS));
    m21log("data size", ad_get_data().size());
//    ad_get_data().log("data");
}

void test_tensor_wrapper() {
    int i = 0;
    timer time;
    time.start();
    {
        m21log("i", i);
        TenR A;
        A.setDeviceType(m21_device_type_gpu);
        A.setSize(20000000);
        A.letters();
        TenR B;
        B.setDeviceType(m21_device_type_gpu);
        B = A;
        B.log("B");
        math21_op_vector_kx_onto(3, B);

        time.start();
        math21_op_container_sin(A, B);
        time.end();
        m21log("\ntime 1 used", time.time());

        B.log("B");
    }
}

int main(int argc, char **) {
    timer time;
    time.start();

//    test_std();
//    test_ad_sinusoid();
    test_ad_fv();
//    test_ad_tanh_01();
//    test_ad_gmm_log_likelihood();
//    test_ad_m21point_gmm_log_likelihood();
//     math21_data_rnn_read_text();
//    test_ad_arithmetic_gpu_01();
//    test_tensor_wrapper();
    math21_destroy();

    time.end();
    if (time.time() > 0) {
        m21log("\ntime used", time.time());
    }
    printf("\nmath21 unit test finish!\n");
    return 0;
}