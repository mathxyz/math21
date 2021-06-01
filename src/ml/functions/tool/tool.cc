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

#include "../files_c.h"
#include "tool_c.h"
#include "tool_cc.h"
#include "inner_cc.h"

using namespace math21;

void math21_ml_function_debug_node_save_state(const mlfunction_node *fnode) {
    if (math21_time_is_debug()) {
        int startTime = math21_time_index_get_start_time();
        int endTime = math21_time_index_get_end_time();
        if (math21_time_index_get() > endTime) {
            exit(0);
        }
        if (math21_time_index_get() >= startTime) {
            char buffer[1024] = {0};
            sprintf(buffer, "%s/a/k_%05d.bin", MATH21_WORKING_PATH, math21_time_index_get());
            const char *path = buffer;

            FILE *file = fopen(path, "wb");
            if (!file) {
                math21_file_error(path);
                return;
            }

            if (fnode->saveState) {
                fnode->saveState(fnode, file);
            }
            fclose(file);
        }
    }
}

//void math21_ml_function_debug_function_save_state(data *d) {
void math21_ml_function_debug_function_save_state(void *f0, void* f2) {
    auto *f = (mlfunction_lstm *)f0;
//    auto *f = (mlfunction_fully_connected *)f0;
//    auto *f = (mlfunction_node *)f0;
//    auto *f = (float *)f0;
//    NumN n = *(NumN *)f2;
    int test = 0;
//    if (test || math21_time_index_get_debug_time() == math21_time_index_get()
    if (test || math21_time_index_get_debug_time() < math21_time_index_get()
        || math21_time_is_debug()) {

        int startTime = math21_time_index_get_start_time();
        int endTime = math21_time_index_get_end_time();
        if (math21_time_index_get() > endTime) {
            m21exit(0);
        }
        if (test || math21_time_index_get() >= startTime) {
            char buffer[1024] = {0};
            fprintf(stderr, "math21_time_index_get() = %d\n", math21_time_index_get());
            sprintf(buffer, "%s/%s", MATH21_WORKING_PATH, "a/k_4.bin");

            const char *path = buffer;

            FILE *file = fopen(path, "wb");
            if (!file) {
                math21_file_error(path);
                return;
            }

//            math21_vector_serialize_c_wrapper(file, f, n);
//            math21_vector_serialize_c_wrapper(file, f, n);
//            math21_vector_serialize_c_wrapper(file, f->weights, f->nweights);
//            math21_vector_serialize_c_wrapper(file, f->output, f->batch * f->outputs);
//            math21_ml_function_rnn_saveState(f, file);
            math21_ml_function_lstm_saveState(f, file);

//            math21_ml_function_fully_connected_saveState(f, file);
//            math21_ml_net_data_save(d, file);
//            math21_vector_serialize_c_cpu(file, net->input, net->batch*net->inputs);
//            math21_vector_serialize_c_wrapper(file, finput->y, finput->mini_batch_size*finput->y_size);
//            math21_vector_serialize_c_wrapper(file, f->output, f->batch * f->outputs);
//            math21_vector_serialize_c_wrapper(file, f->delta, f->batch * f->outputs);
//            math21_vector_serialize_c_wrapper(file, f->weights, f->nweights);
//            math21_vector_serialize_c_wrapper(file, f->weight_updates, f->nweights);
//            if (f->batch_normalize) {
//                math21_ml_function_batchnorm_saveState(f->bn, file);
//            } else{
//                math21_vector_serialize_c_wrapper(file, f->biases, f->out_c);
//                math21_vector_serialize_c_wrapper(file, f->bias_updates, f->out_c);
//            }


            fclose(file);
            m21log("exit from ", __FUNCTION__);
            m21exit(0);
        }
    }
}

NumB _ml_function_tool_is_debug = 0;
//NumB _ml_function_tool_is_debug = 1;
NumB math21_ml_function_tool_is_debug(){
//    m21log(__FUNCTION__, _ml_function_tool_is_debug);
    return _ml_function_tool_is_debug;
}
NumB math21_ml_function_tool_varName_check(const char *name, const char *varName, std::string& varNameNew){
    if (!varName) {
        return 0;
    }
    std::string name_op, name_var;
    math21_string_split_by_first_separator(varName, name_op, name_var);
    if (name &&
        !math21_string_is_equal(name_op, name) &&
        !math21_string_is_equal(name_op, "*")) {
        return 0;
    }
    varNameNew = name_var;
    return 1;
}
