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
#include "rnn.h"

void math21_ml_function_rnn_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                  const mlfunction_node *finput, m21list *options) {
    int output = math21_function_option_find_int(options, "output", 1);
    const char *activation_s = math21_function_option_find_str(options, "activation", "logistic");
    MATH21_FUNCTION_ACTIVATION_TYPE activation = math21_function_activation_get_type(activation_s);
    int is_use_bias = math21_function_option_find_int_quiet(options, "is_use_bias", 1);
    int is_batch_normalize = math21_function_option_find_int_quiet(options, "batch_normalize", 0);

    mlfunction_rnn *f = math21_ml_function_rnn_create(fnode, fnet->mini_batch_size, finput->y_size, output,
                                                      fnet->n_time_step_in_rnn, activation,
                                                      is_use_bias, is_batch_normalize, fnet->adam);
}

void math21_ml_function_rnn_node_saveState(const mlfunction_node *fnode, FILE *file) {
    mlfunction_rnn *f = (mlfunction_rnn *) fnode->function;
    math21_ml_function_rnn_saveState(f, file);
}

// todo: may deprecate this.
void math21_ml_function_rnn_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    mlfunction_rnn *f = (mlfunction_rnn *) fnode->function;
    math21_ml_function_rnn_set_mbs(f, mini_batch_size);
}

void math21_ml_function_rnn_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (mlfunction_rnn *) fnode->function;
    math21_ml_function_rnn_forward(f, finput, net->is_train);
}

void math21_ml_function_rnn_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    mlfunction_rnn *f = (mlfunction_rnn *) fnode->function;
    math21_ml_function_rnn_backward(f, finput, net->is_train);
}

void math21_ml_function_rnn_node_update(mlfunction_node *fnode, OptUpdate *optUpdate) {
    mlfunction_rnn *f = (mlfunction_rnn *) fnode->function;
    math21_ml_function_rnn_update(f, optUpdate);
}

void math21_ml_function_rnn_node_reset(mlfunction_node *fnode) {
    auto *f = (mlfunction_rnn *) fnode->function;
    fnode->mini_batch_size = f->steps * f->batch;
    fnode->x_dim[0] = f->inputs;
    fnode->x_dim[1] = 1;
    fnode->x_dim[2] = 1;
    fnode->y_dim[0] = f->outputs;
    fnode->y_dim[1] = 1;
    fnode->y_dim[2] = 1;
    fnode->x_size = fnode->x_dim[0] * fnode->x_dim[1] * fnode->x_dim[2];
    fnode->y_size = fnode->y_dim[0] * fnode->y_dim[1] * fnode->y_dim[2];
    fnode->y = f->output;
    fnode->dy = f->delta;
}

mlfunction_rnn *math21_ml_function_rnn_create(
        mlfunction_node *fnode, int batch_size, int input_size, int output_size,
        int n_time_step, MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias, int is_batch_normalize,
        int is_adam) {
    fprintf(stdout, "rnn layer: input_size %d, output_size %d\n", input_size, output_size);
    int rnn_batch_size = batch_size / n_time_step;
    mlfunction_rnn *f = (mlfunction_rnn *) math21_vector_calloc_cpu(1, sizeof(mlfunction_rnn));
    f->batch = rnn_batch_size;
    f->steps = n_time_step;
    f->inputs = input_size;

    // added by YE
    f->state = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
    f->prev_state = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);

    fprintf(stdout, "\t\t");

    f->input_layer = math21_ml_function_fully_connected_with_n_time_step_create(0, rnn_batch_size, n_time_step,
                                                                                input_size,
                                                                                output_size, activation,
                                                                                is_use_bias,
                                                                                is_batch_normalize,
                                                                                is_adam, "fc_input");
    fprintf(stdout, "\t\t");
    f->self_layer = math21_ml_function_fully_connected_with_n_time_step_create(0, rnn_batch_size, n_time_step,
                                                                               output_size,
                                                                               output_size, activation,
                                                                               is_use_bias,
                                                                               is_batch_normalize, is_adam, "fc_self");
    fprintf(stdout, "\t\t");
    f->output_layer = math21_ml_function_fully_connected_with_n_time_step_create(0, rnn_batch_size, n_time_step,
                                                                                 output_size,
                                                                                 output_size, activation,
                                                                                 is_use_bias,
                                                                                 is_batch_normalize,
                                                                                 is_adam, "fc_output");
    f->outputs = output_size;
    f->output = f->output_layer->output;
    f->delta = f->output_layer->delta;
    if (fnode) {
        fnode->type = mlfnode_type_rnn;
        fnode->function = f;
        fnode->saveState = math21_ml_function_rnn_node_saveState;
        fnode->set_mbs = math21_ml_function_rnn_node_set_mbs;
        fnode->forward = math21_ml_function_rnn_node_forward;
        fnode->backward = math21_ml_function_rnn_node_backward;
        fnode->update = math21_ml_function_rnn_node_update;
        math21_ml_function_rnn_node_reset(fnode);
    }
    return f;
}

// h(t) = tanh( Whh * h(t-1) + Wxh * x(t) + bh)
// y(t) = Why*h(t) + by
// y_x = W_xh * x + b_x
// y_h = W_hh * h(t-1) + b_h
// h(t) <- y_h + y_x
// y = W_hy * h(t) + b_y
// h_pre, h(1), ..., h(t), ..., h(T)
void math21_ml_function_rnn_forward(mlfunction_rnn *f, mlfunction_node *finput, int is_train) {

    int i;

    mlfunction_fully_connected input_layer = *f->input_layer;
    mlfunction_fully_connected self_layer = *f->self_layer;
    mlfunction_fully_connected output_layer = *f->output_layer;

    if (is_train) {
        math21_vector_set_wrapper(f->outputs * f->batch * f->steps, 0, f->delta, 1);
        // h_pre <- h
        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->state, 1, f->prev_state, 1);
    }

    mlfunction_node finput_fc0 = {0};
    mlfunction_node *finput_fc = &finput_fc0;
    for (i = 0; i < f->steps; ++i) {

        // y_x = W_xh * x + b_x
        finput_fc->y = finput->y + i * f->inputs * f->batch;;
        math21_ml_function_fully_connected_forward(
                &input_layer, finput_fc, is_train);


        // y_h = W_hh * h(t-1) + b_h
        finput_fc->y = f->state;
        math21_ml_function_fully_connected_forward(
                &self_layer, finput_fc, is_train);

        // h <- y_h + y_x
        // h(t) <- y_h(t) + y_x(t)
        math21_vector_set_wrapper(f->outputs * f->batch, 0, f->state, 1);
        math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, input_layer.output, 1, f->state, 1);
        math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, self_layer.output, 1, f->state, 1);


        // y = W_hy * h(t) + b_y
        finput_fc->y = f->state;
        math21_ml_function_fully_connected_forward(
                &output_layer, finput_fc, is_train);

        math21_ml_function_fully_connected_increase_by_time(&input_layer, 1);
        math21_ml_function_fully_connected_increase_by_time(&self_layer, 1);
        math21_ml_function_fully_connected_increase_by_time(&output_layer, 1);

    }

    math21_ml_function_rnn_reset(f);

}

// cpu not check
// => dL/dWhy, dL/dby, dL/dWhh, dL/dWxh, dL/dbh
void math21_ml_function_rnn_backward(mlfunction_rnn *f, mlfunction_node *finput, int is_train) {

    int i;
    mlfunction_fully_connected input_layer = *f->input_layer;
    mlfunction_fully_connected self_layer = *f->self_layer;
    mlfunction_fully_connected output_layer = *f->output_layer;

    math21_ml_function_fully_connected_increase_by_time(&input_layer, f->steps - 1);
    math21_ml_function_fully_connected_increase_by_time(&self_layer, f->steps - 1);
    math21_ml_function_fully_connected_increase_by_time(&output_layer, f->steps - 1);

    PointerFloatWrapper last_input = input_layer.output;
    PointerFloatWrapper last_self = self_layer.output;
    mlfunction_node finput_fc0 = {0};
    mlfunction_node *finput_fc = &finput_fc0;
    for (i = f->steps - 1; i >= 0; --i) {
        // todo: check and remove the following lines because f->state is computed already elsewhere.
        // checked once
        // h = y_h + y_x
        // h(t) = y_h(t) + y_x(t)
        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, input_layer.output, 1, f->state, 1);
        math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, self_layer.output, 1, f->state, 1);

        finput_fc->y = f->state;
        finput_fc->dy = self_layer.delta; // dL/dh = dL/dy_h
        // dL/dy => dL/dh(t) => dL/dy_h
        // dL/dh(t) = (dL/dy)*(dy/dh(t)) + (dL/dh(t) at t+1)
        math21_ml_function_fully_connected_backward(&output_layer, finput_fc, is_train);

        if (i != 0) {
            // h = y_h + y_x
            // h(t-1) = y_h(t-1) + y_x(t-1)
            math21_vector_set_wrapper(f->outputs * f->batch, 0, f->state, 1);
            math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, input_layer.output - f->outputs * f->batch, 1,
                                           f->state, 1);
            math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, self_layer.output - f->outputs * f->batch, 1,
                                           f->state, 1);
        } else {
            // h <- h(0) <- h_pre
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->prev_state, 1, f->state, 1);
        }

        // dL/dh = dL/dy_h = dL/dy_x => dL/dy_x
        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, self_layer.delta, 1, input_layer.delta,
                                                 1);

        // y_h = W_hh * h(t-1) + b_h
        // dL/y_h => (dL/dh(t-1) at t)
        finput_fc->y = f->state;
        finput_fc->dy = (i > 0) ? self_layer.delta - f->outputs * f->batch : math21_vector_getEmpty_R32_wrapper();
        math21_ml_function_fully_connected_backward(&self_layer, finput_fc, is_train);

        // y_x = W_xh * x + b_x
        // dL/y_x => dL/dx(t)
        finput_fc->y = finput->y + i * f->inputs * f->batch;
        if (!math21_vector_isEmpty_wrapper(finput->dy)) finput_fc->dy = finput->dy + i * f->inputs * f->batch;
        else finput_fc->dy = math21_vector_getEmpty_R32_wrapper();
        math21_ml_function_fully_connected_backward(&input_layer, finput_fc, is_train);

        math21_ml_function_fully_connected_increase_by_time(&input_layer, -1);
        math21_ml_function_fully_connected_increase_by_time(&self_layer, -1);
        math21_ml_function_fully_connected_increase_by_time(&output_layer, -1);
    }

    // restore h for next forward or just restore it.
    // h <- h(T) = y_h(T) + y_x(T)
    math21_vector_set_wrapper(f->outputs * f->batch, 0, f->state, 1);
    math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, last_input, 1, f->state, 1);
    math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, last_self, 1, f->state, 1);

    math21_ml_function_rnn_reset(f);
}

void math21_ml_function_rnn_update(mlfunction_rnn *f, OptUpdate *optUpdate) {
    math21_ml_function_fully_connected_update(f->input_layer, optUpdate);
    math21_ml_function_fully_connected_update(f->self_layer, optUpdate);
    math21_ml_function_fully_connected_update(f->output_layer, optUpdate);
}

void math21_ml_function_rnn_saveState(const mlfunction_rnn *f, FILE *file) {
    math21_vector_serialize_c_wrapper(file, f->state, f->batch * f->outputs);
//    math21_ml_function_fully_connected_saveState(f->input_layer, file);
//    math21_ml_function_fully_connected_saveState(f->self_layer, file);
//    math21_ml_function_fully_connected_saveState(f->output_layer, file);
}

void math21_ml_function_rnn_reset(mlfunction_rnn *f) {
    math21_ml_function_fully_connected_reset(f->input_layer);
    math21_ml_function_fully_connected_reset(f->self_layer);
    math21_ml_function_fully_connected_reset(f->output_layer);
}

void math21_ml_function_rnn_set_mbs(mlfunction_rnn *f, int mini_batch_size) {
    f->batch = mini_batch_size;
    math21_ml_function_fully_connected_set_mbs(f->input_layer, mini_batch_size);
    math21_ml_function_fully_connected_set_mbs(f->self_layer, mini_batch_size);
    math21_ml_function_fully_connected_set_mbs(f->output_layer, mini_batch_size);
}
