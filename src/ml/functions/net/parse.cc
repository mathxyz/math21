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
#include "inner_cc.h"
#include "net.h"
#include "parse.h"
#include "detail.h"

using namespace math21;

NumB math21_ml_is_function_paras_file(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) math21_file_error(filename);

    NumR64 lib_name = '0';
    fread(&lib_name, sizeof(NumR64), 1, fp);
    NumR64 lib_name_digit = 0;
    fread(&lib_name_digit, sizeof(NumR64), 1, fp);
    fclose(fp);
    if ((char) lib_name != 'm' || (int) lib_name_digit != 21) {
        return 0;
    }
    return 1;
}

void math21_ml_function_net_parse_options(mlfunction_net *fnet, m21list *options) {
    const char *name = math21_function_option_find_str_quiet(options, "name", "net");
    fnet->name = math21_string_create_from_string(name);
    NumN logLevel = math21_function_option_find_NumN_quiet(options, "logLevel", 0);
    math21_ml_function_net_setlogLevel(logLevel);
    fnet->mini_batch_size_in_opt = math21_function_option_find_int(options, "batch", 1);
    fnet->n_time_step_in_rnn = math21_function_option_find_int_quiet(options, "time_steps", 1);
    fnet->mini_batch_size_in_opt *= fnet->n_time_step_in_rnn;

    int subdivs = math21_function_option_find_int(options, "subdivisions", 1);
    math21_tool_assert(fnet->mini_batch_size_in_opt % subdivs == 0);
    fnet->mini_batch_size = fnet->mini_batch_size_in_opt / subdivs;

    // pre-processing
    fnet->saturation = math21_function_option_find_float_quiet(options, "saturation", 1);
    fnet->exposure = math21_function_option_find_float_quiet(options, "exposure", 1);
    fnet->hue = math21_function_option_find_float_quiet(options, "hue", 0);

    fnet->alpha = math21_function_option_find_float(options, "learning_rate", .001);
    fnet->momentum = math21_function_option_find_float(options, "momentum", .9);
    fnet->decay = math21_function_option_find_float(options, "decay", .0001);

    fnet->adam = math21_function_option_find_int_quiet(options, "adam", 0);
    if (fnet->adam) {
        fnet->B1 = math21_function_option_find_float(options, "B1", .9);
        fnet->B2 = math21_function_option_find_float(options, "B2", .999);
        fnet->eps = math21_function_option_find_float(options, "eps", .0000001);
    }

    fnet->data_x_dim[0] = math21_function_option_find_int_quiet(options, "channels", 0);
    fnet->data_x_dim[1] = math21_function_option_find_int_quiet(options, "height", 0);
    fnet->data_x_dim[2] = math21_function_option_find_int_quiet(options, "width", 0);

    fnet->data_x_size = math21_function_option_find_int_quiet(options, "inputs",
                                                              fnet->data_x_dim[0] * fnet->data_x_dim[1] *
                                                              fnet->data_x_dim[2]);

    float clip = math21_function_option_find_float_quiet(options, "clip", 0);
    math21_tool_assert(clip == 0 && "clip set to net and passed to layer");

    if (!fnet->data_x_size && !(fnet->data_x_dim[0] && fnet->data_x_dim[1] && fnet->data_x_dim[2]))
        math21_error("No input parameters supplied");

    const char *policy_s = math21_function_option_find_str(options, "policy", "constant");
    fnet->alphaPolicy = math21_opt_learning_rate_policy_get_from_name(policy_s);
    fnet->burn_in = math21_function_option_find_int_quiet(options, "burn_in", 0);
    fnet->power = math21_function_option_find_float_quiet(options, "power", 4);
    if (fnet->alphaPolicy == OptAlphaPolicy_STEP) {
        fnet->step = math21_function_option_find_int(options, "step", 1);
        fnet->scale = math21_function_option_find_float(options, "scale", 1);
    } else if (fnet->alphaPolicy == OptAlphaPolicy_STEPS) {
        char *l = math21_function_option_find(options, "steps");
        char *p = math21_function_option_find(options, "scales");
        if (!l || !p) math21_error("STEPS policy must have steps and scales in cfg file");

        int len = strlen(l);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (l[i] == ',') ++n;
        }
        int *steps = (int *) math21_vector_calloc_cpu(n, sizeof(int));
        float *scales = (float *) math21_vector_calloc_cpu(n, sizeof(float));
        for (i = 0; i < n; ++i) {
            int step = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',') + 1;
            p = strchr(p, ',') + 1;
            steps[i] = step;
            scales[i] = scale;
        }
        fnet->scales = scales;
        fnet->steps = steps;
        fnet->num_steps = n;
    } else if (fnet->alphaPolicy == OptAlphaPolicy_EXP) {
        fnet->gamma = math21_function_option_find_float(options, "gamma", 1);
    } else if (fnet->alphaPolicy == OptAlphaPolicy_SIG) {
        fnet->gamma = math21_function_option_find_float(options, "gamma", 1);
        fnet->step = math21_function_option_find_int(options, "step", 1);
    } else if (fnet->alphaPolicy == OptAlphaPolicy_POLY || fnet->alphaPolicy == OptAlphaPolicy_RANDOM) {
    }
    fnet->n_mini_batch_max_in_opt = math21_function_option_find_int(options, "max_batches", 0);
}

void
math21_ml_function_net_save_function_paras_upto(mlfunction_net *fnet, const char *filename, int index_node_cutoff) {
#ifndef MATH21_FLAG_USE_CPU
    if (fnet->gpuDevice >= 0) {
        math21_gpu_set_device_wrapper(fnet->gpuDevice);
    }
#endif
    fprintf(stdout, "Saving function paras to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    if (!fp) math21_file_error(filename);

    NumR64 x = 'm';
    fwrite(&x, sizeof(NumR64), 1, fp);
    x = 21;
    fwrite(&x, sizeof(NumR64), 1, fp);

    NumN major = 2;
    NumN minor = 1;
    NumN revision = 8;
    fwrite(&major, sizeof(NumN), 1, fp);
    fwrite(&minor, sizeof(NumN), 1, fp);
    fwrite(&revision, sizeof(NumN), 1, fp);
    fwrite(&fnet->n_seen, sizeof(NumSize), 1, fp);

    int i;
    for (i = 0; i < fnet->n_node && i < index_node_cutoff; ++i) {
        mlfunction_node *fnode = fnet->nodes[i];
        if (fnode->dontsave) continue;
        if (fnode->type == mlfnode_type_conv) {
            auto *f = (mlfunction_conv *) fnode->function;
            math21_ml_function_conv_save_theta(f, fp);
        } else if (fnode->type == mlfnode_type_fully_connected) {
            auto *f = (mlfunction_fully_connected *) fnode->function;
            math21_ml_function_fully_connected_save_theta(f, fp);
        } else if (fnode->type == mlfnode_type_rnn) {
            auto *f = (mlfunction_rnn *) fnode->function;
            math21_ml_function_fully_connected_save_theta(f->input_layer, fp);
            math21_ml_function_fully_connected_save_theta(f->self_layer, fp);
            math21_ml_function_fully_connected_save_theta(f->output_layer, fp);
        } else if (fnode->type == mlfnode_type_lstm) {
            auto *f = (mlfunction_lstm *) fnode->function;
            if (f->implementationMode == 1) {
                math21_ml_function_fully_connected_save_theta(f->fcWi, fp);
                math21_ml_function_fully_connected_save_theta(f->fcWf, fp);
                math21_ml_function_fully_connected_save_theta(f->fcWo, fp);
                math21_ml_function_fully_connected_save_theta(f->fcWg, fp);
                math21_ml_function_fully_connected_save_theta(f->fcUi, fp);
                math21_ml_function_fully_connected_save_theta(f->fcUf, fp);
                math21_ml_function_fully_connected_save_theta(f->fcUo, fp);
                math21_ml_function_fully_connected_save_theta(f->fcUg, fp);
            } else if (f->implementationMode == 2) {
                math21_ml_function_fully_connected_save_theta(f->fcWx, fp);
                math21_ml_function_fully_connected_save_theta(f->fcUh, fp);
            } else {
                math21_ml_function_fully_connected_save_theta(f->fcW, fp);
            }
        } else if (fnode->type == mlfnode_type_batchnorm) {
            math21_tool_assert(0 && "not used!");
            auto *f = (mlfunction_batchnorm *) fnode->function;
            math21_ml_function_batchnorm_save_theta(f, fp, 1);
        }
    }
    fclose(fp);
}

// save theta of the function f(x, theta)
void math21_ml_function_net_save_function_paras(mlfunction_net *fnet, const char *filename) {
    math21_ml_function_net_save_function_paras_upto(fnet, filename, fnet->n_node);
}

void math21_ml_function_net_load_function_paras_from_config_upto(mlfunction_net *fnet, const char *filename,
                                                                 int index_node_start,
                                                                 int index_node_cutoff) {
#ifndef MATH21_FLAG_USE_CPU
    if (fnet->gpuDevice >= 0) {
        math21_gpu_set_device_wrapper(fnet->gpuDevice);
    }
#endif
    fprintf(stdout, "Loading function paras from %s\n", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if (!fp) math21_file_error(filename);

    NumR64 lib_name = '0';
    fread(&lib_name, sizeof(NumR64), 1, fp);
    NumR64 lib_name_digit = 0;
    fread(&lib_name_digit, sizeof(NumR64), 1, fp);

    NumN major;
    NumN minor;
    NumN revision;
    fread(&major, sizeof(NumN), 1, fp);
    fread(&minor, sizeof(NumN), 1, fp);
    fread(&revision, sizeof(NumN), 1, fp);
    if (major >= 2) {
        NumSize n_seen;
        fread(&n_seen, sizeof(NumSize), 1, fp);
        fnet->n_seen = n_seen;
    } else {
        NumN n_seen = 0;
        fread(&n_seen, sizeof(NumN), 1, fp);
        fnet->n_seen = n_seen;
    }


    if ((char) lib_name != 'm' || (int) lib_name_digit != 21) {
        fprintf(stderr, "not math21 file!\n");
        exit(0);
    }
#if 1
    math21_log_char("lib_name", (char) lib_name);
    m21log("lib_name_digit", (int) lib_name_digit);
    m21log("major", major);
    m21log("minor", minor);
    m21log("revision", revision);
    m21log("fnet->n_seen", fnet->n_seen);
#endif

    int i;
    for (i = index_node_start; i < fnet->n_node && i < index_node_cutoff; ++i) {
        mlfunction_node *fnode = fnet->nodes[i];
        if (fnode->dontload) continue;
        if (fnode->type == mlfnode_type_conv) {
            auto *f = (mlfunction_conv *) fnode->function;
            math21_ml_function_conv_load_theta(f, fp);
        } else if (fnode->type == mlfnode_type_fully_connected) {
            auto *f = (mlfunction_fully_connected *) fnode->function;
            math21_ml_function_fully_connected_load_theta(f, fp);
        } else if (fnode->type == mlfnode_type_rnn) {
            auto *f = (mlfunction_rnn *) fnode->function;
            math21_ml_function_fully_connected_load_theta(f->input_layer, fp);
            math21_ml_function_fully_connected_load_theta(f->self_layer, fp);
            math21_ml_function_fully_connected_load_theta(f->output_layer, fp);
        } else if (fnode->type == mlfnode_type_lstm) {
            auto *f = (mlfunction_lstm *) fnode->function;
            if (f->implementationMode == 1) {
                math21_ml_function_fully_connected_load_theta(f->fcWi, fp);
                math21_ml_function_fully_connected_load_theta(f->fcWf, fp);
                math21_ml_function_fully_connected_load_theta(f->fcWo, fp);
                math21_ml_function_fully_connected_load_theta(f->fcWg, fp);
                math21_ml_function_fully_connected_load_theta(f->fcUi, fp);
                math21_ml_function_fully_connected_load_theta(f->fcUf, fp);
                math21_ml_function_fully_connected_load_theta(f->fcUo, fp);
                math21_ml_function_fully_connected_load_theta(f->fcUg, fp);
            } else if (f->implementationMode == 2) {
                math21_ml_function_fully_connected_load_theta(f->fcWx, fp);
                math21_ml_function_fully_connected_load_theta(f->fcUh, fp);
            } else {
                math21_ml_function_fully_connected_load_theta(f->fcW, fp);
            }
        } else if (fnode->type == mlfnode_type_batchnorm) {
            math21_tool_assert(0);
            auto *f = (mlfunction_batchnorm *) fnode->function;
            math21_ml_function_batchnorm_load_theta(f, fp, 1);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void math21_ml_function_net_load_function_paras_from_config(mlfunction_net *fnet, const char *filename) {
    math21_ml_function_net_load_function_paras_from_config_upto(fnet, filename, 0, fnet->n_node);
}

int math21_ml_function_net_config_is_net_name(m21section *s) {
    return (strcmp(s->type, "[net]") == 0
            || strcmp(s->type, "[network]") == 0);
}

mlfunction_net *math21_ml_function_net_load_function_form_from_config(const char *filename) {
    m21list *sections = math21_function_option_cfg_read(filename);
    m21node *current_node = sections->first;
    if (!current_node) math21_error("Config file has no sections");
    mlfunction_net *fnet = math21_ml_function_net_create(sections->size - 1);
    auto *f_detail = (mlfunction_net_detail *) fnet->detail;
    mlfunction_node **fnodes = fnet->nodes;
#if defined(MATH21_FLAG_USE_CPU)
    printf("loading net from cpu\n");
#else
    fnet->gpuDevice = math21_gpu_get_global_variable_wrapper();
    printf("loading net from device %d\n", fnet->gpuDevice);
#endif

    m21section *current_section = (m21section *) current_node->val;
    m21list *function_options = current_section->options;
    if (!math21_ml_function_net_config_is_net_name(current_section))
        math21_error("First section must be [net] or [network]");

    math21_ml_function_net_parse_options(fnet, function_options);

    mlfunction_node finput0 = {0};
    mlfunction_node *finput = &finput0;

    finput->mini_batch_size = fnet->mini_batch_size;
    finput->y_dim[0] = fnet->data_x_dim[0];
    finput->y_dim[1] = fnet->data_x_dim[1];
    finput->y_dim[2] = fnet->data_x_dim[2];
//    finput->y_size = finput->y_dim[0] * finput->y_dim[1] * finput->y_dim[2];
    finput->y_size = fnet->data_x_size;
//    math21_tool_assert(finput->y_size == fnet->data_x_size);
//    finput->y = f->output;
//    finput->dy = f->delta;


    size_t workspace_size = 0;
    current_node = current_node->next;
    math21_function_option_free_section(current_section);
    if (math21_ml_function_net_getlogLevel()) {
        fprintf(stdout, "%s architecture\n", fnet->name);
    }
    fprintf(stdout, "fnode.                weightShape              input                output\n");
    int index = 1;
    while (current_node) {
        current_section = (m21section *) current_node->val;
        function_options = current_section->options;
        mlfunction_node *fnode = fnodes[index - 1];
        mlfnode_type fnodeType = math21_ml_function_node_type_string_to_type(current_section->type);
        if (fnodeType == mlfnode_type_conv) {
            math21_ml_function_conv_parse(fnode, fnet, finput, function_options);
        } else if (fnodeType == mlfnode_type_sample) {
            math21_ml_function_sample_parse(fnode, fnet, finput, function_options);
        } else if (fnodeType == mlfnode_type_res) {
            math21_ml_function_res_parse(fnode, fnet, finput, function_options);
        } else if (fnodeType == mlfnode_type_route) {
            math21_ml_function_route_parse(fnode, fnet, finput, function_options);
        } else if (fnodeType == mlfnode_type_yolo) {
            math21_ml_function_yolo_parse(fnode, fnet, finput, function_options);
        } else if (fnodeType == mlfnode_type_max_pooling) {
            math21_ml_function_max_pooling_parse(fnode, fnet, finput, function_options);
        } else if (fnodeType == mlfnode_type_average_pooling) {
            math21_ml_function_average_pooling_parse(fnode, fnet, finput, function_options);
        } else if (fnodeType == mlfnode_type_softmax) {
            math21_ml_function_softmax_parse(fnode, fnet, finput, function_options);
            auto *f = (mlfunction_softmax *) fnode->function;
            math21_tool_assert(f->softmax_tree == 0);
        } else if (fnodeType == mlfnode_type_fully_connected) {
            math21_ml_function_fully_connected_parse(fnode, fnet, finput, function_options);
        } else if (fnodeType == mlfnode_type_dropout) {
            math21_ml_function_dropout_parse(fnode, fnet, finput, function_options);
        } else if (fnodeType == mlfnode_type_rnn) {
            math21_ml_function_rnn_parse(fnode, fnet, finput, function_options);
        } else if (fnodeType == mlfnode_type_batchnorm) {
            assert(0 && "");
            math21_ml_function_batchnorm_parse(fnode, fnet, finput, function_options);
        } else if (fnodeType == mlfnode_type_lstm) {
            math21_ml_function_lstm_parse(fnode, fnet, finput, function_options);
        } else if (fnodeType == mlfnode_type_cost) {
            math21_ml_function_cost_parse(fnode, fnet, finput, function_options);
        } else {
            fprintf(stderr, "Type not recognized: %s\n", current_section->type);
        }

        fnode->stopbackward = math21_function_option_find_int_quiet(function_options, "stopbackward", 0);
        fnode->dontsave = math21_function_option_find_int_quiet(function_options, "dontsave", 0);
        fnode->dontload = math21_function_option_find_int_quiet(function_options, "dontload", 0);
        math21_function_option_log_unused(function_options);

        if (fnode->getGlobalSpaceSize) {
            if (fnode->getGlobalSpaceSize(fnode) > workspace_size) workspace_size = fnode->getGlobalSpaceSize(fnode);
        }

        math21_function_option_free_section(current_section);
        current_node = current_node->next;
        if (math21_ml_function_net_getlogLevel()) {
            fprintf(stdout, "%5d ", index);
            if (fnode->log) {
                fnode->log(fnode, "*/summary");
            }
        }
        ++index;
        if (current_node) {
            finput = fnode;
        }
    }
    math21_data_structure_list_free(sections);

    // used by rnn
    mlfunction_node *fnode_out = math21_ml_function_net_get_output_node(fnet);
    fnet->y_size = fnode_out->y_size;
    fnet->data_y_size = fnode_out->y_size;
    fnet->mbs_y = fnode_out->mini_batch_size;

    mlfunction_node *fnode = fnet->nodes[fnet->n_node - 1];
    if (fnode->type == mlfnode_type_yolo) {
        auto *fyolo = (mlfunction_yolo *) fnode->function;
        if (fyolo->truths) {
            fnet->data_y_size = fyolo->truths;
        }
    }
    fnet->y_wrapper = fnode_out->y;
#ifdef MATH21_FLAG_USE_CPU
#else
    fnet->y_cpu = math21_vector_create_with_default_value_cpu(fnet->y_size * fnet->mbs_y, 0);
#endif

    fnet->data_x_wrapper = math21_vector_create_with_default_value_wrapper(fnet->data_x_size * fnet->mini_batch_size,
                                                                           0);
    fnet->data_y_wrapper = math21_vector_create_with_default_value_wrapper(fnet->data_y_size * fnet->mbs_y,
                                                                           0);
    if (fnet->n_time_step_in_rnn == 1) {
        f_detail->data_x_wrapper.setWrapper(fnet->data_x_wrapper, fnet->mini_batch_size / fnet->n_time_step_in_rnn,
                                            fnet->data_x_size);
    } else {
        f_detail->data_x_wrapper.setWrapper(fnet->data_x_wrapper, fnet->n_time_step_in_rnn,
                                            fnet->mini_batch_size / fnet->n_time_step_in_rnn, fnet->data_x_size);
    }
    f_detail->data_y_wrapper.setWrapper(fnet->data_y_wrapper, fnet->mbs_y, fnet->data_y_size);
#ifdef MATH21_FLAG_USE_CPU
#else
    fnet->data_x_cpu = math21_vector_create_with_default_value_cpu(fnet->data_x_size * fnet->mini_batch_size, 0);
    fnet->data_y_cpu = math21_vector_create_with_default_value_cpu(fnet->data_y_size * fnet->mbs_y, 0);

#endif
    if (workspace_size) {
        fnet->workspace = math21_vector_create_with_default_value_wrapper((workspace_size - 1) / sizeof(float) + 1, 0);
    }
    return fnet;
}

mlfunction_net *
math21_ml_function_net_create_from_file(const char *function_form, const char *function_paras, int isClear) {
    mlfunction_net *fnet = math21_ml_function_net_load_function_form_from_config(function_form);
    if (function_paras && function_paras[0] != 0) {
        math21_ml_function_net_load_function_paras_from_config(fnet, function_paras);
    }
    if (isClear) fnet->n_seen = 0;
    return fnet;
}

// todo: use abstract
void math21_ml_function_net_reset_rnn_state_when_gpu(mlfunction_net *fnet, int b) {
    int i;
    for (i = 0; i < fnet->n_node; ++i) {
#ifndef MATH21_FLAG_USE_CPU
        mlfunction_node *fnode = fnet->nodes[i];
        if (fnode->type == mlfnode_type_rnn) {
            auto f = (mlfunction_rnn *) fnode->function;
            math21_vector_set_wrapper(f->outputs, 0, f->state + f->outputs * b, 1);
        } else if (fnode->type == mlfnode_type_gru) {
            math21_tool_assert(0);
        } else if (fnode->type == mlfnode_type_lstm) {
            auto f = (mlfunction_lstm *) fnode->function;
            math21_vector_set_wrapper(f->outputs, 0, f->h + f->outputs * b, 1);
        }
#endif
    }
}

int math21_ml_function_net_resize(mlfunction_net *fnet, int w, int h) {
#if !defined(MATH21_FLAG_USE_CPU)
    math21_gpu_set_device_wrapper(fnet->gpuDevice);
#endif

    int i;
    //if(w == net->w && h == net->h) return 0;

    fnet->data_x_dim[1] = h;
    fnet->data_x_dim[2] = w;
//    mlfunction_node finput0={0};
//    mlfunction_node* finput=&finput0;
//    math21_vector_assign_from_vector_int_cpu(3, fnet->data_x_dim, 1, finput->y_dim, 1);
    int inputs = 0;
    size_t workspace_size = 0;
//    fprintf(stderr, "Resizing to %d x %d...\n", w, h);
    //fflush(stderr);
    for (i = 0; i < fnet->n_node; ++i) {
        mlfunction_node *fnode = fnet->nodes[i];
        if (fnode->type == mlfnode_type_conv) {
            mlfunction_conv *f = (mlfunction_conv *) fnode->function;
            math21_ml_function_conv_resize(fnode, f, w, h);
        } else if (fnode->type == mlfnode_type_sample) {
            mlfunction_sample *f = (mlfunction_sample *) fnode->function;
            math21_ml_function_sample_resize(fnode, f, w, h);
        } else if (fnode->type == mlfnode_type_res) {
            mlfunction_res *f = (mlfunction_res *) fnode->function;
            math21_ml_function_res_resize(fnode, f, w, h);
        } else if (fnode->type == mlfnode_type_route) {
            mlfunction_route *f = (mlfunction_route *) fnode->function;
            math21_ml_function_route_resize(fnode, f, fnet);
        } else if (fnode->type == mlfnode_type_yolo) {
            mlfunction_yolo *f = (mlfunction_yolo *) fnode->function;
            math21_ml_function_yolo_resize(fnode, f, w, h);
        } else if (fnode->type == mlfnode_type_max_pooling) {
            mlfunction_max_pooling *f = (mlfunction_max_pooling *) fnode->function;
            math21_ml_function_max_pooling_resize(fnode, f, w, h);
        } else if (fnode->type == mlfnode_type_average_pooling) {
            mlfunction_average_pooling *f = (mlfunction_average_pooling *) fnode->function;
            math21_ml_function_average_pooling_resize(fnode, f, w, h);
        } else if (fnode->type == mlfnode_type_cost) {
            mlfunction_cost *f = (mlfunction_cost *) fnode->function;
            math21_ml_function_cost_resize(fnode, f, w, h);
        } else {
            m21error("Cannot resize this type of layer");
        }
        if (fnode->getGlobalSpaceSize) {
            if (fnode->getGlobalSpaceSize(fnode) > workspace_size) workspace_size = fnode->getGlobalSpaceSize(fnode);
            if (fnode->getGlobalSpaceSize(fnode) > 2000000000) assert(0);
        }
        inputs = fnode->y_size;
//        net->layers[i] = l;
        h = fnode->y_dim[1];
        w = fnode->y_dim[2];
        // why break?
        if (fnode->type == mlfnode_type_average_pooling) break;
    }

    mlfunction_node *fnode_out = math21_ml_function_net_get_output_node(fnet);
    // some fnode has not set x_size
    math21_tool_assert(fnet->nodes[0]->x_size > 0);
    fnet->data_x_size = fnet->nodes[0]->x_size;

    fnet->y_size = fnode_out->y_size;
    fnet->data_y_size = fnet->y_size;

    mlfunction_node *fnode = fnet->nodes[fnet->n_node - 1];
    if (fnode->type == mlfnode_type_yolo) {
        mlfunction_yolo *fyolo = (mlfunction_yolo *) fnode->function;
        if (fyolo->truths) {
            fnet->data_y_size = fyolo->truths;
        }
    }

#if defined(MATH21_FLAG_USE_CPU)
#else
    fnet->y_cpu = math21_vector_resize_with_default_value_cpu(fnet->y_cpu, fnet->y_size * fnet->mbs_y, 0);
#endif
    fnet->data_x_wrapper = math21_vector_resize_with_default_value_wrapper(fnet->data_x_wrapper,
                                                                           fnet->data_x_size * fnet->mini_batch_size,
                                                                           0);
    fnet->data_y_wrapper = math21_vector_resize_with_default_value_wrapper(fnet->data_y_wrapper,
                                                                           fnet->data_y_size * fnet->mbs_y, 0);
#if defined(MATH21_FLAG_USE_CPU)
#else
    fnet->data_x_cpu = math21_vector_resize_with_default_value_cpu(fnet->data_x_cpu,
                                                                   fnet->data_x_size * fnet->mini_batch_size, 0);

    fnet->data_y_cpu = math21_vector_resize_with_default_value_cpu(fnet->data_y_cpu, fnet->data_y_size * fnet->mbs_y,
                                                                   0);
#endif
    if (workspace_size) {
        fnet->workspace = math21_vector_resize_with_default_value_wrapper(fnet->workspace,
                                                                          (workspace_size - 1) / sizeof(float) + 1, 0);
    }
    return 0;
}
