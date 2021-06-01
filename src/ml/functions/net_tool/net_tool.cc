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
#include "net_tool.h"


int math21_ml_function_net_get_detections_num(mlfunction_net *fnet, float thresh) {
    int i;
    int s = 0;
    for (i = 0; i < fnet->n_node; ++i) {
        mlfunction_node *fnode = fnet->nodes[i];
        if (fnode->type == mlfnode_type_yolo) {
            mlfunction_yolo *fyolo = (mlfunction_yolo *) fnode->function;
            s += math21_ml_function_yolo_num_detections(fyolo, thresh);
        }
    }
    return s;
}

mldetection *math21_ml_function_net_boxes_create(mlfunction_net *fnet, float thresh, int *ndetections) {
    mlfunction_node *fnode = fnet->nodes[fnet->n_node - 1];
    math21_tool_assert(fnode->type == mlfnode_type_yolo);
    mlfunction_yolo *fyolo = (mlfunction_yolo *) fnode->function;

    int i;
    int nboxes = math21_ml_function_net_get_detections_num(fnet, thresh);
    if (ndetections) *ndetections = nboxes;
    mldetection *dets = (mldetection *) math21_vector_create_buffer_cpu(nboxes, sizeof(mldetection));
    for (i = 0; i < nboxes; ++i) {
        dets[i].prob = (float *) math21_vector_create_buffer_cpu(fyolo->classes, sizeof(float));
    }
    return dets;
}

void math21_ml_function_net_boxes_destroy(mldetection *dets, int ndetections) {
    int iobject;
    for (iobject = 0; iobject < ndetections; ++iobject) {
        math21_vector_free_cpu(dets[iobject].prob);
        if (dets[iobject].mask) math21_vector_free_cpu(dets[iobject].mask);
    }
    free(dets);
}

void math21_ml_function_net_boxes_set(mlfunction_net *fnet, int nr, int nc, float thresh, int relative,
                                      mldetection *dets) {
    int j;
    for (j = 0; j < fnet->n_node; ++j) {
        mlfunction_node *fnode = fnet->nodes[j];
        if (fnode->type == mlfnode_type_yolo) {
            mlfunction_yolo *fyolo = (mlfunction_yolo *) fnode->function;
            int count = math21_ml_function_yolo_get_detections(fyolo, nc, nr, fnet->data_x_dim[2], fnet->data_x_dim[1],
                                                               thresh, relative,
                                                               dets);
            dets += count;
        }
    }
}

mldetection *math21_ml_function_net_boxes_get(
        mlfunction_net *fnet, int nr, int nc, float thresh, int relative, int *ndetections) {
    mldetection *dets = math21_ml_function_net_boxes_create(fnet, thresh, ndetections);
    math21_ml_function_net_boxes_set(fnet, nr, nc, thresh, relative, dets);
    return dets;
}
