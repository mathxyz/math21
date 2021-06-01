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

#include "../../../image/files_c.h"
#include "box.h"

void math21_ml_box_detection_to_results(NumN ndetections, const mldetection *detections,
                                        NumN nresults, mlobject *results) {
    int idet, iclass;
    int iresult = 0;
    for (idet = 0; idet < ndetections; ++idet) {
        for (iclass = 0; iclass < detections[idet].classes; ++iclass) {
            if (detections[idet].prob[iclass] > 0) {
                mlbox b = detections[idet].bbox;
                if (iresult < nresults) {
                    results[iresult].bbox = b;
                    results[iresult].prob = detections[idet].prob[iclass];
                    results[iresult].id = iclass;
                    ++iresult;
                } else {
                    return;
                }
            }
        }
    }
}

void math21_ml_box_log_object(mlobject *results, int n) {
    int iresult;
    for (iresult = 0; iresult < n; ++iresult) {
        mlbox b = results[iresult].bbox;
        printf("%d, %f, %f, %f, %f, %f\n", results[iresult].id, results[iresult].prob,
               b.x, b.y, b.w, b.h);
    }
    printf("\n");
}

int _math21_ml_box_nms_comparator(const void *pa, const void *pb) {
    mldetection a = *(mldetection *) pa;
    mldetection b = *(mldetection *) pb;
    float diff = 0;
    if (b.sort_class >= 0) {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

void math21_ml_box_do_nms_obj(mldetection *detections, int ndetections, float thresh) {
    int i, j, k;
    k = ndetections - 1;
    for (i = 0; i <= k; ++i) {
        if (detections[i].objectness == 0) {
            mldetection swap = detections[i];
            detections[i] = detections[k];
            detections[k] = swap;
            --k;
            --i;
        }
    }
    ndetections = k + 1;

    for (i = 0; i < ndetections; ++i) {
        detections[i].sort_class = -1;
    }

    qsort(detections, ndetections, sizeof(mldetection), _math21_ml_box_nms_comparator);
    for (i = 0; i < ndetections; ++i) {
        if (detections[i].objectness == 0) continue;
        mlbox a = detections[i].bbox;
        for (j = i + 1; j < ndetections; ++j) {
            if (detections[j].objectness == 0) continue;
            mlbox b = detections[j].bbox;
            if (math21_ml_box_iou(a, b) > thresh) {
                detections[j].objectness = 0;
                for (k = 0; k < detections[j].classes; ++k) {
                    detections[j].prob[k] = 0;
                }
            }
        }
    }
}

void math21_ml_box_do_nms_sort(mldetection *detections, int ndetections, int classes, float thresh) {
    int i, j, k;
    k = ndetections - 1;
    for (i = 0; i <= k; ++i) {
        if (detections[i].objectness == 0) {
            mldetection swap = detections[i];
            detections[i] = detections[k];
            detections[k] = swap;
            --k;
            --i;
        }
    }
    ndetections = k + 1;

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < ndetections; ++i) {
            detections[i].sort_class = k;
        }
        qsort(detections, ndetections, sizeof(mldetection), _math21_ml_box_nms_comparator);
        for (i = 0; i < ndetections; ++i) {
            if (detections[i].prob[k] == 0) continue;
            mlbox a = detections[i].bbox;
            for (j = i + 1; j < ndetections; ++j) {
                mlbox b = detections[j].bbox;
                if (math21_ml_box_iou(a, b) > thresh) {
                    detections[j].prob[k] = 0;
                }
            }
        }
    }
}

float math21_ml_box_rmse(mlbox a, mlbox b) {
    return sqrt(pow(a.x - b.x, 2) +
                pow(a.y - b.y, 2) +
                pow(a.w - b.w, 2) +
                pow(a.h - b.h, 2));
}

mlbox math21_ml_box_vector_to_box(float *f, int stride) {
    mlbox b = {0};
    b.x = f[0];
    b.y = f[1 * stride];
    b.w = f[2 * stride];
    b.h = f[3 * stride];
    return b;
}


float math21_ml_box_overlap(float x1, float w1, float x2, float w2) {
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float math21_ml_box_intersection(mlbox a, mlbox b) {
    float w = math21_ml_box_overlap(a.x, a.w, b.x, b.w);
    float h = math21_ml_box_overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    float area = w * h;
    return area;
}

float math21_ml_box_union(mlbox a, mlbox b) {
    float i = math21_ml_box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}

float math21_ml_box_iou(mlbox a, mlbox b) {
    return math21_ml_box_intersection(a, b) / math21_ml_box_union(a, b);
}

mldbox math21_ml_box_derivative(mlbox a, mlbox b) {
    mldbox d;
    d.dx = 0;
    d.dw = 0;
    float l1 = a.x - a.w / 2;
    float l2 = b.x - b.w / 2;
    if (l1 > l2) {
        d.dx -= 1;
        d.dw += .5;
    }
    float r1 = a.x + a.w / 2;
    float r2 = b.x + b.w / 2;
    if (r1 < r2) {
        d.dx += 1;
        d.dw += .5;
    }
    if (l1 > r2) {
        d.dx = -1;
        d.dw = 0;
    }
    if (r1 < l2) {
        d.dx = 1;
        d.dw = 0;
    }

    d.dy = 0;
    d.dh = 0;
    float t1 = a.y - a.h / 2;
    float t2 = b.y - b.h / 2;
    if (t1 > t2) {
        d.dy -= 1;
        d.dh += .5;
    }
    float b1 = a.y + a.h / 2;
    float b2 = b.y + b.h / 2;
    if (b1 < b2) {
        d.dy += 1;
        d.dh += .5;
    }
    if (t1 > b2) {
        d.dy = -1;
        d.dh = 0;
    }
    if (b1 < t2) {
        d.dy = 1;
        d.dh = 0;
    }
    return d;
}

mldbox math21_ml_box_dintersect(mlbox a, mlbox b) {
    float w = math21_ml_box_overlap(a.x, a.w, b.x, b.w);
    float h = math21_ml_box_overlap(a.y, a.h, b.y, b.h);
    mldbox dover = math21_ml_box_derivative(a, b);
    mldbox di;

    di.dw = dover.dw * h;
    di.dx = dover.dx * h;
    di.dh = dover.dh * w;
    di.dy = dover.dy * w;

    return di;
}

mldbox math21_ml_box_dunion(mlbox a, mlbox b) {
    mldbox du;

    mldbox di = math21_ml_box_dintersect(a, b);
    du.dw = a.h - di.dw;
    du.dh = a.w - di.dh;
    du.dx = -di.dx;
    du.dy = -di.dy;

    return du;
}

void _math21_ml_box_dunion_test() {
    mlbox a = {0, 0, 1, 1};
    mlbox dxa = {0 + .0001, 0, 1, 1};
    mlbox dya = {0, 0 + .0001, 1, 1};
    mlbox dwa = {0, 0, 1 + .0001, 1};
    mlbox dha = {0, 0, 1, 1 + .0001};

    mlbox b = {.5, .5, .2, .2};
    mldbox di = math21_ml_box_dunion(a, b);
    printf("Union: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter = math21_ml_box_union(a, b);
    float xinter = math21_ml_box_union(dxa, b);
    float yinter = math21_ml_box_union(dya, b);
    float winter = math21_ml_box_union(dwa, b);
    float hinter = math21_ml_box_union(dha, b);
    xinter = (xinter - inter) / (.0001);
    yinter = (yinter - inter) / (.0001);
    winter = (winter - inter) / (.0001);
    hinter = (hinter - inter) / (.0001);
    printf("Union Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}

void _math21_ml_box_dintersect_test() {
    mlbox a = {0, 0, 1, 1};
    mlbox dxa = {0 + .0001, 0, 1, 1};
    mlbox dya = {0, 0 + .0001, 1, 1};
    mlbox dwa = {0, 0, 1 + .0001, 1};
    mlbox dha = {0, 0, 1, 1 + .0001};

    mlbox b = {.5, .5, .2, .2};
    mldbox di = math21_ml_box_dintersect(a, b);
    printf("Inter: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter = math21_ml_box_intersection(a, b);
    float xinter = math21_ml_box_intersection(dxa, b);
    float yinter = math21_ml_box_intersection(dya, b);
    float winter = math21_ml_box_intersection(dwa, b);
    float hinter = math21_ml_box_intersection(dha, b);
    xinter = (xinter - inter) / (.0001);
    yinter = (yinter - inter) / (.0001);
    winter = (winter - inter) / (.0001);
    hinter = (hinter - inter) / (.0001);
    printf("Inter Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}

mldbox math21_ml_box_diou(mlbox a, mlbox b) {
    float u = math21_ml_box_union(a, b);
    float i = math21_ml_box_intersection(a, b);
    mldbox di = math21_ml_box_dintersect(a, b);
    mldbox du = math21_ml_box_dunion(a, b);
    mldbox dd = {0, 0, 0, 0};

    if (i <= 0 || 1) {
        dd.dx = b.x - a.x;
        dd.dy = b.y - a.y;
        dd.dw = b.w - a.w;
        dd.dh = b.h - a.h;
        return dd;
    }

    dd.dx = 2 * pow((1 - (i / u)), 1) * (di.dx * u - du.dx * i) / (u * u);
    dd.dy = 2 * pow((1 - (i / u)), 1) * (di.dy * u - du.dy * i) / (u * u);
    dd.dw = 2 * pow((1 - (i / u)), 1) * (di.dw * u - du.dw * i) / (u * u);
    dd.dh = 2 * pow((1 - (i / u)), 1) * (di.dh * u - du.dh * i) / (u * u);
    return dd;
}

void _math21_ml_box_test() {
    _math21_ml_box_dintersect_test();
    _math21_ml_box_dunion_test();
    mlbox a = {0, 0, 1, 1};
    mlbox dxa = {0 + .00001, 0, 1, 1};
    mlbox dya = {0, 0 + .00001, 1, 1};
    mlbox dwa = {0, 0, 1 + .00001, 1};
    mlbox dha = {0, 0, 1, 1 + .00001};

    mlbox b = {.5, 0, .2, .2};

    float iou = math21_ml_box_iou(a, b);
    iou = (1 - iou) * (1 - iou);
    printf("%f\n", iou);
    mldbox d = math21_ml_box_diou(a, b);
    printf("%f %f %f %f\n", d.dx, d.dy, d.dw, d.dh);

    float xiou = math21_ml_box_iou(dxa, b);
    float yiou = math21_ml_box_iou(dya, b);
    float wiou = math21_ml_box_iou(dwa, b);
    float hiou = math21_ml_box_iou(dha, b);
    xiou = ((1 - xiou) * (1 - xiou) - iou) / (.00001);
    yiou = ((1 - yiou) * (1 - yiou) - iou) / (.00001);
    wiou = ((1 - wiou) * (1 - wiou) - iou) / (.00001);
    hiou = ((1 - hiou) * (1 - hiou) - iou) / (.00001);
    printf("manual %f %f %f %f\n", xiou, yiou, wiou, hiou);
}

void math21_ml_box_01centerbox_to_rectangle(mlbox b, int nr, int nc, m21rectangle *rectangle) {
    rectangle->top = (int) ((b.y - b.h / 2.) * nr);
    rectangle->bottom = (int) ((b.y + b.h / 2.) * nr);
    rectangle->left = (int) ((b.x - b.w / 2.) * nc);
    rectangle->right = (int) ((b.x + b.w / 2.) * nc);
}

void math21_ml_box_draw_detections(m21image image, mldetection *detections, int ndetections, float thresh) {
    int i, j;

    for (i = 0; i < ndetections; ++i) {
        int nclasses = detections[i].classes;
        int iclass = -1;
        for (j = 0; j < nclasses; ++j) {
            if (detections[i].prob[j] > thresh) {
                if (iclass < 0) {
                    iclass = j;
                    break;
                }
            }
        }
        if (iclass >= 0) {
            int width = image.nr * .008;

            float red = 1;
            float green = 0;
            float blue = 0;
            float rgb[3];

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            mlbox b = detections[i].bbox;

            m21rectangle rectangle;
            math21_ml_box_01centerbox_to_rectangle(b, image.nr, image.nc, &rectangle);

            math21_number_interval_intersect_to_int(0, image.nr - 1, &rectangle.top, &rectangle.bottom);
            math21_number_interval_intersect_to_int(0, image.nc - 1, &rectangle.left, &rectangle.right);
            math21_image_draw_box_with_width(image, rectangle.top, rectangle.bottom, rectangle.left, rectangle.right,
                                             width, red, green, blue);
        }
    }
}
