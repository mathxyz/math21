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

#pragma once

#include "inner_h.h"

namespace math21 {

    struct ImageDraw {

        // deprecate
        // draw and save
        virtual void draw(const TenR &A, const char *name, NumB isBatch = 1) {
            MATH21_ASSERT(0, "Please override to use!");
        }

        virtual void setDir(const char *name) {
            MATH21_ASSERT(0, "Please override to use!");
        }

        // reset image to bg color.
        virtual void reset() {
            MATH21_ASSERT(0, "Please override to use!");
        }

        // don't save
        virtual void plot(const TenR &A, const char *name, NumB isBatch = 1) {
            MATH21_ASSERT(0, "Please override to use!");
        }

        virtual void save(const char *name) {
            MATH21_ASSERT(0, "Please override to use!");
        }
    };

    struct ImageDraw_Dummy : public ImageDraw {
        ImageDraw_Dummy() = default;

        void draw(const TenR &A, const char *name, NumB isBatch = 1) override {
            m21log("ImageDraw_Dummy draw!");
        }

        void setDir(const char *name) override {
            m21log("ImageDraw_Dummy setDir!");
        }

        void reset() override {
            m21log("ImageDraw_Dummy reset!");
        }

        // don't save
        void plot(const TenR &A, const char *name, NumB isBatch = 1) override {
            m21log("ImageDraw_Dummy plot!");
        }

        void save(const char *name) override {
            m21log("ImageDraw_Dummy save!");
        }
    };


    enum {
        img_resize_method_default = 1,
        img_resize_method_sampling,
        img_resize_method_pooling,//
    };

    void math21_img_resize(const MatR &src, MatR &dst, NumN img_resize_method = img_resize_method_default);

    void math21_img_resize(const Seqce <TenR> &srcs, Seqce <TenR> &dsts,
                           NumN img_resize_method = img_resize_method_default);

    void math21_img_resize(Seqce <TenR> &images, const VecN &d, NumN img_resize_method = img_resize_method_default);

    void math21_img_resize(TenR &image, const VecN &d, NumN img_resize_method = img_resize_method_default);

    void math21_img_rgb_to_gray(Seqce <TenR> &images);

    void math21_img_rgb_to_gray(MatR &image);

    void math21_img_rgb_to_gray(const MatR &src, MatR &dst);

    void math21_img_gray_cluster_by_value(const MatR &src, MatN &dst, NumN K, VecN &num_in_clusters);

    void math21_img_gray_to_binary(MatR &image);

    void math21_img_gray_to_binary(const MatR &src, MatR &dst);

    // slow, so use for small image.
    void math21_img_gray_to_binary(Seqce <TenR> &images);

    void math21_img_histogram(const MatR &src, MatR &histogram);

    // image nch*nr*nc -> nr*nc*nch
    template<typename T, typename S>
    void math21_img_planar_to_interleaved(const Tensor <T> &A, Tensor <S> &B) {
        MATH21_ASSERT(A.dims() == 3)
        NumN nr, nc, nch;
        nch = A.dim(1);
        nr = A.dim(2);
        nc = A.dim(3);
        if (A.dims() == 3) {
            MATH21_ASSERT(nch == 1 || nch == 3 || nch == 4,
                          "not 1, 3 or 4 channels, channels: " << nch);
        }
        if (B.isSameSize(nr, nc, nch) == 0) {
            B.setSize(nr, nc, nch);
        }
        math21_op_as_matrix_trans(A, B, nch, nr * nc);
    }

    // image nr*nc*nch -> nch*nr*nc
    template<typename T, typename S>
    void math21_img_interleaved_to_planar(const Tensor <T> &A, Tensor <S> &B) {
        MATH21_ASSERT(A.dims() == 3)
        NumN nr, nc, nch;
        nch = A.dim(3);
        nr = A.dim(1);
        nc = A.dim(2);
        if (A.dims() == 3) {
            MATH21_ASSERT(nch == 1 || nch == 3 || nch == 4,
                          "not 1, 3 or 4 channels, channels: " << nch);
        }
        if (B.isSameSize(nch, nr, nc) == 0) {
            B.setSize(nch, nr, nc);
        }
        math21_op_as_matrix_trans(A, B, nr * nc, nch);
    }

    template<typename T>
    void math21_image_get_nr_nc(const Tensor <T> &A, NumN &nr, NumN &nc, NumN &nch) {
        MATH21_ASSERT(A.dims() == 3 || A.dims() == 2)
        if (A.dims() == 3) {
            nr = A.dim(2);
            nc = A.dim(3);
            nch = A.dim(1);
        } else {
            nr = A.dim(1);
            nc = A.dim(2);
            nch = 1;
        }
    }

    template<typename T>
    NumN math21_image_get_nch(const Tensor <T> &A) {
        MATH21_ASSERT(A.dims() == 3 || A.dims() == 2)
        if (A.dims() == 3) {
            return A.dim(1);
        } else {
            return 1;
        }
    }

    template<typename T>
    NumN math21_image_get_nr(const Tensor <T> &A) {
        MATH21_ASSERT(A.dims() == 3 || A.dims() == 2)
        if (A.dims() == 3) {
            return A.dim(2);
        } else {
            return A.dim(1);
        }
    }

    template<typename T>
    NumN math21_image_get_nc(const Tensor <T> &A) {
        MATH21_ASSERT(A.dims() == 3 || A.dims() == 2)
        if (A.dims() == 3) {
            return A.dim(3);
        } else {
            return A.dim(2);
        }
    }

    template<typename T>
    NumN math21_image_get_area(const Tensor <T> &A) {
        return math21_image_get_nr(A) * math21_image_get_nc(A);
    }

    template<typename T>
    void math21_image_set_size(Tensor <T> &m, NumN nr, NumN nc, NumN nch, NumN dims) {
        if (nch == 1 && dims == 2) {
            m.setSize(nr, nc);
        } else {
            m.setSize(nch, nr, nc);
        }
    }


    namespace detail {
        void math21_image_argb8888_to_tensor(NumN32 *data, TenR &m, NumN nr, NumN nc, NumN nch);

#ifdef MATH21_FLAG_USE_CUDA

        void math21_image_argb8888_to_tensor_cuda(NumN32 *data, TenR &m, NumN nr, NumN nc, NumN nch);

#endif
    }

    void math21_image_argb8888_to_tensor(NumN32 *data, TenR &m, NumN nr, NumN nc, NumN nch = 4);

    namespace detail {
        void math21_image_tensor_to_argb8888(NumN32 *data, const TenR &m);

#ifdef MATH21_FLAG_USE_CUDA

        void math21_image_tensor_to_argb8888_cuda(NumN32 *data, const TenR &m);

#endif
    }

    // maybe to do next: to read all argb data we saved, we store them as bgra no matter which endian is.
    // Note argb8888 is stored as bgra in little endian, as argb in big endian.
    // rgba to argb8888
    void math21_image_tensor_to_argb8888(NumN32 *data, const TenR &m);

    // read argb8888 to rgba
    NumB math21_io_argb8888_read(TenR &m, NumN nr, NumN nc, NumN nch, const char *path);

    NumB math21_io_argb8888_write(const TenR &m, const char *path);

    void math21_image_convert_to_argb8888(const TenR &m, Tensor <NumN32> &B);

    void math21_image_size_to_Interval2D(NumN nr, NumN nc, Interval2D &I);

    void math21_image_size_to_Interval2D(const TenR &A, Interval2D &I);

    template<typename T>
    void math21_image_set_pixel(Tensor <T> &m, const VecN &pixel) {
        MATH21_ASSERT(m.dims() == 3);
        for (NumN i1 = 1; i1 <= m.dim(1); i1++) {
            for (NumN i2 = 1; i2 <= m.dim(2); i2++) {
                for (NumN i3 = 1; i3 <= m.dim(3); i3++) {
                    m(i1, i2, i3) = pixel(i1);
                }
            }
        }
    }

    template<typename T, typename T2>
    void math21_image_convert_ch_r_c_to_r_c_ch(const T *src, T2 *dst, NumN nr, NumN nc, NumN nch) {
        NumN ic, ir, ich;
        for (ich = 0; ich < nch; ++ich) {
            for (ir = 0; ir < nr; ++ir) {
                for (ic = 0; ic < nc; ++ic) {
                    NumN src_index = ich * nr * nc + ir * nc + ic;
                    NumN dst_index = ir * nc * nch + ic * nch + ich;
                    dst[dst_index] = (T2) src[src_index];
                }
            }
        }
    }

    template<typename T, typename T2>
    void math21_image_convert_r_c_ch_to_ch_r_c(const T *src, T2 *dst, NumN nr, NumN nc, NumN nch) {
        NumN ic, ir, ich;
        for (ich = 0; ich < nch; ++ich) {
            for (ir = 0; ir < nr; ++ir) {
                for (ic = 0; ic < nc; ++ic) {
                    NumN src_index = ir * nc * nch + ic * nch + ich;
                    NumN dst_index = ich * nr * nc + ir * nc + ic;
                    dst[dst_index] = (T2) src[src_index];
                }
            }
        }
    }
}