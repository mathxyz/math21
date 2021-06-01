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

#include "inner.h"
#include "commonFunctions.h"

namespace math21 {
    enum {
        //cnn nonlinear which must be element-wise.

        cnn_type_hn_linear = 1,
        cnn_type_hn_tanh,//
        cnn_type_hn_ReLU,
        cnn_type_hn_LogSigmoid,//

        cnn_type_fn_fully,
        cnn_type_fn_locally,//
        cnn_type_fn_conv,
        cnn_type_fn_pooling,//

        cnn_type_pooling_average,
        cnn_type_pooling_max,//
        cnn_type_pooling_cluster
    };

    inline std::string math21_type2string_cnn(NumN cnn_type) {
#define MATH21_LOCAL_F(a) case a: return MATH21_STRINGIFY(a);
        switch (cnn_type) {
            MATH21_LOCAL_F(cnn_type_hn_linear)
            MATH21_LOCAL_F(cnn_type_hn_tanh)
            MATH21_LOCAL_F(cnn_type_hn_ReLU)
            MATH21_LOCAL_F(cnn_type_hn_LogSigmoid)
            MATH21_LOCAL_F(cnn_type_fn_fully)
            MATH21_LOCAL_F(cnn_type_fn_locally)
            MATH21_LOCAL_F(cnn_type_fn_conv)
            MATH21_LOCAL_F(cnn_type_fn_pooling)
            MATH21_LOCAL_F(cnn_type_pooling_average)
            MATH21_LOCAL_F(cnn_type_pooling_max)
            MATH21_LOCAL_F(cnn_type_pooling_cluster)
            default:
                return "UNKNOWN";
        }
#undef MATH21_LOCAL_F
    }

//######################### common functions

    void math21_operator_ml_pooling_get_mk_ms(NumN m1, NumN n1, NumN m2, NumN n2,
                                              NumN &mk, NumN &nk, NumN &ms, NumN &ns);

    void math21_operator_ml_pooling_valueAt(const TenR &x, TenR &xn_next, NumN cnn_type_pooling,
                                            NumN mk, NumN nk, NumN ms, NumN ns,
                                            Seqce <TenN> *p_xn_argmax = 0,
                                            NumB isUsingDiff = 0);

    ////######################### config
    struct cnn_config_fn {
    public:
        VecN d2;
        NumN cnn_type_hn;

        cnn_config_fn(const VecN &_d2, NumN cnn_type_hn) : cnn_type_hn(cnn_type_hn) {
            d2.copyFrom(_d2);
            MATH21_ASSERT(d2.size() == 3, "output shape is not 3-D tensor!");
        }

        virtual ~cnn_config_fn() {

        }

        virtual NumN getType() const = 0;
    };

    struct cnn_config_fn_fully : public cnn_config_fn {
    public:
        cnn_config_fn_fully(const VecN &d2, NumN cnn_type_hn) : cnn_config_fn(d2, cnn_type_hn) {

        }

        NumN getType() const override {
            return cnn_type_fn_fully;
        }
    };

    struct cnn_config_fn_locally : public cnn_config_fn {
    public:
        NumN mk;
        NumN nk;
        NumN ms;
        NumN ns;

        cnn_config_fn_locally(const VecN &d2, NumN cnn_type_hn,
                              NumN mk, NumN nk, NumN ms, NumN ns)
                : cnn_config_fn(d2, cnn_type_hn),
                  mk(mk), nk(nk), ms(ms), ns(ns) {

        }

        NumN getType() const override {
            return cnn_type_fn_locally;
        }
    };

    //tiled convolution.
    struct cnn_config_fn_conv : public cnn_config_fn {
    public:
        NumN mk;
        NumN nk;
        NumN ms;
        NumN ns;
        NumN mt;
        NumN nt;

        cnn_config_fn_conv(const VecN &d2, NumN cnn_type_hn,
                           NumN mk, NumN nk, NumN ms, NumN ns, NumN mt, NumN nt)
                : cnn_config_fn(d2, cnn_type_hn),
                  mk(mk), nk(nk), ms(ms), ns(ns), mt(mt), nt(nt) {
        }

        NumN getType() const override {
            return cnn_type_fn_conv;
        }
    };

    struct cnn_config_fn_pooling : public cnn_config_fn {
    public:
        NumN cnn_type_pooling;

        cnn_config_fn_pooling(const VecN &d2, NumN cnn_type_pooling)
                : cnn_config_fn(d2, cnn_type_hn_linear),
                  cnn_type_pooling(cnn_type_pooling) {
        }

        NumN getType() const override {
            return cnn_type_fn_pooling;
        }
    };

////#########################
    class cnn_fn : public think::Operator {
    protected:
        VecN d1; //input shape.
        VecN d2; //output shape.
        NumN cnn_type_hn;
        NumB isUsingDiff;
        TenR xn_next;
        TenR dxn;
        TenR dyn;
        Function *h;
    public:
        //
        cnn_fn(const cnn_fn &fn) {
            setSize(fn.d1, fn.d2, fn.cnn_type_hn, fn.isUsingDiff);
        }

        //Because it is one of cnn components, we just set output shape.
        cnn_fn(const VecN &_d1, const VecN &_d2, NumN _cnn_type_hn, NumB _isUsingDiff) {
            setSize(_d1, _d2, _cnn_type_hn, _isUsingDiff);
        }

        cnn_fn() {}

        // another construction
        virtual cnn_fn *clone() const = 0;

        virtual void serialize(std::ostream &out, SerializeNumInterface &sn) {
            math21_io_serialize(out, d1, sn);
            math21_io_serialize(out, d2, sn);
            sn.serialize(out, cnn_type_hn);
            sn.serialize(out, (NumN) isUsingDiff);
        }

        virtual void deserialize(std::istream &in, DeserializeNumInterface &sn) {
            VecN d1; //input shape.
            VecN d2; //output shape.
            NumN cnn_type_hn;
            NumN isUsingDiff;

            math21_io_deserialize(in, d1, sn);
            math21_io_deserialize(in, d2, sn);
            sn.deserialize(in, cnn_type_hn);
            sn.deserialize(in, isUsingDiff);

            setSize(d1, d2, cnn_type_hn, (NumB) isUsingDiff);
        }

        void setSize(const VecN &_d1, const VecN &_d2, NumN _cnn_type_hn, NumB _isUsingDiff) {
            MATH21_ASSERT(_d1.size() == 3, "input shape is not 3-D tensor!");
            MATH21_ASSERT(_d2.size() == 3, "output shape is not 3-D tensor!");
            d1.setSize(_d1.size());
            d2.setSize(_d2.size());
            d1.assign(_d1);
            d2.assign(_d2);
            xn_next.setSize(d2);
            cnn_type_hn = _cnn_type_hn;
            isUsingDiff = _isUsingDiff;
            if (cnn_type_hn == cnn_type_hn_linear) {
                h = new Function_linear();
            } else if (cnn_type_hn == cnn_type_hn_tanh) {
                h = new Function_tanh();
            } else if (cnn_type_hn == cnn_type_hn_ReLU) {
                h = new Function_LeakyReLU();
            } else if (cnn_type_hn == cnn_type_hn_LogSigmoid) {
                h = new Function_LogSigmoid();
            } else {
                MATH21_ASSERT(0, "current version check nonlinear hn fail!");
            }
            if (isUsingDiff) {
                dxn.setSize(d1);
                dyn.setSize(d2);
            }
        }

        virtual ~cnn_fn() {
            if (h != 0) {
                delete h;
            }
        }

        NumB isEmpty() const {
            return d1.isEmpty() ? 1 : 0;
        }

        //from left to right.
        const VecN &getOutputShape() const {
            return d2;
        }

        virtual NumN getType() const = 0;

        virtual void a() = 0;

        //must be called in child class constructor.
        virtual void setSize() = 0;

        virtual NumN getThetaSize() const = 0;

        virtual void setThetaSpace(const SpaceParas &paras) = 0;

//        virtual void setDtheta(const TenR &_theta) = 0;
        virtual void setDthetaSpace(const SpaceParas &paras) = 0;

        virtual const TenR &valueAt(const TenR &x) = 0;

        const TenR &get_derivativeValue_J() {
            return dxn;
        }

        TenR &getValue() {
            return xn_next;
        }

        virtual void derivativeValueAtTheta_and_xn_J(const TenR &xn, const TenR &dxn_next, NumR alpha) = 0;

        virtual void log() const = 0;

        virtual NumR calWeightNormSquare(NumN norm) const = 0;
    };

    class cnn_fn_fully : public cnn_fn {
    private:
        TenR W, b;
        TenR dW, db;

        void thetaToInner(const SpaceParas &paras, TenR &W, TenR &b) {
            MATH21_ASSERT(paras.size == getThetaSize() * sizeof(NumR));
            NumN offset = 0;
            SpaceParas paras_dst;
            math21_memory_getSpace(paras, paras_dst, offset, W.volume(), sizeof(NumR));
            W.setSpace(paras_dst);

            offset = offset + W.volume();
            math21_memory_getSpace(paras, paras_dst, offset, b.volume(), sizeof(NumR));
            b.setSpace(paras_dst);
        }

    public:
        cnn_fn_fully(const cnn_fn_fully &fn) : cnn_fn(fn) {
            setSize();
        }

        cnn_fn_fully(const VecN &d1, const VecN &d2, NumN cnn_type_hn, NumB isUsingDiff) : cnn_fn(d1, d2,
                                                                                                  cnn_type_hn,
                                                                                                  isUsingDiff) {
            setSize();
        }

        cnn_fn_fully() {}

        cnn_fn *clone() const override {
            cnn_fn *fn = new cnn_fn_fully(*this);
            return fn;
        }

        void serialize(std::ostream &out, SerializeNumInterface &sn) override {
            cnn_fn::serialize(out, sn);
        }

        void deserialize(std::istream &in, DeserializeNumInterface &sn) override {
            cnn_fn::deserialize(in, sn);
            setSize();
        }

        void a() override {
        }

        NumN getThetaSize() const override {
            return W.volume() + b.volume();
        }

        void setThetaSpace(const SpaceParas &paras) override {
            thetaToInner(paras, W, b);
        }

        void setDthetaSpace(const SpaceParas &paras) override {
            MATH21_ASSERT(!dW.isEmpty() && !db.isEmpty(), "maybe you forgot to enable diff in constructor");
            thetaToInner(paras, dW, db);
        }

        void log() const override {
            if (isEmpty()) {
                return;
            }
            W.logInfo("W");
            b.logInfo("b");
        }

        void setSize() override {

            VecN index(d1.size() + d2.size());
            for (NumN i = 1; i <= d2.size(); i++) {
                index(i) = d2(i);
            }
            for (NumN i = 1, j = d2.size() + 1; i <= d1.size(); i++, j++) {
                index(j) = d1(i);
            }
            W.setSize(index);
            b.setSize(d2);

            if (isUsingDiff) {
                dW.setSize(index);
                db.setSize(d2);
            }
        }

        void derivativeValueAtTheta_and_xn_J(const TenR &xn, const TenR &dxn_next, NumR alpha) override {
            NumN j1, j2, j3, i1, i2, i3;
            // dyn
            for (j1 = 1; j1 <= d2(1); ++j1) {
                for (j2 = 1; j2 <= d2(2); ++j2) {
                    for (j3 = 1; j3 <= d2(3); ++j3) {
                        dyn(j1, j2, j3) = dxn_next(j1, j2, j3) * h->derivativeValue_using_y(xn_next(j1, j2, j3));
                    }
                }
            }

            // dW, db
            NumR val_dy;
            for (j1 = 1; j1 <= d2(1); ++j1) {
                for (j2 = 1; j2 <= d2(2); ++j2) {
                    for (j3 = 1; j3 <= d2(3); ++j3) {
                        val_dy = dyn(j1, j2, j3);
                        for (i1 = 1; i1 <= d1(1); ++i1) {
                            for (i2 = 1; i2 <= d1(2); ++i2) {
                                for (i3 = 1; i3 <= d1(3); ++i3) {
                                    dW(j1, j2, j3, i1, i2, i3) = val_dy * xn(i1, i2, i3) +
                                                                 alpha * W(j1, j2, j3, i1, i2, i3);
                                }
                            }
                        }
                        db(j1, j2, j3) = val_dy;
                    }
                }
            }

            // dxn, clear first, then reverse compute.
            dxn = 0;
            for (j1 = 1; j1 <= d2(1); ++j1) {
                for (j2 = 1; j2 <= d2(2); ++j2) {
                    for (j3 = 1; j3 <= d2(3); ++j3) {
                        val_dy = dyn(j1, j2, j3);
                        for (i1 = 1; i1 <= d1(1); ++i1) {
                            for (i2 = 1; i2 <= d1(2); ++i2) {
                                for (i3 = 1; i3 <= d1(3); ++i3) {
                                    NumR &val_dxn = dxn.valueAt(i1, i2, i3);
                                    val_dxn = val_dxn + W(j1, j2, j3, i1, i2, i3) * val_dy;
                                }
                            }
                        }
                    }
                }
            }

            // clip
            math21_clip(dxn);
        }

        const TenR &valueAt(const TenR &x) override {
            NumN j1, j2, j3, i1, i2, i3;
            for (j1 = 1; j1 <= d2(1); ++j1) {
                for (j2 = 1; j2 <= d2(2); ++j2) {
                    for (j3 = 1; j3 <= d2(3); ++j3) {
                        NumR y = 0;
                        for (i1 = 1; i1 <= d1(1); ++i1) {
                            for (i2 = 1; i2 <= d1(2); ++i2) {
                                for (i3 = 1; i3 <= d1(3); ++i3) {
                                    y = y + x(i1, i2, i3) * W(j1, j2, j3, i1, i2, i3);
                                }
                            }
                        }
                        y = y + b(j1, j2, j3);
                        xn_next(j1, j2, j3) = h->valueAt(y);
                    }
                }
            }
            return xn_next;
        }

        NumR calWeightNormSquare(NumN norm) const override {
            NumR sum = math21_operator_norm(W, norm);
            if (norm == 1) {
            } else if (norm == 2) {
                sum = xjsquare(sum);
            } else {
                MATH21_ASSERT(0, "norm other than 1, 2 not supported currently");
            }
            math21_clip(sum);
            return sum;
        }

        NumN getType() const override {
            return cnn_type_fn_fully;
        }
    };

    class cnn_fn_locally : public cnn_fn {
    private:
        TenR K, b;
        TenR dK, db;

        NumN mk;
        NumN nk;
        NumN ms;
        NumN ns;
        NumN mpad;
        NumN npad;

        void thetaToInner(const SpaceParas &paras, TenR &W, TenR &b) {
            MATH21_ASSERT(paras.size == getThetaSize() * sizeof(NumR));
            NumN offset = 0;
            SpaceParas paras_dst;
            math21_memory_getSpace(paras, paras_dst, offset, W.volume(), sizeof(NumR));
            W.setSpace(paras_dst);

            offset = offset + W.volume();
            math21_memory_getSpace(paras, paras_dst, offset, b.volume(), sizeof(NumR));
            b.setSpace(paras_dst);
        }

    public:
        cnn_fn_locally(const cnn_fn_locally &fn) :
                cnn_fn(fn),
                mk(fn.mk), nk(fn.nk), ms(fn.ms), ns(fn.ns) {
            setSize();
        }

        cnn_fn_locally(const VecN &d1, const VecN &d2, NumN cnn_type_hn,
                       NumN mk, NumN nk, NumN ms, NumN ns,
                       NumB isUsingDiff) :
                cnn_fn(d1, d2, cnn_type_hn, isUsingDiff),
                mk(mk), nk(nk), ms(ms), ns(ns) {
            setSize();
        }

        cnn_fn_locally() {}

        cnn_fn *clone() const override {
            cnn_fn *fn = new cnn_fn_locally(*this);
            return fn;
        }

        void serialize(std::ostream &out, SerializeNumInterface &sn) override {
            cnn_fn::serialize(out, sn);
            sn.serialize(out, mk);
            sn.serialize(out, nk);
            sn.serialize(out, ms);
            sn.serialize(out, ns);
        }

        void deserialize(std::istream &in, DeserializeNumInterface &sn) override {
            cnn_fn::deserialize(in, sn);
            sn.deserialize(in, mk);
            sn.deserialize(in, nk);
            sn.deserialize(in, ms);
            sn.deserialize(in, ns);
            setSize();
        }

        void a() override {
        }

        NumN getThetaSize() const override {
            return K.volume() + b.volume();
        }

        void setThetaSpace(const SpaceParas &paras) override {
            thetaToInner(paras, K, b);
        }

        void setDthetaSpace(const SpaceParas &paras) override {
            MATH21_ASSERT(!dK.isEmpty() && !db.isEmpty(), "maybe you forgot to enable diff in constructor");
            thetaToInner(paras, dK, db);
        }

        void log() const override {
            if (isEmpty()) {
                return;
            }
            K.logInfo("K");
            b.logInfo("b");
        }


        void setSize() override {
            MATH21_ASSERT(mk >= 1 && nk >= 1 && ms >= 1 && ns >= 1);

            //
            if (mk > d1(2)) {
                mk = d1(2);
                m21warn("mk is too large. Adjust to ", mk);
            }
            if (nk > d1(3)) {
                nk = d1(3);
                m21warn("nk is too large. Adjust to ", nk);
            }
            NumZ mpad2 = (d2(2) - 1) * ms + mk - d1(2);
            NumZ npad2 = (d2(3) - 1) * ns + nk - d1(3);
            if (mpad2 >= 0) {
                if (xjIsEven(mpad2)) {
                    mpad = (NumN) mpad2 / 2;
                } else {
                    mpad = (NumN) ((mpad2 - 1) / 2);
                }
            } else {
                mk = d1(2) - (d2(2) - 1) * ms;
                mpad = 0;
                m21warn("mk is too small. Adjust to ", mk);
            }
            if (npad2 >= 0) {
                if (xjIsEven(npad2)) {
                    npad = npad2 / 2;
                } else {
                    npad = (npad2 - 1) / 2;
                }
            } else {
                nk = d1(3) - (d2(3) - 1) * ns;
                npad = 0;
                m21warn("nk is too small. Adjust to ", nk);
            }

            ///
            VecN index(d1.size() + d2.size());
            index = d2(1), d2(2), d2(3), d1(1), mk, nk;
            K.setSize(index);
            b.setSize(d2);

            if (isUsingDiff) {
                dK.setSize(index);
                db.setSize(d2);
            }
        }

        void derivativeValueAtTheta_and_xn_J(const TenR &xn, const TenR &dxn_next, NumR alpha) override {
            NumN j1, j2, j3, i1, i2, i3;
            NumZ ii2, ii3; //absolute index w.r.t. x.
            NumZ ia, ic;
            NumR val;

            // dyn
            for (j1 = 1; j1 <= d2(1); ++j1) {
                for (j2 = 1; j2 <= d2(2); ++j2) {
                    for (j3 = 1; j3 <= d2(3); ++j3) {
                        dyn(j1, j2, j3) = dxn_next(j1, j2, j3) * h->derivativeValue_using_y(xn_next(j1, j2, j3));
                    }
                }
            }
            // dW, db
            NumR val_dy;
            for (j1 = 1; j1 <= d2(1); ++j1) {
                for (j2 = 1; j2 <= d2(2); ++j2) {
                    for (j3 = 1; j3 <= d2(3); ++j3) {
                        val_dy = dyn(j1, j2, j3);
                        ia = (j2 - 1) * ms - mpad;
                        ic = (j3 - 1) * ns - npad;
                        for (i1 = 1; i1 <= d1(1); ++i1) {
                            for (i2 = 1; i2 <= mk; ++i2) {
                                ii2 = ia + i2;
                                for (i3 = 1; i3 <= nk; ++i3) {
                                    ii3 = ic + i3;
                                    if (xjIsIn(ii2, 1, d1(2)) && xjIsIn(ii3, 1, d1(3))) {
                                        val = xn(i1, ii2, ii3);
                                    } else {
                                        val = 0;
                                    }
                                    dK(j1, j2, j3, i1, i2, i3) = val_dy * val +
                                                                 alpha * K(j1, j2, j3, i1, i2, i3);
                                }
                            }
                        }
                        db(j1, j2, j3) = val_dy;
                    }
                }
            }
            // dxn, clear first, then reverse compute.
            dxn = 0;
            for (j1 = 1; j1 <= d2(1); ++j1) {
                for (j2 = 1; j2 <= d2(2); ++j2) {
                    for (j3 = 1; j3 <= d2(3); ++j3) {
                        val_dy = dyn(j1, j2, j3);
                        ia = (j2 - 1) * ms - mpad;
                        ic = (j3 - 1) * ns - npad;
                        for (i1 = 1; i1 <= d1(1); ++i1) {
                            for (i2 = 1; i2 <= mk; ++i2) {
                                ii2 = ia + i2;
                                for (i3 = 1; i3 <= nk; ++i3) {
                                    ii3 = ic + i3;
                                    if (xjIsIn(ii2, 1, d1(2)) && xjIsIn(ii3, 1, d1(3))) {
                                        NumR &val_dxn = dxn.valueAt(i1, ii2, ii3);
                                        val_dxn = val_dxn + K(j1, j2, j3, i1, i2, i3) * val_dy;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // clip
            math21_clip(dxn);
        }

        const TenR &valueAt(const TenR &x) override {
            NumN j1, j2, j3, i1, i2, i3;
            NumZ ii2, ii3; //absolute index w.r.t. x.
            NumZ ia, ic;
            NumR val;
            NumR y;
            for (j1 = 1; j1 <= d2(1); ++j1) {
                for (j2 = 1; j2 <= d2(2); ++j2) {
                    for (j3 = 1; j3 <= d2(3); ++j3) {
                        y = 0;
                        ia = (j2 - 1) * ms - mpad;
                        ic = (j3 - 1) * ns - npad;
                        for (i1 = 1; i1 <= d1(1); ++i1) {
                            for (i2 = 1; i2 <= mk; ++i2) {
                                ii2 = ia + i2;
                                for (i3 = 1; i3 <= nk; ++i3) {
                                    ii3 = ic + i3;
                                    if (xjIsIn(ii2, 1, d1(2)) && xjIsIn(ii3, 1, d1(3))) {
                                        val = x(i1, ii2, ii3);
                                    } else {
                                        val = 0;
                                    }
                                    y = y + val * K(j1, j2, j3, i1, i2, i3);
                                }
                            }
                        }
                        y = y + b(j1, j2, j3);
                        xn_next(j1, j2, j3) = h->valueAt(y);
                    }
                }
            }
            return xn_next;
        }

        NumR calWeightNormSquare(NumN norm) const override {
            NumR sum = math21_operator_norm(K, norm);
            if (norm == 1) {
            } else if (norm == 2) {
                sum = xjsquare(sum);
            } else {
                MATH21_ASSERT(0, "norm other than 1, 2 not supported currently");
            }
            math21_clip(sum);
            return sum;
        }

        NumN getType() const override {
            return cnn_type_fn_locally;
        }
    };

    class cnn_fn_conv : public cnn_fn {
    private:
        TenR K, b;
        TenR dK, db;

        NumN mk;
        NumN nk;
        NumN ms;
        NumN ns;
        NumN mt;
        NumN nt;
        NumN mpad;
        NumN npad;

        void thetaToInner(const SpaceParas &paras, TenR &W, TenR &b) {
            MATH21_ASSERT(paras.size == getThetaSize() * sizeof(NumR));
            NumN offset = 0;
            SpaceParas paras_dst;
            math21_memory_getSpace(paras, paras_dst, offset, W.volume(), sizeof(NumR));
            W.setSpace(paras_dst);

            offset = offset + W.volume();
            math21_memory_getSpace(paras, paras_dst, offset, b.volume(), sizeof(NumR));
            b.setSpace(paras_dst);
        }

    public:
        cnn_fn_conv(const cnn_fn_conv &fn) : cnn_fn(fn),
                                             mk(fn.mk), nk(fn.nk), ms(fn.ms), ns(fn.ns), mt(fn.mt), nt(fn.nt) {
            setSize();
        }

        cnn_fn_conv(const VecN &d1, const VecN &d2, NumN cnn_type_hn,
                    NumN mk, NumN nk, NumN ms, NumN ns, NumN mt, NumN nt,
                    NumB isUsingDiff) :
                cnn_fn(d1, d2, cnn_type_hn, isUsingDiff),
                mk(mk), nk(nk), ms(ms), ns(ns), mt(mt), nt(nt) {
            setSize();
        }

        cnn_fn_conv() {
        }

        cnn_fn *clone() const override {
            cnn_fn *fn = new cnn_fn_conv(*this);
            return fn;
        }

        void serialize(std::ostream &out, SerializeNumInterface &sn) override {
            cnn_fn::serialize(out, sn);
            sn.serialize(out, mk);
            sn.serialize(out, nk);
            sn.serialize(out, ms);
            sn.serialize(out, ns);
            sn.serialize(out, mt);
            sn.serialize(out, nt);
        }

        void deserialize(std::istream &in, DeserializeNumInterface &sn) override {
            cnn_fn::deserialize(in, sn);
            sn.deserialize(in, mk);
            sn.deserialize(in, nk);
            sn.deserialize(in, ms);
            sn.deserialize(in, ns);
            sn.deserialize(in, mt);
            sn.deserialize(in, nt);
            setSize();
        }


        void a() override {
        }

        NumN getThetaSize() const override {
            return K.volume() + b.volume();
        }

        void setThetaSpace(const SpaceParas &paras) override {
            thetaToInner(paras, K, b);
        }

        void setDthetaSpace(const SpaceParas &paras) override {
            MATH21_ASSERT(!dK.isEmpty() && !db.isEmpty(), "maybe you forgot to enable diff in constructor");
            thetaToInner(paras, dK, db);
        }

        void log() const override {
            if (isEmpty()) {
                return;
            }
            K.logInfo("K");
            b.logInfo("b");
        }

        void setSize() override {
            MATH21_ASSERT(mk >= 1 && nk >= 1 && ms >= 1 && ns >= 1 && mt >= 1 && nt >= 1);

            //
            if (mk > d1(2)) {
                mk = d1(2);
                m21warn("mk is too large. Adjust to ", mk);
            }
            if (nk > d1(3)) {
                nk = d1(3);
                m21warn("nk is too large. Adjust to ", nk);
            }
            if (mt > d2(2)) {
                mt = d2(2);
                m21warn("mt is too large. Adjust to ", mt);
            }
            if (nt > d2(3)) {
                nt = d2(3);
                m21warn("nt is too large. Adjust to ", nt);
            }

            NumZ mpad2 = (d2(2) - 1) * ms + mk - d1(2);
            NumZ npad2 = (d2(3) - 1) * ns + nk - d1(3);
            if (mpad2 >= 0) {
                if (xjIsEven(mpad2)) {
                    mpad = mpad2 / 2;
                } else {
                    mpad = (mpad2 - 1) / 2;
                }
            } else {
                mk = d1(2) - (d2(2) - 1) * ms;
                mpad = 0;
                m21warn("mk is too small. Adjust to ", mk);
            }
            if (npad2 >= 0) {
                if (xjIsEven(npad2)) {
                    npad = npad2 / 2;
                } else {
                    npad = (npad2 - 1) / 2;
                }
            } else {
                nk = d1(3) - (d2(3) - 1) * ns;
                npad = 0;
                m21warn("nk is too small. Adjust to ", nk);
            }

            //
            VecN index(d1.size() + d2.size());
            index = d2(1), mt, nt, d1(1), mk, nk;
            K.setSize(index);
            b.setSize(d2);

            if (isUsingDiff) {
                dK.setSize(index);
                db.setSize(d2);
            }
        }

        void derivativeValueAtTheta_and_xn_J(const TenR &xn, const TenR &dxn_next, NumR alpha) override {
            NumN j1, j2, j3, i1, i2, i3;
            NumN jj2, jj3; // index of K.
            NumZ ii2, ii3; //absolute index w.r.t. x.
            NumZ ia, ic;
            NumR val;

            // dyn
            for (j1 = 1; j1 <= d2(1); ++j1) {
                for (j2 = 1; j2 <= d2(2); ++j2) {
                    for (j3 = 1; j3 <= d2(3); ++j3) {
                        dyn(j1, j2, j3) = dxn_next(j1, j2, j3) * h->derivativeValue_using_y(xn_next(j1, j2, j3));
                    }
                }
            }

            // dW, db
            NumR val_dy;
            for (j1 = 1; j1 <= d2(1); ++j1) {
                for (j2 = 1; j2 <= d2(2); ++j2) {
                    jj2 = j2 % mt + 1;
                    for (j3 = 1; j3 <= d2(3); ++j3) {
                        jj3 = j3 % nt + 1;
                        val_dy = dyn(j1, j2, j3);
                        ia = (j2 - 1) * ms - mpad;
                        ic = (j3 - 1) * ns - npad;
                        for (i1 = 1; i1 <= d1(1); ++i1) {
                            for (i2 = 1; i2 <= mk; ++i2) {
                                ii2 = ia + i2;
                                for (i3 = 1; i3 <= nk; ++i3) {
                                    ii3 = ic + i3;
                                    if (xjIsIn(ii2, 1, d1(2)) && xjIsIn(ii3, 1, d1(3))) {
                                        val = xn(i1, ii2, ii3);
                                    } else {
                                        val = 0;
                                    }
                                    dK(j1, jj2, jj3, i1, i2, i3) = val_dy * val +
                                                                   alpha * K(j1, jj2, jj3, i1, i2, i3);
                                }
                            }
                        }
                        db(j1, j2, j3) = val_dy;
                    }
                }
            }

            // dxn, clear first, then reverse compute.
            dxn = 0;
            for (j1 = 1; j1 <= d2(1); ++j1) {
                for (j2 = 1; j2 <= d2(2); ++j2) {
                    jj2 = j2 % mt + 1;
                    for (j3 = 1; j3 <= d2(3); ++j3) {
                        jj3 = j3 % nt + 1;
                        val_dy = dyn(j1, j2, j3);
                        ia = (j2 - 1) * ms - mpad;
                        ic = (j3 - 1) * ns - npad;
                        for (i1 = 1; i1 <= d1(1); ++i1) {
                            for (i2 = 1; i2 <= mk; ++i2) {
                                ii2 = ia + i2;
                                for (i3 = 1; i3 <= nk; ++i3) {
                                    ii3 = ic + i3;
                                    if (xjIsIn(ii2, 1, d1(2)) && xjIsIn(ii3, 1, d1(3))) {
                                        NumR &val_dxn = dxn.valueAt(i1, ii2, ii3);
                                        val_dxn = val_dxn + K(j1, jj2, jj3, i1, i2, i3) * val_dy;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // clip
            math21_clip(dxn);
        }

        const TenR &valueAt(const TenR &x) override {
            NumN j1, j2, j3, i1, i2, i3;
            NumN jj2, jj3; // index of K.
            NumZ ii2, ii3; //absolute index w.r.t. x.
            NumZ ia, ic;
            NumR val;
            NumR y;
            for (j1 = 1; j1 <= d2(1); ++j1) {
                for (j2 = 1; j2 <= d2(2); ++j2) {
                    jj2 = j2 % mt + 1;
                    for (j3 = 1; j3 <= d2(3); ++j3) {
                        jj3 = j3 % nt + 1;
                        y = 0;
                        ia = (j2 - 1) * ms - mpad;
                        ic = (j3 - 1) * ns - npad;
                        for (i1 = 1; i1 <= d1(1); ++i1) {
                            for (i2 = 1; i2 <= mk; ++i2) {
                                ii2 = ia + i2;
                                for (i3 = 1; i3 <= nk; ++i3) {
                                    ii3 = ic + i3;
                                    if (xjIsIn(ii2, 1, d1(2)) && xjIsIn(ii3, 1, d1(3))) {
                                        val = x(i1, ii2, ii3);
                                    } else {
                                        val = 0;
                                    }
                                    y = y + val * K(j1, jj2, jj3, i1, i2, i3);
                                }
                            }
                        }
                        y = y + b(j1, j2, j3);
                        xn_next(j1, j2, j3) = h->valueAt(y);
                    }
                }
            }
            return xn_next;
        }

        NumR calWeightNormSquare(NumN norm) const override {
            NumR sum = math21_operator_norm(K, norm);
            if (norm == 1) {
            } else if (norm == 2) {
                sum = xjsquare(sum);
            } else {
                MATH21_ASSERT(0, "norm other than 1, 2 not supported currently");
            }
            math21_clip(sum);
            return sum;
        }

        NumN getType() const override {
            return cnn_type_fn_conv;
        }
    };


    class cnn_fn_pooling : public cnn_fn {
    private:
        Seqce <TenN> xn_argmax; //just for train max pooling
        NumN cnn_type_pooling;
        NumN mk;
        NumN nk;
        NumN ms;
        NumN ns;

        void thetaToInner(const SpaceParas &paras, TenR &W, TenR &b) {
        }

    public:
        cnn_fn_pooling(const cnn_fn_pooling &fn) : cnn_fn(fn),
                                                   cnn_type_pooling(fn.cnn_type_pooling) {
            setSize();
        }

        cnn_fn_pooling(const VecN &d1, const VecN &d2,
                       NumN cnn_type_pooling,
                       NumB isUsingDiff) :
                cnn_fn(d1, d2, cnn_type_hn_linear, isUsingDiff),
                cnn_type_pooling(cnn_type_pooling) {
            setSize();
        }

        cnn_fn_pooling() {}

        cnn_fn *clone() const override {
            cnn_fn *fn = new cnn_fn_pooling(*this);
            return fn;
        }

        void serialize(std::ostream &out, SerializeNumInterface &sn) override {
            cnn_fn::serialize(out, sn);
            sn.serialize(out, cnn_type_pooling);
        }

        void deserialize(std::istream &in, DeserializeNumInterface &sn) override {
            cnn_fn::deserialize(in, sn);
            sn.deserialize(in, cnn_type_pooling);
            setSize();
        }

        void a() override {
        }

        NumN getThetaSize() const override {
            return 0;
        }

        void setThetaSpace(const SpaceParas &paras) override {
        }

        void setDthetaSpace(const SpaceParas &paras) override {
        }

        void log() const override {
            if (isEmpty()) {
                return;
            }
            m21log("mk, nk, ms, ns", mk, nk, ms, ns);
        }

        void setSize() override {
            MATH21_ASSERT(cnn_type_pooling == cnn_type_pooling_average || cnn_type_pooling == cnn_type_pooling_max);
            MATH21_ASSERT(d1(1) == d2(1), "pooling requires the number of output features be equal to that of input");
            MATH21_ASSERT(d1(2) >= d2(2) && d1(3) >= d2(3),
                          "pooling requires output shape not be larger than input shape");

            //
            math21_operator_ml_pooling_get_mk_ms(d1(2), d1(3), d2(2), d2(3), mk, nk, ms, ns);

            // just for training
            if (isUsingDiff) {
                if (cnn_type_pooling == cnn_type_pooling_max) {
                    xn_argmax.setSize(2);
                    xn_argmax(1).setSize(d2);
                    xn_argmax(2).setSize(d2);
                }
            }
        }

        void derivativeValueAtTheta_and_xn_J(const TenR &xn, const TenR &dxn_next, NumR alpha) override {
            NumN j1, j2, j3, i1, i2, i3;
            NumZ ii2, ii3; //absolute index w.r.t. x.
            NumZ ia, ic;
//            NumR val;
            NumN ii2_max; // argmax index of x
            NumN ii3_max;
            // dyn
            for (j1 = 1; j1 <= d2(1); ++j1) {
                for (j2 = 1; j2 <= d2(2); ++j2) {
                    for (j3 = 1; j3 <= d2(3); ++j3) {
                        dyn(j1, j2, j3) = dxn_next(j1, j2, j3);
                    }
                }
            }

            NumR val_dy;
            // dxn, clear first, then reverse compute.
            dxn = 0;
            if (cnn_type_pooling == cnn_type_pooling_average) {
                for (j1 = 1; j1 <= d2(1); ++j1) {
                    for (j2 = 1; j2 <= d2(2); ++j2) {
                        for (j3 = 1; j3 <= d2(3); ++j3) {
                            val_dy = dyn(j1, j2, j3);
                            ia = (j2 - 1) * ms;
                            ic = (j3 - 1) * ns;
                            i1 = j1;
                            for (i2 = 1; i2 <= mk; ++i2) {
                                ii2 = ia + i2;
                                for (i3 = 1; i3 <= nk; ++i3) {
                                    ii3 = ic + i3;
                                    NumR &val_dxn = dxn.valueAt(i1, ii2, ii3);
                                    val_dxn = val_dxn + val_dy / (mk * nk);
                                }
                            }
                        }
                    }
                }
            } else if (cnn_type_pooling == cnn_type_pooling_max) {
                for (j1 = 1; j1 <= d2(1); ++j1) {
                    for (j2 = 1; j2 <= d2(2); ++j2) {
                        for (j3 = 1; j3 <= d2(3); ++j3) {
                            val_dy = dyn(j1, j2, j3);
                            i1 = j1;
                            ii2_max = xn_argmax(1)(j1, j2, j3);
                            ii3_max = xn_argmax(2)(j1, j2, j3);
                            NumR &val_dxn = dxn.valueAt(i1, ii2_max, ii3_max);
                            val_dxn = val_dxn + val_dy;
                        }
                    }
                }
            }

            // clip
            math21_clip(dxn);
        }

        const TenR &valueAt(const TenR &x) override {
            math21_operator_ml_pooling_valueAt(x, xn_next, cnn_type_pooling,
                                               mk, nk, ms, ns,
                                               &xn_argmax, isUsingDiff);
            return xn_next;
        }


        NumR calWeightNormSquare(NumN norm) const override {
            return 0;
        }

        NumN getType() const override {
            return cnn_type_fn_pooling;
        }
    };

    /*
     * Note 1: data sharing space
     * We hold a vector theta containing all parameters. When created, we distribute its space to inner structure.
     * This step is done only once. After this distribution, if we change the vector theta, its inner parameters will
     * change correspondingly, and vice versa.
     * */
    class cnn : public think::Operator {
    private:
        VecN d0;
        NumN N;
        NumB isUsingDiff;
        Seqce<cnn_fn *> fns;
        //theta doesn't have space currently.
        VecR theta;
        //dtheta has space currently.
        VecR dtheta;//performance theta

        void init() {
            reset();
        }

        void reset() {
            N = 0;
            isUsingDiff = 0;
        }

        // called only once
        void thetaToInner(const VecR &theta) {
            NumN offset = 0;
            for (NumN i = 1; i <= fns.size(); i++) {
                cnn_fn &fn = *fns(i);
                SpaceParas paras = theta.getSpace(offset, fn.getThetaSize(), sizeof(NumR));
                offset = offset + fn.getThetaSize();
                fn.setThetaSpace(paras);
            }
        }

        // called only once
        void dthetaToInner(const VecR &theta) {
            NumN offset = 0;
            for (NumN i = 1; i <= fns.size(); i++) {
                cnn_fn &fn = *fns(i);
                SpaceParas paras = theta.getSpace(offset, fn.getThetaSize(), sizeof(NumR));
                offset = offset + fn.getThetaSize();
                fn.setDthetaSpace(paras);
            }
        }

    public:
        cnn();

        //useDif: solve differentiation, or just use function.
        cnn(const VecN &_d0, const Seqce<cnn_config_fn *> &config_fns, NumB isUsingDiff) {
            init();
            setSize(_d0, config_fns, isUsingDiff);
        }

        void setSize(const VecN &_d0, const Seqce<cnn_config_fn *> &config_fns,
                     NumB isUsingDiff) {
            if (config_fns.isEmpty()) {
                return;
            }
            for (NumN i = 1; i <= config_fns.size(); i++) {
                MATH21_ASSERT(config_fns(i) != 0, "null config_fn " << i);
            }

            clear();
            d0.setSize(_d0.size());
            d0.assign(_d0);
            N = config_fns.size();
            this->isUsingDiff = isUsingDiff;

            fns.setSize(N);
            const VecN *d1;
            for (NumN i = 1; i <= N; i++) {
                cnn_config_fn &config_fn = *config_fns(i);
                if (i == 1) {
                    d1 = &d0;
                } else {
                    cnn_fn &fn_pre = *fns(i - 1);
                    d1 = &fn_pre.getOutputShape();
                }
                if (config_fn.getType() == cnn_type_fn_fully) {
                    fns(i) = new cnn_fn_fully(*d1, config_fn.d2, config_fn.cnn_type_hn, isUsingDiff);
                } else if (config_fn.getType() == cnn_type_fn_locally) {
                    cnn_config_fn_locally &config_fn_locally = (cnn_config_fn_locally &) config_fn;
                    fns(i) = new cnn_fn_locally(*d1, config_fn_locally.d2, config_fn_locally.cnn_type_hn,
                                                config_fn_locally.mk, config_fn_locally.nk,
                                                config_fn_locally.ms, config_fn_locally.ns,
                                                isUsingDiff);
                } else if (config_fn.getType() == cnn_type_fn_conv) {
                    cnn_config_fn_conv &config_fn_conv = (cnn_config_fn_conv &) config_fn;
                    fns(i) = new cnn_fn_conv(*d1, config_fn_conv.d2, config_fn_conv.cnn_type_hn,
                                             config_fn_conv.mk, config_fn_conv.nk,
                                             config_fn_conv.ms, config_fn_conv.ns,
                                             config_fn_conv.mt, config_fn_conv.nt,
                                             isUsingDiff);
                } else if (config_fn.getType() == cnn_type_fn_pooling) {
                    cnn_config_fn_pooling &config_fn_pooling = (cnn_config_fn_pooling &) config_fn;
                    fns(i) = new cnn_fn_pooling(*d1, config_fn_pooling.d2,
                                                config_fn_pooling.cnn_type_pooling,
                                                isUsingDiff);
                } else {
                    MATH21_ASSERT(0, "cnn_config_fn not support");
                }
            }
            ////////////// test
//            theta.setSize(getThetaSize());
//            thetaToInner(theta);
            ///////////////
            if (isUsingDiff) {
                dtheta.setSize(getThetaSize());
//                dtheta.letters();
//                dtheta.log("1");
                dthetaToInner(dtheta);
            }

            logInfo();
        }

        void setSize(const VecN &_d0, const Seqce<cnn_fn *> &_fns,
                     NumB _isUsingDiff) {
            if (_fns.isEmpty()) {
                return;
            }
            for (NumN i = 1; i <= _fns.size(); i++) {
                MATH21_ASSERT(_fns(i) != 0, "null fn " << i);
            }

            clear();
            d0.setSize(_d0.size());
            d0.assign(_d0);
            N = _fns.size();
            isUsingDiff = _isUsingDiff;

            fns.setSize(N);
//            const VecN *d1;
            for (NumN i = 1; i <= N; i++) {
                fns(i) = _fns(i)->clone();
            }
            ////////////// test
//            theta.setSize(getThetaSize());
//            thetaToInner(theta);
            ///////////////
            if (isUsingDiff) {
                dtheta.setSize(getThetaSize());
//                dtheta.letters();
//                dtheta.log("1");
                dthetaToInner(dtheta);
            }

            logInfo();
        }

        void serialize(std::ostream &out, SerializeNumInterface &sn) const {
            math21_io_serialize(out, d0, sn);
            sn.serialize(out, N);
            sn.serialize(out, (NumN) isUsingDiff);
            for (NumN i = 1; i <= N; i++) {
                sn.serialize(out, fns(i)->getType());
                fns(i)->serialize(out, sn);
            }
        }

        void deserialize(std::istream &in, DeserializeNumInterface &sn) {
            VecN d0;
            NumN N;
            NumN isUsingDiff;
            Seqce<cnn_fn *> fns;
            NumN type;

            math21_io_deserialize(in, d0, sn);
            sn.deserialize(in, N);
            sn.deserialize(in, isUsingDiff);

            fns.setSize(N);

            for (NumN i = 1; i <= N; i++) {
                sn.deserialize(in, type);
                switch (type) {
                    case cnn_type_fn_fully:
                        fns(i) = new cnn_fn_fully();
                        break;
                    case cnn_type_fn_locally:
                        fns(i) = new cnn_fn_locally();
                        break;
                    case cnn_type_fn_conv:
                        fns(i) = new cnn_fn_conv();
                        break;
                    case cnn_type_fn_pooling:
                        fns(i) = new cnn_fn_pooling();
                        break;
                    default:
                        MATH21_ASSERT_NOT_CALL(0, "UNKNOWN TYPE " << type);
                        break;
                }
                fns(i)->deserialize(in, sn);
            }
            setSize(d0, fns, (NumB) isUsingDiff);
            for (NumN i = 1; i <= fns.size(); i++) {
                delete fns(i);
            }
        }

        //used when finishing training.
        void setTheta(const VecR &_theta) {
            if (theta.isSameSize(_theta.shape()) == 0) {
                theta.setSize(_theta.shape());
            }
            theta.assign(_theta);
            thetaToInner(theta);
        }

        //used when evaluate
        NumR calWeightNormSquare(NumN norm) {
            MATH21_ASSERT(!theta.isEmpty());
            return calWeightNormSquare(theta, norm);
        }

        //used when train
        //calculate weight norm square, square can be 1, 2, p.
        NumR calWeightNormSquare(const VecR &theta, NumN norm) {
            thetaToInner(theta);
            NumR sum = 0.0;
            for (NumN n = 1; n <= N; n++) {
                cnn_fn &fn = *fns(n);
                sum = sum + fn.calWeightNormSquare(norm);
            }
            math21_clip(sum);
            return sum;
        }

        //use when predict
        const TenR &valueAt(const TenR &x) {
            MATH21_ASSERT(!theta.isEmpty());
            return valueAt(x, theta);
        }

        //used by loss function
        const TenR &valueAt(const TenR &x, const VecR &theta) {
            thetaToInner(theta);
            const TenR *xn = &x;
            const TenR *xn_next;
            for (NumN n = 1; n <= N; n++) {
                cnn_fn &fn = *fns(n);
                xn_next = &fn.valueAt(*xn);
                xn = xn_next;
            }
            return *xn_next;
        }

        const VecR &derivativeValueAtTheta(const TenR &x, const VecR &theta, Functional &f, NumR alpha) {
            MATH21_ASSERT(!dtheta.isEmpty(), "empty, you can set when create cnn");

            valueAt(x, theta);
            const TenR *xn_p;
            const TenR *dxn_next_p;
            for (NumN n = N; n >= 1; n--) {
                if (n == N) {
                    cnn_fn &fn = *fns(n);
                    dxn_next_p = (const TenR *) &f.derivativeValueAt(fn.getValue());
                } else {
                    cnn_fn &fn_next = *fns(n + 1);
                    dxn_next_p = &fn_next.get_derivativeValue_J();
                }
                if (n > 1) {
                    cnn_fn &fn_pre = *fns(n - 1);
                    xn_p = &fn_pre.getValue();
                } else {
                    xn_p = &x;
                }
                cnn_fn &fn = *fns(n);
                fn.derivativeValueAtTheta_and_xn_J(*xn_p, *dxn_next_p, alpha);
            }

            math21_clip(dtheta);
            return dtheta;
        }

        NumN getThetaSize() const {
            if (theta.isEmpty()) {
                NumN thetaSize = 0;
                for (NumN i = 1; i <= fns.size(); i++) {
                    const cnn_fn &fn = *fns(i);
                    thetaSize = thetaSize + fn.getThetaSize();
                }
                return thetaSize;
            } else {
                return theta.size();
            }
        }

        void logInfo() {
            std::cout << "cnn size " << getThetaSize() << ", architecture: " << N << " layers (from bottom to top)"
                      << "\n>> input: ";
            d0.log(0, 0, 0);
            for (NumN i = 1; i <= fns.size(); i++) {
                std::cout << ">> layer " << i << ", " << math21_type2string_cnn(fns(i)->getType()) << ", ";
                fns(i)->getOutputShape().log(0, 0, 0);
                fns(i)->log();
            }
        }

        NumB isEmpty() const {
            return N == 0 ? 1 : 0;
        }

        virtual ~cnn() {
            clear();
        }

        void clear() {
            if (!isEmpty()) {
                for (NumN i = 1; i <= fns.size(); i++) {
                    delete fns(i);
                }
                reset();
            }
        }
    };

    struct cnn_cost_class_paras {
        NumR &lambda;

        cnn_cost_class_paras(NumR &lambda) : lambda(lambda) {}
    };

    //negative log likelihood cost function with respect to theta.
    //here we use softmax function and cross-entropy together.
    class cnn_cost_class : public Functional {
    private:
        cnn &f;
        CostFunctional_class &L;

        const Seqce <TenR> &X;
        const Seqce <TenR> &Y;
        NumN start;
        NumN size;
        VecR dtheta;
        NumN Xsize;
        NumR lambda;

        NumB isUsingDiff;

        // not good
        NumN stride = 33;
    public:
        //Todo: ruffle X
        //cnn requires data to have same size, rnn doesn't require it.
        cnn_cost_class(cnn &_f,
                       CostFunctional_class &_L,
                       const Seqce <TenR> &_X,
                       const Seqce <TenR> &_Y,
                       NumN start,
                       NumN minibatch_size, NumB _isUsingDiff = 1)
                : f(_f),
                  L(_L),
                  X(_X),
                  Y(_Y),
                  isUsingDiff(_isUsingDiff) {
            MATH21_ASSERT(!X.isEmpty(), "X is empty");
            MATH21_ASSERT(X.size() == Y.size(), "X and Y must contain same number of points");
            MATH21_ASSERT(X(1).dims() == 3 && Y(1).dims() == 3, "data point must be 3-D tensor");
            for (NumN i = 1; i <= X.size(); ++i) {
                MATH21_ASSERT(X(i).isSameSize(X(1).shape()) && Y(i).isSameSize(Y(1).shape()),
                              "data points must have same size currently.\n"
                              "You can use pooling as first layer to remove the restriction.");
            }
            lambda = 0.0001;
            this->size = minibatch_size;
            this->start = start;
            Xsize = X.size();
            if (this->start > Xsize) {
                this->start = this->start % Xsize;
                if (this->start == 0) {
                    this->start = Xsize;
                }
            }
            if (isUsingDiff) {
                dtheta.setSize(getXDim());
            }
        }

        virtual ~cnn_cost_class() {}

        cnn_cost_class_paras getParas() {
            cnn_cost_class_paras paras(lambda);
            return paras;
        }

        NumR valueAt(const VecR &theta) override {
            NumR value = 0;
            NumN index = start;
            for (NumN i = 1; i <= size; i++) {
                if (index > Xsize) {
                    index = index % Xsize;
                    if (index == 0) {
                        index = Xsize;
                    }
                }
                L.setParas(Y(index));
//                value = value + L.valueAt(f.valueAt(X(index), theta));

                value = value + L.valueAt(f.valueAt(X(index), theta)) +
                        (lambda / 2) * f.calWeightNormSquare(theta, 2);
//                index++;
                index = index + stride;
//                break;
            }
            value = value / size;
            MATH21_ASSERT(!math21_check_clip(value), value << " should be clipped!");
            return value;
        }

        NumR valueAt_cnn_y(cnn &f, const TenR &cnn_y, const TenR &y, NumB isUsingPenalty = 1) {
            L.setParas(y);
            if (isUsingPenalty) {
                return L.valueAt(cnn_y) + (lambda / 2) * f.calWeightNormSquare(2);
            } else {
                return L.valueAt(cnn_y);
            }
        }

        const VecR &derivativeValueAt(const VecR &theta) override {
            NumN index = start;
            dtheta = 0;
            for (NumN i = 1; i <= size; i++) {
                if (index > Xsize) {
                    index = index % Xsize;
                    if (index == 0) {
                        index = Xsize;
                    }
                }

                L.setParas(Y(index));
                const VecR &tmp = f.derivativeValueAtTheta(X(index), theta, L, lambda);
                math21_operator_addToA(dtheta, tmp);
//                index++;
                index = index + stride;
            }
            math21_operator_linear_to(1.0 / size, dtheta);
            MATH21_ASSERT(!math21_check_clip(dtheta), "dtheta should be clipped!");
            return dtheta;
        }

        void updateParas() {
            start = start + size;
            if (start > Xsize) {
                start = start % Xsize;
                if (start == 0) {
                    start = Xsize;
                }
            }
        }

        NumN getXDim() override {
            return f.getThetaSize();
        }

        cnn &get_cnn() {
            return f;
        }
    };


    void math21_serialize_model(std::ofstream &out, cnn &f, const VecR &theta);

    void math21_deserialize_model(std::ifstream &in, cnn &f, VecR &theta);

    class OptimizationInterface_cnn : public OptimizationInterface {
    private:
        std::string name;

        void save_function(think::Optimization &opt);

    public:
        OptimizationInterface_cnn() {
            name = "model_cnn_opt.bin";
        }

        void onFinishOneInteration(think::Optimization &opt);

        void setName(const char *name) {
            this->name = name;
        }
    };


    void evaluate_cnn(cnn &f,
                      cnn_cost_class &J,
                      const Seqce <TenR> &X,
                      const Seqce <TenR> &Y, NumB isUsePr = 0, NumB isUsingPenalty = 1);

    void evaluate_cnn_error_rate(cnn &f,
                                 cnn_cost_class &J,
                                 const Seqce <TenR> &X,
                                 const Seqce <TenR> &Y, NumB isUsePr = 0, NumB isUsingPenalty = 1);

}