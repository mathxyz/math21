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

namespace math21 {
    namespace _random_detail {
        struct Ran {
        private:
            NumN64 u, v, w;

            void set(NumN64 j) {
                v = 4101842887655102017LL;
                w = 1;
                u = j ^ v;
                int64();
                v = u;
                int64();
                w = v;
                int64();
            }

        public:

            // j is seed.
            Ran(NumN64 j) {
                set(j);
            }

            inline NumN64 int64() {
                u = u * 2862933555777941757LL + 7046029254386353087LL;
                v ^= v >> 17;
                v ^= v << 31;
                v ^= v >> 8;
                w = 4294957665U * (w & 0xffffffff) + (w >> 32);
                NumN64 x = u ^(u << 21);
                x ^= x >> 35;
                x ^= x << 4;
                return (x + v) ^ w;
            }

            // [0.0, 1.0]
            inline NumR64 doub() { return 5.42101086242752217E-20 * int64(); }

            inline NumN32 int32() { return (NumN32) int64(); }
        };
    }

    struct DefaultRandomEngine : public think::RandomEngine {
    private:
        _random_detail::Ran ran;
    public:
        DefaultRandomEngine(NumN seed) : ran(seed) {
        }

        virtual ~DefaultRandomEngine() {
        }

        // [0.0, 1.0]
        virtual NumR draw_0_1() { return ran.doub(); }

        virtual NumN draw_NumN() { return (NumN) ran.int64(); }
    };

    struct RanUniform : public think::Random {
    private:
        NumR a;
        NumR b;
        think::RandomEngine &engine;
    public:
        // Constructor arguments are mu, sigma, and a random sequence seed.
        RanUniform(think::RandomEngine &engine);

        RanUniform();

        virtual ~RanUniform() {
        }

        void set(NumR a, NumR b);

        void draw(NumR &x) override;

        void draw(NumN &x) override;

        void draw(NumZ &x) override;
    };

    // univariate normal distribution
    // N(mu, sig^2) = lim binomial(k|n), with n -> inf.
    // default is standard normal distribution.
    // pdf = P(x) = N(mu, sig^2) = k*exp^(-0.5*((x-mu)/sig)^2), where k = 1/(sqrt(2*pi)*sig)
    // mu is mean, sig is the standard deviation, sig^2 is variance
    // property: X ~ N(mu, sig^2), Y = aX + b => Y ~ N(a * mu + b, (a * sig)^2)
    struct RanNormal : public think::Random {
    private:
        NumR64 mu, sig;
        think::RandomEngine &engine;

        // Return a normal deviate.
        NumR64 dev();

    public:
        // a random number generator
        RanNormal(think::RandomEngine &engine);

        RanNormal();

        virtual ~RanNormal() {
        }

        // arguments are mu, sigma
        void set(NumR mu, NumR sigma);

        void draw(NumR &x) override;

        void draw(NumN &x) override;

        void draw(NumZ &x) override;

    };

    // draw from any discrete distribution.
    struct RanDiscrete : public think::Random {
    private:
        RanUniform ran;
//        const VecR &v;
        const VecR *pv;

        NumN get();

    public:
        // a random number generator
        // just set v, not get size.
        // Todo: put v to set method.
        RanDiscrete(const VecR &v, think::RandomEngine &engine);

        RanDiscrete(think::RandomEngine &engine);

        RanDiscrete();

        virtual ~RanDiscrete() {
        }

        void set(const VecR &v);

        void draw(NumR &x) override;

        void draw(NumN &x) override;

        void draw(NumZ &x) override;
    };

    // draw samples from the parameterized binomial distribution
    struct RanBinomial : public think::Random {
    private:
        RanDiscrete ran;
        VecR v;

        void init(NumN n, NumR p);

        NumN get();

    public:
        // a random number generator
        RanBinomial(NumN n, NumR p, think::RandomEngine &engine);

        RanBinomial(NumN n, NumR p);

        RanBinomial(think::RandomEngine &engine);

        RanBinomial();

        void set(NumN n, NumR p);

        virtual ~RanBinomial() {
        }

        void draw(NumR &x) override;

        void draw(NumN &x) override;

        void draw(NumZ &x) override;
    };

    //////////////////////////////
    template<typename T, template<typename> class Container>
    void math21_pr_log_histogram_Y(const Container<T> &Y, const char *name = 0, std::ostream &io = std::cout) {
        if (name) {
            io << name << "\n";
        }
        NumN nstars = 100;
        NumN n;
        n = (NumN) math21_operator_container_sum(Y, 1);
        if (n < 1) {
            return;
        }
        for (NumN i = 1; i <= Y.size(); ++i) {
            std::cout << i << ": ";
            std::cout << std::string(Y(i) * nstars / n, '*') << std::endl;
        }
    }

    template<typename T, typename S, template<typename> class Container>
    void math21_pr_log_histogram_XY(const Container<T> &X, const Container<S> &Y, const char *name = 0,
                                    std::ostream &io = std::cout) {
        if (name) {
            io << name << "\n";
        }
        MATH21_ASSERT(X.size() == Y.size())
        NumN nstars = 100;
        NumN n;
        n = (NumN) math21_operator_container_sum(Y, 1);
        if (n < 1) {
            return;
        }

        for (NumN i = 1; i <= X.size(); ++i) {
            std::cout << X(i) << ": ";
            std::cout << std::string(Y(i) * nstars / n, '*') << std::endl;
        }
    }

    void math21_pr_rand_VecSize(VecSize &v, NumN n);
}