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

#include "ran_c.h"
#include "ran.h"
#include "distributions.h"

namespace math21 {
    DefaultRandomEngine global_engine(21);

    //////////////////////////////
    RanUniform::RanUniform(think::RandomEngine &engine) : engine(engine) {
        set(0, 1);
    }

    RanUniform::RanUniform() : engine(global_engine) {
        set(0, 1);
    }

    void RanUniform::set(NumR a, NumR b) {
        this->a = a;
        this->b = b;
    }

    //////////////////////////////
    RanNormal::RanNormal(think::RandomEngine &engine) : engine(engine) {
        set(0, 1);
    }

    RanNormal::RanNormal() : engine(global_engine) {
        set(0, 1);
    }

    NumR64 RanNormal::dev() {
        NumR64 u, v, x, y, q;
        do {
            u = engine.draw_0_1();
            v = 1.7156 * (engine.draw_0_1() - 0.5);
            x = u - 0.449871;
            y = m21_abs(v) + 0.386595;
            q = m21_sqr(x) + y * (0.19600 * y - 0.25472 * x);
        } while (q > 0.27597
                 && (q > 0.27846 || m21_sqr(v) > -4. * m21_log(u) * m21_sqr(u)));
        return mu + sig * v / u;
    }

    // arguments are mu, sigma
    void RanNormal::set(NumR mu, NumR sigma) {
        this->mu = mu;
        this->sig = sigma;
    }

    void RanNormal::draw(NumR &x) {
        x = (NumR) dev();
    }

    void RanNormal::draw(NumN &x) {
        MATH21_ASSERT(0)
    }

    void RanNormal::draw(NumZ &x) {
        MATH21_ASSERT(0)
    }

    //////////////////////////////
    RanDiscrete::RanDiscrete(const VecR &v, think::RandomEngine &engine)
            : pv(&v), ran(engine) {
        ran.set(0, 1);
    }

    RanDiscrete::RanDiscrete(think::RandomEngine &engine)
            : pv(0), ran(engine) {
        ran.set(0, 1);
    }

    RanDiscrete::RanDiscrete()
            : pv(0), ran(global_engine) {
        ran.set(0, 1);
    }

    NumN RanDiscrete::get() {
        MATH21_ASSERT(pv != 0)
        const VecR &v = *pv;
        NumR u;
        math21_random_draw(u, ran);
        NumN n = v.size();

        // method one
        /*NumR s = 0;
        for (NumN i = 1; i <= n; ++i) {
            s = s + v(i);
            if (u < s) {
                return i;
            }
        }
        return n;*/

        // method two
        for (NumN i = 1; i <= n; ++i) {
            u = u - v(i);
            if (u <= 0) {
                return i;
            }
        }
        return n;
    }

    void RanDiscrete::set(const VecR &v) {
        pv = &v;
    }

    void RanDiscrete::draw(NumR &x) {
        x = get();
    }

    void RanDiscrete::draw(NumN &x) {
        x = get();
    }

    void RanDiscrete::draw(NumZ &x) {
        x = get();
    }

    //////////////////////////////
    RanBinomial::RanBinomial(NumN n, NumR p, think::RandomEngine &engine) : v(n + 1), ran(v, engine) {
        init(n, p);
    }

    RanBinomial::RanBinomial(think::RandomEngine &engine) : ran(engine) {
        v.clear();
    }

    RanBinomial::RanBinomial(NumN n, NumR p) : v(n + 1), ran(v, global_engine) {
        init(n, p);
    }

    RanBinomial::RanBinomial() : ran(global_engine) {
        v.clear();
    }

    void RanBinomial::set(NumN n, NumR p) {
        v.setSize(n + 1);
        ran.set(v);
        init(n, p);
    }

    void RanBinomial::init(NumN n, NumR p) {
        MATH21_ASSERT(n >= 1 && p >= 0 && p <= 1)
        for (NumN i = 1; i <= v.size(); ++i) {
            v(i) = math21_pr_binomial(n, p, i - 1);
        }
    }

    NumN RanBinomial::get() {
        MATH21_ASSERT(!v.isEmpty())
        NumN k;
        ran.draw(k);
        return k - 1;
    }

    void RanBinomial::draw(NumR &x) {
        x = get();
    }

    void RanBinomial::draw(NumN &x) {
        x = get();
    }

    void RanBinomial::draw(NumZ &x) {
        x = get();
    }
}

using namespace math21;

void *math21_random_DefaultRandomEngine_create(NumN seed) {
    think::RandomEngine *engine = new DefaultRandomEngine(seed);
    return engine;
}

void math21_random_DefaultRandomEngine_destroy(void *engine) {
    delete (DefaultRandomEngine *) engine;
}

void *math21_random_RanUniform_create(void *engine0) {
    think::RandomEngine &engine = *(think::RandomEngine *) engine0;
    auto *ranUniform = new RanUniform(engine);
    return ranUniform;
}

void math21_random_RanUniform_destroy(void *ranUniform) {
    delete (RanUniform *) ranUniform;
}

void math21_random_RanUniform_set(void *ranUniform0, NumR a, NumR b) {
    RanUniform &ranUniform = *(RanUniform *) ranUniform0;
    ranUniform.set(a, b);
}

void *math21_random_RanNormal_create(void *engine0) {
    think::RandomEngine &engine = *(think::RandomEngine *) engine0;
    auto *ranNormal = new RanNormal(engine);
    return ranNormal;
}

void math21_random_RanNormal_destroy(void *ranNormal) {
    delete (RanNormal *) ranNormal;
}

void math21_random_RanNormal_set(void *ranNormal0, NumR mu, NumR sigma) {
    RanNormal &ranNormal = *(RanNormal *) ranNormal0;
    ranNormal.set(mu, sigma);
}

NumN32 math21_pr_rand_NumN32() {
    return ((NumN32) (rand() & 0xff) << 24) |
           ((NumN32) (rand() & 0xff) << 16) |
           ((NumN32) (rand() & 0xff) << 8) |
           ((NumN32) (rand() & 0xff) << 0);
}

NumN64 math21_pr_rand_NumN64() {
    return ((NumN64) (rand() & 0xff) << 56) |
           ((NumN64) (rand() & 0xff) << 48) |
           ((NumN64) (rand() & 0xff) << 40) |
           ((NumN64) (rand() & 0xff) << 32) |
           ((NumN64) (rand() & 0xff) << 24) |
           ((NumN64) (rand() & 0xff) << 16) |
           ((NumN64) (rand() & 0xff) << 8) |
           ((NumN64) (rand() & 0xff) << 0);
}

size_t math21_pr_rand_size_t() {
    if (math21_type_size_t_is_4_bytes()) {
        return (size_t) math21_pr_rand_NumN32();
    } else {
        return (size_t) math21_pr_rand_NumN64();
    }
}

NumSize math21_pr_rand_NumSize() {
    if (math21_type_NumSize_is_4_bytes()) {
        return (NumSize) math21_pr_rand_NumN32();
    } else {
        return (NumSize) math21_pr_rand_NumN64();
    }
}

namespace math21 {

    void math21_pr_rand_VecSize(VecSize &v, NumN n) {
        v.setSize(n);
        auto *x = v.getDataAddress();
        int j;
        for (j = 0; j < n; ++j) {
            x[j] = math21_pr_rand_NumSize();
        }
    }

}