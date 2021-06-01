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

#include <math.h>
#include <cstdio>

struct tiling_config {
public:
    int m1, n1, m2, n2;
    int mk_min, nk_min;
    float alpha_m, alpha_n;
    bool noRedundant;

    tiling_config() {
        alpha_m = 0.1f;
        alpha_n = 0.1f;
        m1 = 2000;
        n1 = 1000;
        m2 = 5;
        n2 = 4;
        mk_min = 300;
        nk_min = 300;
        noRedundant = true;
    }

    virtual ~tiling_config() {
    }
};

// results should have size at least 2+m2+n2+2 provided by caller.
// results: m2, n2, coordinates in m direction, coordinates in n direction, mk, nk.
// coordinates should subtract by one if index starts from 0.
int tiling(const tiling_config *pconfig, int *results) {
    int m1, n1, m2, n2, mk, nk, ms, ns;
    int mk_min, nk_min;
    float alpha_m, alpha_n;
    int m2_new, n2_new;
    bool noRedundant;

    const tiling_config &config = *pconfig;
    alpha_m = config.alpha_m;
    alpha_n = config.alpha_n;
    m1 = config.m1;
    n1 = config.n1;
    m2 = config.m2;
    n2 = config.n2;
    mk_min = config.mk_min;
    nk_min = config.nk_min;
    noRedundant = config.noRedundant;

    // step 1: check
    if (m2 > m1) {
        m2 = m1;
    }
    if (n2 > n1) {
        n2 = n1;
    }

//    bool DEBUG= false;
    bool DEBUG = true;
    m2_new = m2;
    n2_new = n2;
    if (DEBUG) {
        printf("input: \n"
               "m1: %d, n1: %d, m2: %d, n2: %d,\n"
               "mk_min: %d, nk_min: %d, alpha_m: %f, alpha_n: %f\n",
               m1, n1, m2, n2, mk_min, nk_min, alpha_m, alpha_n);
    }

    // step 2
    ms = (int) floor(fmin((m1 - mk_min) / (m2 - 1.0), ((double) m1 / (m2 + (alpha_m / (1 - alpha_m))))));
    if (ms == 0) {
        ms = 1;
    }
    mk = m1 - (m2 - 1) * ms;
    ns = (int) floor(fmin((n1 - nk_min) / (n2 - 1.0), ((double) n1 / (n2 + (alpha_n / (1 - alpha_n))))));
    if (ns == 0) {
        ns = 1;
    }
    nk = n1 - (n2 - 1) * ns;

    if (DEBUG) {
        printf("step 2:\n");
        printf("mk, nk, ms, ns: %d, %d, %d, %d\n", mk, nk, ms, ns);
    }

    // step 3
    if (mk * nk_min < nk * mk_min) {
        int mk_tmp;
        mk_tmp = nk * mk_min / nk_min;
        if (mk_tmp > m1) {
            printf("aspect ratio not kept in m direction\n");
        } else {
            mk = mk_tmp;
            ms = (int) ceil((m1 - mk) / (double) (m2 - 1));
            if (ms == 0) {
                ms = 1;
            }
            int x = ((m2 - 1) * ms + mk - m1) / ms;
            if (x >= 1) {
                printf("there is %d redundant tiles in m axis", x);
            }
            if (noRedundant) {
                m2_new = m2 - x;
            }
        }
    } else if (mk * nk_min > nk * mk_min) {
        int nk_tmp;
        nk_tmp = mk * nk_min / mk_min;
        if (nk_tmp > n1) {
            printf("aspect ratio not kept in n direction\n");
        } else {
            nk = nk_tmp;
            ns = (int) ceil((n1 - nk) / (double) (n2 - 1));
            if (ns == 0) {
                ns = 1;
            }
            int x = ((n2 - 1) * ns + nk - n1) / ns;
            if (x >= 1) {
                printf("there is %d redundant tiles in n axis", x);
            }
            if (noRedundant) {
                n2_new = n2 - x;
            }
        }
    }

    if (DEBUG) {
        printf("step 3:\n");
        printf("mk, nk, ms, ns: %d, %d, %d, %d\n", mk, nk, ms, ns);
        printf("m2_new: %d, n2_new: %d\n", m2_new, n2_new);
    }

    results[0] = m2_new;
    results[1] = n2_new;

    int index = 2;
    // step 4: get all tiles
    int ia, ic;
    for (int j2 = 1; j2 <= m2_new; ++j2) {
        ia = (j2 - 1) * ms;
        if (ia + mk > m1) {
            ia = ia - (ia + mk - m1);
        }
        results[index] = ia + 1;
        ++index;
    }
    for (int j3 = 1; j3 <= n2_new; ++j3) {
        ic = (j3 - 1) * ns;
        if (ic + nk > n1) {
            ic = ic - (ic + nk - n1);
        }
        results[index] = ic + 1;
        ++index;
    }
    results[index] = mk;
    ++index;
    results[index] = nk;

    if (DEBUG) {
        printf("results: \n");
        printf("%d, ", results[0]);
        printf("%d, ", results[1]);
        printf("\n");
        index = 2;
        for (int j2 = 1; j2 <= results[0]; ++j2) {
            printf("%d, ", results[index]);
            ++index;
        }
        printf("\n");
        for (int j3 = 1; j3 <= results[1]; ++j3) {
            printf("%d, ", results[index]);
            ++index;
        }
        printf("\n");
        printf("%d, ", results[index]);
        ++index;
        printf("%d, ", results[index]);
    }
    return 1;
}