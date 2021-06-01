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

#include <limits>
#include "eigen_sym.h"

namespace math21 {
    namespace numerical_recipes {
        void eigsrt(ShiftedVecR &d, ShiftedMatR *v = NULL) {
            NumZ k;
            NumZ n = d.size();
            for (NumZ i = 0; i < n - 1; i++) {
                NumR p = d(k = i);
                for (NumZ j = i; j < n; j++)
                    if (d(j) >= p) p = d(k = j);
                if (k != i) {
                    d(k) = d(i);
                    d(i) = p;
                    if (v != NULL)
                        for (NumZ j = 0; j < n; j++) {
                            p = (*v)(j, i);
                            (*v)(j, i) = (*v)(j, k);
                            (*v)(j, k) = p;
                        }
                }
            }
        }

        Jacobi::Jacobi(const ShiftedMatR &aa) : n(aa.nrows()), a(aa), v(n, n), d(n), nrot(0),
                                                EPS(std::numeric_limits<NumR>::epsilon()) {
            NumZ i, j, ip, iq;
            NumR tresh, theta, tau, t, sm, s, h, g, c;
            ShiftedVecR b(n), z(n);
            for (ip = 0; ip < n; ip++) {
                for (iq = 0; iq < n; iq++) v(ip, iq) = 0.0;
                v(ip, ip) = 1.0;
            }
            for (ip = 0; ip < n; ip++) {
                b(ip) = d(ip) = a(ip, ip);
                z(ip) = 0.0;
            }
            for (i = 1; i <= 50; i++) {
                sm = 0.0;
                for (ip = 0; ip < n - 1; ip++) {
                    for (iq = ip + 1; iq < n; iq++)
                        sm += abs(a(ip, iq));
                }
                if (sm == 0.0) {
                    eigsrt(d, &v);
                    return;
                }
                if (i < 4)
                    tresh = 0.2 * sm / (n * n);
                else
                    tresh = 0.0;
                for (ip = 0; ip < n - 1; ip++) {
                    for (iq = ip + 1; iq < n; iq++) {
                        g = 100.0 * abs(a(ip, iq));
                        if (i > 4 && g <= EPS * abs(d(ip)) && g <= EPS * abs(d(iq)))
                            a(ip, iq) = 0.0;
                        else if (abs(a(ip, iq)) > tresh) {
                            h = d(iq) - d(ip);
                            if (g <= EPS * abs(h))
                                t = (a(ip, iq)) / h;
                            else {
                                theta = 0.5 * h / (a(ip, iq));
                                t = 1.0 / (abs(theta) + sqrt(1.0 + theta * theta));
                                if (theta < 0.0) t = -t;
                            }
                            c = 1.0 / sqrt(1 + t * t);
                            s = t * c;
                            tau = s / (1.0 + c);
                            h = t * a(ip, iq);
                            z(ip) -= h;
                            z(iq) += h;
                            d(ip) -= h;
                            d(iq) += h;
                            a(ip, iq) = 0.0;
                            for (j = 0; j < ip; j++)
                                rot(a, s, tau, j, ip, j, iq);
                            for (j = ip + 1; j < iq; j++)
                                rot(a, s, tau, ip, j, j, iq);
                            for (j = iq + 1; j < n; j++)
                                rot(a, s, tau, ip, j, iq, j);
                            for (j = 0; j < n; j++)
                                rot(v, s, tau, j, ip, j, iq);
                            ++nrot;
                        }
                    }
                }
                for (ip = 0; ip < n; ip++) {
                    b(ip) += z(ip);
                    d(ip) = b(ip);
                    z(ip) = 0.0;
                }
            }
            MATH21_ASSERT("Too many iterations in routine jacobi");
        }

        inline void Jacobi::rot(ShiftedMatR &a, NumR s, NumR tau, NumZ i,
                                NumZ j, NumZ k, NumZ l) {
            NumR g = a(i, j);
            NumR h = a(k, l);
            a(i, j) = g - s * (h + g * tau);
            a(k, l) = h + s * (g - h * tau);
        }

        Symmeig::Symmeig(const ShiftedMatR &a, NumB yesvec) : n(a.nrows()), z(a), d(n),
                                                              e(n), yesvecs(yesvec) {
            tred2();
            tqli();
            sort();
        }

        Symmeig::Symmeig(const ShiftedVecR &dd, const ShiftedVecR &ee, NumB yesvec) :
                n(dd.size()), d(dd), e(ee), z(n, n, 0.0), yesvecs(yesvec) {
            for (NumZ i = 0; i < n; i++) z(i, i) = 1.0;
            tqli();
            sort();
        }

        void Symmeig::sort() {
            if (yesvecs)
                eigsrt(d, &z);
            else
                eigsrt(d);
        }

        void Symmeig::tred2() {
            NumZ l, k, j, i;
            NumR scale, hh, h, g, f;
            for (i = n - 1; i > 0; i--) {
                l = i - 1;
                h = scale = 0.0;
                if (l > 0) {
                    for (k = 0; k < i; k++)
                        scale += xjabs(z(i, k));
                    if (scale == 0.0)
                        e(i) = z(i, l);
                    else {
                        for (k = 0; k < i; k++) {
                            z(i, k) /= scale;
                            h += z(i, k) * z(i, k);
                        }
                        f = z(i, l);
                        g = (f >= 0.0 ? -xjsqrt(h) : xjsqrt(h));
                        e(i) = scale * g;
                        h -= f * g;
                        z(i, l) = f - g;
                        f = 0.0;
                        for (j = 0; j < i; j++) {
                            if (yesvecs)
                                z(j, i) = z(i, j) / h;
                            g = 0.0;
                            for (k = 0; k < j + 1; k++)
                                g += z(j, k) * z(i, k);
                            for (k = j + 1; k < i; k++)
                                g += z(k, j) * z(i, k);
                            e(j) = g / h;
                            f += e(j) * z(i, j);
                        }
                        hh = f / (h + h);
                        for (j = 0; j < i; j++) {
                            f = z(i, j);
                            e(j) = g = e(j) - hh * f;
                            for (k = 0; k < j + 1; k++)
                                z(j, k) -= (f * e(k) + g * z(i, k));
                        }
                    }
                } else
                    e(i) = z(i, l);
                d(i) = h;
            }
            if (yesvecs) d(0) = 0.0;
            e(0) = 0.0;
            for (i = 0; i < n; i++) {
                if (yesvecs) {
                    if (d(i) != 0.0) {
                        for (j = 0; j < i; j++) {
                            g = 0.0;
                            for (k = 0; k < i; k++)
                                g += z(i, k) * z(k, j);
                            for (k = 0; k < i; k++)
                                z(k, j) -= g * z(k, i);
                        }
                    }
                    d(i) = z(i, i);
                    z(i, i) = 1.0;
                    for (j = 0; j < i; j++) z(j, i) = z(i, j) = 0.0;
                } else {
                    d(i) = z(i, i);
                }
            }
        }

        void Symmeig::tqli() {
            NumZ m, l, iter, i, k;
            NumR s, r, p, g, f, dd, c, b;
            const NumR EPS = std::numeric_limits<NumR>::epsilon();
            for (i = 1; i < n; i++) e(i - 1) = e(i);
            e(n - 1) = 0.0;
            for (l = 0; l < n; l++) {
                iter = 0;
                do {
                    for (m = l; m < n - 1; m++) {
                        dd = xjabs(d(m)) + xjabs(d(m + 1));
                        if (xjabs(e(m)) <= EPS * dd) break;
                    }
                    if (m != l) {
//                        NumN iter_max = 30;
                        NumN iter_max = 20000;
                        if (iter++ == iter_max) {
                            MATH21_ASSERT(0, "Too many iterations in tqli"
                                    << ", iter = " << iter << ", m = " << m << ", l = " << l);
                        }
                        g = (d(l + 1) - d(l)) / (2.0 * e(l));
                        r = pythag(g, 1.0);
                        g = d(m) - d(l) + e(l) / (g + xjchangeSign(r, g));
                        s = c = 1.0;
                        p = 0.0;
                        for (i = m - 1; i >= l; i--) {
                            f = s * e(i);
                            b = c * e(i);
                            e(i + 1) = (r = pythag(f, g));
                            if (r == 0.0) {
                                d(i + 1) -= p;
                                e(m) = 0.0;
                                break;
                            }
                            s = f / r;
                            c = g / r;
                            g = d(i + 1) - p;
                            r = (d(i) - g) * s + 2.0 * c * b;
                            d(i + 1) = g + (p = s * r);
                            g = c * r - b;
                            if (yesvecs) {
                                for (k = 0; k < n; k++) {
                                    f = z(k, i + 1);
                                    z(k, i + 1) = s * z(k, i) + c * f;
                                    z(k, i) = c * z(k, i) - s * f;
                                }
                            }
                        }
                        if (r == 0.0 && i >= l) continue;
                        d(l) -= p;
                        e(l) = g;
                        e(m) = 0.0;
                    }
                } while (m != l);
            }
        }

        NumR Symmeig::pythag(NumR a, NumR b) {
            NumR absa = xjabs(a), absb = xjabs(b);
            return (absa > absb ? absa * xjsqrt(1.0 + xjsquare(absb / absa)) :
                    (absb == 0.0 ? 0.0 : absb * xjsqrt(1.0 + xjsquare(absa / absb))));
        }
    }
}