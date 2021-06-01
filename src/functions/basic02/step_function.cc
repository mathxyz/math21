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

#include "inner.h"
#include "step_function.h"

namespace math21 {
    void
    math21_function_interval_to_smaller(const Seqce<Interval> &Is, const Seqce<Interval> &Js, Seqce<Interval> &Ls) {
        SetR S;
        for (NumN i = 1; i <= Is.size(); ++i) {
            S.add(Is(i).left());
            S.add(Is(i).right());
        }
        for (NumN i = 1; i <= Js.size(); ++i) {
            S.add(Js(i).left());
            S.add(Js(i).right());
        }
        S.sort();
        Ls.clear();
        if (S.isEmpty()) {
            return;
        }
        Interval interval;
        NumN i;
        for (i = 1; i < S.size(); ++i) {
            interval.set(S(i), S(i), 1, 1);
            Ls.push(interval);
            interval.set(S(i), S(i + 1), 0, 0);
            Ls.push(interval);
        }
        interval.set(S(i), S(i), 1, 1);
        Ls.push(interval);
    }

    void math21_function_step_add(const Step &f, const Step &g, Step &h) {
        f.add(g, h);
    }

    void Step::add(const Step &g, Step &h) const {
        const Step &f = *this;
        MATH21_ASSERT(f.isStandard() && g.isStandard())

        if (f.isZero()) {
            g.copyTo(h);
            return;
        }
        if (g.isZero()) {
            f.copyTo(h);
            return;
        }

        Seqce<Interval> intervals;
        math21_function_interval_to_smaller(f.intervals, g.intervals, intervals);
        h.setSize(intervals.size());
        NumR x, y;
        for (NumN i = 1; i <= intervals.size(); ++i) {
            const Interval &I = intervals.operator()(i);
            x = (I.left() + I.right()) / 2;
            y = f.valueAt(x) + g.valueAt(x);
            h.set(i, y, I);
        }
        h.toStandard_no_overlap();
    }

    void Step::remove_overlap() {
        if (size() == 0) {
            return;
        }
        Step g;
        g.push(values(1), intervals(1));
        g.toStandard();
        Step h, g_tmp;
        for (NumN i = 2; i <= size(); ++i) {
            h.clear();
            h.push(values(i), intervals(i));
            h.toStandard();
            g.add(h, g_tmp);
            g_tmp.swap(g);
        }
        g.swap(*this);
    }

    void Step::toStandard() {
        if (isZero() || (size() == 1 && values(1) == 0)) {
            clear();
            setStandard(1);
            return;
        } else if (size() == 1) {
            setStandard(1);
            return;
        }
        remove_overlap();
        toStandard_no_overlap();
    }

    void Step::toStandard_no_overlap() {
        if (isStandard()) {
            return;
        }
        if (size() == 0) {
            return;
        }

        // sort intervals
        if (size() > 1) {
            VecN idx;
            math21_operator_interval_sort(intervals, idx);
            if (math21_operator_isContinuousIntegers(idx) == 0) {
                Step step;
                step.setSize(size());
                for (NumN i = 1; i <= size(); ++i) {
                    step.values(i) = values(idx(i));
                    step.intervals(i) = intervals(idx(i));
                }
                step.copyTo(*this);
            }
        }

        NumN k = 1;
        NumN i;
        Step step;
        Interval interval;
        while (k <= size()) {
            while (k <= size() && values(k) == 0) {
                ++k;
            }
            if (k > size()) {
                break;
            }
            if (k == size()) {
                step.push(values(k), intervals(k));
                break;
            }
            i = k;
            while (k < size() && values(k) == values(k + 1)
                   && intervals(k).right() == intervals(k + 1).left()
                   && (intervals(k).isRightClosed() || intervals(k + 1).isLeftClosed())) {
                ++k;
            }
            interval.set(intervals(i).left(), intervals(k).right(), intervals(i).isLeftClosed(),
                         intervals(k).isRightClosed());
            NumR val = values(i);
            step.push(val, interval);
            ++k;
        }
        step.copyTo(*this);
        setStandard(1);
    }

    void get_step_function_01(Step &f) {
        f.clear();
        f.push(0, Interval(0, 1, 1, 1));
        f.push(2, Interval(1, 2, 0, 1));
        f.push(4, Interval(-10, -8, 1, 0));
        f.push(4, Interval(-8, -2, 1, 0));
//    f.log("f1");
    }

    void get_step_function_02(Step &f) {
        f.clear();
        f.push(1, Interval(0, 1, 1, 1));
        f.push(2, Interval(1, 2, 0, 1));
        f.push(4, Interval(-10, -8, 0, 1));
//    f.log("f2");
    }

    void get_step_function_03(Step &f) {
        f.clear();
        f.push(1, Interval(0, 1, 1, 1));
        f.push(1, Interval(1, 2, 1, 1));
//    f.log("f3");
    }

    void test_step_function_01() {
        Step f1;
        get_step_function_01(f1);
        f1.toStandard();
        f1.log("f1_standard");

        Step f2;
        get_step_function_02(f2);
        f2.toStandard();
        f2.log("f2_standard");

        Step f;
        math21_function_step_add(f1, f2, f);

        f.log("f");
    }

    void test_step_function_02() {
        Step f1;
        get_step_function_03(f1);
        f1.toStandard();
        f1.log("f1_standard");

        Step f2;
        get_step_function_03(f2);
        f2.toStandard();
        f2.log("f2_standard");

        Step f;
        math21_function_step_add(f1, f2, f);

        f.log("f");
    }

    void test_step_function() {
//    test_step_function_01();
        test_step_function_02();
    }

}