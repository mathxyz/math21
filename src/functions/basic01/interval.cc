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
#include "interval.h"

namespace math21 {
    NumB Interval::isEmpty() const {
        if (math21_operator_num_isEqual(a, b) && !(include_a && include_b)) {
            return 1;
        } else {
            return 0;
        }
    }

// Todo: learn std_comp, remove +1 index.
    struct Compare_Interval {
        const Seqce<Interval> &intervals;

        Compare_Interval(const Seqce<Interval> &intervals0) : intervals(intervals0) {
        }

        bool operator()(int i1, int i2) {
            NumN i, j;
            i = (NumN) i1 + 1;
            j = (NumN) i2 + 1;
            if (intervals(i).left() < intervals(j).left()) {
                return 1;
            } else if (intervals(i).left() == intervals(j).left()) {
                if (intervals(i).isLeftClosed()) {
                    return 1;
                }
            }
            return 0;
        }
    };

    VecN &math21_operator_interval_sort(const Seqce<Interval> &intervals, VecN &idx) {
        Compare_Interval comp(intervals);
        return math21_operator_sort_indexes(intervals.size(), idx, comp);
    }

    NumB math21_point_isEqual(const Interval &x, const Interval &y, NumR epsilon) {
        return x.isEqual(y);
    }

    std::ostream &operator<<(std::ostream &out, const Interval &m) {
        m.log(out, 0, 1);
        return out;
    }

    NumB math21_point_isEqual(const Interval2D &x, const Interval2D &y, NumR epsilon) {
        return x.isEqual(y);
    }

    // C is the minimum interval which encloses A and B
    void math21_algebra_interval1d_union(const Interval &A, const Interval &B, Interval &C) {
        NumR a;
        NumR b;
        NumB include_a;
        NumB include_b;

        if (A.left() < B.left()) {
            a = A.left();
            include_a = A.isLeftClosed();
        } else if (B.left() < A.left()) {
            a = B.left();
            include_a = B.isLeftClosed();
        } else {
            a = B.left();
            if (A.isLeftClosed() || B.isLeftClosed()) {
                include_a = 1;
            } else {
                include_a = 0;
            }
        }

        if (A.right() > B.right()) {
            b = A.right();
            include_b = A.isRightClosed();
        } else if (B.right() > A.right()) {
            b = B.right();
            include_b = B.isRightClosed();
        } else {
            b = B.right();
            if (A.isRightClosed() || B.isRightClosed()) {
                include_b = 1;
            } else {
                include_b = 0;
            }
        }
        C.set(a, b, include_a, include_b);
    }

    // C is the minimum 2d interval which encloses A and B
    void math21_algebra_interval2d_union(const Interval2D &A, const Interval2D &B, Interval2D &C) {
        math21_algebra_interval1d_union(A(1), B(1), C.at(1));
        math21_algebra_interval1d_union(A(2), B(2), C.at(2));
    }

    NumN math21_type_get(const Interval &m) {
        return m21_type_default;
    }

    NumN math21_type_get(const Interval2D &m) {
        return m21_type_default;
    }

    void math21_io_serialize(std::ostream &out, const Interval &m, SerializeNumInterface &sn) {
        math21_io_serialize(out, m.left(), sn);
        math21_io_serialize(out, m.right(), sn);
        math21_io_serialize(out, (NumN) m.isLeftClosed(), sn);
        math21_io_serialize(out, (NumN) m.isRightClosed(), sn);
    }

    void math21_io_deserialize(std::istream &in, Interval &m, DeserializeNumInterface &sn) {
        NumR a;
        NumR b;
        NumN include_a;
        NumN include_b;
        math21_io_deserialize(in, a, sn);
        math21_io_deserialize(in, b, sn);
        math21_io_deserialize(in, include_a, sn);
        math21_io_deserialize(in, include_b, sn);
        m.set(a, b, (NumB) include_a, (NumB) include_b);
    }

    void math21_io_serialize(std::ostream &out, const Interval2D &m, SerializeNumInterface &sn) {
        math21_io_serialize(out, m(1), sn);
        math21_io_serialize(out, m(2), sn);
    }

    void math21_io_deserialize(std::istream &in, Interval2D &m, DeserializeNumInterface &sn) {
        math21_io_deserialize(in, m.at(1), sn);
        math21_io_deserialize(in, m.at(2), sn);
    }

    NumB math21_device_interval_is_include(NumR x, NumR a,
                                           NumR b,
                                           NumB include_a,
                                           NumB include_b) {
        if ((x > a || (x == a && include_a))
            && (x < b || (x == b && include_b))) {
            return 1;
        }
        return 0;
    }

    NumB math21_device_interval2d_is_include(NumR x1, NumR x2,
                                             NumR I1_a,
                                             NumR I1_b,
                                             NumB I1_include_a,
                                             NumB I1_include_b,
                                             NumR I2_a,
                                             NumR I2_b,
                                             NumB I2_include_a,
                                             NumB I2_include_b) {
        if ((x1 > I1_a || (x1 == I1_a && I1_include_a))
            && (x1 < I1_b || (x1 == I1_b && I1_include_b))
            && (x2 > I2_a || (x2 == I2_a && I2_include_a))
            && (x2 < I2_b || (x2 == I2_b && I2_include_b))) {
            return 1;
        }
        return 0;
    }
}