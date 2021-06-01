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

#include "inner_cc.h"
#include "DepthFirstDirectedPaths.h"

namespace math21 {
    namespace data_structure {
        void DepthFirstDirectedPaths::pathTo(int v, VecZ &p) const {
            validateVertex(v);
            MATH21_ASSERT(hasPathTo(v))
            Stack<int> path;
            for (int x = v; x != s; x = edgeTo[x]) {
                path.push(x);
            }
            path.push(s);
            math21_data_structure_stack_type_convert_to_vector_type(path, p);
        }

        void DepthFirstDirectedPaths::log(std::ostream &io, const char *name) const {
            if (name) {
                io << "DepthFirstDirectedPaths: " << name << "\n";
                io << V << " vertices\n";
            }
            for (int v = 0; v < V; v++) {
                if (hasPathTo(v)) {
                    io << s << " to " << v << ":  ";
                    VecZ path;
                    pathTo(v, path);
                    for (NumN k = 1; k <= path.size(); ++k) {
                        int x = path(k);
                        if (x == s) io << x;
                        else io << "-" << x;
                    }
                    io << "\n";
                } else {
                    io << s << " to " << v << ":  not connected\n";
                }
            }
        }
    }
}