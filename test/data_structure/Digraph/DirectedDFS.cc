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

#include "data.h"
#include "DirectedDFS.h"

namespace math21 {
    namespace data_structure {
        void math21_data_structure_DirectedDFS_test() {
            std::string text = text_tinyDG;

            std::istringstream in;
            in.str(text);
            DeserializeNumInterface_text dsn;
            Digraph G(in, dsn);
            {
                std::ostringstream out;
                std::string result = "0 1 2 3 4 5 ";
                int s = 2;
                DirectedDFS dfs(G, s);
                // print out vertices reachable from sources
                for (int v = 0; v < G.V(); v++) {
                    if (dfs.isMarked(v)) {
                        out << v << " ";
                    }
                }
                MATH21_PASS(out.str() == result,
                            ""
                                    << "\noutput size: " << out.str().size()
                                    << "\noutput: \n" << out.str()
                                    << "\nresult size: " << result.size()
                                    << "\nresult: \n" << result
                )
            }

            {
                std::ostringstream out;
                std::string result = "0 1 2 3 4 5 6 8 9 10 11 12 ";
                Bag<int> sources;
                sources.add(1);
                sources.add(2);
                sources.add(6);
                // multiple-source reachability
                DirectedDFS dfs(G, sources.getIterator());
                // print out vertices reachable from sources
                for (int v = 0; v < G.V(); v++) {
                    if (dfs.isMarked(v)) {
                        out << v << " ";
                    }
                }
                MATH21_PASS(out.str() == result,
                            ""
                                    << "\noutput size: " << out.str().size()
                                    << "\noutput: \n" << out.str()
                                    << "\nresult size: " << result.size()
                                    << "\nresult: \n" << result
                )
            }
        }
    }
}