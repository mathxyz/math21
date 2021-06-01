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
#include "DepthFirstDirectedPaths.h"
namespace math21 {
    namespace data_structure {
        void math21_data_structure_DepthFirstDirectedPaths_test() {
            std::string text = text_tinyDG;

            std::istringstream in;
            in.str(text);
            DeserializeNumInterface_text dsn;
            Digraph G(in, dsn);

            {
                int s = 3;
                DepthFirstDirectedPaths dfs(G, s);
                std::ostringstream out;
                dfs.log(out);

                std::string result = "3 to 0:  3-5-4-2-0\n"
                                     "3 to 1:  3-5-4-2-0-1\n"
                                     "3 to 2:  3-5-4-2\n"
                                     "3 to 3:  3\n"
                                     "3 to 4:  3-5-4\n"
                                     "3 to 5:  3-5\n"
                                     "3 to 6:  not connected\n"
                                     "3 to 7:  not connected\n"
                                     "3 to 8:  not connected\n"
                                     "3 to 9:  not connected\n"
                                     "3 to 10:  not connected\n"
                                     "3 to 11:  not connected\n"
                                     "3 to 12:  not connected\n";
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