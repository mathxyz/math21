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
#include "Digraph.h"

namespace math21 {
    namespace data_structure {

        // https://algs4.cs.princeton.edu
        void math21_data_structure_digraph_test() {
            std::string text = text_tinyDG;
            std::string result =
                    "Digraph: G\n"
                    "13 vertices, 22 edges\n"
                    "0: 5 1 \n"
                    "1: \n"
                    "2: 0 3 \n"
                    "3: 5 2 \n"
                    "4: 3 2 \n"
                    "5: 4 \n"
                    "6: 9 4 8 0 \n"
                    "7: 6 9 \n"
                    "8: 6 \n"
                    "9: 11 10 \n"
                    "10: 12 \n"
                    "11: 4 12 \n"
                    "12: 9 \n";

            {
                std::istringstream in;
                in.str(text);
                DeserializeNumInterface_text dsn;
                Digraph G(in, dsn);
                std::ostringstream out;
                G.log(out, "G");
                MATH21_PASS(out.str() == result,
                            ""
                                    << "\noutput size: " << out.str().size()
                                    << "\noutput: \n" << out.str()
                                    << "\nresult size: " << result.size()
                                    << "\nresult: \n" << result
                )
                math21_io_generic_type_write_to_file(G, "G.txt", 1, 0);
            }

            {
                Digraph G;
                std::ostringstream out;
                math21_io_generic_type_read_from_file(G, "G.txt", 1, 0);
                G.log(out, "G");
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