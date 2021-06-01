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

#include <fstream>
#include "files.h"
#include "inner.h"

namespace math21 {

    void test_dict() {
        Dict<NumZ, std::string> f;
        f.add(1, "love");
        f.add(-1, "I");
        f.add(-2, "Hi,");
        f.add(2, "math21!");
        f.log("f");


        NumZ x = 1;
        m21log("x", x);
        NumN id = f.has(x);
        m21log("id", id);
        if (id != 0) {
            m21log("a(id)", f.keyAtIndex(id));
            m21log("b(id)", f.valueAtIndex(id));
        }
    }

    void test_dict_serialize() {
        Dict<NumZ, std::string> f;
        f.add(1, "love");
        f.add(-1, "I");
        f.add(-2, "Hi,");
        f.add(2, "math21!");
        f.log("f");

        std::ofstream out;
        out.open("z_tmp", std::ofstream::binary);
        SerializeNumInterface_simple sn;
        math21_io_serialize(out, f, sn);
        out.close();

        Dict<NumZ, std::string> f2;
        std::ifstream in;
        in.open("z_tmp", std::ifstream::binary);
        DeserializeNumInterface_simple dsn;
        math21_io_deserialize(in, f2, dsn);
        in.close();
        f2.log("f2");
    }

    void test_algebra() {
//        test_dict();
        test_dict_serialize();
    }
}