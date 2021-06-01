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

#include "Stack.h"

namespace math21 {
    // https://algs4.cs.princeton.edu
    namespace data_structure {
        void math21_data_structure_stack_test() {
            std::string text = "to be or not to - be - - that - - - is";
            std::istringstream io;
            io.str(text);
            std::string item;
            std::string output = "";
            std::string result = "to be not that or be (2 left on stack)";
            Stack<std::string> stack;
            while (!io.eof()) {
                io >> item;
                if (item != "-") {
                    stack.push(item);
                } else if (!stack.isEmpty()) {
                    output += (stack.pop() + " ");
                }
            }
            output += "(";
            output += math21_string_to_string(stack.size()) + " left on stack)";
            MATH21_PASS(output == result, "" << output)
        }

        void math21_data_structure_queue_test() {
            std::string text = "to be or not to - be - - that - - - is";
            std::istringstream io;
            io.str(text);
            std::string item;
            std::string output = "";
            std::string result = "to be or not to be (2 left on queue)";
            Queue<std::string> queue;
            while (!io.eof()) {
                io >> item;
                if (item != "-") {
                    queue.enqueue(item);
                } else if (!queue.isEmpty()) {
                    output += (queue.dequeue() + " ");
                }
            }
            output += "(";
            output += math21_string_to_string(queue.size()) + " left on queue)";
            MATH21_PASS(output == result, "" << output)
        }

        void math21_data_structure_bag_test() {
            std::string text = "to be or not to - be - - that - - - is";
            std::istringstream io;
            io.str(text);
            std::string item;
            std::string output = "";
            std::string result = "is - - - that - - be - to not or be to \nsize of bag = 14";
            Bag<std::string> bag;
            while (!io.eof()) {
                io >> item;
                bag.add(item);
            }

            auto iterator = bag.getIterator();
            while (iterator.hasNext()) {
                output += (iterator.next() + " ");
            }
            output += ("\nsize of bag = " + math21_string_to_string(bag.size()));
            MATH21_PASS(output == result, "" << output)
        }
    }

}