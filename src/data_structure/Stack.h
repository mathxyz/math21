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

#pragma once

#include "inner.h"

namespace math21 {
    template<typename StackType, typename VecType>
    void math21_data_structure_stack_type_convert_to_vector_type(const StackType &s, VecType &v) {
        if (s.isEmpty()) {
            v.clear();
            return;
        }
        v.setSize(s.size());
        NumN k = 1;
        auto iterator = s.getIterator();
        while (iterator.hasNext()) {
            v(k) = iterator.next();
            ++k;
        }
    }

    namespace data_structure {

// helper linked list class
        template<typename Item>
        class Node {
        public:
            Node() {
                next = 0;
            }

            Item item;
            Node<Item> *next;
        };

        template<typename Item>
        class ListIterator {
        private:
            Node<Item> *current;
        public:
            ListIterator() {
                current = 0;
            }

            ListIterator(const ListIterator &i) {
                assign(i);
            }

            ListIterator &operator=(const ListIterator &i) {
                assign(i);
                return *this;
            }

            ListIterator &assign(const ListIterator &i) {
                current = i.current;
                return *this;
            }

            ListIterator(Node<Item> *first) {
                current = first;
            }

            virtual ~ListIterator() {
                current = 0;
            }

            bool isEmpty() const {
                return current == 0;
            }

            bool hasNext() const {
                return current != 0;
            }

            void remove() {
                MATH21_ASSERT(0, "UnsupportedOperationException")
            }

            Item next() {
                MATH21_ASSERT(hasNext(), "NoSuchElementException")
                Item item = current->item;
                current = current->next;
                return item;
            }

            void log(const char *name = 0) const {
                log(std::cout, name);
            }

            void log(std::ostream &io, const char *name = 0) const {
                if (name) {
                    io << "ListIterator: " << name << "\n";
                    if (current) {
                        io << current << "\n";
                    } else {
                        io << "null pointer\n";
                    }
                }
                ListIterator<Item> i(*this);
                while (i.hasNext()) {
                    auto w = i.next();
                    io << w << " ";
                }
            }
        };

        template<typename Item>
        class Bag {
        private:
            Node<Item> *first;    // beginning of bag
            int n;               // number of elements in bag

            Item pop() {
                MATH21_ASSERT(!isEmpty(), "Bag underflow")
                Item item = first->item;        // save item to return
                delete first;
                first = first->next;            // delete first node
                n--;
                return item;                   // return the saved item
            }

        public:
            Bag() {
                first = 0;
                n = 0;
            }

            virtual ~Bag() {
                while (!isEmpty()) {
                    pop();
                }
            }

            bool isEmpty() const {
                return first == 0;
            }

            int size() const {
                return n;
            }

            void add(Item item) {
                Node<Item> *oldfirst = first;
                first = new Node<Item>();
                first->item = item;
                first->next = oldfirst;
                n++;
            }

            void log(const char *name = 0) const {
                log(std::cout, name);
            }

            void log(std::ostream &io, const char *name = 0) const {
                if (name) {
                    io << "Bag: " << name << "\n";
                }
                auto iterator = getIterator();
                while (iterator.hasNext()) {
                    io << iterator.next() << " ";
                }
                io << "\n";
            }

            ListIterator<Item> getIterator() const {
                ListIterator<Item> iterator(first);
                return iterator;
            }
        };


        template<typename Item>
        class Queue {
        private:
            Node<Item> *first;    // beginning of queue
            Node<Item> *last;     // end of queue
            int n;               // number of elements on queue
        public:
            Queue() {
                first = 0;
                last = 0;
                n = 0;
            }

            virtual ~Queue() {
                while (!isEmpty()) {
                    dequeue();
                }
            }

            bool isEmpty() const {
                return first == 0;
            }

            int size() const {
                return n;
            }

            Item peek() const {
                MATH21_ASSERT(!isEmpty(), "Queue underflow")
                return first->item;
            }

            void enqueue(Item item) {
                Node<Item> *oldlast = last;
                last = new Node<Item>();
                last->item = item;
                last->next = 0;
                if (isEmpty()) {
                    first = last;
                } else {
                    oldlast->next = last;
                }
                n++;
            }

            Item dequeue() {
                MATH21_ASSERT(!isEmpty(), "Queue underflow")
                Item item = first->item;
                delete first;
                first = first->next;
                n--;
                if (isEmpty()) last = 0;   // to avoid loitering
                return item;
            }

            void log(const char *name = 0) const {
                log(std::cout, name);
            }

            void log(std::ostream &io, const char *name = 0) const {
                if (name) {
                    io << "Queue: " << name << "\n";
                }
                auto iterator = getIterator();
                while (iterator.hasNext()) {
                    io << iterator.next() << " ";
                }
                io << "\n";
            }

            ListIterator<Item> getIterator() const {
                ListIterator<Item> iterator(first);
                return iterator;
            }
        };

        template<typename Item>
        class Stack {
        private:
            Node<Item> *first;     // top of stack
            int n;                // size of the stack
        public:
            Stack() {
                first = 0;
                n = 0;
            }

            virtual ~Stack() {
                while (!isEmpty()) {
                    pop();
                }
            }

            bool isEmpty() const {
                return first == 0;
            }

            int size() const {
                return n;
            }

            void push(Item item) {
                Node<Item> *oldfirst = first;
                first = new Node<Item>();
                first->item = item;
                first->next = oldfirst;
                n++;
            }

            Item pop() {
                MATH21_ASSERT(!isEmpty(), "Stack underflow")
                Item item = first->item;        // save item to return
                delete first;
                first = first->next;            // delete first node
                n--;
                return item;                   // return the saved item
            }

            Item peek() const {
                MATH21_ASSERT(!isEmpty(), "Stack underflow")
                return first->item;
            }

            void log(const char *name = 0) const {
                log(std::cout, name);
            }

            void log(std::ostream &io, const char *name = 0) const {
                if (name) {
                    io << "Stack: " << name << "\n";
                }
                auto iterator = getIterator();
                while (iterator.hasNext()) {
                    io << iterator.next() << " ";
                }
                io << "\n";
            }

            ListIterator<Item> getIterator() const {
                ListIterator<Item> iterator(first);
                return iterator;
            }
        };
    }
}