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

    namespace data_structure {

        class Digraph {
        private:
            int _V;           // number of vertices in this digraph
            int _E;                 // number of edges in this digraph
            Bag<int> **adj;    // adj[v] = adjacency list for vertex v
            int *indeg;        // indegree[v] = indegree of vertex v

            void init() {
                _V = 0;
                _E = 0;
                adj = 0;
                indeg = 0;
            }
            void validateVertex(int v) const {
                if (v < 0 || v >= V()) {
                    MATH21_ASSERT(0, "vertex " << v << " is not between 0 and " << (V() - 1));
                }
            }
        public:
            Digraph() {
                init();
            }

            Digraph(int V) {
                init();
                setSize(V);
            }

            virtual ~Digraph() {
                clear();
            }

            void swap(Digraph &digraph) {
                m21_swap(_V, digraph._V);
                m21_swap(_E, digraph._E);
                m21_swap(adj, digraph.adj);
                m21_swap(indeg, digraph.indeg);
            }

            void clear() {
                if (indeg) {
                    delete[] indeg;
                    indeg = 0;
                }
                if (adj) {
                    for (int v = 0; v < _V; v++) {
                        delete adj[v];
                    }
                    delete[] adj;
                    adj = 0;
                }
                _V = 0;
                _E = 0;
            }

            bool isEmpty() const {
                return _V == 0;
            }

            // Initializes an empty digraph with V vertices.
            void setSize(int V) {
                if (!isEmpty()) {
                    clear();
                }
                MATH21_ASSERT(V >= 0, "Number of vertices in a Digraph must be nonnegative")
                this->_V = V;// V can be 0
                this->_E = 0;
                if (_V > 0) {
                    indeg = new int[_V];
                    for (int v = 0; v < _V; v++) {
                        indeg[v] = 0;
                    }
                    adj = new Bag<int> *[_V];
                    for (int v = 0; v < _V; v++) {
                        adj[v] = new Bag<int>();
                    }
                }
            }


            /**  Initializes a digraph from the specified input stream.
            * The format is the number of vertices <em>V</em>,
            * followed by the number of edges <em>E</em>,
            * followed by <em>E</em> pairs of vertices, with each entry separated by whitespace.
            * */
            Digraph(std::istream &io, DeserializeNumInterface &dsn) {
                init();
                deserialize(io, dsn);
            }

            void deserialize(std::istream &io, DeserializeNumInterface &dsn) {
                MATH21_ASSERT(!io.eof(), "argument is null")
                if (!isEmpty()) {
                    clear();
                }
                NumZ x;
                math21_io_deserialize(io, x, dsn);
//                try {
                    this->_V = x;
                    MATH21_ASSERT(_V >= 0, "number of vertices in a Digraph must be nonnegative")
                    indeg = new int[_V];
                    for (int v = 0; v < _V; v++) {
                        indeg[v] = 0;
                    }
                    adj = new Bag<int> *[_V];
                    for (int v = 0; v < _V; v++) {
                        adj[v] = new Bag<int>();
                    }
                    math21_io_deserialize(io, x, dsn);
                    int E = x;
                    MATH21_ASSERT(E >= 0, "number of edges in a Digraph must be nonnegative")
                    for (int i = 0; i < E; i++) {
                        math21_io_deserialize(io, x, dsn);
                        int v = x;
                        math21_io_deserialize(io, x, dsn);
                        int w = x;
                        addEdge(v, w);
                    }
//                } catch (const math21::fatal_error &e) {
//                    MATH21_ASSERT(0, "invalid input format in Digraph constructor")
//                }
            }

            void serialize(std::ostream &io, SerializeNumInterface &sn) const {
                MATH21_ASSERT(!io.eof(), "argument is null")
                NumZ x;
                x = _V;
                math21_io_serialize(io, x, sn);
                x = _E;
                math21_io_serialize(io, x, sn);
                for (int v = 0; v < V(); v++) {
                    // reverse so that adjacency list is in same order as original
                    Stack<int> reverse;
                    auto i = adjacent(v);
                    while (i.hasNext()) {
                        reverse.push(i.next());
                    }
                    auto j = reverse.getIterator();
                    while (j.hasNext()) {
                        x = v;
                        math21_io_serialize(io, x, sn);
                        x = j.next();
                        math21_io_serialize(io, x, sn);
                    }
                }

            }

            Digraph(const Digraph &G) {
                init();
                assign(G);
            }

            Digraph &operator=(const Digraph &G) {
                assign(G);
                return *this;
            }

            // deep copy
            Digraph &assign(const Digraph &G) {
                if(!isEmpty()){
                    clear();
                }
                this->_V = G.V();
                this->_E = G.E();
                MATH21_ASSERT(_V >= 0, "Number of vertices in a Digraph must be nonnegative")

                // update indegrees
                indeg = new int[_V];
                for (int v = 0; v < _V; v++) {
                    indeg[v] = G.indegree(v);
                }

                // update adjacency lists
                adj = new Bag<int> *[_V];
                for (int v = 0; v < _V; v++) {
                    adj[v] = new Bag<int>();
                }

                for (int v = 0; v < G.V(); v++) {
                    // reverse so that adjacency list is in same order as original
                    Stack<int> reverse;
                    auto i = G.adj[v]->getIterator();
                    while (i.hasNext()) {
                        reverse.push(i.next());
                    }
                    auto j = reverse.getIterator();
                    while (j.hasNext()) {
                        adj[v]->add(j.next());
                    }
                }
                return *this;
            }

            int V() const {
                return _V;
            }

            int E() const {
                return _E;
            }

            // Adds the directed edge v→w to this digraph.
            void addEdge(int v, int w) {
                validateVertex(v);
                validateVertex(w);
                adj[v]->add(w);
                indeg[w]++;
                _E++;
            }

            // Returns the vertices adjacent from vertex v in this digraph.
            ListIterator<int> adjacent(int v) const {
                validateVertex(v);
                return adj[v]->getIterator();
            }

            // Returns the number of directed edges incident from vertex v.
            int outdegree(int v) const {
                validateVertex(v);
                return adj[v]->size();
            }

            // Returns the number of directed edges incident to vertex v.
            int indegree(int v) const {
                validateVertex(v);
                return indeg[v];
            }

            void reverse(Digraph &reverse) const {
                if (!reverse.isEmpty()) {
                    reverse.clear();
                }
                reverse.setSize(_V);
                for (int v = 0; v < _V; v++) {
                    auto i = adjacent(v);
                    while (i.hasNext()) {
                        reverse.addEdge(i.next(), v);
                    }
                }
            }

            void log(const char *name = 0) const {
                log(std::cout, name);
            }

            void log(std::ostream &io, const char *name = 0) const {
                if (name) {
                    io << "Digraph: " << name << "\n";
                    io << V() << " vertices, " << E() << " edges\n";
                }
                for (int v = 0; v < _V; v++) {
                    io << v << ": ";
                    auto i = adjacent(v);
                    while (i.hasNext()) {
                        io << i.next() << " ";
                    }
                    io << "\n";
                }
            }
        };

        NumN math21_type_get(const Digraph &m);

        void math21_io_serialize(std::ostream &out, const Digraph &m, SerializeNumInterface &sn);

        void math21_io_deserialize(std::istream &in, Digraph &m, DeserializeNumInterface &sn);
   }
}