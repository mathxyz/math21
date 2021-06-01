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

#include "files.h"

namespace math21 {
    using namespace detail_li;
    using namespace data_structure;

    void test_data_structure() {
//        li_test_heapsort();
//        li_test_heapsort_2();
//        li_test_heapsort_3();

//        li_test_insertion_sort();
//        li_test_insertion_sort_2();
//        li_test_insertion_sort_3();

//        li_test_binary_search();
//        li_test_binary_search_2();
//        li_test_binary_search_3();

//        li_test_shuffle();
//        li_test_shuffle_2();

        math21_data_structure_stack_test();
        math21_data_structure_queue_test();
        math21_data_structure_bag_test();
//        math21_data_structure_digraph_test();
        math21_data_structure_DirectedDFS_test();
//        math21_data_structure_DepthFirstDirectedPaths_test();
    }


//    /*
//     *  Run breadth-first search on a digraph.
//     *  Runs in O(E + V) time.
//     *  The {@code BreadthDirectedFirstPaths} class represents a data type for
//     *  finding shortest paths (number of edges) from a source vertex <em>s</em>
//     *  (or set of source vertices) to every other vertex in the digraph.
//     * */
//    class BreadthFirstDirectedPaths {
//    private:
//        static const int _INFINITY = NumZ32_MAX;
//        bool *marked;  // marked[v] = is there an s->v path?
//        int *edgeTo;      // edgeTo[v] = last edge on shortest s->v path
//        int *distTo;      // distTo[v] = length of shortest s->v path
//        int V;
//
//        // BFS from single source
//        void bfs(const Digraph &G, int s) {
//            Queue<int> q;
//            marked[s] = true;
//            distTo[s] = 0;
//            q.enqueue(s);
//            while (!q.isEmpty()) {
//                int v = q.dequeue();
//                for (int w : G.adj(v)) {
//                    if (!marked[w]) {
//                        edgeTo[w] = v;
//                        distTo[w] = distTo[v] + 1;
//                        marked[w] = true;
//                        q.enqueue(w);
//                    }
//                }
//            }
//        }
//
//        // BFS from multiple sources
//        void bfs(const Digraph &G, ListIterator<int> sources) {
//            Queue<int> q;
//            for (int s : sources) {
//                marked[s] = true;
//                distTo[s] = 0;
//                q.enqueue(s);
//            }
//            while (!q.isEmpty()) {
//                int v = q.dequeue();
//                for (int w : G.adj(v)) {
//                    if (!marked[w]) {
//                        edgeTo[w] = v;
//                        distTo[w] = distTo[v] + 1;
//                        marked[w] = true;
//                        q.enqueue(w);
//                    }
//                }
//            }
//        }
//        void validateVertex(int v) const {
//            if (v < 0 || v >= V) {
//                MATH21_ASSERT(0, "vertex " << v << " is not between 0 and " << (V - 1))
//            }
//        }
//
//        void validateVertices(ListIterator<int> vertices) const {
//            while (vertices.hasNext()) {
//                auto v = vertices.next();
//                if (v < 0 || v >= V) {
//                    MATH21_ASSERT(0, "vertex " << v << " is not between 0 and " << (V - 1))
//                }
//            }
//        }
//
//    public:
//
//        // Computes the shortest path from {@code s} and every other vertex in graph {@code G}.
//        BreadthFirstDirectedPaths(const Digraph &G, int s) {
//            V = G.V();
//            marked = new bool[V];
//            distTo = new int[V];
//            edgeTo = new int[V];
//            for (int v = 0; v < V; ++v) {
//                marked[v] = false;
//            }
//            for (int v = 0; v < V; v++)
//            {
//                distTo[v] = _INFINITY;
//            }
//            validateVertex(s);
//            bfs(G, s);
//        }
//
//        // Computes the shortest path from any one of the source vertices in {@code sources}
//        // to every other vertex in graph {@code G}.
//        BreadthFirstDirectedPaths(const Digraph &G, ListIterator<int> sources) {
//            V = G.V();
//            marked = new bool[V];
//            distTo = new int[V];
//            edgeTo = new int[V];
//            for (int v = 0; v < V; ++v) {
//                marked[v] = false;
//            }
//            for (int v = 0; v < V; v++)
//            {
//                distTo[v] = _INFINITY;
//            }
//            validateVertices(sources);
//            bfs(G, sources);
//        }
//
//        virtual ~BreadthFirstDirectedPaths(){
//            delete []marked;
//            delete []distTo;
//            delete []edgeTo;
//        }
//
//        bool hasPathTo(int v) const{
//            validateVertex(v);
//            return marked[v];
//        }
//
//        // Returns the number of edges in a shortest path from the source {@code s}
//        // (or sources) to vertex {@code v}?
//        int distTo(int v) const{
//            validateVertex(v);
//            return distTo[v];
//        }
//
//        ListIterator<int> pathTo(int v) {
//            validateVertex(v);
//
//            if (!hasPathTo(v)) return null;
//            Stack<int> path;
//            int x;
//            for (x = v; distTo[x] != 0; x = edgeTo[x])
//                path.push(x);
//            path.push(x);
//            return path;
//        }
//
//    };
//
//    void math21_data_structure_BreadthFirstDirectedPaths_test() {
////        std::string text = text_tinyDG;
//        std::string text = "";
//
//        std::istringstream in;
//        in.str(text);
//        DeserializeNumInterface_text dsn;
//        Digraph G(in, dsn);
//
//        int s = 3;
//
//        std::string result = "3 to 0 (2):  3->2->0\n"
//                             "3 to 1 (3):  3->2->0->1\n"
//                             "3 to 2 (1):  3->2\n"
//                             "3 to 3 (0):  3\n"
//                             "3 to 4 (2):  3->5->4\n"
//                             "3 to 5 (1):  3->5\n"
//                             "3 to 6 (-):  not connected\n"
//                             "3 to 7 (-):  not connected\n"
//                             "3 to 8 (-):  not connected\n"
//                             "3 to 9 (-):  not connected\n"
//                             "3 to 10 (-):  not connected\n"
//                             "3 to 11 (-):  not connected\n"
//                             "3 to 12 (-):  not connected\n";
//
//        for (int v = 0; v < G.V(); v++) {
//            if (bfs.hasPathTo(v)) {
//                StdOut.printf("%d to %d (%d):  ", s, v, bfs.distTo(v));
//                for (int x : bfs.pathTo(v)) {
//                    if (x == s) StdOut.print(x);
//                    else        StdOut.print("->" + x);
//                }
//                StdOut.println();
//            }
//
//            else {
//                StdOut.printf("%d to %d (-):  not connected\n", s, v);
//            }
//
//        }
//    }

}