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

// Implementation of the KMeans Algorithm
// reference: http://mnemstudio.org/clustering-k-means-example-1.htm

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <fstream>

using namespace std;

#include <vector>
#include <algorithm>
#include "kmeans.h"

//#define IS_PRINT 1
#define IS_PRINT 0
namespace math21 {
    namespace kmeans {
        class Cluster {
        private:
            int id_cluster;
            vector<double> central_values;
            int total_points;

        public:

            Cluster(int id_cluster, const VecR &point) {
                this->id_cluster = id_cluster;

                NumN total_values = point.size();

                for (NumN i = 1; i <= total_values; ++i)
                    central_values.push_back(point(i));
            }

            double getCentralValue(int index) {
                return central_values[index];
            }

            void setCentralValue(int index, double value) {
                central_values[index] = value;
            }

            void addTo(const VecR &point) {
                for (int j = 0; j < central_values.size(); j++) {
                    central_values[j] += point(j + 1);
                }
            }

            void addAfter() {
                if (total_points > 0) {
                    for (int j = 0; j < central_values.size(); j++) {
                        central_values[j] /= total_points;
                    }
                }
            }

            void zeroTotalPoints() {
                total_points = 0;
            }

            void addOneTotalPoints() {
                ++total_points;
            }

            // set all clusters to zero so that we can increment.
            void addBefore() {
                if (total_points > 0) {
                    for (int j = 0; j < central_values.size(); j++) {
                        central_values[j] = 0;
                    }

                }
            }

            int getTotalPoints() {
                return total_points;
            }


            int getID() {
                return id_cluster;
            }
        };

        class KMeans {
        private:
            NumN K; // number of clusters
            NumN total_values, total_points, max_iterations;
            vector<Cluster> clusters;

            // return ID of nearest center (uses euclidean distance)
            int getIDNearestCenter(const VecR &point) {
                double sum = 0.0, min_dist;
                int id_cluster_center = 1;

                for (int i = 0; i < total_values; i++) {
                    sum += pow(clusters[0].getCentralValue(i) -
                               point(i + 1), 2.0);
                }

                min_dist = sqrt(sum);

                for (int i = 1; i < K; i++) {
                    double dist;
                    sum = 0.0;

                    for (int j = 0; j < total_values; j++) {
                        sum += pow(clusters[i].getCentralValue(j) -
                                   point(j + 1), 2.0);
                    }

                    dist = sqrt(sum);

                    if (dist < min_dist) {
                        min_dist = dist;
                        id_cluster_center = i + 1;
                    }
                }
                return id_cluster_center;
            }

        public:
            KMeans(NumN K, NumN total_points, NumN total_values, NumN max_iterations) {
                this->K = K;
                this->total_points = total_points;
                this->total_values = total_values;
                this->max_iterations = max_iterations;
            }

            // data TenR is vector, label is NumN.
//            void run(vector<Point> &points, const Seqce<TenR> &data, Seqce<NumZ> &labels) {
            void
            run(const Seqce<TenR> &data, VecN &labels, Seqce<VecN> *p_points_in_clusters, VecN *p_num_in_clusters) {
                if (K > total_points)
                    return;

                vector<int> prohibited_indexes;

                // choose K distinct values for the centers of the clusters
                for (NumN i = 0; i < K; i++) {
                    while (true) {
                        int index_point = rand() % total_points;

                        if (find(prohibited_indexes.begin(), prohibited_indexes.end(),
                                 index_point) == prohibited_indexes.end()) {
                            prohibited_indexes.push_back(index_point);
                            labels(index_point + 1) = i + 1;
                            const VecR &A = data(index_point + 1);
                            Cluster cluster(i + 1, A);
                            clusters.push_back(cluster);
                            break;
                        }
                    }
                }

                int iter = 1;

                while (true) {
                    bool done = true;

                    // associates each point to the nearest center
                    for (int i = 0; i < K; i++) {
                        clusters[i].zeroTotalPoints();
                    }
                    for (int i = 0; i < total_points; i++) {
                        int id_old_cluster = labels(i + 1);
                        int id_nearest_center = getIDNearestCenter(data(i + 1));

                        if (id_old_cluster != id_nearest_center) {

//                            points[i].setCluster(id_nearest_center);
                            labels(i + 1) = id_nearest_center;
                            if (done) {
                                done = false;
                            }
                        }
                        clusters[id_nearest_center - 1].addOneTotalPoints();
                    }

                    // recalculating the center of each cluster
                    for (int i = 0; i < K; i++) {
                        clusters[i].addBefore();
                    }
                    for (int i = 0; i < total_points; i++) {
//                        int id_cluster = points[i].getCluster();
                        int id_cluster = labels(i + 1);
//                        clusters[id_cluster].addTo(points[i]);
                        clusters[id_cluster - 1].addTo(data(i + 1));
                    }
                    for (int i = 0; i < K; i++) {
                        clusters[i].addAfter();
                    }


                    if (done || iter >= max_iterations) {
                        cout << "Break in iteration " << iter << "\n\n";
                        break;
                    }

                    iter++;
                }

                if (p_num_in_clusters) {
                    VecN &num_in_clusters = *p_num_in_clusters;
                    num_in_clusters.setSize(K);
                    for (NumN i = 1; i <= K; ++i) {
                        NumN total_points_cluster = clusters[i - 1].getTotalPoints();
                        num_in_clusters(i) = total_points_cluster;
                    }
                }

                // shows elements of clusters
                if (p_points_in_clusters) {
                    less_than_id_cluster comp = less_than_id_cluster(labels);
                    std::vector<int> idx;
                    math21_operator_sort_indexes(labels.size(), idx, comp);

                    Seqce<VecN> &points_in_clusters = *p_points_in_clusters;
                    int k = 0;
                    if (points_in_clusters.size() != K) {
                        points_in_clusters.setSize(K);
                    }
                    for (int i = 0; i < K; i++) {
                        NumN total_points_cluster = clusters[i].getTotalPoints();
                        VecN &points_cluster = points_in_clusters.at(i + 1);
                        points_cluster.setSize(total_points_cluster);
                        for (NumN j = 0; j < total_points_cluster; ++j) {
                            points_cluster(j + 1) = idx[k] + 1;
                            ++k;
                        }
                    }
#if IS_PRINT
                    for (int i = 0; i < K; i++) {
                        NumN total_points_cluster = clusters[i].getTotalPoints();
                        const VecN &points_cluster = points_in_clusters(i + 1);
                        cout << "Cluster " << clusters[i].getID() << endl;
                        for (NumN j = 0; j < total_points_cluster; ++j) {
                            const VecR &point = data(points_cluster(j + 1));
                            cout << "point " << points_cluster(j + 1) << ": ";
                            for (int p = 0; p < total_values; p++) {
                                cout << point(p + 1) << " ";
                            }

                            string point_name = point.getName();

                            if (point_name != "")
                                cout << "- " << point_name;

                            cout << endl;
                            ++k;
                        }

                        cout << "Cluster values: ";

                        for (int j = 0; j < total_values; j++)
                            cout << clusters[i].getCentralValue(j) << " ";

                        cout << "\n\n";
                    }
#endif
                }
            }
        };
    }


    void ml_kmeans(const Seqce<TenR> &data, VecN &labels,
                   const ml_kmeans_config &config) {
        math21_c_seed_random_generator_by_current_time();
        kmeans::KMeans kmeans(config.K, config.total_points, config.total_values, config.max_iterations);
        kmeans.run(data, labels, 0, 0);
    }

    void ml_kmeans(const Seqce<TenR> &data, VecN &labels, VecN &num_in_clusters,
                   const ml_kmeans_config &config) {
        math21_c_seed_random_generator_by_current_time();
        kmeans::KMeans kmeans(config.K, config.total_points, config.total_values, config.max_iterations);
        kmeans.run(data, labels, 0, &num_in_clusters);
    }

    void ml_kmeans(const Seqce<TenR> &data, VecN &labels, Seqce<VecN> &points_in_clusters,
                   const ml_kmeans_config &config) {
        math21_c_seed_random_generator_by_current_time();
        kmeans::KMeans kmeans(config.K, config.total_points, config.total_values, config.max_iterations);
        kmeans.run(data, labels, &points_in_clusters, 0);
    }
}