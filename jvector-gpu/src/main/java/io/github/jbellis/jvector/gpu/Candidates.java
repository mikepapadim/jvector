/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.gpu;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import uk.ac.manchester.tornado.cuvs.CuVS;
import uk.ac.manchester.tornado.cuvs.CuVSAllNeighborsAlgo;
import uk.ac.manchester.tornado.cuvs.CuVSAllNeighborsOptions;
import uk.ac.manchester.tornado.cuvs.CuVSDistance;

/**
 * Candidate neighbors of every row, from cuVS all-neighbors through TornadoVM's cuVS provider, with JVector
 * similarity scores.
 */
final class Candidates {

    /** {@code k} candidates per row ({@code -1} padded), nearest first, as JVector similarity scores. */
    record TopK(int k, int[] ids, float[] scores) {
    }

    /** NN-Descent iterations ({@code -Djvector.gpu.nndIterations}); cuVS's default (20) gives the same graphs, slower. */
    static final long ITERATIONS = Long.getLong("jvector.gpu.nndIterations", 8);
    /** NN-Descent intermediate graph degree: cuVS's default (128) costs more and gives the same graphs. */
    static final long INTERMEDIATE_DEGREE = 64;

    private Candidates() {
    }

    /** How the rows are scored: the cuVS metric and the conversion of its distances to JVector scores. */
    enum Metric {
        /** unit vectors (COSINE, normalised) or DOT_PRODUCT: (1 + dot) / 2 */
        INNER_PRODUCT(CuVSDistance.INNER_PRODUCT),
        /** COSINE over vectors that may not be unit length: cuVS gives 1 - cos */
        COSINE(CuVSDistance.COSINE_EXPANDED),
        /** EUCLIDEAN: cuVS gives the squared distance */
        EUCLIDEAN(CuVSDistance.L2_EXPANDED);

        final CuVSDistance distance;

        Metric(CuVSDistance distance) {
            this.distance = distance;
        }

        static Metric of(VectorSimilarityFunction similarity, boolean unitVectors) {
            switch (similarity) {
                case EUCLIDEAN:
                    return EUCLIDEAN;
                case DOT_PRODUCT:
                    return INNER_PRODUCT;
                default:
                    return unitVectors ? INNER_PRODUCT : COSINE;
            }
        }

        /** The JVector score of a cuVS distance. */
        float score(float distance) {
            switch (this) {
                case EUCLIDEAN:
                    return 1.0f / (1.0f + Math.max(distance, 0.0f));
                case COSINE:
                    return (2.0f - distance) / 2.0f;
                default:
                    return (1.0f + distance) / 2.0f;
            }
        }
    }

    /**
     * Approximate candidates with NN-Descent.
     *
     * @param rows     {@code n x dim} row-major float32 in host memory
     * @param clusters batches of cuVS's host-dataset build (1: the whole dataset at once)
     * @param overlap  clusters each row joins when {@code clusters > 1}
     */
    static TopK nnDescent(MemorySegment rows, int n, int dim, int k, Metric metric, long clusters, long overlap) {
        var options = new CuVSAllNeighborsOptions()
                .withIntermediateGraphDegree(Math.max(INTERMEDIATE_DEGREE, k + 1))
                .withMaxIterations(ITERATIONS);
        if (clusters > 1) {
            options.withClusters(clusters, overlap);
        }
        return allNeighbors(rows, n, dim, k, metric, CuVSAllNeighborsAlgo.NN_DESCENT, options);
    }

    /** Exact candidates (for the small upper layers). */
    static TopK exact(MemorySegment rows, int n, int dim, int k, Metric metric) {
        return allNeighbors(rows, n, dim, k, metric, CuVSAllNeighborsAlgo.BRUTE_FORCE, null);
    }

    private static TopK allNeighbors(MemorySegment rows, int n, int dim, int k, Metric metric, CuVSAllNeighborsAlgo algo, CuVSAllNeighborsOptions options) {
        int kk = k + 1; // cuVS includes every row itself
        int[] ids = new int[n * k];
        float[] scores = new float[n * k];
        try (Arena arena = Arena.ofShared()) {
            MemorySegment neighbors = arena.allocate((long) n * kk * Long.BYTES, 64);
            MemorySegment distances = arena.allocate((long) n * kk * Float.BYTES, 64);
            CuVS.allNeighborsOnHost(rows, n, dim, kk, algo, metric.distance, neighbors, distances, options);
            IntStream.range(0, n).parallel().forEach(i -> {
                int c = 0;
                for (int j = 0; j < kk && c < k; j++) {
                    long id = neighbors.getAtIndex(ValueLayout.JAVA_LONG, (long) i * kk + j);
                    if (id == i || id < 0) {
                        continue;
                    }
                    ids[i * k + c] = (int) id;
                    scores[i * k + c] = metric.score(distances.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * kk + j));
                    c++;
                }
                for (; c < k; c++) {
                    ids[i * k + c] = -1;
                    scores[i * k + c] = -1;
                }
            });
        }
        return new TopK(k, ids, scores);
    }
}
