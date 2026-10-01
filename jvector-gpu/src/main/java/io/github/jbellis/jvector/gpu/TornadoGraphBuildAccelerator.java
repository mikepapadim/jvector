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
import java.util.Arrays;
import java.util.Locale;
import java.util.logging.Logger;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.graph.GraphBuildAccelerator;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.NodeArray;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Builds JVector graphs on an NVIDIA GPU with TornadoVM and NVIDIA cuVS. Select it with
 * {@code -Djvector.graph.accelerator=tornadovm} and run under the TornadoVM launcher.
 * <ol>
 * <li>Candidates: the approximate k-NN graph of all vectors, from cuVS NN-Descent. Datasets larger than device
 * memory are built in overlapping batches from host memory.</li>
 * <li>Diversity pruning with JVector's rule and the builder's alpha, on the GPU: the candidates' Gram matrix with
 * tensor cores (FP16 in, FP32 accumulate), then one work-group per node.</li>
 * <li>Reverse edges are merged, and the lists that grew are pruned again.</li>
 * <li>With a hierarchy, the upper layers are built the same way from exact candidates, with JVector's level
 * distribution.</li>
 * </ol>
 * Supports COSINE, DOT_PRODUCT and EUCLIDEAN, any dimension, and max degrees up to {@value #MAX_DEGREE}. For
 * datasets in a {@link MappedVectors} file, the vectors are read in place.
 */
public final class TornadoGraphBuildAccelerator implements GraphBuildAccelerator {

    private static final Logger LOG = Logger.getLogger(TornadoGraphBuildAccelerator.class.getName());

    public static final String NAME = "tornadovm";

    /** Largest max degree: lists hold {@link PruneKernels#SLOTS} entries before pruning. */
    static final int MAX_DEGREE = 64;
    /** NN-Descent candidates per node ({@code -Djvector.gpu.candidates}). */
    static final int CANDIDATES = Integer.getInteger("jvector.gpu.candidates", 96);
    /** cuVS batches for the candidate graph, or 0 to choose from the device memory ({@code -Djvector.gpu.clusters}). */
    static final long CLUSTERS = Long.getLong("jvector.gpu.clusters", 0);
    /** Batches every row joins when the candidate graph is built in batches. */
    static final long OVERLAP = 2;
    /** Device memory for the pruner besides the vectors: Gram matrices and lists of one batch, with slack. */
    private static final long PRUNE_WORKSPACE = 512L << 20;

    private static final boolean TRACE = Boolean.getBoolean("jvector.gpu.trace");

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public boolean supports(GraphIndexBuilder builder, RandomAccessVectorValues vectors, VectorSimilarityFunction similarity) {
        int n = vectors.size();
        int dim = vectors.dimension();
        if (builder.getMaxDegree() > MAX_DEGREE) {
            LOG.info(() -> "GPU build supports max degree up to " + MAX_DEGREE + ", not " + builder.getMaxDegree());
            return false;
        }
        if (n < 2 || !PackedVectors.fits(n, dim)) {
            return false;
        }
        long needed = PackedVectors.bytes(n, dim) + PRUNE_WORKSPACE;
        long budget = deviceMemoryBudget();
        if (needed > budget) {
            LOG.info(() -> String.format(Locale.ROOT, "GPU build of %d x %d vectors needs %.1f GB of device memory: run with -Dtornado.device.memory=%dGB",
                    n, dim, needed / 1e9, (needed >> 30) + 1));
            return false;
        }
        return runtimeAvailable();
    }

    private static boolean runtimeAvailable() {
        try {
            Class.forName("uk.ac.manchester.tornado.runtime.TornadoCoreRuntime");
            return uk.ac.manchester.tornado.cuvs.CuVS.isAvailable();
        } catch (Throwable t) {
            return false;
        }
    }

    /** TornadoVM's device memory budget, {@code -Dtornado.device.memory} (default 4GB). */
    static long deviceMemoryBudget() {
        String size = System.getProperty("tornado.device.memory", "4GB").trim().toUpperCase(Locale.ROOT);
        long unit = size.endsWith("GB") ? 1L << 30 : size.endsWith("MB") ? 1L << 20 : size.endsWith("KB") ? 1L << 10 : 1L;
        String digits = size.replaceAll("[^0-9]", "");
        return digits.isEmpty() ? 0 : Long.parseLong(digits) * unit;
    }

    @Override
    public void build(GraphIndexBuilder builder, RandomAccessVectorValues vectors, VectorSimilarityFunction similarity) {
        int n = vectors.size();
        int dim = vectors.dimension();
        long t0 = System.nanoTime();
        try (Arena arena = Arena.ofShared()) {
            // contiguous float32 rows: the mapping itself, or a copy (normalised for COSINE)
            MemorySegment rows;
            boolean unit;
            if (vectors instanceof MappedVectors mapped) {
                rows = mapped.segment();
                unit = false;
            } else {
                unit = similarity == VectorSimilarityFunction.COSINE;
                rows = copy(vectors, arena, unit);
            }
            Candidates.Metric metric = Candidates.Metric.of(similarity, unit);
            boolean normalise = similarity == VectorSimilarityFunction.COSINE && !unit;
            boolean euclidean = similarity == VectorSimilarityFunction.EUCLIDEAN;
            long t1 = System.nanoTime();

            long clusters = CLUSTERS > 0 ? CLUSTERS : clusters(n, dim);
            var candidates = Candidates.nnDescent(rows, n, dim, Math.min(CANDIDATES, n - 1), metric, clusters, OVERLAP);
            long t2 = System.nanoTime();
            var adjacency = GpuGraphBuilder.build(PackedVectors.of(rows, n, dim, normalise, euclidean), n, candidates,
                    builder.getMaxDegree(), builder.getAlpha());
            candidates = null; // free before the install
            long t3 = System.nanoTime();

            install(builder, 0, adjacency, null);
            adjacency = null;
            long t4 = System.nanoTime();
            boolean hierarchy = buildUpperLayers(builder, rows, n, dim, metric, normalise, euclidean, arena);
            // with a hierarchy the entry point is the node on the top layer; a single layer starts at the medoid
            builder.completeExternalBuild(hierarchy ? 0 : medoid(rows, n, dim, similarity));
            long t5 = System.nanoTime();
            if (TRACE) {
                System.out.printf("GPU build of %d x %d (%s): rows %.2f s, candidates %.2f s (%d batches), prune %.2f s, install %.2f s, upper layers %.2f s, total %.2f s%n",
                        n, dim, similarity, (t1 - t0) / 1e9, (t2 - t1) / 1e9, clusters, (t3 - t2) / 1e9, (t4 - t3) / 1e9, (t5 - t4) / 1e9, (t5 - t0) / 1e9);
            }
        }
    }

    /**
     * cuVS batches for the candidate graph: one if a single batch fits in 70% of the device memory, else enough that
     * each (holding about {@code OVERLAP / clusters} of the rows) fits. A batch needs the rows in FP32 and FP16 and
     * the NN-Descent graphs, about 1 KB per row.
     */
    static long clusters(long n, int dim) {
        long perRow = (long) dim * 6 + 1024;
        long available;
        try {
            available = (long) (uk.ac.manchester.tornado.api.TornadoExecutionPlan.getDevice(0, 0).getMaxGlobalMemory() * 0.7);
        } catch (RuntimeException e) {
            available = 8L << 30;
        }
        if (n * perRow <= available) {
            return 1;
        }
        return Math.max(OVERLAP + 1, (n * perRow * OVERLAP + available - 1) / available);
    }

    /** Copies the vectors into one row-major float32 segment, scaled to unit length if {@code normalise}. */
    private static MemorySegment copy(RandomAccessVectorValues vectors, Arena arena, boolean normalise) {
        int n = vectors.size();
        int dim = vectors.dimension();
        MemorySegment rows = arena.allocate((long) n * dim * Float.BYTES, 64);
        var supplier = vectors.threadLocalSupplier();
        ThreadLocal<float[]> buffers = ThreadLocal.withInitial(() -> new float[dim]);
        IntStream.range(0, n).parallel().forEach(i -> {
            VectorFloat<?> v = supplier.get().getVector(i);
            float[] row = buffers.get();
            for (int d = 0; d < dim; d++) {
                row[d] = v.get(d);
            }
            if (normalise) {
                float scale = (float) (1.0 / Math.sqrt(VectorUtil.dotProduct(v, v)));
                for (int d = 0; d < dim; d++) {
                    row[d] *= scale;
                }
            }
            MemorySegment.copy(row, 0, rows, ValueLayout.JAVA_FLOAT, (long) i * dim * Float.BYTES, dim);
        });
        return rows;
    }

    /** Installs the lists of one layer; {@code members[x]} is the node of row {@code x}, or row x itself if null. */
    private static void install(GraphIndexBuilder builder, int level, GpuGraphBuilder.Adjacency adjacency, int[] members) {
        int slots = PruneKernels.SLOTS;
        IntStream.range(0, adjacency.sizes().length).parallel().forEach(x -> {
            int size = adjacency.sizes()[x];
            var list = new NodeArray(Math.max(size, 1));
            for (int c = 0; c < size; c++) {
                int id = adjacency.ids()[x * slots + c];
                list.addInOrder(members == null ? id : members[id], adjacency.scores()[x * slots + c]);
            }
            builder.addGraphNodeWithNeighbors(level, members == null ? x : members[x], list);
        });
    }

    /**
     * The upper layers, built like the base layer: the nodes of each layer (levels drawn by JVector) get exact
     * candidates among themselves and the same pruning. Without a hierarchy every level is 0.
     *
     * @return whether there are upper layers
     */
    private static boolean buildUpperLayers(GraphIndexBuilder builder, MemorySegment rows, int n, int dim, Candidates.Metric metric,
                                            boolean normalise, boolean euclidean, Arena arena) {
        int[] levels = builder.sampleGraphLevels(n);
        int maxLevel = Arrays.stream(levels).max().orElse(0);
        for (int level = 1; level <= maxLevel; level++) {
            int lvl = level;
            int[] members = IntStream.range(0, n).filter(i -> levels[i] >= lvl).toArray();
            int m = members.length;
            if (m == 1) {
                builder.addGraphNodeWithNeighbors(lvl, members[0], new NodeArray(1));
                continue;
            }
            long rowBytes = (long) dim * Float.BYTES;
            MemorySegment sub = arena.allocate(m * rowBytes, 64);
            IntStream.range(0, m).parallel().forEach(x -> MemorySegment.copy(rows, members[x] * rowBytes, sub, x * rowBytes, rowBytes));
            var candidates = Candidates.exact(sub, m, dim, Math.min(CANDIDATES, m - 1), metric);
            var adjacency = GpuGraphBuilder.build(PackedVectors.of(sub, m, dim, normalise, euclidean), m, candidates,
                    builder.getMaxDegree(), builder.getAlpha());
            install(builder, lvl, adjacency, members);
        }
        return maxLevel > 0;
    }

    /** The node most similar to the centroid, the usual entry point of a single-layer graph. */
    static int medoid(MemorySegment rows, int n, int dim, VectorSimilarityFunction similarity) {
        boolean cosine = similarity == VectorSimilarityFunction.COSINE;
        double[] centroid = IntStream.range(0, n).parallel().collect(() -> new double[dim], (acc, i) -> {
            double scale = cosine ? 1.0 / Math.sqrt(dot(rows, i, i, dim)) : 1.0;
            for (int d = 0; d < dim; d++) {
                acc[d] += value(rows, i, d, dim) * scale;
            }
        }, (a, b) -> {
            for (int d = 0; d < dim; d++) {
                a[d] += b[d];
            }
        });
        for (int d = 0; d < dim; d++) {
            centroid[d] /= n;
        }
        return IntStream.range(0, n).parallel().boxed().max((a, b) -> Double.compare(closeness(rows, a, centroid, dim, similarity),
                closeness(rows, b, centroid, dim, similarity))).orElse(0);
    }

    /** Larger is closer: cosine, dot product, or negative squared distance. */
    private static double closeness(MemorySegment rows, int i, double[] c, int dim, VectorSimilarityFunction similarity) {
        double dot = 0;
        double norm = 0;
        for (int d = 0; d < dim; d++) {
            double x = value(rows, i, d, dim);
            dot += x * c[d];
            norm += x * x;
        }
        switch (similarity) {
            case EUCLIDEAN:
                return 2 * dot - norm; // -|x - c|^2 + |c|^2
            case DOT_PRODUCT:
                return dot;
            default:
                return dot / Math.sqrt(norm);
        }
    }

    private static double dot(MemorySegment rows, int i, int j, int dim) {
        double s = 0;
        for (int d = 0; d < dim; d++) {
            s += (double) value(rows, i, d, dim) * value(rows, j, d, dim);
        }
        return s;
    }

    private static float value(MemorySegment rows, int i, int d, int dim) {
        return rows.get(ValueLayout.JAVA_FLOAT_UNALIGNED, ((long) i * dim + d) * Float.BYTES);
    }
}
