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
import java.util.List;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The host-side parts of the GPU build, without a GPU. */
class TestGpuBuildHost {

    private static MemorySegment rows(Arena arena, float[][] values) {
        int dim = values[0].length;
        MemorySegment rows = arena.allocate((long) values.length * dim * Float.BYTES);
        for (int i = 0; i < values.length; i++) {
            MemorySegment.copy(values[i], 0, rows, ValueLayout.JAVA_FLOAT, (long) i * dim * Float.BYTES, dim);
        }
        return rows;
    }

    @Test
    void packsToFp16PaddedToSixteenDimensions() {
        float[][] values = new float[3][20];
        for (int i = 0; i < 3; i++) {
            for (int d = 0; d < 20; d++) {
                values[i][d] = (i + 1) * 0.1f * (d - 10);
            }
        }
        try (Arena arena = Arena.ofShared()) {
            var packed = PackedVectors.of(rows(arena, values), 3, 20, false, true);
            assertEquals(32, packed.paddedDim);
            for (int i = 0; i < 3; i++) {
                float norm = 0;
                for (int d = 0; d < 32; d++) {
                    float expected = d < 20 ? Float.float16ToFloat(Float.floatToFloat16(values[i][d])) : 0;
                    assertEquals(expected, packed.get(i, d), 0);
                    norm += expected * expected;
                }
                assertEquals(norm, packed.squaredNorms[i], 1e-4f);
            }
        }
    }

    @Test
    void normalisesWhilePacking() {
        float[][] values = { { 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
        try (Arena arena = Arena.ofShared()) {
            var packed = PackedVectors.of(rows(arena, values), 1, 16, true, false);
            assertEquals(0.6f, packed.get(0, 0), 1e-3f);
            assertEquals(0.8f, packed.get(0, 1), 1e-3f);
        }
    }

    @Test
    void scoresMatchJVector() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        VectorFloat<?> a = vts.createFloatVector(new float[] { 0.6f, 0.8f });
        VectorFloat<?> b = vts.createFloatVector(new float[] { 1.0f, 0.0f });
        float dot = 0.6f;
        float squaredDistance = 0.16f + 0.64f;
        assertEquals(VectorSimilarityFunction.DOT_PRODUCT.compare(a, b), Candidates.Metric.INNER_PRODUCT.score(dot), 1e-6f);
        assertEquals(VectorSimilarityFunction.COSINE.compare(a, b), Candidates.Metric.COSINE.score(1 - dot), 1e-6f);
        assertEquals(VectorSimilarityFunction.EUCLIDEAN.compare(a, b), Candidates.Metric.EUCLIDEAN.score(squaredDistance), 1e-6f);
    }

    @Test
    void ordersScoresDescending() {
        float[] scores = { -1f, -0.5f, -0f, 0f, 0.25f, 0.9999f, 1f };
        for (int i = 0; i + 1 < scores.length; i++) {
            assertTrue(GpuGraphBuilder.descendingKey(scores[i]) >= GpuGraphBuilder.descendingKey(scores[i + 1]));
        }
    }

    @Test
    void declinesWithoutTheRuntimeOrForLargeDegrees() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        List<VectorFloat<?>> data = List.of(vts.createFloatVector(new float[] { 1, 0 }), vts.createFloatVector(new float[] { 0, 1 }));
        var vectors = new ListRandomAccessVectorValues(data, 2);
        var accelerator = new TornadoGraphBuildAccelerator();
        // no TornadoVM runtime on the test class path
        assertFalse(accelerator.supports(new GraphIndexBuilder(vectors, VectorSimilarityFunction.COSINE, 16, 100, 1.2f, 1.2f, false),
                vectors, VectorSimilarityFunction.COSINE));
        assertFalse(accelerator.supports(new GraphIndexBuilder(vectors, VectorSimilarityFunction.COSINE, 128, 100, 1.2f, 1.2f, false),
                vectors, VectorSimilarityFunction.COSINE));
    }

    @Test
    void parsesTheDeviceMemoryBudget() {
        String old = System.getProperty("tornado.device.memory");
        try {
            System.setProperty("tornado.device.memory", "22GB");
            assertEquals(22L << 30, TornadoGraphBuildAccelerator.deviceMemoryBudget());
            System.setProperty("tornado.device.memory", "512MB");
            assertEquals(512L << 20, TornadoGraphBuildAccelerator.deviceMemoryBudget());
        } finally {
            if (old == null) {
                System.clearProperty("tornado.device.memory");
            } else {
                System.setProperty("tornado.device.memory", old);
            }
        }
    }
}
