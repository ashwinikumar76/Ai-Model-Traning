package com.ashwi.vectordb.algorithm;

import java.util.List;

public final class DistanceMetrics {
    private DistanceMetrics() {
    }

    public static DistanceFunction byName(String metric) {
        if ("manhattan".equalsIgnoreCase(metric)) {
            return DistanceMetrics::manhattan;
        }
        if ("euclidean".equalsIgnoreCase(metric)) {
            return DistanceMetrics::euclidean;
        }
        return DistanceMetrics::cosine;
    }

    public static float euclidean(List<Float> a, List<Float> b) {
        float sum = 0;
        for (int i = 0; i < a.size(); i++) {
            float diff = a.get(i) - b.get(i);
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }

    public static float cosine(List<Float> a, List<Float> b) {
        float dot = 0;
        float normA = 0;
        float normB = 0;
        for (int i = 0; i < a.size(); i++) {
            dot += a.get(i) * b.get(i);
            normA += a.get(i) * a.get(i);
            normB += b.get(i) * b.get(i);
        }
        if (normA < 1e-9f || normB < 1e-9f) {
            return 1.0f;
        }
        return 1.0f - dot / ((float) Math.sqrt(normA) * (float) Math.sqrt(normB));
    }

    public static float manhattan(List<Float> a, List<Float> b) {
        float sum = 0;
        for (int i = 0; i < a.size(); i++) {
            sum += Math.abs(a.get(i) - b.get(i));
        }
        return sum;
    }
}
