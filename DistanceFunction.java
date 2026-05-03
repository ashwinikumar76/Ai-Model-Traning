package com.ashwi.vectordb.algorithm;

import java.util.List;

@FunctionalInterface
public interface DistanceFunction {
    float apply(List<Float> a, List<Float> b);
}
