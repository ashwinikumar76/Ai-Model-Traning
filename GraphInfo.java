package com.ashwi.vectordb.model;

import java.util.List;

public record GraphInfo(
        int topLayer,
        int nodeCount,
        List<Integer> nodesPerLayer,
        List<Integer> edgesPerLayer,
        List<NodeView> nodes,
        List<EdgeView> edges
) {
    public record NodeView(int id, String metadata, String category, int maxLyr) {
    }

    public record EdgeView(int src, int dst, int lyr) {
    }
}
