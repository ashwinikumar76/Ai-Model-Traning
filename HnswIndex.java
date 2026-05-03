package com.ashwi.vectordb.algorithm;

import com.ashwi.vectordb.model.GraphInfo;
import com.ashwi.vectordb.model.ScoredId;
import com.ashwi.vectordb.model.VectorItem;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

public class HnswIndex {
    private final Map<Integer, Node> graph = new HashMap<>();
    private final int m;
    private final int m0;
    private final int efBuild;
    private final double levelMultiplier;
    private final Random random = new Random(42);
    private int topLayer = -1;
    private int entryPoint = -1;

    public HnswIndex() {
        this(16, 200);
    }

    public HnswIndex(int m, int efBuild) {
        this.m = m;
        this.m0 = m * 2;
        this.efBuild = efBuild;
        this.levelMultiplier = 1.0 / Math.log(m);
    }

    public void insert(VectorItem item, DistanceFunction distance) {
        int id = item.id();
        int level = randomLevel();
        graph.put(id, new Node(item, level));

        if (entryPoint == -1) {
            entryPoint = id;
            topLayer = level;
            return;
        }

        int ep = entryPoint;
        for (int layer = topLayer; layer > level; layer--) {
            if (hasLayer(ep, layer)) {
                List<ScoredId> nearest = searchLayer(item.embedding(), ep, 1, layer, distance);
                if (!nearest.isEmpty()) {
                    ep = nearest.getFirst().id();
                }
            }
        }

        for (int layer = Math.min(topLayer, level); layer >= 0; layer--) {
            List<ScoredId> candidates = searchLayer(item.embedding(), ep, efBuild, layer, distance);
            int maxConnections = layer == 0 ? m0 : m;
            List<Integer> selected = selectNeighbors(candidates, maxConnections);
            graph.get(id).neighbors.set(layer, new ArrayList<>(selected));

            for (Integer neighborId : selected) {
                Node neighbor = graph.get(neighborId);
                if (neighbor == null) {
                    continue;
                }
                ensureLayer(neighbor, layer);
                List<Integer> connections = neighbor.neighbors.get(layer);
                connections.add(id);
                if (connections.size() > maxConnections) {
                    List<ScoredId> ranked = new ArrayList<>();
                    for (Integer connectedId : connections) {
                        Node connected = graph.get(connectedId);
                        if (connected != null) {
                            ranked.add(new ScoredId(distance.apply(neighbor.item.embedding(), connected.item.embedding()), connectedId));
                        }
                    }
                    ranked.sort(Comparator.naturalOrder());
                    connections.clear();
                    selectNeighbors(ranked, maxConnections).forEach(connections::add);
                }
            }
            if (!candidates.isEmpty()) {
                ep = candidates.getFirst().id();
            }
        }

        if (level > topLayer) {
            topLayer = level;
            entryPoint = id;
        }
    }

    public List<ScoredId> knn(List<Float> query, int k, int ef, DistanceFunction distance) {
        if (entryPoint == -1) {
            return List.of();
        }
        int ep = entryPoint;
        for (int layer = topLayer; layer > 0; layer--) {
            if (hasLayer(ep, layer)) {
                List<ScoredId> nearest = searchLayer(query, ep, 1, layer, distance);
                if (!nearest.isEmpty()) {
                    ep = nearest.getFirst().id();
                }
            }
        }
        List<ScoredId> found = searchLayer(query, ep, Math.max(ef, k), 0, distance);
        return found.stream().limit(k).toList();
    }

    public void remove(int id) {
        if (!graph.containsKey(id)) {
            return;
        }
        for (Node node : graph.values()) {
            for (List<Integer> layer : node.neighbors) {
                layer.removeIf(neighborId -> neighborId == id);
            }
        }
        graph.remove(id);
        if (entryPoint == id) {
            entryPoint = graph.keySet().stream().findFirst().orElse(-1);
        }
        topLayer = graph.values().stream().mapToInt(node -> node.maxLayer).max().orElse(-1);
    }

    public GraphInfo info() {
        int layerCount = Math.max(topLayer + 1, 1);
        List<Integer> nodesPerLayer = new ArrayList<>(Collections.nCopies(layerCount, 0));
        List<Integer> edgesPerLayer = new ArrayList<>(Collections.nCopies(layerCount, 0));
        List<GraphInfo.NodeView> nodes = new ArrayList<>();
        List<GraphInfo.EdgeView> edges = new ArrayList<>();

        for (Node node : graph.values()) {
            nodes.add(new GraphInfo.NodeView(node.item.id(), node.item.metadata(), node.item.category(), node.maxLayer));
            for (int layer = 0; layer <= node.maxLayer && layer < layerCount; layer++) {
                nodesPerLayer.set(layer, nodesPerLayer.get(layer) + 1);
                if (layer < node.neighbors.size()) {
                    for (Integer neighborId : node.neighbors.get(layer)) {
                        if (node.item.id() < neighborId) {
                            edgesPerLayer.set(layer, edgesPerLayer.get(layer) + 1);
                            edges.add(new GraphInfo.EdgeView(node.item.id(), neighborId, layer));
                        }
                    }
                }
            }
        }
        return new GraphInfo(topLayer, graph.size(), nodesPerLayer, edgesPerLayer, nodes, edges);
    }

    private List<ScoredId> searchLayer(List<Float> query, int entry, int ef, int layer, DistanceFunction distance) {
        Set<Integer> visited = new HashSet<>();
        PriorityQueue<ScoredId> candidates = new PriorityQueue<>();
        PriorityQueue<ScoredId> found = new PriorityQueue<>(Comparator.reverseOrder());

        float firstDistance = distance.apply(query, graph.get(entry).item.embedding());
        ScoredId first = new ScoredId(firstDistance, entry);
        visited.add(entry);
        candidates.offer(first);
        found.offer(first);

        while (!candidates.isEmpty()) {
            ScoredId current = candidates.poll();
            if (found.size() >= ef && current.distance() > found.peek().distance()) {
                break;
            }
            Node currentNode = graph.get(current.id());
            if (currentNode == null || layer >= currentNode.neighbors.size()) {
                continue;
            }
            for (Integer neighborId : currentNode.neighbors.get(layer)) {
                if (visited.contains(neighborId) || !graph.containsKey(neighborId)) {
                    continue;
                }
                visited.add(neighborId);
                float neighborDistance = distance.apply(query, graph.get(neighborId).item.embedding());
                if (found.size() < ef || neighborDistance < found.peek().distance()) {
                    ScoredId scored = new ScoredId(neighborDistance, neighborId);
                    candidates.offer(scored);
                    found.offer(scored);
                    if (found.size() > ef) {
                        found.poll();
                    }
                }
            }
        }

        List<ScoredId> results = new ArrayList<>(found);
        results.sort(Comparator.naturalOrder());
        return results;
    }

    private List<Integer> selectNeighbors(List<ScoredId> candidates, int maxConnections) {
        return candidates.stream().limit(maxConnections).map(ScoredId::id).toList();
    }

    private int randomLevel() {
        return (int) Math.floor(-Math.log(random.nextFloat()) * levelMultiplier);
    }

    private boolean hasLayer(int id, int layer) {
        Node node = graph.get(id);
        return node != null && layer < node.neighbors.size();
    }

    private void ensureLayer(Node node, int layer) {
        while (node.neighbors.size() <= layer) {
            node.neighbors.add(new ArrayList<>());
        }
    }

    private static class Node {
        private final VectorItem item;
        private final int maxLayer;
        private final List<List<Integer>> neighbors = new ArrayList<>();

        private Node(VectorItem item, int maxLayer) {
            this.item = item;
            this.maxLayer = maxLayer;
            for (int i = 0; i <= maxLayer; i++) {
                neighbors.add(new ArrayList<>());
            }
        }
    }
}
