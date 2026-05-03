package com.ashwi.vectordb.algorithm;

import com.ashwi.vectordb.model.ScoredId;
import com.ashwi.vectordb.model.VectorItem;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public class KdTreeIndex {
    private final int dims;
    private Node root;

    public KdTreeIndex(int dims) {
        this.dims = dims;
    }

    public void insert(VectorItem item) {
        root = insert(root, item, 0);
    }

    public List<ScoredId> knn(List<Float> query, int k, DistanceFunction distance) {
        PriorityQueue<ScoredId> heap = new PriorityQueue<>(Comparator.reverseOrder());
        knn(root, query, k, 0, distance, heap);
        List<ScoredId> results = new ArrayList<>(heap);
        results.sort(Comparator.naturalOrder());
        return results;
    }

    public void rebuild(List<VectorItem> items) {
        root = null;
        for (VectorItem item : items) {
            insert(item);
        }
    }

    private Node insert(Node node, VectorItem item, int depth) {
        if (node == null) {
            return new Node(item);
        }
        int axis = depth % dims;
        if (item.embedding().get(axis) < node.item.embedding().get(axis)) {
            node.left = insert(node.left, item, depth + 1);
        } else {
            node.right = insert(node.right, item, depth + 1);
        }
        return node;
    }

    private void knn(Node node, List<Float> query, int k, int depth, DistanceFunction distance,
                     PriorityQueue<ScoredId> heap) {
        if (node == null) {
            return;
        }

        float currentDistance = distance.apply(query, node.item.embedding());
        if (heap.size() < k || currentDistance < heap.peek().distance()) {
            heap.offer(new ScoredId(currentDistance, node.item.id()));
            if (heap.size() > k) {
                heap.poll();
            }
        }

        int axis = depth % dims;
        float diff = query.get(axis) - node.item.embedding().get(axis);
        Node closer = diff < 0 ? node.left : node.right;
        Node farther = diff < 0 ? node.right : node.left;

        knn(closer, query, k, depth + 1, distance, heap);
        if (heap.size() < k || Math.abs(diff) < heap.peek().distance()) {
            knn(farther, query, k, depth + 1, distance, heap);
        }
    }

    private static class Node {
        private final VectorItem item;
        private Node left;
        private Node right;

        private Node(VectorItem item) {
            this.item = item;
        }
    }
}
