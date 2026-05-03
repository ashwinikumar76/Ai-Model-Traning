package com.ashwi.vectordb.algorithm;

import com.ashwi.vectordb.model.ScoredId;
import com.ashwi.vectordb.model.VectorItem;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class BruteForceIndex {
    private final List<VectorItem> items = new ArrayList<>();

    public void insert(VectorItem item) {
        items.add(item);
    }

    public List<ScoredId> knn(List<Float> query, int k, DistanceFunction distance) {
        return items.stream()
                .map(item -> new ScoredId(distance.apply(query, item.embedding()), item.id()))
                .sorted()
                .limit(k)
                .toList();
    }

    public void remove(int id) {
        items.removeIf(item -> item.id() == id);
    }

    public void rebuild(List<VectorItem> values) {
        items.clear();
        items.addAll(values.stream().sorted(Comparator.comparingInt(VectorItem::id)).toList());
    }
}
