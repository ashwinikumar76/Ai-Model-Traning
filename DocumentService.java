package com.ashwi.vectordb.service;

import com.ashwi.vectordb.algorithm.BruteForceIndex;
import com.ashwi.vectordb.algorithm.DistanceMetrics;
import com.ashwi.vectordb.algorithm.HnswIndex;
import com.ashwi.vectordb.model.DocItem;
import com.ashwi.vectordb.model.ScoredId;
import com.ashwi.vectordb.model.VectorItem;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
public class DocumentService {
    private final Map<Integer, DocItem> store = new LinkedHashMap<>();
    private final HnswIndex hnsw = new HnswIndex();
    private final BruteForceIndex bruteForce = new BruteForceIndex();
    private int nextId = 1;
    private int dims = 0;

    public synchronized int insert(String title, String text, List<Float> embedding) {
        if (dims == 0) {
            dims = embedding.size();
        }
        DocItem item = new DocItem(nextId++, title, text, List.copyOf(embedding));
        store.put(item.id(), item);
        VectorItem vectorItem = new VectorItem(item.id(), item.title(), "doc", item.embedding());
        hnsw.insert(vectorItem, DistanceMetrics::cosine);
        bruteForce.insert(vectorItem);
        return item.id();
    }

    public synchronized List<DocHit> search(List<Float> query, int k) {
        if (store.isEmpty()) {
            return List.of();
        }
        List<ScoredId> raw = store.size() < 10
                ? bruteForce.knn(query, k, DistanceMetrics::cosine)
                : hnsw.knn(query, k, 50, DistanceMetrics::cosine);
        List<DocHit> hits = new ArrayList<>();
        for (ScoredId scored : raw) {
            DocItem item = store.get(scored.id());
            if (item != null && scored.distance() <= 0.7f) {
                hits.add(new DocHit(scored.distance(), item));
            }
        }
        return hits;
    }

    public synchronized boolean remove(int id) {
        if (!store.containsKey(id)) {
            return false;
        }
        store.remove(id);
        hnsw.remove(id);
        bruteForce.remove(id);
        return true;
    }

    public synchronized List<DocItem> all() {
        return List.copyOf(store.values());
    }

    public synchronized int size() {
        return store.size();
    }

    public synchronized int dims() {
        return dims;
    }

    public record DocHit(float distance, DocItem item) {
    }
}
