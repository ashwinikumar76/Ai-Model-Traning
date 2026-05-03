package com.ashwi.vectordb.service;

import com.ashwi.vectordb.algorithm.BruteForceIndex;
import com.ashwi.vectordb.algorithm.DistanceFunction;
import com.ashwi.vectordb.algorithm.DistanceMetrics;
import com.ashwi.vectordb.algorithm.HnswIndex;
import com.ashwi.vectordb.algorithm.KdTreeIndex;
import com.ashwi.vectordb.model.GraphInfo;
import com.ashwi.vectordb.model.ScoredId;
import com.ashwi.vectordb.model.SearchHit;
import com.ashwi.vectordb.model.VectorItem;
import org.springframework.stereotype.Service;

import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
public class VectorDbService {
    public static final int DEMO_DIMS = 16;

    private final Map<Integer, VectorItem> store = new LinkedHashMap<>();
    private final BruteForceIndex bruteForce = new BruteForceIndex();
    private final KdTreeIndex kdTree = new KdTreeIndex(DEMO_DIMS);
    private final HnswIndex hnsw = new HnswIndex();
    private int nextId = 1;

    public synchronized int insert(String metadata, String category, List<Float> embedding) {
        VectorItem item = new VectorItem(nextId++, metadata, category, List.copyOf(embedding));
        store.put(item.id(), item);
        bruteForce.insert(item);
        kdTree.insert(item);
        hnsw.insert(item, DistanceMetrics::cosine);
        return item.id();
    }

    public synchronized boolean remove(int id) {
        if (!store.containsKey(id)) {
            return false;
        }
        store.remove(id);
        bruteForce.remove(id);
        hnsw.remove(id);
        kdTree.rebuild(allSorted());
        return true;
    }

    public synchronized SearchResponse search(List<Float> query, int k, String metric, String algo) {
        DistanceFunction distance = DistanceMetrics.byName(metric);
        long start = System.nanoTime();
        List<ScoredId> raw = switch (algo) {
            case "bruteforce" -> bruteForce.knn(query, k, distance);
            case "kdtree" -> kdTree.knn(query, k, distance);
            default -> hnsw.knn(query, k, 50, distance);
        };
        long latencyUs = (System.nanoTime() - start) / 1_000;
        List<SearchHit> hits = raw.stream()
                .map(hit -> Map.entry(hit, store.get(hit.id())))
                .filter(entry -> entry.getValue() != null)
                .map(entry -> new SearchHit(
                        entry.getValue().id(),
                        entry.getValue().metadata(),
                        entry.getValue().category(),
                        entry.getKey().distance(),
                        entry.getValue().embedding()))
                .toList();
        return new SearchResponse(hits, latencyUs, algo, metric);
    }

    public synchronized BenchmarkResponse benchmark(List<Float> query, int k, String metric) {
        DistanceFunction distance = DistanceMetrics.byName(metric);
        long bf = timeUs(() -> bruteForce.knn(query, k, distance));
        long kd = timeUs(() -> kdTree.knn(query, k, distance));
        long hw = timeUs(() -> hnsw.knn(query, k, 50, distance));
        return new BenchmarkResponse(bf, kd, hw, store.size());
    }

    public synchronized List<VectorItem> all() {
        return allSorted();
    }

    public synchronized GraphInfo hnswInfo() {
        return hnsw.info();
    }

    public synchronized int size() {
        return store.size();
    }

    private List<VectorItem> allSorted() {
        return store.values().stream().sorted(Comparator.comparingInt(VectorItem::id)).toList();
    }

    private long timeUs(Runnable runnable) {
        long start = System.nanoTime();
        runnable.run();
        return (System.nanoTime() - start) / 1_000;
    }

    public record SearchResponse(List<SearchHit> results, long latencyUs, String algo, String metric) {
    }

    public record BenchmarkResponse(long bruteforceUs, long kdtreeUs, long hnswUs, int itemCount) {
    }
}
