package com.ashwi.vectordb.model;

public record ScoredId(float distance, int id) implements Comparable<ScoredId> {
    @Override
    public int compareTo(ScoredId other) {
        int byDistance = Float.compare(distance, other.distance);
        return byDistance != 0 ? byDistance : Integer.compare(id, other.id);
    }
}
