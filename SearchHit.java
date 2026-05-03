package com.ashwi.vectordb.model;

import java.util.List;

public record SearchHit(int id, String metadata, String category, float distance, List<Float> embedding) {
}
