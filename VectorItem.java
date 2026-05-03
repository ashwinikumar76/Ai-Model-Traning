package com.ashwi.vectordb.model;

import java.util.List;

public record VectorItem(int id, String metadata, String category, List<Float> embedding) {
}
