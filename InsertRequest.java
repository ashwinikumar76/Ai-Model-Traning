package com.ashwi.vectordb.dto;

import java.util.List;

public record InsertRequest(String metadata, String category, List<Float> embedding) {
}
