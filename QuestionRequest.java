package com.ashwi.vectordb.dto;

public record QuestionRequest(String question, Integer k) {
    public int safeK() {
        return k == null || k < 1 ? 3 : k;
    }
}
