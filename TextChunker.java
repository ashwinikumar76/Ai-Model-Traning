package com.ashwi.vectordb.service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class TextChunker {
    private TextChunker() {
    }

    public static List<String> chunk(String text, int chunkWords, int overlapWords) {
        List<String> words = Arrays.stream(text.trim().split("\\s+"))
                .filter(word -> !word.isBlank())
                .toList();
        if (words.isEmpty()) {
            return List.of();
        }
        if (words.size() <= chunkWords) {
            return List.of(text);
        }

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int start = 0; start < words.size(); start += step) {
            int end = Math.min(start + chunkWords, words.size());
            chunks.add(String.join(" ", words.subList(start, end)));
            if (end == words.size()) {
                break;
            }
        }
        return chunks;
    }
}
