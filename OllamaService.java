package com.ashwi.vectordb.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

import java.util.List;
import java.util.Map;

@Service
public class OllamaService {
    private final RestClient restClient;
    private final String embedModel;
    private final String generateModel;

    public OllamaService(
            @Value("${ollama.base-url}") String baseUrl,
            @Value("${ollama.embed-model}") String embedModel,
            @Value("${ollama.generate-model}") String generateModel
    ) {
        this.restClient = RestClient.builder().baseUrl(baseUrl).build();
        this.embedModel = embedModel;
        this.generateModel = generateModel;
    }

    public boolean isAvailable() {
        try {
            restClient.get().uri("/api/tags").retrieve().toBodilessEntity();
            return true;
        } catch (RestClientException ex) {
            return false;
        }
    }

    public List<Float> embed(String text) {
        try {
            EmbeddingResponse response = restClient.post()
                    .uri("/api/embeddings")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(Map.of("model", embedModel, "prompt", text))
                    .retrieve()
                    .body(EmbeddingResponse.class);
            return response == null || response.embedding() == null ? List.of() : response.embedding();
        } catch (RestClientException ex) {
            return List.of();
        }
    }

    public String generate(String prompt) {
        try {
            GenerateResponse response = restClient.post()
                    .uri("/api/generate")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(Map.of("model", generateModel, "prompt", prompt, "stream", false))
                    .retrieve()
                    .body(GenerateResponse.class);
            return response == null || response.response() == null ? "" : response.response();
        } catch (RestClientException ex) {
            return "ERROR: Ollama unavailable. Run: ollama serve";
        }
    }

    public String embedModel() {
        return embedModel;
    }

    public String generateModel() {
        return generateModel;
    }

    private record EmbeddingResponse(List<Float> embedding) {
    }

    private record GenerateResponse(String response) {
    }
}
