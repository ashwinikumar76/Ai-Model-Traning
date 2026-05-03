# VectorDB Spring Boot Port

This is the Java/Spring Boot version of the original C++ VectorDB project.

## Stack

- Java 21
- Spring Boot 3
- Maven
- In-memory vector indexes: Brute Force, KD-Tree, HNSW
- Ollama for embeddings and RAG
- Existing `index.html` served from `src/main/resources/static`

## Run

```powershell
mvn spring-boot:run
```

Then open:

```text
http://localhost:8081
```

For document embedding and RAG features, keep Ollama running and install the models:

```powershell
ollama pull nomic-embed-text
ollama pull llama3
```

## Architecture

```text
controller/
  VectorController.java      demo vector REST API
  DocumentController.java    document embedding + RAG API
  StatusController.java      health/status API

service/
  VectorDbService.java       owns demo vector database behavior
  DocumentService.java       owns document chunk store/search
  OllamaService.java         calls local Ollama HTTP API
  DemoDataLoader.java        seeds the 20 demo vectors

algorithm/
  BruteForceIndex.java
  KdTreeIndex.java
  HnswIndex.java
  DistanceMetrics.java

model/
  VectorItem.java
  DocItem.java
  SearchHit.java
  GraphInfo.java

dto/
  InsertRequest.java
  DocumentInsertRequest.java
  QuestionRequest.java
```

Special thanks to yt:Padho with pratyush for teach these kind of project.
