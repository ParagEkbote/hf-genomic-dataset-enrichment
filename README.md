# hf-genomic-dataset-enrichment

cpu enrichment:

```bash
dagster job execute \
  -m carbon_enrichment.definitions \
  -j carbon_cpu_job \
  -c config/dev-gpu.yaml
```

for gpu enrichment:

```bash
dagster job execute \
  -m carbon_enrichment.definitions \
  -j carbon_gpu_job \
  -c config/dev-gpu.yaml
```

| Question type                        | Faceberg | DuckDB | Qdrant | Elasticsearch | LlamaIndex |
| ------------------------------------ | :------: | :----: | :----: | :-----------: | :--------: |
| What tables exist?                   |   **●**  |        |        |               |            |
| What is in the corpus?               |          |  **●** |        |               |            |
| What is the distribution of species? |          |  **●** |        |               |            |
| What is the data quality?            |          |  **●** |        |               |            |
| What sequences are similar?          |          |        |  **●** |               |            |
| Find exact metadata matches          |          |        |        |     **●**     |            |
| Hybrid semantic + metadata retrieval |          |        |  **●** |     **●**     |            |
| Compare retrieval approaches         |          |  **●** |  **●** |     **●**     |            |
| Explain a retrieved neighborhood     |          |  **●** |  **●** |     **●**     |    **●**   |
| Answer multi-step research questions |          |        |        |               |    **●**   |
