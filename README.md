# ScholarSynth AI

ScholarSynth AI is an autonomous research assistant for research paper exploration, literature review generation, paper Q&A, technical explanation, and research gap analysis.

The system retrieves papers from arXiv and Semantic Scholar, preprocesses academic text, stores metadata in SQLite, stores embeddings in ChromaDB, and evaluates baseline generation strategies before PEFT fine-tuning.

## Current Tech Stack

- Baseline model: `google/flan-t5-base`
- Prompt-engineered baseline: `google/flan-t5-base`
- Fine-tuned model: `google/flan-t5-base` with LoRA adapter
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector DB: `ChromaDB`
- Metadata DB: `SQLite`
- Frontend: `Streamlit`
- Evaluation: BLEU, ROUGE-1, ROUGE-2, ROUGE-L, BERTScore

## What Has Been Completed

1. Project scaffold created with `data/`, `notebooks/`, `src/`, `outputs/`, and `report/`.
2. Paper retrieval pipeline implemented in `src/paper_search.py`.
3. Dataset preprocessing, chunking, train/validation/test split, and fine-tuning dataset creation implemented in `src/preprocessing.py`.
4. Data collection notebook completed and run: `notebooks/01_data_collection.ipynb`.
5. Large-scale 5000-paper collection added via `src/large_data_collection.py` and `notebooks/06_large_data_collection_5000.ipynb`.
6. SQLite metadata storage and ChromaDB vector indexing implemented in `src/vector_store.py`.
7. Vector database rebuilt for the 5000-paper dataset.
8. Semantic retrieval tested across LoRA, chatbots, RAG hallucination, long-context transformers, scholarly search, and literature review queries.
9. RAG and agent scaffolding created in `src/rag_pipeline.py` and `src/agents.py`.
10. Baseline evaluation module created in `src/baseline_eval.py`.
11. 200-example baseline evaluation completed.
12. Generated baseline evaluation outputs are saved under `outputs/`.
13. LoRA adapter files are present under `models/flan_t5_lora/`.
14. Streamlit app UI redesigned with light/dark theme support, dataset status cards, safer index controls, and output/evidence tabs.
15. Project Jupyter kernel registered as `Gen AI Research Assistant (.venv)`.

## Current Dataset Status

Line counts include CSV headers where applicable.

| File | Rows / Lines |
| --- | ---: |
| `data/raw_papers.csv` | 5001 |
| `data/raw_papers_checkpoint.csv` | 5019 |
| `data/processed_papers.csv` | 10340 |
| `data/train.csv` | 7755 |
| `data/val.csv` | 1035 |
| `data/test.csv` | 1552 |
| `data/finetune_dataset.jsonl` | 8631 |
| `data/finetune_train.jsonl` | 6473 |
| `data/finetune_val.jsonl` | 863 |
| `data/finetune_test.jsonl` | 1295 |

Approximate actual dataset sizes:

- Raw papers: 5,000
- Processed chunks: 10,339
- Fine-tuning examples: 8,631
- Chunk split: 7,754 train / 1,034 validation / 1,551 test
- Fine-tuning split: 6,473 train / 863 validation / 1,295 test
- Split ratio: 75% train / 10% validation / 15% test
- Topics covered: 94
- Source split: 4,673 arXiv papers and 327 Semantic Scholar papers
- SQLite metadata rows: 5,000
- ChromaDB indexed chunks: 10,339

## Baseline Evaluation Results

The latest baseline evaluation was run on 200 balanced examples using:

- `pretrained`: plain `flan-t5-base`
- `prompt_engineered`: structured prompt using the same `flan-t5-base`
- `rag_system`: Chroma retrieval + `flan-t5-base`

Results are saved in:

- `outputs/baseline_200_eval_data.csv`
- `outputs/baseline_200_metrics.csv`
- `outputs/baseline_200_generations.csv`
- `outputs/baseline_200_comparison_table.csv`
- `outputs/baseline_200_comparison.md`

### Aggregate Comparison Table

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pretrained | 0.0117 | 0.2100 | 0.1011 | 0.1675 | 0.7849 |
| rag_system | 0.0116 | 0.1649 | 0.0749 | 0.1355 | 0.7632 |
| prompt_engineered | 0.0026 | 0.1384 | 0.0727 | 0.1200 | 0.7537 |

Interpretation:

- The pretrained baseline currently scores highest on lexical metrics because many reference answers are derived from the same input abstracts.
- RAG still provides useful retrieved evidence and is important for grounded answers, citations, and hallucination reduction.
- These are baseline results before LoRA fine-tuning. Final improvement should be measured after adding `fine_tuned_lora` and `rag_plus_lora`.

### Baseline Metrics Chart

```text
BERTScore F1
pretrained         0.7849 | ██████████████████████████████
rag_system         0.7632 | █████████████████████████████
prompt_engineered  0.7537 | ████████████████████████████

ROUGE-L
pretrained         0.1675 | ██████████████████████████████
rag_system         0.1355 | ████████████████████████
prompt_engineered  0.1200 | █████████████████████

ROUGE-2
pretrained         0.1011 | ██████████████████████████████
rag_system         0.0749 | ██████████████████████
prompt_engineered  0.0727 | █████████████████████
```

## Fine-Tuned LoRA Status

The LoRA adapter exists locally:

```text
models/flan_t5_lora/adapter_config.json
models/flan_t5_lora/adapter_model.safetensors
models/flan_t5_lora/tokenizer.json
models/flan_t5_lora/tokenizer_config.json
```

Notebook `notebooks/04_peft_finetuning_colab.ipynb` is the Colab-oriented training notebook. It installs dependencies, mounts Google Drive, loads the fine-tuning JSONL files, configures LoRA, trains `google/flan-t5-base`, and saves the adapter.

Notebook `notebooks/07_finetuned_lora_comparison.ipynb` is intended to compare:

- `pretrained`
- `prompt_engineered`
- `rag_system`
- `fine_tuned_lora`
- `rag_plus_lora`

The LoRA adapter is present, but final full LoRA comparison outputs should be regenerated and saved before the final report.

## Streamlit App Changes

`app.py` now provides a more user-friendly demo interface:

- Light/dark mode selector in the sidebar
- Dataset status indicators for raw papers, processed chunks, SQLite, and Chroma
- Hero section for ScholarSynth AI
- Status cards for raw papers, text chunks, vector chunks, and generator
- Safer dataset/index management inside an expander



## Project Layout

```text
project/
├── data/
│   ├── raw_papers.csv
│   ├── processed_papers.csv
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── finetune_dataset.jsonl
│   ├── finetune_train.jsonl
│   ├── finetune_val.jsonl
│   ├── finetune_test.jsonl
│   ├── papers.db
│   └── chroma/
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_vector_database.ipynb
│   ├── 03_baseline_comparison.ipynb
│   ├── 04_peft_finetuning_colab.ipynb
│   ├── 05_evalute_baseline.ipynb
│   ├── 06_large_data_collection_5000.ipynb
│   └── 07_finetuned_lora_comparison.ipynb
├── src/
│   ├── paper_search.py
│   ├── preprocessing.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   ├── agents.py
│   ├── evaluation.py
│   ├── baseline_eval.py
│   ├── large_data_collection.py
│   └── topic_bank.py
├── outputs/
├── models/
│   └── flan_t5_lora/
├── report/
├── app.py
├── requirements.txt
└── README.md
```


