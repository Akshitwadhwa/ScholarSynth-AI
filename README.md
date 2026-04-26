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
- Evaluation: BLEU/SacreBLEU, ROUGE-1, ROUGE-2, ROUGE-L, plus guardrail/error-analysis checks

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
11. 200-example baseline evaluation completed across `pretrained`, `prompt_engineered`, `rag_system`, and `fine_tuned_lora`.
12. 1,200-example LoRA test generation and metrics files added under `outputs/`.
13. Qualitative error analysis and guardrail reports generated in `outputs/error_analysis.md`.
14. RAG + LoRA generation artifacts saved with retrieved titles/chunks in `outputs/rag_plus_lora_200_generations.csv`.
15. LoRA adapter files are present under `models/flan_t5_lora/`.
16. Streamlit app UI redesigned with light/dark theme support, dataset status cards, LoRA generator selection, safer index controls, output/evidence tabs, and guardrail warnings.
17. Project Jupyter kernel registered as `Gen AI Research Assistant (.venv)`.

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
- `fine_tuned_lora`: `flan-t5-base` with the trained LoRA adapter

Results are saved in:

- `outputs/baseline_200_eval_data.csv`
- `outputs/baseline_200_comparison_table.csv`
- `outputs/baseline_200_task_metrics.csv`
- `outputs/baseline_200_comparison.md`
- `outputs/lora_test_metrics.csv`
- `outputs/lora_test_generations.csv`
- `outputs/lora_test_best_cases.csv`
- `outputs/lora_test_worst_cases.csv`
- `outputs/error_analysis.md`
- `outputs/error_analysis_cases.csv`
- `outputs/error_analysis_failure_summary.csv`
- `outputs/error_analysis_guardrail_summary.csv`
- `outputs/rag_plus_lora_200_generations.csv`
- `outputs/rag_plus_lora_200_metrics.csv`
- `outputs/rag_plus_lora_200_task_metrics.csv`
- `outputs/rag_plus_lora_200_summary.json`
- `outputs/rag_plus_lora_200_retrieval_cache.json`

### Aggregate Comparison Table

| Model | Eval Examples | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: |
| fine_tuned_lora | 200 | 29.0810 | 0.5131 | 0.3699 | 0.4376 |
| rag_plus_lora | 200 | 22.3723 | 0.4296 | 0.2673 | 0.3621 |
| pretrained | 200 | 0.3151 | 0.1679 | 0.0820 | 0.1363 |
| rag_system | 200 | 0.1292 | 0.1458 | 0.0816 | 0.1294 |
| prompt_engineered | 200 | 0.0005 | 0.1026 | 0.0594 | 0.0964 |

Separate 1,200-example LoRA test:

| Model | Eval Examples | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: |
| fine_tuned_lora | 1200 | 47.2745 | 0.6561 | 0.5404 | 0.6031 |

### RAG + LoRA Generation Artifacts

`outputs/rag_plus_lora_200_generations.csv` contains 200 saved `rag_plus_lora` generations with retrieved titles and retrieved chunks. This makes the hallucination and grounding analysis stronger than the LoRA-only generation file, which only contains references and predictions.

### Error Analysis Summary

Guardrail findings on the 1,200-example LoRA test generations:

| Finding | Count |
| --- | ---: |
| No retrieved evidence was supplied to the output guardrail; hallucination checks are limited. | 1200 |
| The answer uses strong quantitative or comparative language without an explicit citation. | 65 |

Guardrail findings on the 200-example RAG + LoRA generations:

| Finding | Count |
| --- | ---: |
| Weak lexical overlap with retrieved evidence | 40 |
| Repetitive wording detected | 6 |
| Strong unsupported quantitative/comparative language | 5 |

Failure proxy counts on the 200-example RAG + LoRA generations:

| Failure Type | Count |
| --- | ---: |
| low_bleu | 105 |
| low_rougeL | 90 |
| no_major_proxy_failure | 78 |
| weak_evidence_overlap | 40 |
| repetition | 6 |
| hallucination_risk | 5 |

Interpretation:

- `fine_tuned_lora` is now the strongest saved model on the 200-example comparison.
- `rag_plus_lora` is second overall and is the best candidate for evidence-grounded demo outputs because its generation file includes retrieved titles/chunks.
- The largest LoRA gains are on `evidence_based_qa`, `technical_explanation`, and `paper_summary`.
- `comparative_analysis`, `literature_review`, and `research_gap_analysis` remain harder tasks.
- RAG + LoRA is strongest on `evidence_based_qa`, but retrieval can reduce lexical overlap on some synthesis tasks because the generated wording shifts toward retrieved evidence.

### Baseline Metrics Chart

```text
ROUGE-L
fine_tuned_lora    0.4376 | ████████████████████████████
rag_plus_lora      0.3621 | ███████████████████████
pretrained         0.1363 | █████████
rag_system         0.1294 | ████████
prompt_engineered  0.0964 | ██████

ROUGE-1
fine_tuned_lora    0.5131 | ████████████████████████████
rag_plus_lora      0.4296 | ███████████████████████
pretrained         0.1679 | █████████
rag_system         0.1458 | ████████
prompt_engineered  0.1026 | ██████

BLEU
fine_tuned_lora   29.0810 | ████████████████████████████
rag_plus_lora     22.3723 | ██████████████████████
pretrained         0.3151 | █
rag_system         0.1292 | █
prompt_engineered  0.0005 | █
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

The LoRA adapter has been evaluated in two ways:

- 200-example comparison against `pretrained`, `prompt_engineered`, and `rag_system`
- 1,200-example LoRA-only test using `outputs/lora_test_generations.csv`
- 200-example `rag_plus_lora` evaluation using retrieved Chroma evidence and the trained LoRA adapter

The next useful evaluation upgrade is prompt tuning for `rag_plus_lora`, especially on comparative analysis and research-gap tasks.

## Streamlit App Changes

`app.py` now provides a more user-friendly demo interface:

- Light/dark mode selector in the sidebar
- Dataset status indicators for raw papers, processed chunks, SQLite, and Chroma
- Hero section for ScholarSynth AI
- Status cards for raw papers, text chunks, vector chunks, and LoRA adapter availability
- Generator selector for `Base FLAN-T5`, `Fine-tuned LoRA`, and `RAG + Fine-tuned LoRA`
- Safer dataset/index management inside an expander
- Input and output guardrail warnings/errors shown in expandable panels

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
│   ├── 07_finetuned_lora_comparison.ipynb
│   └── 08_error_analysis_and_guardrails.ipynb
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
