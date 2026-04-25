# ScholarSynth AI

ScholarSynth AI is an autonomous research assistant for research paper exploration, literature review generation, paper Q&A, technical explanation, and research gap analysis.

The system retrieves papers from arXiv and Semantic Scholar, preprocesses academic text, stores metadata in SQLite, stores embeddings in ChromaDB, and evaluates baseline generation strategies before PEFT fine-tuning.

## Current Tech Stack

- Baseline model: `google/flan-t5-base`
- Prompt-engineered baseline: `google/flan-t5-base`
- Planned fine-tuned model: `google/flan-t5-base` with LoRA
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
5. SQLite metadata storage and ChromaDB vector indexing implemented in `src/vector_store.py`.
6. Vector database notebook completed and run: `notebooks/02_vector_database.ipynb`.
7. Semantic retrieval tested across LoRA, chatbots, RAG hallucination, long-context transformers, scholarly search, and literature review queries.
8. RAG and agent scaffolding created in `src/rag_pipeline.py` and `src/agents.py`.
9. Baseline evaluation module created in `src/baseline_eval.py`.
10. 40-example baseline evaluation notebook created and executed: `notebooks/05_evalute_baseline.ipynb`.
11. Generated baseline evaluation outputs are saved under `outputs/`.

## Current Dataset Status

Line counts include CSV headers where applicable.

| File | Rows / Lines |
| --- | ---: |
| `data/raw_papers.csv` | 449 |
| `data/processed_papers.csv` | 929 |
| `data/train.csv` | 650 |
| `data/val.csv` | 140 |
| `data/test.csv` | 141 |
| `data/finetune_dataset.jsonl` | 1566 |
| `data/finetune_train.jsonl` | 1252 |
| `data/finetune_val.jsonl` | 157 |
| `data/finetune_test.jsonl` | 157 |

Approximate actual dataset sizes:

- Raw papers: 448
- Processed chunks: 928
- Fine-tuning examples: 1,566
- Fine-tuning split: 1,252 train / 157 validation / 157 test

## Baseline Evaluation Results

The latest baseline evaluation was run on 40 balanced examples using:

- `pretrained`: plain `flan-t5-base`
- `prompt_engineered`: structured prompt using the same `flan-t5-base`
- `rag_system`: Chroma retrieval + `flan-t5-base`

Results are saved in:

- `outputs/baseline_40_metrics.csv`
- `outputs/baseline_40_generations.csv`
- `outputs/baseline_40_comparison_table.csv`
- `outputs/baseline_40_comparison.md`

### Aggregate Comparison Table

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pretrained | 0.0126 | 0.1926 | 0.0788 | 0.1515 | 0.7770 |
| rag_system | 0.0034 | 0.1553 | 0.0549 | 0.1177 | 0.7627 |
| prompt_engineered | 0.0031 | 0.1204 | 0.0579 | 0.1045 | 0.7447 |

Interpretation:

- The pretrained baseline currently scores highest on lexical metrics because many reference answers are derived from the same input abstracts.
- RAG still provides useful retrieved evidence and is important for grounded answers, citations, and hallucination reduction.
- These are baseline results before LoRA fine-tuning. Final improvement should be measured after adding `fine_tuned_lora` and `rag_plus_lora`.

## Important Artifacts

```text
data/raw_papers.csv
data/processed_papers.csv
data/train.csv
data/val.csv
data/test.csv
data/finetune_dataset.jsonl
data/finetune_train.jsonl
data/finetune_val.jsonl
data/finetune_test.jsonl
data/papers.db
data/chroma/
outputs/retrieval_examples.csv
outputs/baseline_40_comparison.md
outputs/baseline_40_metrics.csv
outputs/baseline_40_generations.csv
```

Note: generated data, databases, model artifacts, and vector indexes are ignored by `.gitignore` to avoid committing large files.

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
│   ├── 04_peft_finetuning.ipynb
│   └── 05_evalute_baseline.ipynb
├── src/
│   ├── paper_search.py
│   ├── preprocessing.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   ├── agents.py
│   ├── evaluation.py
│   └── baseline_eval.py
├── outputs/
├── report/
├── app.py
├── requirements.txt
└── README.md
```

## Setup

```bash
cd "/Users/Lenovo/Desktop/sem 6/Gen_AI Project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If using VS Code notebooks, select this interpreter:

```text
/Users/Lenovo/Desktop/sem 6/Gen_AI Project/.venv/bin/python
```

## Useful Commands

Check core files:

```bash
ls -lh data outputs notebooks src
```

Compile Python files:

```bash
python -m compileall src app.py
```

Check dataset counts:

```bash
wc -l data/raw_papers.csv data/processed_papers.csv data/train.csv data/val.csv data/test.csv data/finetune_dataset.jsonl data/finetune_train.jsonl data/finetune_val.jsonl data/finetune_test.jsonl
```

Run Streamlit app:

```bash
streamlit run app.py
```

Test the saved LoRA adapter locally after copying it back from Colab:

```bash
python -m src.load_lora_model
```

Expected local adapter folder:

```text
models/flan_t5_lora/
```

The Streamlit/RAG generator automatically loads `models/flan_t5_lora` when the adapter files exist. If that folder is missing, it falls back to the base `google/flan-t5-base` model.

Run the 40-example baseline evaluation from Python:

```bash
python -c "from src.baseline_eval import run_baseline_evaluation; metrics, aggregate, generations = run_baseline_evaluation(sample_size=40); print(aggregate.round(4).to_string(index=False))"
```

Execute the baseline evaluation notebook and save outputs into the notebook:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/05_evalute_baseline.ipynb --ExecutePreprocessor.timeout=3600
```

## Next Steps

1. Replace `notebooks/04_peft_finetuning.ipynb` with a Colab-ready LoRA fine-tuning notebook.
2. Fine-tune `google/flan-t5-base` using:
   - `data/finetune_train.jsonl`
   - `data/finetune_val.jsonl`
3. Save the LoRA adapter to:

```text
models/flan_t5_lora/
```

4. Add final evaluation for:
   - `fine_tuned_lora`
   - `rag_plus_lora`
5. Update the report with:
   - baseline comparison
   - LoRA fine-tuning justification
   - quantitative metric table
   - qualitative/error analysis
   - hallucination and guardrail cases

## Notes for the Next Developer

- Prefer running PEFT fine-tuning in Google Colab with GPU.
- Keep the local MacBook workflow for data processing, ChromaDB, SQLite, retrieval testing, and Streamlit.
- The current baseline scores are not expected to prove final improvement yet. They are pre-fine-tuning baselines.
- The final improvement story should compare `pretrained`, `prompt_engineered`, and `rag_system` against `fine_tuned_lora` and `rag_plus_lora`.
