# Autonomous Research Assistant

A multi-agent Generative AI system for research paper exploration, literature review generation, and research gap analysis.

## Stack

- Baseline model: `google/flan-t5-base`
- Prompt-engineered baseline: `google/flan-t5-base`
- Fine-tuned model: `google/flan-t5-base` with LoRA
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector DB: `ChromaDB`
- Metadata DB: `SQLite`
- Frontend: `Streamlit`

## Project Layout

```text
project/
├── data/
├── notebooks/
├── src/
├── outputs/
├── report/
├── app.py
├── requirements.txt
└── README.md
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
