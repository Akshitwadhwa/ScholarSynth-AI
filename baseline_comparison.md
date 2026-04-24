# Baseline Comparison Results

## Aggregate Metrics
| model | bleu | rouge1 | rouge2 | rougeL | bertscore_f1 |
| --- | --- | --- | --- | --- | --- |
| prompt_engineered | 0.0007 | 0.1359 | 0.0540 | 0.1145 | 0.7674 |
| pretrained | 0.0019 | 0.1552 | 0.0458 | 0.1141 | 0.7672 |
| rag_system | 0.0071 | 0.1549 | 0.0706 | 0.1322 | 0.7543 |

## Example 1: literature_review
Topic: RAG for citation-grounded text generation

Reference:
Research on RAG for citation-grounded text generation shows a clear focus on methods represented by papers such as From RAG to QA-RAG: Integrating Generative AI for Pharmaceutical Regulatory Compliance Process; Telco-RAG: Navigating the Challenges of Retrieval-Augmented Language Models for Telecommunications; Collab-RAG: Boosting Retrieval-Augmented Generation for Complex Question Answering via White-Box and Black-Box LLM Collaboration; Auto-RAG: Autonomous Retrieval-Augmented Generation for Large Language Models. Across these papers, the common pattern is the use of model-based techniques to improve retrieval, generation, explanation, or evaluation quality. The evidence suggests that stronger grounding, better evaluation, and domain adaptation are recurring themes in this area.

### pretrained
Abstract: The authors propose a collaborative training framework for RAG for open domain question answering.

### prompt_engineered
We introduce a chatbot model that utilizes generative AI and the Retrieval Augmented Generation method.

### rag_system
Retrieval Augmented Generation: A Framework for Reference-Free Evaluation of Retrieval Augmented Generation

Retrieved titles: Automated Literature Review Using NLP Techniques and LLM-Based Retrieval-Augmented Generation | Ragas: Automated Evaluation of Retrieval Augmented Generation | Reconstructing Context: Evaluating Advanced Chunking Strategies for Retrieval-Augmented Generation | AI Literature Review Suite | HySemRAG: A Hybrid Semantic Retrieval-Augmented Generation Framework for Automated Literature Synthesis and Methodological Gap Analysis

## Example 2: research_gap_analysis
Topic: Open-source large language models

Reference:
For Open-source large language models, the retrieved papers suggest several research gaps: more reliable evaluation protocols, stronger evidence grounding, better handling of domain-specific terminology, and clearer comparison between retrieval-based, fine-tuned, and prompt-engineered systems. Future work should test these methods on larger and more diverse scientific corpora.

### pretrained
We propose a new approach to self-cognition in large language models.

### prompt_engineered
We propose a new approach to self-cognition in large language models.

### rag_system
Open-source large language models are a promising approach to re-engineering human coders, they are also vulnerable to re-engineering and re-learning.

Retrieved titles: Rethinking Scale: The Efficacy of Fine-Tuned Open-Source LLMs in Large-Scale Reproducible Social Science Research | LOLA -- An Open-Source Massively Multilingual Large Language Model | ChatGPT's One-year Anniversary: Are Open-Source Large Language Models Catching up? | Building a Strong Instruction Language Model for a Less-Resourced Language | Open foundation models for Azerbaijani language

## Example 3: technical_explanation
Topic: Neural information retrieval for scholarly search

Reference:
In simple terms, this paper is about Ranking models are the main components of information retrieval systems. The main value is that it helps readers understand a specific method, system, or evaluation problem within the broader research area.

### pretrained
We compare the proposed neural ranking models in the literature and propose future research directions.

### prompt_engineered
Neural ranking models for document retrieval

### rag_system
Neural information retrieval is a method of retrieving information from a corpus of documents, e.g., a document, a answer, a video, a document, a page, a page, a page, a page, a page, a page, a page, a page, a page, a page, a page, a page, a page, a page, a page, a page, a page, a page, a page, a page, a page,

Retrieved titles: Neural ranking models for document retrieval | Bibliometric-enhanced Information Retrieval: 2nd International BIR Workshop | Editorial for the Bibliometric-enhanced Information Retrieval Workshop at ECIR 2014 | Bibliometric-enhanced Information Retrieval: 2nd International BIR Workshop | SF-RAG: Structure-Fidelity Retrieval-Augmented Generation for Academic Question Answering
