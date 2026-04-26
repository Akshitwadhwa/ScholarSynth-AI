# Baseline Evaluation on 200 Examples

Evaluated examples: 200

This report reflects the latest uploaded baseline outputs. The 200-example comparison now includes `fine_tuned_lora` in addition to `pretrained`, `prompt_engineered`, and `rag_system`.

## Aggregate Comparison Table

| model | eval_examples | bleu | rouge1 | rouge2 | rougeL |
| --- | --- | --- | --- | --- | --- |
| fine_tuned_lora | 200 | 29.0810 | 0.5131 | 0.3699 | 0.4376 |
| pretrained | 200 | 0.3151 | 0.1679 | 0.0820 | 0.1363 |
| rag_system | 200 | 0.1292 | 0.1458 | 0.0816 | 0.1294 |
| prompt_engineered | 200 | 0.0005 | 0.1026 | 0.0594 | 0.0964 |

## Task-Level Metrics

| model | task | eval_examples | bleu | rouge1 | rouge2 | rougeL |
| --- | --- | --- | --- | --- | --- | --- |
| fine_tuned_lora | comparative_analysis | 33 | 4.8471 | 0.3049 | 0.0922 | 0.2093 |
| pretrained | comparative_analysis | 33 | 0.2410 | 0.1314 | 0.0233 | 0.0953 |
| prompt_engineered | comparative_analysis | 33 | 0.0000 | 0.0566 | 0.0182 | 0.0511 |
| rag_system | comparative_analysis | 33 | 0.0151 | 0.0745 | 0.0070 | 0.0625 |
| fine_tuned_lora | evidence_based_qa | 33 | 70.4928 | 0.8542 | 0.7826 | 0.8364 |
| pretrained | evidence_based_qa | 33 | 0.0350 | 0.1617 | 0.0812 | 0.1332 |
| prompt_engineered | evidence_based_qa | 33 | 0.0029 | 0.1503 | 0.1019 | 0.1441 |
| rag_system | evidence_based_qa | 33 | 0.1306 | 0.1922 | 0.1160 | 0.1695 |
| fine_tuned_lora | literature_review | 33 | 5.6744 | 0.3580 | 0.1417 | 0.2182 |
| pretrained | literature_review | 33 | 0.0563 | 0.1493 | 0.0593 | 0.1113 |
| prompt_engineered | literature_review | 33 | 0.0000 | 0.0784 | 0.0430 | 0.0750 |
| rag_system | literature_review | 33 | 0.0191 | 0.1186 | 0.0484 | 0.0984 |
| fine_tuned_lora | paper_summary | 35 | 44.6741 | 0.6037 | 0.4987 | 0.5433 |
| pretrained | paper_summary | 35 | 0.5662 | 0.2698 | 0.2051 | 0.2524 |
| prompt_engineered | paper_summary | 35 | 0.0101 | 0.1865 | 0.1482 | 0.1827 |
| rag_system | paper_summary | 35 | 0.3440 | 0.2816 | 0.2480 | 0.2794 |
| fine_tuned_lora | research_gap_analysis | 33 | 8.7164 | 0.2825 | 0.1459 | 0.2008 |
| pretrained | research_gap_analysis | 33 | 0.5854 | 0.1422 | 0.0623 | 0.1053 |
| prompt_engineered | research_gap_analysis | 33 | 0.0001 | 0.0758 | 0.0182 | 0.0710 |
| rag_system | research_gap_analysis | 33 | 0.0146 | 0.0767 | 0.0113 | 0.0605 |
| fine_tuned_lora | technical_explanation | 33 | 49.5945 | 0.6696 | 0.5440 | 0.6147 |
| pretrained | technical_explanation | 33 | 0.0593 | 0.1442 | 0.0541 | 0.1067 |
| prompt_engineered | technical_explanation | 33 | 0.0000 | 0.0582 | 0.0175 | 0.0506 |
| rag_system | technical_explanation | 33 | 0.0220 | 0.1244 | 0.0481 | 0.0935 |

## Metric Charts

```text
rougeL
fine_tuned_lora      0.4376 | ████████████████████████████
pretrained           0.1363 | █████████
rag_system           0.1294 | ████████
prompt_engineered    0.0964 | ██████

rouge1
fine_tuned_lora      0.5131 | ████████████████████████████
pretrained           0.1679 | █████████
rag_system           0.1458 | ████████
prompt_engineered    0.1026 | ██████

bleu
fine_tuned_lora     29.0810 | ████████████████████████████
pretrained           0.3151 | █
rag_system           0.1292 | █
prompt_engineered    0.0005 | █
```

## Separate LoRA Test Set

The uploaded LoRA test files evaluate the fine-tuned adapter on 1,200 examples.

| model | eval_examples | bleu | rouge1 | rouge2 | rougeL |
| --- | --- | --- | --- | --- | --- |
| fine_tuned_lora | 1200 | 47.2745 | 0.6561 | 0.5404 | 0.6031 |

## Interpretation

- `fine_tuned_lora` is the strongest model in the 200-example comparison, with ROUGE-L `0.4376` compared with `0.1363` for `pretrained`, `0.1294` for `rag_system`, and `0.0964` for `prompt_engineered`.
- The largest LoRA gains are on `evidence_based_qa`, `technical_explanation`, and `paper_summary`.
- `comparative_analysis`, `literature_review`, and `research_gap_analysis` remain harder tasks, even though LoRA still leads the saved baselines.
- The 1,200-example LoRA-only test reports BLEU `47.2745` and ROUGE-L `0.6031`, which is stronger than the 200-example LoRA aggregate. Treat this as a separate evaluation split, not a direct row in the 200-example comparison.

## Qualitative LoRA Samples

### LoRA Test Example 1

Reference:
In simple terms, this paper is about Retrieval Augmented Generation (RAG) has emerged as a promising solution to address hallucination issues in Large Language Models (LLMs). The main value is that it helps readers understand a specific method, system, or evaluation problem within the broader research area.

Prediction:
In simple terms, this paper is on the main in a specific method. The main idea is that it helps readers understand a particular problem and to identify a successful solution to address hallucination issues in Large Language Models (LLMs).

### LoRA Test Example 2

Reference:
The AI Research Assistant: Promise, Peril, and a Proof of Concept studies Can artificial intelligence truly contribute to creative mathematical research, or does it merely automate routine calculations while introducing risks of error? We provide empirical evidence through a detailed case study: the discovery of novel error representations and bounds for ...

Prediction:
In a detail, this paper is: Can artificial intelligence truly contribute to creative mathematical research, or does it merely automate routine calculations while introducing risks of error. We provide empirical evidence through a detailed case study: the discovery of novel error representations and bounds for Hermite quadrature rules via systematic human-...

### LoRA Test Example 3

Reference:
In simple terms, this paper is about We present the BioImage$.$IO Chatbot, an AI assistant powered by Large Language Models and supported by a community-driven knowledge base and toolset. The main value is that it helps readers understand a specific method, system, or evaluation problem within the broader research area.

Prediction:
In simple terms, this paper is on the main in a specific method. The main idea is that it helps readers understand a different research area, and to provide a more clear for-intie study.

### LoRA Test Example 4

Reference:
A Review on Neural Network Models of Schizophrenia and Autism Spectrum Disorder studies This survey presents the most relevant neural network models of autism spectrum disorder and schizophrenia, from the first connectionist models to recent deep network architectures. We analyzed and compared the most representative symptoms with its neural model counter...

Prediction:
In a review, the paper is on this survey: This research presents the most relevant neural network models of autism spectrum disorder and schizophrenia, from the first connectionist model to recent deep network architectures. We analyzed and compared the more representative symptoms with its neural model counterpart, detailing the alteration introduced in ...

### LoRA Test Example 5

Reference:
Based on the abstract, the paper addresses this problem: This paper addresses the task of (complex) conversational question answering over a knowledge graph. For this task, we propose LASAGNE (muLti-task semAntic parSing with trAnsformer and Graph atteNtion nEtworks).

Prediction:
In the abstract, this paper addresses a problem: This paper addresses the task of (complex) conversational question answering over a knowledge graph. For this task, we propose LASAGNE (muLti-task semAnsformer and Graph atteNtion nEdworks).

