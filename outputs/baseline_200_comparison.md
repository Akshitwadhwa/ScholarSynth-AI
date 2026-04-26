# Baseline Evaluation on 200 Examples

Evaluated examples: 200

This report reflects the latest evaluation outputs. The 200-example comparison now includes the final intended system, `rag_plus_lora`, along with `fine_tuned_lora`, `pretrained`, `prompt_engineered`, and `rag_system`.

## Aggregate Comparison Table

| model | eval_examples | bleu | rouge1 | rouge2 | rougeL |
| --- | --- | --- | --- | --- | --- |
| fine_tuned_lora | 200 | 29.0810 | 0.5131 | 0.3699 | 0.4376 |
| rag_plus_lora | 200 | 22.3723 | 0.4296 | 0.2673 | 0.3621 |
| pretrained | 200 | 0.3151 | 0.1679 | 0.0820 | 0.1363 |
| rag_system | 200 | 0.1292 | 0.1458 | 0.0816 | 0.1294 |
| prompt_engineered | 200 | 0.0005 | 0.1026 | 0.0594 | 0.0964 |

## Task-Level Metrics

| model | task | eval_examples | bleu | rouge1 | rouge2 | rougeL |
| --- | --- | --- | --- | --- | --- | --- |
| fine_tuned_lora | comparative_analysis | 33 | 4.8471 | 0.3049 | 0.0922 | 0.2093 |
| rag_plus_lora | comparative_analysis | 33 | 1.2212 | 0.2379 | 0.0408 | 0.1644 |
| pretrained | comparative_analysis | 33 | 0.2410 | 0.1314 | 0.0233 | 0.0953 |
| rag_system | comparative_analysis | 33 | 0.0151 | 0.0745 | 0.0070 | 0.0625 |
| prompt_engineered | comparative_analysis | 33 | 0.0000 | 0.0566 | 0.0182 | 0.0511 |
| rag_plus_lora | evidence_based_qa | 35 | 78.8669 | 0.8864 | 0.8376 | 0.8796 |
| fine_tuned_lora | evidence_based_qa | 33 | 70.4928 | 0.8542 | 0.7826 | 0.8364 |
| rag_system | evidence_based_qa | 33 | 0.1306 | 0.1922 | 0.1160 | 0.1695 |
| prompt_engineered | evidence_based_qa | 33 | 0.0029 | 0.1503 | 0.1019 | 0.1441 |
| pretrained | evidence_based_qa | 33 | 0.0350 | 0.1617 | 0.0812 | 0.1332 |
| fine_tuned_lora | literature_review | 33 | 5.6744 | 0.3580 | 0.1417 | 0.2182 |
| rag_plus_lora | literature_review | 33 | 1.5932 | 0.2904 | 0.0570 | 0.1728 |
| pretrained | literature_review | 33 | 0.0563 | 0.1493 | 0.0593 | 0.1113 |
| rag_system | literature_review | 33 | 0.0191 | 0.1186 | 0.0484 | 0.0984 |
| prompt_engineered | literature_review | 33 | 0.0000 | 0.0784 | 0.0430 | 0.0750 |
| fine_tuned_lora | paper_summary | 35 | 44.6741 | 0.6037 | 0.4987 | 0.5433 |
| rag_plus_lora | paper_summary | 33 | 18.2987 | 0.3642 | 0.2318 | 0.3111 |
| rag_system | paper_summary | 35 | 0.3440 | 0.2816 | 0.2480 | 0.2794 |
| pretrained | paper_summary | 35 | 0.5662 | 0.2698 | 0.2051 | 0.2524 |
| prompt_engineered | paper_summary | 35 | 0.0101 | 0.1865 | 0.1482 | 0.1827 |
| fine_tuned_lora | research_gap_analysis | 33 | 8.7164 | 0.2825 | 0.1459 | 0.2008 |
| rag_plus_lora | research_gap_analysis | 33 | 2.7345 | 0.2472 | 0.0407 | 0.1673 |
| pretrained | research_gap_analysis | 33 | 0.5854 | 0.1422 | 0.0623 | 0.1053 |
| prompt_engineered | research_gap_analysis | 33 | 0.0001 | 0.0758 | 0.0182 | 0.0710 |
| rag_system | research_gap_analysis | 33 | 0.0146 | 0.0767 | 0.0113 | 0.0605 |
| fine_tuned_lora | technical_explanation | 33 | 49.5945 | 0.6696 | 0.5440 | 0.6147 |
| rag_plus_lora | technical_explanation | 33 | 28.0952 | 0.5236 | 0.3612 | 0.4462 |
| pretrained | technical_explanation | 33 | 0.0593 | 0.1442 | 0.0541 | 0.1067 |
| rag_system | technical_explanation | 33 | 0.0220 | 0.1244 | 0.0481 | 0.0935 |
| prompt_engineered | technical_explanation | 33 | 0.0000 | 0.0582 | 0.0175 | 0.0506 |

## Metric Charts

```text
rougeL
fine_tuned_lora      0.4376 | ████████████████████████████
rag_plus_lora        0.3621 | ███████████████████████
pretrained           0.1363 | █████████
rag_system           0.1294 | ████████
prompt_engineered    0.0964 | ██████

rouge1
fine_tuned_lora      0.5131 | ████████████████████████████
rag_plus_lora        0.4296 | ███████████████████████
pretrained           0.1679 | █████████
rag_system           0.1458 | ████████
prompt_engineered    0.1026 | ██████

bleu
fine_tuned_lora     29.0810 | ████████████████████████████
rag_plus_lora       22.3723 | ██████████████████████
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

- `fine_tuned_lora` remains the strongest lexical scorer in the 200-example comparison, with ROUGE-L `0.4376`.
- `rag_plus_lora` is second overall with ROUGE-L `0.3621` and now provides retrieved titles/chunks for grounding analysis.
- `rag_plus_lora` is strongest on `evidence_based_qa`, where retrieved evidence helps the answer align with the reference.
- Retrieval can reduce pure lexical scores for some tasks because extra evidence changes the wording, but it improves real-world grounding and explainability.
- The 1,200-example LoRA-only test reports BLEU `47.2745` and ROUGE-L `0.6031`; treat it as a separate evaluation split.

## Qualitative RAG + LoRA Samples

### RAG + LoRA Example 1: paper_summary

Topic: transformer models for natural language processing

Reference:
A Holistic Natural Language Generation Framework for the Semantic Web studies With the ever-growing generation of data for the Semantic Web comes an increasing demand for this data to be made available to non-semantic Web experts. One way of achieving this goal is to translate the languages of the Semantic Web into natural language.

Prediction:
In the a broader ad, this paper is on a system based on semantic language processing. The as-in-out, a method is a framework for verbalizing the three key languages of the samantic Web, i.e., RDF, OWL, and SPARQL.

Retrieved titles:
A Holistic Natural Language Generation Framework for the Semantic Web | Transformer Based Implementation for Automatic Book Summarization | A Review of Bangla Natural Language Processing Tasks and the Utility of Transformer Models | Semantic Web for Machine Translation: Challenges and Directions

### RAG + LoRA Example 2: paper_summary

Topic: attention mechanisms in transformers

Reference:
On the Surprising Effectiveness of Attention Transfer for Vision Transformers studies Conventional wisdom suggests that pre-training Vision Transformers (ViT) improves downstream performance by learning useful representations. Is this actually true?

Prediction:
In the a broader ad, this paper is on the underlying method, guiding how information flows between tokens is sufficient for models to learn high quality features from scratch and achieve comparable downstream performance. The ac-intee is a simple method, in that a paper focuses on the main source, a specific method, or results in a more specialized work.

Retrieved titles:
On the Surprising Effectiveness of Attention Transfer for Vision Transformers | Self-attention in Vision Transformers Performs Perceptual Grouping, Not Attention | An analysis of attention mechanisms and its variance in transformer | Self-attention in vision transformers performs perceptual grouping, not attention

### RAG + LoRA Example 3: paper_summary

Topic: Multi-agent systems for literature review generation

Reference:
Context Engineering for Multi-Agent LLM Code Assistants Using Elicit, NotebookLM, ChatGPT, and Claude Code studies Large Language Models (LLMs) have shown promise in automating code generation and software engineering tasks, yet they often struggle with complex, multi-file projects due to context limitations and knowledge gaps. We propose a novel context ...

Prediction:
In the ad-based paper, the paper is a a new system, combining multiple AI components: an intent Translator (GPT-5) for clarifying user requirements, an Elicit-powered semantic literature retrieval for injecting domain knowledge, NotebookLM-based document synthesis for contextual understanding, and a, Claude Code, a. The agent system is able to use a uniqu...

Retrieved titles:
Context Engineering for Multi-Agent LLM Code Assistants Using Elicit, NotebookLM, ChatGPT, and Claude Code | ResearchCodeAgent: An LLM Multi-Agent System for Automated Codification of Research Methodologies | AutoFlow: Automated Workflow Generation for Large Language Model Agents | AutoFlow: Automated Workflow Generation for Large Language Model Agents

### RAG + LoRA Example 4: paper_summary

Topic: attention mechanisms in transformers

Reference:
SCSC: Spatial Cross-scale Convolution Module to Strengthen both CNNs and Transformers studies This paper presents a module, Spatial Cross-scale Convolution (SCSC), which is verified to be effective in improving both CNNs and Transformers. Nowadays, CNNs and Transformers have been successful in a variety of tasks.

Prediction:
In the a ad, the paper is a system, a Spatial Cross-scale Convolution (SCSC), which is verified to be effective in improving both CNNs and Transformers. The receptive field, reced, and dynamic weight are a key aessing a mechanism, i-scale, and resead a more effective system.

Retrieved titles:
SCSC: Spatial Cross-scale Convolution Module to Strengthen both CNNs and Transformers | Co-Scale Conv-Attentional Image Transformers | Preconditioned Attention: Enhancing Efficiency in Transformers | An analysis of attention mechanisms and its variance in transformer

### RAG + LoRA Example 5: paper_summary

Topic: research trend analysis using NLP

Reference:
A Survey on Patent Analysis: From NLP to Multimodal AI studies Recent advances in Pretrained Language Models (PLMs) and Large Language Models (LLMs) have demonstrated transformative capabilities across diverse domains. The field of patent analysis and innovation is not an exception, where natural language processing (NLP) techniques presents opportunities...

Prediction:
In a broader research, the paper is a in-tead source, a more streamlined analysis, and a better aeration of research, based on the broader information, and the ad-in-ressi aes, if a the ssing a-sse, able to make the esss in a larger acing, stad, or sass, is. The as sys is able a to

Retrieved titles:
A Survey on Patent Analysis: From NLP to Multimodal AI | A Survey on Patent Analysis: From NLP to Multimodal AI | A Hybrid Architecture with Efficient Fine Tuning for Abstractive Patent Document Summarization | A Hybrid Architecture with Efficient Fine Tuning for Abstractive Patent Document Summarization

