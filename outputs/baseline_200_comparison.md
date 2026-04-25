# Baseline Evaluation on 200 Examples

Evaluated examples: 200

## Aggregate Comparison Table
| model | bleu | rouge1 | rouge2 | rougeL | bertscore_f1 |
| --- | --- | --- | --- | --- | --- |
| pretrained | 0.0117 | 0.2100 | 0.1011 | 0.1675 | 0.7849 |
| rag_system | 0.0116 | 0.1649 | 0.0749 | 0.1355 | 0.7632 |
| prompt_engineered | 0.0026 | 0.1384 | 0.0727 | 0.1200 | 0.7537 |

## Task-Level Metrics
| model | task | bleu | rougeL | bertscore_f1 |
| --- | --- | --- | --- | --- |
| pretrained | comparative_analysis | 0.0035 | 0.1146 | 0.7594 |
| prompt_engineered | comparative_analysis | 0.0018 | 0.0834 | 0.7344 |
| rag_system | comparative_analysis | 0.0020 | 0.0819 | 0.7347 |
| pretrained | evidence_based_qa | 0.0318 | 0.2033 | 0.7897 |
| prompt_engineered | evidence_based_qa | 0.0041 | 0.1399 | 0.7662 |
| rag_system | evidence_based_qa | 0.0164 | 0.1786 | 0.7815 |
| pretrained | literature_review | 0.0032 | 0.1392 | 0.7821 |
| prompt_engineered | literature_review | 0.0008 | 0.1146 | 0.7694 |
| rag_system | literature_review | 0.0017 | 0.1330 | 0.7847 |
| pretrained | paper_summary | 0.0124 | 0.2683 | 0.8321 |
| prompt_engineered | paper_summary | 0.0073 | 0.2251 | 0.8076 |
| rag_system | paper_summary | 0.0459 | 0.2409 | 0.8105 |
| pretrained | research_gap_analysis | 0.0042 | 0.1394 | 0.7823 |
| prompt_engineered | research_gap_analysis | 0.0015 | 0.0872 | 0.7332 |
| rag_system | research_gap_analysis | 0.0011 | 0.0869 | 0.7322 |
| pretrained | technical_explanation | 0.0141 | 0.1379 | 0.7634 |
| prompt_engineered | technical_explanation | 0.0003 | 0.0686 | 0.7104 |
| rag_system | technical_explanation | 0.0023 | 0.0887 | 0.7343 |

## Qualitative Samples

### Example 1: paper_summary
Topic: transformer models for natural language processing

Reference:
A Holistic Natural Language Generation Framework for the Semantic Web studies With the ever-growing generation of data for the Semantic Web comes an increasing demand for this data to be made available to non-semantic Web experts. One way of achieving this goal is to translate the languages of the Semantic Web into natural language.

#### pretrained
A Holistic Natural Language Generation Framework for the Semantic Web

#### prompt_engineered
LD2NL

#### rag_system
Non-experts.

Retrieved titles: A Holistic Natural Language Generation Framework for the Semantic Web | Transformer Based Implementation for Automatic Book Summarization | A Review of Bangla Natural Language Processing Tasks and the Utility of Transformer Models | Semantic Web for Machine Translation: Challenges and Directions


### Example 2: paper_summary
Topic: attention mechanisms in transformers

Reference:
On the Surprising Effectiveness of Attention Transfer for Vision Transformers studies Conventional wisdom suggests that pre-training Vision Transformers (ViT) improves downstream performance by learning useful representations. Is this actually true?

#### pretrained
Attention transfer for Vision Transformers improves downstream performance by learning useful representations.

#### prompt_engineered
Attention transfer for Vision Transformers

#### rag_system
Pre-trained teacher ViT are transferred to a student, either by copying or distilling the attention maps.

Retrieved titles: On the Surprising Effectiveness of Attention Transfer for Vision Transformers | Self-attention in Vision Transformers Performs Perceptual Grouping, Not Attention | An analysis of attention mechanisms and its variance in transformer | Self-attention in vision transformers performs perceptual grouping, not attention


### Example 3: paper_summary
Topic: Multi-agent systems for literature review generation

Reference:
Context Engineering for Multi-Agent LLM Code Assistants Using Elicit, NotebookLM, ChatGPT, and Claude Code studies Large Language Models (LLMs) have shown promise in automating code generation and software engineering tasks, yet they often struggle with complex, multi-file projects due to context limitations and knowledge gaps. We propose a novel context engineering workflow that combines multiple AI components: an Intent Translator (GPT-5) for clarifying user requirements, an Elicit-powered semantic literature retrieval for injecting domain knowledge, NotebookLM-based document synthesis for contextual understanding, and a Claude Code multi-agent system for code generation and validation.

#### pretrained
A novel context engineering workflow combining multiple AI components: an Intent Translator, Elicit-powered semantic literature retrieval, NotebookLM-based document synthesis, and Claude Code multi-agent system.

#### prompt_engineered
A novel context engineering workflow combining multiple AI components: an Intent Translator, Elicit-powered semantic literature retrieval, NotebookLM-based document synthesis, and Claude Code

#### rag_system
A novel context engineering workflow that combines multiple AI components: an Intent Translator (GPT-5) for clarifying user requirements, an Elicit-powered semantic literature retrieval for injecting domain knowledge, NotebookLM-based document synthesis for contextual understanding, and a Claude Code multi-agent system for code generation and validation.

Retrieved titles: Context Engineering for Multi-Agent LLM Code Assistants Using Elicit, NotebookLM, ChatGPT, and Claude Code | ResearchCodeAgent: An LLM Multi-Agent System for Automated Codification of Research Methodologies | AutoFlow: Automated Workflow Generation for Large Language Model Agents | AutoFlow: Automated Workflow Generation for Large Language Model Agents


### Example 4: paper_summary
Topic: attention mechanisms in transformers

Reference:
SCSC: Spatial Cross-scale Convolution Module to Strengthen both CNNs and Transformers studies This paper presents a module, Spatial Cross-scale Convolution (SCSC), which is verified to be effective in improving both CNNs and Transformers. Nowadays, CNNs and Transformers have been successful in a variety of tasks.

#### pretrained
A module to improve both CNNs and Transformers.

#### prompt_engineered
Spatial Cross-scale Convolution Module to Strengthen Both CNNs and Transformers

#### rag_system
We design a general module taking in these design keys to enhance both CNNs and Transformers.

Retrieved titles: SCSC: Spatial Cross-scale Convolution Module to Strengthen both CNNs and Transformers | Co-Scale Conv-Attentional Image Transformers | Preconditioned Attention: Enhancing Efficiency in Transformers | An analysis of attention mechanisms and its variance in transformer


### Example 5: paper_summary
Topic: research trend analysis using NLP

Reference:
A Survey on Patent Analysis: From NLP to Multimodal AI studies Recent advances in Pretrained Language Models (PLMs) and Large Language Models (LLMs) have demonstrated transformative capabilities across diverse domains. The field of patent analysis and innovation is not an exception, where natural language processing (NLP) techniques presents opportunities to streamline and enhance important tasks -- such as patent classification and patent retrieval -- in the patent cycle.

#### pretrained
A survey of recent NLP-based methods in patent analysis.

#### prompt_engineered
A survey on patent analysis: from NLP to Multimodal AI

#### rag_system
A Survey on Patent Analysis: From NLP to Multimodal AI

Retrieved titles: A Survey on Patent Analysis: From NLP to Multimodal AI | A Survey on Patent Analysis: From NLP to Multimodal AI | A Hybrid Architecture with Efficient Fine Tuning for Abstractive Patent Document Summarization | A Hybrid Architecture with Efficient Fine Tuning for Abstractive Patent Document Summarization
