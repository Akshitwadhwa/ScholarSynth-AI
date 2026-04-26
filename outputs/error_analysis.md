# Error Analysis and Guardrails

Generated from the latest uploaded evaluation outputs:

- `outputs/baseline_200_comparison_table.csv`
- `outputs/baseline_200_task_metrics.csv`
- `outputs/lora_test_metrics.csv`
- `outputs/lora_test_generations.csv`

## What Was Implemented

- Input guardrails block empty, overlong, prompt-injection, and credential-leaking queries.
- Input guardrails warn on very short, off-scope, or personal-data-like queries.
- Output guardrails flag empty answers, repetitive generations, very short answers, weak evidence availability, generic model disclaimers, citation-like text, and strong unsupported claims.
- `app.py` displays input and output guardrail findings in expandable UI panels.

## 200-Example Aggregate Metrics

| model | eval_examples | bleu | rouge1 | rouge2 | rougeL |
| --- | --- | --- | --- | --- | --- |
| fine_tuned_lora | 200 | 29.0810 | 0.5131 | 0.3699 | 0.4376 |
| pretrained | 200 | 0.3151 | 0.1679 | 0.0820 | 0.1363 |
| rag_system | 200 | 0.1292 | 0.1458 | 0.0816 | 0.1294 |
| prompt_engineered | 200 | 0.0005 | 0.1026 | 0.0594 | 0.0964 |

## 200-Example Task Metrics

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

## 1,200-Example LoRA Test Metrics

| model | eval_examples | bleu | rouge1 | rouge2 | rougeL |
| --- | --- | --- | --- | --- | --- |
| fine_tuned_lora | 1200 | 47.2745 | 0.6561 | 0.5404 | 0.6031 |

## Guardrail Finding Counts on LoRA Test Generations

| strategy | message | count |
| --- | --- | --- |
| fine_tuned_lora_test | No retrieved evidence was supplied to the output guardrail; hallucination checks are limited. | 1200 |
| fine_tuned_lora_test | The answer uses strong quantitative or comparative language without an explicit citation. | 65 |

## Failure Proxy Counts on LoRA Test Generations

| strategy | failure_type | count |
| --- | --- | --- |
| fine_tuned_lora_test | no_major_proxy_failure | 1007 |
| fine_tuned_lora_test | low_bleu | 124 |
| fine_tuned_lora_test | low_rougeL | 77 |
| fine_tuned_lora_test | hallucination_risk | 65 |

## Key Observations

- The new 200-example results show `fine_tuned_lora` clearly outperforming the other saved baselines on BLEU/ROUGE.
- The 1,200-example LoRA test set confirms the adapter is much stronger than the earlier smoke-test behavior suggested.
- The remaining qualitative risks are mostly incomplete or generic answers, plus cases where automatic hallucination checks are limited because the LoRA generation file does not include retrieved evidence chunks.
- For a stronger hallucination analysis, save retrieved chunks/titles beside each generation in the next evaluation run.

## Worst LoRA Test Failure Cases

### Example 204 - fine_tuned_lora_test
Failure tags: `low_rougeL, low_bleu`  
BLEU: 0.0049; ROUGE-L: 0.0889

Prediction: In the papers, this paper is a problem: Bibliometric-enhanced Information Retrieval: 2nd International BIR Workshop. This workshop brings together experts of communities which often have been perceived as different once: bibliometrics / scientometris or inf...

Reference: For Neural information retrieval for scholarly search, the retrieved papers suggest several research gaps: more reliable evaluation protocols, stronger evidence grounding, better handling of domain-specific terminology, and clearer comparison between retrie...

### Example 168 - fine_tuned_lora_test
Failure tags: `low_rougeL, low_bleu`  
BLEU: 0.0047; ROUGE-L: 0.0980

Prediction: In this paper, a new method is that it provides the information in a specific research area. The purpose is to provide a more accurate and clearer study, as a better understanding of a problem, able for an effective evaluation, and on-thestining.

Reference: LayoutLLM: Layout Instruction Tuning with Large Language Models for Document Understanding studies Recently, leveraging large language models (LLMs) or multimodal large language models (MLLMs) for document understanding has been proven very promising. Howev...

### Example 472 - fine_tuned_lora_test
Failure tags: `low_rougeL, low_bleu`  
BLEU: 0.0087; ROUGE-L: 0.1000

Prediction: In this paper, a more detailed explanation of the data is that in this paper. The purpose is to make it easier on the research team and with an as-sode-intirating method, for a better understanding of MLLMs.

Reference: Position-Enhanced Visual Instruction Tuning for Multimodal Large Language Models studies Recently, Multimodal Large Language Models (MLLMs) that enable Large Language Models (LLMs) to interpret images through visual instruction tuning have achieved signific...

### Example 1044 - fine_tuned_lora_test
Failure tags: `low_rougeL, low_bleu, hallucination_risk`  
BLEU: 0.0050; ROUGE-L: 0.1053

Prediction: In the papers, this paper is a about A Hybrid Knowledge-Grounded Framework for Safety and Traceability in Prescription Verification. This paper is on an important method to identify a significant threat to patient safety, making pharmacist verification (PV)...

Reference: The papers on evidence grounded generation are related because they address complementary parts of the same research area. Some papers emphasize model or system design, while others focus on evaluation, reliability, or domain-specific application. Together,...

### Example 811 - fine_tuned_lora_test
Failure tags: `low_rougeL, low_bleu`  
BLEU: 0.0061; ROUGE-L: 0.1124

Prediction: In the papers, this paper is a problem: Medical Question Understanding and Answering with Knowledge Grounding and Semantic Self-Supervision. This issue is that it helps readers understand a specific method, system, or evaluation problems in the broader rese...

Reference: For grounded question answering, the retrieved papers suggest several research gaps: more reliable evaluation protocols, stronger evidence grounding, better handling of domain-specific terminology, and clearer comparison between retrieval-based, fine-tuned,...

### Example 847 - fine_tuned_lora_test
Failure tags: `low_rougeL, low_bleu`  
BLEU: 0.0072; ROUGE-L: 0.1143

Prediction: In a specific study, this paper is in the to-insere. The purpose is that it helps readers understand a different method, system, or research problem in the broader research area.

Reference: H2O Open Ecosystem for State-of-the-art Large Language Models studies Large Language Models (LLMs) represent a revolution in AI. However, they also pose many significant risks, such as the presence of biased, private, copyrighted or harmful text.

### Example 24 - fine_tuned_lora_test
Failure tags: `low_rougeL, low_bleu`  
BLEU: 0.0058; ROUGE-L: 0.1149

Prediction: In a specific paper, this papers is on multiple documents in the research area: AgreeSum: Agreement-Oriented Multi-Document Summarization. The purpose is to provide abstractive summaries that represent information common and faithful to all input articles.

Reference: For multi document scientific summarization, the retrieved papers suggest several research gaps: more reliable evaluation protocols, stronger evidence grounding, better handling of domain-specific terminology, and clearer comparison between retrieval-based,...

### Example 401 - fine_tuned_lora_test
Failure tags: `low_rougeL, low_bleu`  
BLEU: 0.0054; ROUGE-L: 0.1149

Prediction: In the papers, this paper is in that Personalized Search Via Neural Contextual Semantic Relevance Ranking. This paper shows a research gaps for Smantically-Enriched Research Engine for Geoportals: A Case Study with ArcGIs Online.

Reference: For Semantic search for research paper exploration, the retrieved papers suggest several research gaps: more reliable evaluation protocols, stronger evidence grounding, better handling of domain-specific terminology, and clearer comparison between retrieval...


## Recommended Next Steps

1. Add `task`, `topic`, and retrieved evidence columns to future LoRA generation CSVs.
2. Run the same 200 examples through `rag_plus_lora` so the comparison includes both fine-tuned-only and retrieval-grounded fine-tuned generation.
3. Manually label 30-50 worst cases as `grounded`, `partially grounded`, `hallucinated`, `irrelevant`, `too short`, or `repetitive`.
4. Use the guardrail warnings in the Streamlit app demo to explain failure cases transparently.
