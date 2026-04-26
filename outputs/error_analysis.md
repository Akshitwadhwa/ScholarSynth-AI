# Error Analysis and Guardrails

Generated from the latest evaluation outputs, including `rag_plus_lora` generation-level evidence.

## Source Files

- `outputs/baseline_200_comparison_table.csv`
- `outputs/baseline_200_task_metrics.csv`
- `outputs/lora_test_metrics.csv`
- `outputs/lora_test_generations.csv`
- `outputs/rag_plus_lora_200_generations.csv`
- `outputs/rag_plus_lora_200_metrics.csv`

## What Was Implemented

- Input guardrails block empty, overlong, prompt-injection, and credential-leaking queries.
- Input guardrails warn on very short, off-scope, or personal-data-like queries.
- Output guardrails flag empty answers, repetitive generations, very short answers, weak evidence overlap, generic model disclaimers, citation-like text, and strong unsupported claims.
- `app.py` displays input and output guardrail findings in expandable UI panels.

## 200-Example Aggregate Metrics

| model | eval_examples | bleu | rouge1 | rouge2 | rougeL |
| --- | --- | --- | --- | --- | --- |
| fine_tuned_lora | 200 | 29.0810 | 0.5131 | 0.3699 | 0.4376 |
| rag_plus_lora | 200 | 22.3723 | 0.4296 | 0.2673 | 0.3621 |
| pretrained | 200 | 0.3151 | 0.1679 | 0.0820 | 0.1363 |
| rag_system | 200 | 0.1292 | 0.1458 | 0.0816 | 0.1294 |
| prompt_engineered | 200 | 0.0005 | 0.1026 | 0.0594 | 0.0964 |

## 1,200-Example LoRA Test Metrics

| model | eval_examples | bleu | rouge1 | rouge2 | rougeL |
| --- | --- | --- | --- | --- | --- |
| fine_tuned_lora | 1200 | 47.2745 | 0.6561 | 0.5404 | 0.6031 |

## Guardrail Finding Counts

| strategy | message | count |
| --- | --- | --- |
| fine_tuned_lora_test | No retrieved evidence was supplied to the output guardrail; hallucination checks are limited. | 1200 |
| fine_tuned_lora_test | The answer uses strong quantitative or comparative language without an explicit citation. | 65 |
| rag_plus_lora | The answer has weak lexical overlap with retrieved evidence, so grounding may be low. | 40 |
| rag_plus_lora | Repetitive wording was detected. Treat this answer as a possible generation failure case. | 6 |
| rag_plus_lora | The answer uses strong quantitative or comparative language without an explicit citation. | 5 |

## Failure Proxy Counts

| strategy | failure_type | count |
| --- | --- | --- |
| fine_tuned_lora_test | no_major_proxy_failure | 1007 |
| fine_tuned_lora_test | low_bleu | 124 |
| fine_tuned_lora_test | low_rougeL | 77 |
| fine_tuned_lora_test | hallucination_risk | 65 |
| rag_plus_lora | low_bleu | 105 |
| rag_plus_lora | low_rougeL | 90 |
| rag_plus_lora | no_major_proxy_failure | 78 |
| rag_plus_lora | weak_evidence_overlap | 40 |
| rag_plus_lora | repetition | 6 |
| rag_plus_lora | hallucination_risk | 5 |

## Key Observations

- `fine_tuned_lora` has the best lexical metrics in the 200-example comparison.
- `rag_plus_lora` adds retrieved evidence and lands second overall by ROUGE-L, making it the better candidate for grounded demo outputs.
- `rag_plus_lora` performs especially well on `evidence_based_qa`, but weaker on comparative and research-gap tasks where retrieved context may shift wording away from the reference.
- Hallucination checks are stronger for `rag_plus_lora` because the generation file now includes retrieved titles and chunks.

## Worst RAG + LoRA Failure Cases

### Example 28 - rag_plus_lora / paper_summary
Topic: multi agent collaboration with large language models

Failure tags: `low_rougeL, low_bleu, weak_evidence_overlap`  
BLEU: 0.4301; ROUGE-L: 0.0482

Prediction: In the ad-in-rese, the paper is a a primary source, a specific method, or results in a particular process. The assing aesight is , , and a research process is .

Reference: From Control to Foresight: Simulation as a New Paradigm for Human-Agent Collaboration studies Large Language Models (LLMs) are increasingly used to power autonomous agents for complex, multi-step tasks. However, human-agent interaction remains pointwise and reactive: users approve or correct individual actions to mitigate imm...

Retrieved titles: Agentic Large Language Models, a survey | Multi-Agent Consensus Seeking via Large Language Models | LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey | Multi-Agent Collaboration Mechanisms: A Survey of LLMs

### Example 15 - rag_plus_lora / paper_summary
Topic: instruction tuned large language models

Failure tags: `low_rougeL, low_bleu`  
BLEU: 0.4549; ROUGE-L: 0.0690

Prediction: In a broader ad-ining paper, the paper is a in-tead a source, a, , . The a-sessing a and a to a specific method, s, e, t, i, -s, or results are a. The main aese is that a the aa-in-tuning paper is the s-ssed ,, and .

Reference: LIMIT: Less Is More for Instruction Tuning Across Evaluation Paradigms studies Large Language Models are traditionally finetuned on large instruction datasets. However recent studies suggest that small, high-quality datasets can suffice for general purpose instruction following.

Retrieved titles: Dynamics of Instruction Fine-Tuning for Chinese Large Language Models | Demystifying Instruction Mixing for Fine-tuning Large Language Models | LIMIT: Less Is More for Instruction Tuning Across Evaluation Paradigms | Instruction Tuning for Large Language Models: A Survey

### Example 24 - rag_plus_lora / paper_summary
Topic: QLoRA for efficient language model fine tuning

Failure tags: `low_rougeL, low_bleu, weak_evidence_overlap`  
BLEU: 0.1664; ROUGE-L: 0.0759

Prediction: On the abstract, the paper is a a source, a, , and a result. The ad is , .

Reference: From 'Showgirls' to 'Performers': Fine-tuning with Gender-inclusive Language for Bias Reduction in LLMs studies Gender bias is not only prevalent in Large Language Models (LLMs) and their training data, but also firmly ingrained into the structural aspects of language itself. Therefore, adapting linguistic structures within L...

Retrieved titles: From 'Showgirls' to 'Performers': Fine-tuning with Gender-inclusive Language for Bias Reduction in LLMs | Defining and Evaluating Fair Natural Language Generation | Social Bias in Large Language Models For Bangla: An Empirical Study on Gender and Religious Bias | From 'Showgirls' to 'Performers': Fine-tuning with Gender-inclu...

### Example 138 - rag_plus_lora / research_gap_analysis
Topic: Neural information retrieval for scholarly search

Failure tags: `low_rougeL, low_bleu`  
BLEU: 0.4753; ROUGE-L: 0.0988

Prediction: In the paper, the aps are a ad-inte ae, and the results are able to help the scholarly research team to identify research gaps. The a-se, a, , .

Reference: For Neural information retrieval for scholarly search, the retrieved papers suggest several research gaps: more reliable evaluation protocols, stronger evidence grounding, better handling of domain-specific terminology, and clearer comparison between retrieval-based, fine-tuned, and prompt-engineered systems. Future work shou...

Retrieved titles: Accessibility in Information Retrieval | Bibliometric-enhanced Information Retrieval: 2nd International BIR Workshop | Science Models as Value-Added Services for Scholarly Information Systems | ORKG ASK: a Neuro-symbolic Scholarly Search and Exploration System

### Example 22 - rag_plus_lora / paper_summary
Topic: open source large language models

Failure tags: `low_rougeL, low_bleu`  
BLEU: 0.3661; ROUGE-L: 0.1042

Prediction: In a broader research, the paper is a method, a technique, and a evaluation in a more logical, broader, systy, and more effective aesse. The ad-intee is , , and is .

Reference: Large Language Models Reasoning Abilities Under Non-Ideal Conditions After RL-Fine-Tuning studies Reinforcement learning (RL) has become a key technique for enhancing the reasoning abilities of large language models (LLMs), with policy-gradient algorithms dominating the post-training stage because of their efficiency and effe...

Retrieved titles: Reinforcement Learning Meets Large Language Models: A Survey of Advancements and Applications Across the LLM Lifecycle | Large Language Models Reasoning Abilities Under Non-Ideal Conditions After RL-Fine-Tuning | Large Language Models Reasoning Abilities Under Non-Ideal Conditions After RL-Fine-Tuning | A Literature Survey on...

### Example 29 - rag_plus_lora / paper_summary
Topic: retrieval augmented generation

Failure tags: `low_rougeL, low_bleu, weak_evidence_overlap`  
BLEU: 0.5351; ROUGE-L: 0.1087

Prediction: In the a broader asseal, the paper is a in-tead method, a more logical method, and a better acing. The underlying ad-in-ressia is that a system aeses a specific method, method, or data problem, in a the broader context.

Reference: Context Tuning for Retrieval Augmented Generation studies Large language models (LLMs) have the remarkable ability to solve new tasks with just a few examples, but they need access to the right tools. Retrieval Augmented Generation (RAG) addresses this problem by retrieving a list of relevant tools for a given task.

Retrieved titles: Active Retrieval Augmented Generation | FunnelRAG: A Coarse-to-Fine Progressive Retrieval Paradigm for RAG | A Research of Challenges and Solutions in Retrieval Augmented Generation (RAG) Systems | A Systematic Literature Review of Retrieval-Augmented Generation: Methods, Applications, and Future Research Directions


## Recommended Next Steps

1. For the final demo, show both `Fine-tuned LoRA` and `RAG + Fine-tuned LoRA` to explain the quality-vs-grounding tradeoff.
2. For future experiments, tune the RAG+LoRA prompt separately by task, especially `comparative_analysis` and `research_gap_analysis`.
3. Add manual labels for 30-50 worst cases: `grounded`, `partially grounded`, `hallucinated`, `irrelevant`, `too short`, or `repetitive`.
