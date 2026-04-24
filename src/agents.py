from __future__ import annotations

from dataclasses import dataclass

from src.rag_pipeline import RagGenerator, build_review_prompt, format_retrieved_context


@dataclass
class AgentResponse:
    agent_name: str
    content: str


class LiteratureReviewAgent:
    def __init__(self, generator: RagGenerator) -> None:
        self.generator = generator

    def run(self, topic: str, retrieval_results: dict) -> AgentResponse:
        context = format_retrieved_context(retrieval_results)
        prompt = build_review_prompt(topic, context, mode="literature_review")
        return AgentResponse("LiteratureReviewAgent", self.generator.generate(prompt))


class ResearchGapAgent:
    def __init__(self, generator: RagGenerator) -> None:
        self.generator = generator

    def run(self, topic: str, retrieval_results: dict) -> AgentResponse:
        context = format_retrieved_context(retrieval_results)
        prompt = build_review_prompt(topic, context, mode="gap_analysis")
        return AgentResponse("ResearchGapAgent", self.generator.generate(prompt))


class TechnicalExplainerAgent:
    def __init__(self, generator: RagGenerator) -> None:
        self.generator = generator

    def run(self, topic: str, retrieval_results: dict) -> AgentResponse:
        context = format_retrieved_context(retrieval_results)
        prompt = build_review_prompt(topic, context, mode="technical_explanation")
        return AgentResponse("TechnicalExplainerAgent", self.generator.generate(prompt))


class GuardrailAgent:
    def validate_query(self, query: str) -> tuple[bool, str]:
        normalized = query.strip()
        if not normalized:
            return False, "Please enter a research topic or question."
        if len(normalized) > 300:
            return False, "The query is too long. Please keep it under 300 characters."
        return True, ""

    def validate_output(self, text: str) -> str:
        if "I made this up" in text.lower():
            return "Output blocked because it appears unsupported."
        return text
