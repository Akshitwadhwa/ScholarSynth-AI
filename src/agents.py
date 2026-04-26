from __future__ import annotations

from dataclasses import dataclass
import re

from src.rag_pipeline import RagGenerator, build_review_prompt, format_retrieved_context


@dataclass
class AgentResponse:
    agent_name: str
    content: str


@dataclass
class GuardrailFinding:
    severity: str
    message: str


@dataclass
class GuardrailResult:
    passed: bool
    text: str
    findings: list[GuardrailFinding]

    @property
    def errors(self) -> list[str]:
        return [finding.message for finding in self.findings if finding.severity == "error"]

    @property
    def warnings(self) -> list[str]:
        return [finding.message for finding in self.findings if finding.severity == "warning"]

    @property
    def info(self) -> list[str]:
        return [finding.message for finding in self.findings if finding.severity == "info"]


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
    INJECTION_PATTERNS = [
        r"\bignore (all )?(previous|prior|above) instructions\b",
        r"\breveal (the )?(system|developer|hidden) (prompt|message|instructions)\b",
        r"\b(system|developer) prompt\b",
        r"\bjailbreak\b",
        r"\bact as (dan|an unrestricted|a malicious)\b",
        r"\bdisregard (the )?(rules|guardrails|instructions)\b",
    ]
    SECRET_PATTERNS = [
        r"sk-[A-Za-z0-9_-]{20,}",
        r"(?i)\b(api[_ -]?key|secret|password|token)\s*[:=]\s*\S{8,}",
    ]
    RESEARCH_TERMS = {
        "paper",
        "papers",
        "research",
        "study",
        "studies",
        "literature",
        "review",
        "method",
        "methods",
        "model",
        "models",
        "dataset",
        "evaluation",
        "evidence",
        "rag",
        "transformer",
        "llm",
        "ai",
        "nlp",
        "machine",
        "learning",
    }
    UNSUPPORTED_CLAIM_PATTERNS = [
        r"\b\d+(\.\d+)?\s?%\b",
        r"\bp\s*[<=>]\s*0?\.\d+\b",
        r"\bsignificant(ly)?\b",
        r"\bstate[- ]of[- ]the[- ]art\b",
        r"\boutperform(s|ed)?\b",
        r"\bproves?\b",
        r"\bguarantees?\b",
    ]
    CITATION_PATTERNS = [
        r"\[[0-9,\s-]{1,20}\]",
        r"\b[A-Z][A-Za-z-]+ et al\.,? \d{4}\b",
        r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b",
        r"\barXiv:\d{4}\.\d{4,5}\b",
    ]

    def validate_query(self, query: str) -> GuardrailResult:
        normalized = query.strip()
        findings: list[GuardrailFinding] = []

        if not normalized:
            findings.append(GuardrailFinding("error", "Please enter a research topic or question."))
            return GuardrailResult(False, normalized, findings)

        if len(normalized) > 300:
            findings.append(GuardrailFinding("error", "The query is too long. Please keep it under 300 characters."))

        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in self.INJECTION_PATTERNS):
            findings.append(
                GuardrailFinding(
                    "error",
                    "Prompt-injection style instructions were detected. Ask a normal research question instead.",
                )
            )

        if any(re.search(pattern, normalized) for pattern in self.SECRET_PATTERNS):
            findings.append(
                GuardrailFinding(
                    "error",
                    "The query appears to contain a secret or credential. Remove it before submitting.",
                )
            )

        word_count = len(re.findall(r"\b\w+\b", normalized))
        if word_count < 4:
            findings.append(
                GuardrailFinding(
                    "warning",
                    "The query is very short. A more specific research question usually retrieves better evidence.",
                )
            )

        query_terms = set(re.findall(r"\b[a-zA-Z]{3,}\b", normalized.lower()))
        if query_terms and not query_terms.intersection(self.RESEARCH_TERMS):
            findings.append(
                GuardrailFinding(
                    "warning",
                    "The query may be outside the academic-research scope of this assistant.",
                )
            )

        if re.search(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", normalized) or re.search(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", normalized):
            findings.append(
                GuardrailFinding(
                    "warning",
                    "The query may contain personal data. Avoid entering private user information.",
                )
            )

        return GuardrailResult(not any(finding.severity == "error" for finding in findings), normalized, findings)

    def validate_output(
        self,
        text: str,
        evidence_docs: list[str] | None = None,
        evidence_titles: list[str] | None = None,
    ) -> GuardrailResult:
        normalized = " ".join((text or "").split())
        findings: list[GuardrailFinding] = []
        evidence_docs = evidence_docs or []
        evidence_titles = evidence_titles or []
        has_evidence = bool(evidence_docs or evidence_titles)
        evidence_blob = " ".join(evidence_docs + evidence_titles).lower()

        if not normalized:
            findings.append(GuardrailFinding("error", "The model returned an empty answer."))
            return GuardrailResult(False, "", findings)

        lowered = normalized.lower()
        if "i made this up" in lowered or "unsupported answer" in lowered:
            findings.append(GuardrailFinding("error", "The model output appears to admit unsupported content."))

        if any(marker in lowered for marker in ["as an ai language model", "i do not have access", "cannot answer"]):
            findings.append(
                GuardrailFinding(
                    "warning",
                    "The answer contains generic model-disclaimer language instead of grounded research synthesis.",
                )
            )

        if len(normalized.split()) < 12:
            findings.append(
                GuardrailFinding(
                    "warning",
                    "The answer is very short and may not satisfy the selected research task.",
                )
            )

        if self._has_repetition(normalized):
            findings.append(
                GuardrailFinding(
                    "warning",
                    "Repetitive wording was detected. Treat this answer as a possible generation failure case.",
                )
            )

        if has_evidence:
            answer_terms = self._content_terms(normalized)
            evidence_terms = self._content_terms(evidence_blob)
            overlap = answer_terms.intersection(evidence_terms)
            if answer_terms and len(overlap) / len(answer_terms) < 0.18:
                findings.append(
                    GuardrailFinding(
                        "warning",
                        "The answer has weak lexical overlap with retrieved evidence, so grounding may be low.",
                    )
                )
        else:
            findings.append(
                GuardrailFinding(
                    "info",
                    "No retrieved evidence was supplied to the output guardrail; hallucination checks are limited.",
                )
            )

        has_citation = any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in self.CITATION_PATTERNS)
        if has_citation and evidence_blob:
            findings.append(
                GuardrailFinding(
                    "warning",
                    "Citation-like text was detected. Verify that any citation or identifier appears in the retrieved evidence.",
                )
            )

        unsupported_claim = any(
            re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in self.UNSUPPORTED_CLAIM_PATTERNS
        )
        if unsupported_claim and not has_citation:
            findings.append(
                GuardrailFinding(
                    "warning",
                    "The answer uses strong quantitative or comparative language without an explicit citation.",
                )
            )

        safe_text = normalized if not any(finding.severity == "error" for finding in findings) else (
            "Output blocked by guardrails because it may be unsupported or invalid."
        )
        return GuardrailResult(not any(finding.severity == "error" for finding in findings), safe_text, findings)

    def _has_repetition(self, text: str) -> bool:
        tokens = re.findall(r"\b[a-zA-Z]{1,}\b", text.lower())
        if len(tokens) < 8:
            return False

        for index in range(len(tokens) - 2):
            if tokens[index] == tokens[index + 1] == tokens[index + 2]:
                return True

        trigrams = [" ".join(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
        return any(trigrams.count(trigram) >= 3 for trigram in set(trigrams))

    def _content_terms(self, text: str) -> set[str]:
        stopwords = {
            "about",
            "after",
            "also",
            "and",
            "are",
            "based",
            "because",
            "been",
            "being",
            "can",
            "for",
            "from",
            "has",
            "have",
            "into",
            "its",
            "may",
            "more",
            "only",
            "paper",
            "papers",
            "research",
            "show",
            "shows",
            "such",
            "that",
            "the",
            "their",
            "there",
            "this",
            "through",
            "using",
            "with",
        }
        return {
            token
            for token in re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
            if token not in stopwords
        }
