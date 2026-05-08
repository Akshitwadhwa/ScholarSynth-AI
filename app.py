from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agents import GuardrailAgent
from src.paper_search import collect_papers_for_topic
from src.preprocessing import preprocess_papers
from src.rag_pipeline import RagGenerator
from src.vector_store import ChromaVectorStore, SQLiteMetadataStore, load_embedding_model


st.set_page_config(page_title="ScholarSynth AI", page_icon="SS", layout="wide")

DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "raw_papers.csv"
PROCESSED_CSV = DATA_DIR / "processed_papers.csv"
PAPERS_DB = DATA_DIR / "papers.db"
CHROMA_DB = DATA_DIR / "chroma" / "chroma.sqlite3"
LORA_ADAPTER_DIR = Path("models") / "flan_t5_lora"

TASK_DESCRIPTIONS = {
    "Literature Review": "Synthesize methods, findings, trends, and open questions from retrieved papers.",
    "Research Gap Analysis": "Identify missing evidence, weak evaluation areas, and future research directions.",
    "Technical Explanation": "Explain a technical topic in clear language using retrieved academic context.",
}

TASK_TO_FINETUNE_TASK = {
    "Literature Review": "literature_review",
    "Research Gap Analysis": "research_gap_analysis",
    "Technical Explanation": "technical_explanation",
}

TASK_TO_INSTRUCTION = {
    "Literature Review": "Write a readable mini literature review using the uploaded paper, user question, and retrieved evidence.",
    "Research Gap Analysis": "Identify research gaps using the uploaded paper, user question, and retrieved evidence.",
    "Technical Explanation": "Explain the technical concept in simple terms using the uploaded paper, user question, and retrieved evidence.",
}

MODEL_OPTIONS = {
    "Base FLAN-T5": "base",
    "Fine-tuned LoRA": "lora",
    "RAG + Fine-tuned LoRA": "rag_lora",
}

DETAIL_OPTIONS = {
    "Standard": {
        "tokens": 260,
        "instruction": "Write 2-3 short paragraphs. Use clear academic language.",
    },
    "Detailed": {
        "tokens": 420,
        "instruction": "Write 4-6 readable paragraphs with short section headings and concrete takeaways.",
    },
    "Report Style": {
        "tokens": 560,
        "instruction": "Write a structured report with sections: Overview, Evidence, Key Points, Limitations, and Next Steps.",
    },
}


def format_runtime_error(error: Exception) -> str:
    message = str(error).strip() or error.__class__.__name__
    network_markers = ("huggingface.co", "nodename nor servname provided", "Cannot send a request")
    if any(marker in message for marker in network_markers):
        return "Model download failed. Check internet access, then retry the action."
    return f"{error.__class__.__name__}: {message}"


def apply_theme(theme: str) -> None:
    is_dark = theme == "Dark"
    palette = {
        "bg": "#0f1411" if is_dark else "#f6f3ec",
        "panel": "#171f1a" if is_dark else "#fffdf7",
        "panel_soft": "#202a24" if is_dark else "#eee8dc",
        "text": "#f4f1e8" if is_dark else "#1f2521",
        "muted": "#aab7ad" if is_dark else "#5f6a61",
        "line": "#334139" if is_dark else "#d8d0c2",
        "accent": "#7ccf9b" if is_dark else "#216e4e",
        "accent_text": "#092013" if is_dark else "#ffffff",
        "warning": "#f2c46d" if is_dark else "#8a5a00",
        "danger": "#ff8c7a" if is_dark else "#9f2d20",
    }
    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: {palette["bg"]};
            --app-panel: {palette["panel"]};
            --app-panel-soft: {palette["panel_soft"]};
            --app-text: {palette["text"]};
            --app-muted: {palette["muted"]};
            --app-line: {palette["line"]};
            --app-accent: {palette["accent"]};
            --app-accent-text: {palette["accent_text"]};
            --app-warning: {palette["warning"]};
            --app-danger: {palette["danger"]};
        }}
        .stApp {{
            background:
                radial-gradient(circle at top left, color-mix(in srgb, var(--app-accent) 18%, transparent), transparent 30rem),
                linear-gradient(135deg, var(--app-bg), color-mix(in srgb, var(--app-bg) 85%, var(--app-panel-soft)));
            color: var(--app-text);
        }}
        [data-testid="stSidebar"] {{
            background: var(--app-panel);
            border-right: 1px solid var(--app-line);
        }}
        h1, h2, h3, h4, p, label, span {{
            color: var(--app-text);
        }}
        .hero {{
            border: 1px solid var(--app-line);
            background: linear-gradient(135deg, var(--app-panel), var(--app-panel-soft));
            padding: 1.25rem 1.4rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }}
        .hero-title {{
            font-size: clamp(2rem, 4vw, 3.4rem);
            font-weight: 800;
            line-height: 1;
            margin: 0 0 .55rem 0;
            letter-spacing: 0;
        }}
        .hero-copy {{
            color: var(--app-muted);
            max-width: 62rem;
            margin: 0;
            font-size: 1rem;
        }}
        .metric-card {{
            border: 1px solid var(--app-line);
            background: var(--app-panel);
            border-radius: 8px;
            padding: .9rem 1rem;
            min-height: 5.25rem;
        }}
        .metric-label {{
            color: var(--app-muted);
            font-size: .78rem;
            text-transform: uppercase;
            letter-spacing: .06em;
            margin-bottom: .35rem;
        }}
        .metric-value {{
            color: var(--app-text);
            font-size: 1.45rem;
            font-weight: 760;
            line-height: 1.1;
        }}
        .subtle-box {{
            border: 1px solid var(--app-line);
            background: color-mix(in srgb, var(--app-panel) 82%, transparent);
            border-radius: 8px;
            padding: 1rem;
        }}
        .status-ok {{
            color: var(--app-accent);
            font-weight: 700;
        }}
        .status-warn {{
            color: var(--app-warning);
            font-weight: 700;
        }}
        div.stButton > button:first-child {{
            border-radius: 8px;
            border: 1px solid var(--app-line);
            font-weight: 700;
        }}
        div.stButton > button[kind="primary"] {{
            background: var(--app-accent);
            color: var(--app-accent-text);
            border-color: var(--app-accent);
        }}
        [data-testid="stExpander"] {{
            border-color: var(--app-line);
            background: color-mix(in srgb, var(--app-panel) 70%, transparent);
        }}
        .stMarkdown h3 {{
            font-size: 1.2rem;
            line-height: 1.25;
            margin-top: 1rem;
            margin-bottom: .4rem;
        }}
        .stMarkdown p, .stMarkdown li {{
            font-size: 1rem;
            line-height: 1.48;
        }}
        .stMarkdown ul {{
            margin-top: .25rem;
            margin-bottom: .9rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def file_row(label: str, path: Path) -> str:
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        return f"<span class='status-ok'>Ready</span> - {label}: {size_mb:.1f} MB"
    return f"<span class='status-warn'>Missing</span> - {label}"


def lora_is_available() -> bool:
    return (LORA_ADAPTER_DIR / "adapter_config.json").exists() and (
        LORA_ADAPTER_DIR / "adapter_model.safetensors"
    ).exists()


def get_csv_count(path: Path) -> str:
    if not path.exists():
        return "Missing"
    try:
        return f"{sum(1 for _ in path.open(encoding='utf-8')) - 1:,}"
    except OSError:
        return "Unreadable"


def get_chroma_count() -> str:
    if not CHROMA_DB.exists():
        return "Missing"
    try:
        return f"{get_vector_store().collection.count():,}"
    except Exception:
        return "Unavailable"


def render_metric(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_guardrail_findings(findings, title: str = "Guardrail checks") -> None:
    if not findings:
        return

    with st.expander(title, expanded=True):
        for finding in findings:
            if finding.severity == "error":
                st.error(finding.message)
            elif finding.severity == "warning":
                st.warning(finding.message)
            else:
                st.info(finding.message)


def extract_uploaded_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    suffix = Path(uploaded_file.name).suffix.lower()
    raw_bytes = uploaded_file.getvalue()

    if suffix in {".txt", ".md"}:
        return raw_bytes.decode("utf-8", errors="ignore")

    if suffix == ".csv":
        df = pd.read_csv(io.BytesIO(raw_bytes))
        preview_columns = [column for column in ["title", "abstract", "chunk_text", "text"] if column in df.columns]
        if preview_columns:
            return "\n\n".join(df[preview_columns].fillna("").astype(str).head(20).agg(" ".join, axis=1))
        return df.fillna("").astype(str).head(20).to_csv(index=False)

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader
            except ImportError:
                return (
                    "PDF upload detected, but no PDF parser is installed. "
                    "Install pypdf or upload a .txt/.md version of the paper."
                )

        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = []
        for page in reader.pages[:8]:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages)

    return raw_bytes.decode("utf-8", errors="ignore")


def infer_topic_from_uploaded_text(uploaded_text: str, fallback_query: str) -> str:
    if not uploaded_text.strip():
        return fallback_query

    for line in uploaded_text.splitlines()[:20]:
        cleaned = " ".join(line.split())
        if 8 <= len(cleaned) <= 180 and not cleaned.lower().startswith(("abstract", "keywords")):
            return cleaned
    return fallback_query


def build_base_prompt(task: str, user_query: str, docs: list[str], metas: list[dict], uploaded_text: str, detail: str) -> str:
    evidence_lines = [
        f"Paper {index}: {meta.get('title', 'Untitled')}. Evidence: {doc[:600]}"
        for index, (doc, meta) in enumerate(zip(docs, metas), start=1)
    ]
    uploaded_section = f"\nUploaded paper excerpt:\n{uploaded_text[:1400]}\n" if uploaded_text.strip() else ""
    return (
        "You are ScholarSynth AI, a careful academic research assistant. "
        "Write a useful, readable answer grounded in the user query, uploaded paper if provided, and retrieved evidence. "
        "Do not invent citations, metrics, datasets, or paper details. If evidence is limited, say so.\n\n"
        f"Task: {task}\n"
        f"Question or topic: {user_query}\n"
        f"Output requirements: {DETAIL_OPTIONS[detail]['instruction']}\n"
        f"{uploaded_section}\n"
        "Retrieved evidence:\n"
        + "\n".join(evidence_lines)[:2200]
        + "\n\nAnswer:"
    )


def build_finetuned_prompt(
    task: str,
    user_query: str,
    docs: list[str],
    metas: list[dict],
    include_retrieval: bool,
    uploaded_text: str = "",
    detail: str = "Detailed",
) -> str:
    evidence_lines = []
    if include_retrieval:
        for index, (doc, meta) in enumerate(zip(docs, metas), start=1):
            evidence_lines.append(f"Paper {index}: {meta.get('title', 'Untitled')}. Evidence: {doc[:600]}")

    uploaded_section = ""
    if uploaded_text.strip():
        uploaded_section = f"\nUploaded paper excerpt:\n{uploaded_text[:1400]}\n"

    if evidence_lines:
        input_text = (
            f"User question: {user_query}\n"
            f"{uploaded_section}\n"
            "Retrieved evidence:\n"
            + "\n".join(evidence_lines)
        )
    else:
        input_text = f"User question: {user_query}\n{uploaded_section}"

    return (
        f"Instruction: {TASK_TO_INSTRUCTION[task]}\n"
        f"Task: {TASK_TO_FINETUNE_TASK[task]}\n"
        f"Topic: {user_query}\n"
        f"Output requirements: {DETAIL_OPTIONS[detail]['instruction']}\n"
        "Required answer format:\n"
        "Overview: explain the main idea in plain language.\n"
        "Evidence: connect the answer to the uploaded paper or retrieved papers.\n"
        "Key points: list the most important takeaways.\n"
        "Limitations or gaps: mention uncertainty or missing evidence.\n"
        f"Input:\n{input_text[:3200]}\n\n"
        "Answer:"
    )


def expand_short_answer(
    task: str,
    user_query: str,
    generated_text: str,
    docs: list[str],
    metas: list[dict],
    uploaded_text: str,
) -> str:
    if len(generated_text.split()) >= 90:
        return generated_text

    evidence_title = metas[0].get("title", "the retrieved papers") if metas else "the retrieved papers"
    evidence_note = docs[0][:320] if docs else "Retrieved evidence was limited."
    uploaded_note = uploaded_text[:320] if uploaded_text.strip() else ""

    sections = [
        f"### Overview\n{generated_text}",
        f"### Evidence Used\nThe answer is grounded mainly in `{evidence_title}` and related retrieved paper chunks.",
        f"### What This Means\nFor the task `{task}`, the system is trying to connect the topic `{user_query}` with evidence from the local research corpus.",
        f"### Supporting Context\n{evidence_note}",
    ]
    if uploaded_note:
        sections.append(f"### Uploaded Paper Context\n{uploaded_note}")
    sections.append(
        "### Limitation\nThe local fine-tuned FLAN-T5 LoRA model can produce awkward phrasing, so the retrieved evidence tab should be checked before trusting the answer."
    )
    return "\n\n".join(sections)


def has_low_fluency(text: str) -> bool:
    lowered = f" {text.lower()} "
    broken_markers = [
        " a a ",
        " , ,",
        "a-ss",
        "aess",
        " assing",
        " acss",
        " sastia",
        " ad ",
        "logical and logical",
    ]
    if any(marker in lowered for marker in broken_markers):
        return True

    tokens = re.findall(r"\b[a-zA-Z]+\b", lowered)
    repeated_pairs = sum(1 for left, right in zip(tokens, tokens[1:]) if left == right)
    hyphen_noise = len(re.findall(r"\b[a-zA-Z]{1,3}-[a-zA-Z-]{2,}\b", text))
    return repeated_pairs >= 2 or hyphen_noise >= 2


def extract_readable_sentences(text: str, limit: int = 3) -> list[str]:
    candidates = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    readable = []
    for sentence in candidates:
        if 50 <= len(sentence) <= 280 and not has_low_fluency(sentence):
            readable.append(sentence)
        if len(readable) >= limit:
            break
    return readable


def markdown_list(items: list[str]) -> str:
    clean_items = [" ".join(item.split()) for item in items if item and item.strip()]
    return "\n".join(f"- {item}" for item in clean_items)


def section(title: str, items: list[str] | str) -> str:
    if isinstance(items, str):
        body = items.strip()
    else:
        body = markdown_list(items)
    return f"### {title}\n\n{body.strip()}"


def model_draft_section(model_draft: str) -> str:
    sentences = extract_readable_sentences(model_draft, limit=3)
    if not sentences:
        return ""
    return "\n\n" + section("Model Draft Insight", sentences)


def format_answer_pointwise(text: str) -> str:
    """Normalize generated markdown so Streamlit displays compact point-wise sections."""
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return normalized

    # Ensure headings are on their own lines even if model text joined them inline.
    normalized = re.sub(r"\s*(#{2,3}\s+)", r"\n\n\1", normalized)
    normalized = re.sub(r"(?<!\n)\s+-\s+", "\n- ", normalized)

    chunks = re.split(r"\n\s*###\s+", normalized)
    intro = chunks[0].strip()
    sections = []
    if intro and not intro.startswith("###"):
        sections.append(intro)

    for chunk in chunks[1:]:
        lines = chunk.strip().splitlines()
        if not lines:
            continue
        title = lines[0].strip()
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        existing_bullets = re.findall(r"(?:^|\s)-\s+(.+?)(?=\s+-\s+|$)", body)
        if existing_bullets:
            bullets = [bullet.strip() for bullet in existing_bullets if bullet.strip()]
        else:
            bullets = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", body)
                if sentence.strip()
            ]
        if bullets:
            sections.append(section(title, bullets))
        else:
            sections.append(f"### {title}")

    return "\n\n".join(sections)


def topic_specific_explanation(user_query: str, primary_title: str) -> tuple[str, list[str]]:
    query = user_query.lower()

    if "lora" in query or "low-rank" in query:
        return (
            f"LoRA fine-tuning adapts a transformer by training small low-rank adapter weights instead of updating the full model. "
            f"For {user_query}, the most relevant retrieved source is {primary_title}, which connects LoRA-style adaptation to efficient and calibrated model fine-tuning.",
            [
                "LoRA reduces training cost because only a small number of adapter parameters are updated.",
                "The base transformer remains mostly frozen, which makes fine-tuning faster and easier to store.",
                "In this project, LoRA is used to adapt FLAN-T5 for research-assistant tasks such as summaries, explanations, and gap analysis.",
            ],
        )

    if "instruction" in query or "tuning" in query or "sft" in query:
        return (
            f"Instruction tuning adapts a language model with examples of instructions and desired answers, so it learns to follow user requests more reliably. "
            f"For {user_query}, the most relevant retrieved source is {primary_title}.",
            [
                "It is usually a supervised fine-tuning step using instruction-response pairs.",
                "It helps a base model become more useful for assistant-style tasks such as explanation, summarization, and question answering.",
                "The quality of the instruction dataset strongly affects the quality and reliability of the tuned model.",
            ],
        )

    if "hallucination" in query or "factuality" in query or "faithfulness" in query:
        return (
            f"Factuality and hallucination research studies whether model outputs are supported by evidence. "
            f"For {user_query}, the most relevant retrieved source is {primary_title}.",
            [
                "A hallucination happens when a model states information that is not supported by the source or evidence.",
                "Factuality evaluation checks whether generated claims match retrieved documents, references, or known facts.",
                "Guardrails and evidence retrieval can reduce risk, but human review is still important for high-stakes claims.",
            ],
        )

    if "rag" in query or "retrieval" in query or "augmented generation" in query:
        return (
            f"Retrieval-augmented generation combines search with generation: the system first retrieves relevant documents, then uses them as context for the answer. "
            f"For {user_query}, the strongest retrieved signal comes from {primary_title}.",
            [
                "Retrieval helps ground answers in external evidence instead of relying only on model memory.",
                "The quality of the answer depends on both retrieval relevance and the generator's ability to use the evidence.",
                "RAG is especially useful for literature review, paper Q&A, and citation-grounded research assistance.",
            ],
        )

    if "embedding" in query or "vector" in query or "semantic search" in query:
        return (
            f"Semantic search represents text as embeddings and retrieves documents with similar meaning, not just matching keywords. "
            f"For {user_query}, the most relevant retrieved source is {primary_title}.",
            [
                "Embeddings turn papers, abstracts, or chunks into numeric vectors.",
                "A vector database such as ChromaDB can quickly find nearby chunks for a user query.",
                "This retrieval step gives the generator evidence to use in the final answer.",
            ],
        )

    return (
        f"The topic {user_query} is connected to the retrieved research evidence, especially {primary_title}. "
        "The answer below summarizes the main idea using the local paper corpus.",
        [
            "The retrieved papers provide context for the selected research task.",
            "The generated answer should be checked against the evidence tab for grounding.",
            "The most reliable interpretation comes from combining the generated summary with the retrieved paper chunks.",
        ],
    )


def build_evidence_based_fallback(
    task: str,
    user_query: str,
    docs: list[str],
    metas: list[dict],
    uploaded_text: str,
) -> str:
    primary_title = metas[0].get("title", "the top retrieved paper") if metas else "the top retrieved paper"
    supporting_titles = [meta.get("title", "Untitled") for meta in metas[:3]]
    evidence_sentences = []
    for doc in docs[:3]:
        evidence_sentences.extend(extract_readable_sentences(doc, limit=2))
    if uploaded_text.strip():
        evidence_sentences = extract_readable_sentences(uploaded_text, limit=2) + evidence_sentences
    evidence_sentences = evidence_sentences[:4]

    if not evidence_sentences:
        evidence_sentences = [
            "The retrieved evidence is relevant to the topic, but the available text is not clean enough for a detailed automatic synthesis."
        ]

    if task == "Technical Explanation":
        overview, key_points = topic_specific_explanation(user_query, primary_title)
        limitation = "The local LoRA model can still produce noisy wording, so this fallback answer is built directly from retrieved evidence."
    elif task == "Research Gap Analysis":
        overview = (
            f"The retrieved papers suggest that `{user_query}` is an active research area, but several practical gaps remain."
        )
        key_points = [
            "More reliable evaluation protocols are needed across datasets and domains.",
            "Future work should compare retrieval-only, fine-tuned-only, and RAG plus fine-tuned systems more carefully.",
            "Grounding, citation quality, and failure-case analysis remain important open areas.",
        ]
        limitation = "These gaps are inferred from retrieved paper chunks and should be validated by reading the full papers."
    else:
        overview = (
            f"The literature around `{user_query}` connects model adaptation, retrieval, and evaluation methods. "
            f"The strongest retrieved signal comes from `{primary_title}`."
        )
        key_points = [
            "Recent work studies how language models can be adapted for specialized research-assistant tasks.",
            "Retrieval helps connect generated answers to external paper evidence instead of relying only on model memory.",
            "Evaluation should include both automatic metrics and qualitative failure analysis.",
        ]
        limitation = "This is a compact literature synthesis from retrieved chunks, not a replacement for a full manual review."

    evidence_bullets = "\n".join(f"- {sentence}" for sentence in evidence_sentences)
    title_bullets = "\n".join(f"- {title}" for title in supporting_titles if title)
    key_point_bullets = "\n".join(f"- {point}" for point in key_points)

    return (
        f"### Overview\n{overview}\n\n"
        f"### Evidence Used\n{evidence_bullets}\n\n"
        f"### Key Retrieved Papers\n{title_bullets}\n\n"
        f"### Key Points\n{key_point_bullets}\n\n"
        f"### Limitations\n{limitation}"
    )


def build_chunk_grounded_answer(
    task: str,
    user_query: str,
    docs: list[str],
    metas: list[dict],
    uploaded_text: str,
    model_label: str,
    model_draft: str,
) -> str:
    primary_title = metas[0].get("title", "the top retrieved paper") if metas else "the top retrieved paper"
    supporting_titles = [meta.get("title", "Untitled") for meta in metas[:4]]
    evidence_sentences = []
    for doc in docs[:4]:
        evidence_sentences.extend(extract_readable_sentences(doc, limit=2))
    if uploaded_text.strip():
        evidence_sentences = extract_readable_sentences(uploaded_text, limit=2) + evidence_sentences
    evidence_sentences = evidence_sentences[:5] or [
        "The retrieved chunks are relevant, but the text is too noisy for precise sentence extraction."
    ]

    model_note = ""
    if model_draft.strip() and not has_low_fluency(model_draft) and len(model_draft.split()) >= 20:
        model_note = model_draft_section(model_draft)

    title_bullets = "\n".join(f"- {title}" for title in supporting_titles if title)
    evidence_bullets = markdown_list(evidence_sentences)

    if task == "Technical Explanation":
        overview, key_points = topic_specific_explanation(user_query, primary_title)
        key_point_bullets = markdown_list(key_points)
        return (
            f"{section('Overview', [overview])}\n\n"
            f"{section('Explanation From Retrieved Chunks', evidence_sentences)}\n\n"
            f"{section('Key Retrieved Papers', supporting_titles)}\n\n"
            f"{section('Key Takeaways', key_points)}"
            f"{model_note}\n\n"
            f"{section('Reliability Note', [f'This answer is built from retrieved ChromaDB chunks first. The selected generator was {model_label}, but retrieved evidence is used as the main source.'])}"
        )

    if task == "Research Gap Analysis":
        gap_points = [
            "Evaluation needs to be tested across more datasets, domains, and realistic user workflows.",
            "More work is needed on grounding generated claims in retrieved evidence and citations.",
            "Comparisons between baseline prompting, fine-tuned LoRA, RAG, and RAG plus LoRA should be reported consistently.",
            "Failure cases such as repetition, weak evidence overlap, and noisy generation should be manually reviewed.",
        ]
        return (
            f"{section('Evidence Base', [f'The retrieved chunks for {user_query} are mainly connected to {primary_title} and related papers.'])}\n\n"
            f"{section('What The Retrieved Chunks Say', evidence_sentences)}\n\n"
            f"{section('Likely Research Gaps', gap_points)}\n\n"
            f"{section('Key Retrieved Papers', supporting_titles)}"
            f"{model_note}\n\n"
            f"{section('Reliability Note', ['These gaps are inferred from retrieved chunks and should be validated by reading the full papers.'])}"
        )

    themes = [
        "Methods: the retrieved papers describe model, retrieval, or evaluation approaches related to the query.",
        "Evidence grounding: the strongest outputs should connect generated text to retrieved paper chunks.",
        "Evaluation: automatic metrics are useful, but qualitative review is needed for hallucination and failure cases.",
        "Open issues: fluency, citation quality, and domain-specific reliability remain important limitations.",
    ]
    return (
        f"{section('Literature Review Overview', [f'The retrieved literature for {user_query} is centered on {primary_title} and related work from the local corpus.'])}\n\n"
        f"{section('Main Evidence From Retrieved Chunks', evidence_sentences)}\n\n"
        f"{section('Themes Across The Papers', themes)}\n\n"
        f"{section('Key Retrieved Papers', supporting_titles)}"
        f"{model_note}\n\n"
        f"{section('Short Conclusion', ['Overall, the retrieved papers suggest that this topic is useful for research-assistant systems, but the final answer should remain tied to retrieved evidence rather than unsupported model memory.'])}"
    )


def improve_generated_answer(
    task: str,
    user_query: str,
    generated_text: str,
    docs: list[str],
    metas: list[dict],
    uploaded_text: str,
) -> str:
    if has_low_fluency(generated_text):
        return build_evidence_based_fallback(task, user_query, docs, metas, uploaded_text)
    return expand_short_answer(task, user_query, generated_text, docs, metas, uploaded_text)


@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource
def get_vector_store():
    return ChromaVectorStore()


@st.cache_resource
def get_generator():
    return RagGenerator()


@st.cache_resource
def get_lora_generator():
    return RagGenerator(adapter_path=str(LORA_ADAPTER_DIR))


guardrail = GuardrailAgent()

with st.sidebar:
    st.header("Workspace")
    theme = st.radio("Theme", ["Dark", "Light"], horizontal=True)
    st.divider()
    st.caption("Data status")
    st.markdown(file_row("raw papers", RAW_CSV), unsafe_allow_html=True)
    st.markdown(file_row("processed chunks", PROCESSED_CSV), unsafe_allow_html=True)
    st.markdown(file_row("SQLite metadata", PAPERS_DB), unsafe_allow_html=True)
    st.markdown(file_row("Chroma index", CHROMA_DB), unsafe_allow_html=True)
    st.markdown(file_row("LoRA adapter", LORA_ADAPTER_DIR / "adapter_model.safetensors"), unsafe_allow_html=True)
    st.divider()
    st.caption("Tip")
    st.write("Use the existing index for normal demos. Fetching papers overwrites the current raw dataset.")

apply_theme(theme)

st.markdown(
    """
    <section class="hero">
        <div class="hero-title">ScholarSynth AI</div>
        <p class="hero-copy">
            Search a local research-paper corpus, retrieve evidence with semantic embeddings, and generate grounded academic outputs.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

status_cols = st.columns(4)
with status_cols[0]:
    render_metric("Raw Papers", get_csv_count(RAW_CSV))
with status_cols[1]:
    render_metric("Text Chunks", get_csv_count(PROCESSED_CSV))
with status_cols[2]:
    render_metric("Vector Chunks", get_chroma_count())
with status_cols[3]:
    render_metric("LoRA Adapter", "Ready" if lora_is_available() else "Missing")

st.write("")

with st.expander("Dataset and index management", expanded=False):
    st.warning(
        "Use these controls only when you intentionally want to fetch new papers or rebuild the index. "
        "Fetching papers writes to data/raw_papers.csv."
    )
    topic = st.text_input(
        "Research topic for paper fetching",
        value="Retrieval-augmented generation for scientific assistants",
    )
    semantic_api_key = st.text_input("Semantic Scholar API key (optional)", type="password")

    col1, col2 = st.columns([1, 1])
    with col1:
        fetch_clicked = st.button("Fetch Papers", use_container_width=True)
    with col2:
        build_index_clicked = st.button("Build Local Index", use_container_width=True)

    if fetch_clicked:
        query_check = guardrail.validate_query(topic)
        if not query_check.passed:
            render_guardrail_findings(query_check.findings, "Input guardrail checks")
        else:
            render_guardrail_findings(query_check.findings, "Input guardrail checks")
            with st.spinner("Fetching papers from arXiv and Semantic Scholar..."):
                raw_df = collect_papers_for_topic(topic, semantic_api_key=semantic_api_key or None)
                raw_df.to_csv(RAW_CSV, index=False)
            st.success(f"Saved {len(raw_df)} raw papers to {RAW_CSV}")
            st.dataframe(raw_df.head(10), use_container_width=True)

    if build_index_clicked:
        if not RAW_CSV.exists():
            st.warning("Fetch papers first so the system has data to preprocess.")
        else:
            try:
                with st.spinner("Cleaning papers and building the vector index..."):
                    raw_df = pd.read_csv(RAW_CSV)
                    processed_df = preprocess_papers(raw_df)
                    if processed_df.empty:
                        st.warning("No usable abstracts were found after preprocessing.")
                    else:
                        processed_df.to_csv(PROCESSED_CSV, index=False)

                        metadata_store = SQLiteMetadataStore()
                        metadata_store.upsert_papers(raw_df)

                        embedding_model = get_embedding_model()
                        vector_store = get_vector_store()
                        vector_store.index_chunks(processed_df, embedding_model)
                if not processed_df.empty:
                    st.success(f"Indexed {len(processed_df)} chunks and saved processed data to {PROCESSED_CSV}")
                    st.dataframe(processed_df.head(10), use_container_width=True)
            except Exception as exc:
                st.error(format_runtime_error(exc))

st.subheader("Ask the research assistant")
if not PROCESSED_CSV.exists():
    st.info("Fetch and index papers to activate the RAG workflow.")
elif not CHROMA_DB.exists():
    st.warning("Processed data exists, but the Chroma index is missing. Build the local index before generating answers.")
else:
    task = st.selectbox(
        "Task",
        ["Literature Review", "Research Gap Analysis", "Technical Explanation"],
        help="Choose the kind of academic output you want from the retrieved evidence.",
    )
    st.caption(TASK_DESCRIPTIONS[task])

    available_model_labels = ["Base FLAN-T5"]
    if lora_is_available():
        available_model_labels.extend(["Fine-tuned LoRA", "RAG + Fine-tuned LoRA"])
    model_label = st.selectbox(
        "Generator",
        available_model_labels,
        index=len(available_model_labels) - 1,
        help="Fine-tuned LoRA uses the adapter trained on your instruction-style research dataset.",
    )
    model_mode = MODEL_OPTIONS[model_label]
    st.info(
        "Every task first retrieves paper chunks from ChromaDB. The selected generator creates a draft, "
        "then the final answer is rewritten around retrieved evidence so it is easier to read and verify."
    )

    example_queries = {
        "Literature Review": "RAG for citation-grounded literature review generation",
        "Research Gap Analysis": "research gaps in hallucination mitigation for RAG systems",
        "Technical Explanation": "explain LoRA fine-tuning for transformer language models",
    }
    user_query = st.text_area(
        "Research question or topic",
        value=example_queries[task],
        height=130,
        help="Ask a focused academic question. The answer will be based on retrieved paper chunks.",
    )

    uploaded_file = st.file_uploader(
        "Optional: upload your paper or notes",
        type=["txt", "md", "csv", "pdf"],
        help="Upload a paper excerpt, notes, CSV, or PDF. The app uses it as extra context for the selected task.",
    )
    uploaded_text = extract_uploaded_text(uploaded_file)
    if uploaded_file is not None:
        if uploaded_text.startswith("PDF upload detected"):
            st.warning(uploaded_text)
            uploaded_text = ""
        elif uploaded_text.strip():
            inferred_topic = infer_topic_from_uploaded_text(uploaded_text, user_query)
            st.success(f"Loaded uploaded context from {uploaded_file.name}.")
            with st.expander("Preview uploaded context", expanded=False):
                st.caption(f"Auto-detected topic/title: {inferred_topic}")
                st.write(uploaded_text[:2200])

    detail = st.select_slider(
        "Answer detail",
        options=list(DETAIL_OPTIONS.keys()),
        value="Detailed",
        help="Use Detailed or Report Style when you want a longer, easier-to-read answer.",
    )

    top_k = st.slider("Retrieved evidence chunks", min_value=3, max_value=10, value=5, step=1)

    if st.button("Generate Grounded Output", type="primary", use_container_width=True):
        retrieval_query = infer_topic_from_uploaded_text(uploaded_text, user_query)
        query_check = guardrail.validate_query(retrieval_query)
        if not query_check.passed:
            render_guardrail_findings(query_check.findings, "Input guardrail checks")
        else:
            render_guardrail_findings(query_check.findings, "Input guardrail checks")
            try:
                with st.spinner("Retrieving evidence and generating answer..."):
                    embedding_model = get_embedding_model()
                    vector_store = get_vector_store()
                    retrieval_results = vector_store.semantic_search(retrieval_query, embedding_model, top_k=top_k)
                    docs = retrieval_results.get("documents", [[]])[0]
                    metas = retrieval_results.get("metadatas", [[]])[0]
                    if not docs:
                        st.warning("No relevant evidence was retrieved. Rebuild the index or try a different query.")
                    else:
                        if model_mode == "base":
                            generator = get_generator()
                            prompt = build_base_prompt(
                                task=task,
                                user_query=retrieval_query,
                                docs=docs,
                                metas=metas,
                                uploaded_text=uploaded_text,
                                detail=detail,
                            )
                            generated_text = generator.generate(prompt, max_new_tokens=DETAIL_OPTIONS[detail]["tokens"])
                        else:
                            generator = get_lora_generator()
                            prompt = build_finetuned_prompt(
                                task=task,
                                user_query=retrieval_query,
                                docs=docs,
                                metas=metas,
                                include_retrieval=True,
                                uploaded_text=uploaded_text,
                                detail=detail,
                            )
                            generated_text = generator.generate(prompt, max_new_tokens=DETAIL_OPTIONS[detail]["tokens"])

                        generated_text = build_chunk_grounded_answer(
                            task=task,
                            user_query=retrieval_query,
                            docs=docs,
                            metas=metas,
                            uploaded_text=uploaded_text,
                            model_label=model_label,
                            model_draft=generated_text,
                        )

                        evidence_titles = [meta.get("title", "") for meta in metas]
                        output_check = guardrail.validate_output(
                            generated_text,
                            evidence_docs=docs,
                            evidence_titles=evidence_titles,
                        )

                        output_tab, evidence_tab = st.tabs(["Generated output", "Retrieved evidence"])
                        with output_tab:
                            render_guardrail_findings(output_check.findings, "Output guardrail checks")
                            st.markdown("### Answer")
                            with st.container(border=True):
                                st.markdown(format_answer_pointwise(output_check.text))
                        with evidence_tab:
                            for index, (doc, meta) in enumerate(zip(docs, metas), start=1):
                                with st.expander(f"{index}. {meta.get('title', 'Untitled')}", expanded=index == 1):
                                    st.caption(meta.get("url", ""))
                                    st.write(doc)
            except Exception as exc:
                st.error(format_runtime_error(exc))
