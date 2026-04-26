from __future__ import annotations

import io
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
    if model_mode == "lora":
        st.info("Fine-tuned LoRA mode uses your trained adapter. Retrieval is still shown, but the prompt focuses on the user query.")
    elif model_mode == "rag_lora":
        st.info("RAG + Fine-tuned LoRA uses retrieved paper chunks inside the fine-tuned instruction format.")

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
                                include_retrieval=model_mode == "rag_lora",
                                uploaded_text=uploaded_text,
                                detail=detail,
                            )
                            generated_text = generator.generate(prompt, max_new_tokens=DETAIL_OPTIONS[detail]["tokens"])

                        generated_text = expand_short_answer(
                            task=task,
                            user_query=retrieval_query,
                            generated_text=generated_text,
                            docs=docs,
                            metas=metas,
                            uploaded_text=uploaded_text,
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
                            st.markdown(f"<div class='subtle-box'>{output_check.text}</div>", unsafe_allow_html=True)
                        with evidence_tab:
                            for index, (doc, meta) in enumerate(zip(docs, metas), start=1):
                                with st.expander(f"{index}. {meta.get('title', 'Untitled')}", expanded=index == 1):
                                    st.caption(meta.get("url", ""))
                                    st.write(doc)
            except Exception as exc:
                st.error(format_runtime_error(exc))
