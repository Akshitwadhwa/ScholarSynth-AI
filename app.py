from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.agents import GuardrailAgent, LiteratureReviewAgent, ResearchGapAgent, TechnicalExplainerAgent
from src.paper_search import collect_papers_for_topic
from src.preprocessing import preprocess_papers
from src.rag_pipeline import RagGenerator
from src.vector_store import ChromaVectorStore, SQLiteMetadataStore, load_embedding_model


st.set_page_config(page_title="Autonomous Research Assistant", layout="wide")

DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "raw_papers.csv"
PROCESSED_CSV = DATA_DIR / "processed_papers.csv"


def format_runtime_error(error: Exception) -> str:
    message = str(error).strip() or error.__class__.__name__
    network_markers = ("huggingface.co", "nodename nor servname provided", "Cannot send a request")
    if any(marker in message for marker in network_markers):
        return "Model download failed. Check internet access, then retry the action."
    return f"{error.__class__.__name__}: {message}"


@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource
def get_vector_store():
    return ChromaVectorStore()


@st.cache_resource
def get_generator():
    return RagGenerator()


guardrail = GuardrailAgent()

st.title("Autonomous Research Assistant")
st.caption("Semantic paper retrieval, RAG-based literature review generation, and research gap analysis.")

topic = st.text_input("Enter a research topic", value="Retrieval-augmented generation for scientific assistants")
semantic_api_key = st.text_input("Semantic Scholar API Key (optional)", type="password")

col1, col2 = st.columns([1, 1])
with col1:
    fetch_clicked = st.button("Fetch Papers", use_container_width=True)
with col2:
    build_index_clicked = st.button("Build Local Index", use_container_width=True)

if fetch_clicked:
    is_valid, error = guardrail.validate_query(topic)
    if not is_valid:
        st.error(error)
    else:
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

if PROCESSED_CSV.exists():
    st.subheader("Ask the System")
    task = st.selectbox(
        "Choose a task",
        ["Literature Review", "Research Gap Analysis", "Technical Explanation"],
    )
    user_query = st.text_area("Topic or question", value=topic, height=120)

    if st.button("Generate Output", use_container_width=True):
        is_valid, error = guardrail.validate_query(user_query)
        if not is_valid:
            st.error(error)
        else:
            try:
                embedding_model = get_embedding_model()
                vector_store = get_vector_store()
                retrieval_results = vector_store.semantic_search(user_query, embedding_model, top_k=5)
                docs = retrieval_results.get("documents", [[]])[0]
                metas = retrieval_results.get("metadatas", [[]])[0]
                if not docs:
                    st.warning("No relevant evidence was retrieved. Rebuild the index or try a different query.")
                else:
                    generator = get_generator()

                    if task == "Literature Review":
                        agent = LiteratureReviewAgent(generator)
                    elif task == "Research Gap Analysis":
                        agent = ResearchGapAgent(generator)
                    else:
                        agent = TechnicalExplainerAgent(generator)

                    response = agent.run(user_query, retrieval_results)
                    safe_output = guardrail.validate_output(response.content)
                    st.markdown("### Generated Output")
                    st.write(safe_output)

                    st.markdown("### Retrieved Evidence")
                    for index, (doc, meta) in enumerate(zip(docs, metas), start=1):
                        st.markdown(f"**{index}. {meta.get('title', 'Untitled')}**")
                        st.caption(meta.get("url", ""))
                        st.write(doc)
            except Exception as exc:
                st.error(format_runtime_error(exc))
else:
    st.info("Fetch and index papers to activate the RAG workflow.")
