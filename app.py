"""Streamlit interface for a private, entirely local document workspace."""

from dataclasses import replace
import json

from config import Settings
import streamlit as st

from chat import source_records, stream_answer
from ingest import DocumentLibrary
from local_models import inspect_ollama

st.set_page_config(page_title="Local / Document Chat", page_icon="📚", layout="wide")
st.markdown("""
<style>
    .block-container { max-width: 1100px; padding-top: 2.6rem; }
    [data-testid="stSidebar"] { border-right: 1px solid #dfe5db; }
    h1, h2, h3 { letter-spacing: -0.035em; }
    h1 { font-weight: 600; }
    .eyebrow { font-size: .72rem; font-weight: 700; letter-spacing: .16em; color: #157a6e; }
    .hero-note { font-size: 1.05rem; color: #65736a; margin-bottom: 1.8rem; max-width: 650px; }
    .local-mark { font-size: 1.5rem; font-weight: 750; letter-spacing: -.06em; margin-bottom: .2rem; }
    .local-mark span { color: #157a6e; }
    [data-testid="stChatMessage"] { border: 1px solid #e4e8e0; border-radius: 14px; }
    .step-label { color: #157a6e; font-size: .75rem; font-weight: 700; letter-spacing: .1em; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_library(settings: Settings) -> DocumentLibrary:
    return DocumentLibrary(settings)


@st.cache_data(ttl=10, show_spinner=False)
def get_status(settings: Settings):
    return inspect_ollama(settings)


def show_error(error: Exception, action: str):
    if isinstance(error, ValueError):
        st.error(str(error))
    else:
        st.error(f"Could not {action}. Check that Ollama is running and the required model is installed.")
        with st.expander("Technical details"):
            st.code(str(error), language=None)


def render_sources(sources: list[dict]):
    if sources:
        with st.expander(f"View {len(sources)} retrieved excerpts"):
            st.caption("These passages were supplied to the model. Check them to verify its answer.")
            for source in sources:
                page = f" · page {source['page']}" if source.get("page") else ""
                st.text(f"[{source['number']}] {source['name']}{page}")
                st.text(source["text"])


try:
    base_settings = Settings.from_env()
except ValueError as error:
    st.error(str(error))
    st.stop()

st.session_state.setdefault("messages", [])
status = get_status(base_settings)

with st.sidebar:
    st.markdown('<div class="local-mark">local<span>/</span></div>', unsafe_allow_html=True)
    st.caption("YOUR DOCUMENTS. YOUR MACHINE.")
    st.divider()
    if status.online:
        st.success("Ollama connected", icon="🟢")
    else:
        st.warning("Ollama is offline", icon="🟠")
    if st.button("Refresh connection", width="stretch"):
        get_status.clear()
        st.rerun()

    with st.expander("Models & settings", expanded=not status.online):
        candidates = [base_settings.chat_model] + [
            model for model in status.models
            if model != base_settings.chat_model and "embed" not in model.lower()
        ]
        chat_model = st.selectbox("Chat model", options=list(dict.fromkeys(candidates)))
        top_k = st.slider("Retrieved passages", 1, 6, 3)
        st.caption(f"Embedding model: {base_settings.embedding_model}")
        st.caption(f"Local endpoint: {base_settings.ollama_url}")
        st.caption("Changing EMBEDDING_MODEL creates a separate library; re-upload documents for that model.")

    settings = replace(base_settings, chat_model=chat_model)
    embedding_ready = status.online and status.has(settings.embedding_model)
    chat_ready = status.online and status.has(settings.chat_model)
    if not embedding_ready or not chat_ready:
        with st.expander("Set up local models", expanded=True):
            st.caption("Docker: run these in your project terminal. Native Ollama: omit 'docker compose exec ollama'.")
            if not embedding_ready:
                st.code(f"docker compose exec ollama ollama pull {settings.embedding_model}", language="bash")
            if not chat_ready:
                st.code(f"docker compose exec ollama ollama pull {settings.chat_model}", language="bash")
            st.caption("Downloads need internet once. Refresh the connection when they finish.")

    st.divider()
    st.subheader("Add documents")
    uploads = st.file_uploader(
        "PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True,
        help="Up to 25 MB per file. Text is indexed locally; original files are not retained.",
    )

try:
    # Chat model changes reuse the same embedding collection and resource.
    library = get_library(base_settings)
    documents = library.list_documents()
except Exception as error:
    st.error("The local document database could not be opened. Check DATA_DIR permissions and available disk space.")
    with st.expander("Technical details"):
        st.code(str(error), language=None)
    st.stop()

with st.sidebar:
    if st.button("Index documents", type="primary", width="stretch",
                 disabled=not uploads or not embedding_ready):
        with st.status("Indexing on your machine…", expanded=True) as progress:
            failures = 0
            for upload in uploads:
                st.text(f"Reading {upload.name}")
                try:
                    result = library.ingest(upload.name, upload.getvalue())
                    if result.duplicate:
                        st.info(f"{result.name} is already in your library.")
                    else:
                        st.success(f"{result.name} · {result.chunks} passages indexed")
                except Exception as error:
                    failures += 1
                    show_error(error, f"index {upload.name}")
            progress.update(label="Indexing finished" if not failures else "Some documents could not be indexed",
                            state="complete" if not failures else "error", expanded=True)
        documents = library.list_documents()
    st.caption("25 MB per file · Duplicates are skipped")
    st.divider()
    st.caption("Inference and search use local services. Initial installs and model downloads require internet.")

st.markdown('<div class="eyebrow">PRIVATE DOCUMENT WORKSPACE</div>', unsafe_allow_html=True)
st.title("A conversation with your documents.")
st.markdown('<div class="hero-note">Turn the things you’ve saved into answers you can trace. '
            'Ask, explore, and find the source — all on your machine.</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Documents", len(documents))
col2.metric("Indexed passages", sum(doc["chunks"] for doc in documents))
col3.metric("Inference", "Local Ollama")
chat_tab, library_tab = st.tabs(["Conversation", "Document library"])

with library_tab:
    st.subheader("Your document library")
    st.caption("Indexed text and embeddings stay on disk across restarts. Removing a document deletes its indexed passages.")
    if not documents:
        st.info("Your library is empty. Add a document from the sidebar to get started.")
    for document in documents:
        with st.container(border=True):
            info, action = st.columns([4, 1])
            info.text(document["name"])
            info.caption(f"{document['chunks']} passages · {document['bytes'] / 1024:.1f} KB · "
                         f"Indexed {document['indexed_at'][:10]} · {document['id'][:8]}")
            if action.button("Remove", key=f"delete_{document['id']}", width="stretch"):
                try:
                    library.delete(document["id"])
                    # Old answers may quote deleted text; discard session history as well.
                    st.session_state.messages = []
                    st.rerun()
                except Exception as error:
                    show_error(error, "remove this document")

with chat_tab:
    names = {doc["id"]: f"{doc['name']} · {doc['id'][:8]}" for doc in documents}
    scope, clear = st.columns([4, 1])
    selected = scope.selectbox("Search in", options=["all"] + list(names),
                               format_func=lambda value: "All documents" if value == "all" else names[value],
                               disabled=not documents)
    if clear.button("New chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    if not documents:
        st.subheader("Make room for your next good question.")
        a, b, c = st.columns(3)
        for column, number, title, text in [
            (a, "01", "Add a document", "Upload research, policies, notes, or a report."),
            (b, "02", "Index it locally", "Your local model turns text into searchable passages."),
            (c, "03", "Follow the evidence", "Ask a question and inspect the passages behind the answer."),
        ]:
            with column.container(border=True):
                st.markdown(f'<div class="step-label">{number}</div>', unsafe_allow_html=True)
                st.markdown(f"**{title}**")
                st.write(text)
    elif not embedding_ready or not chat_ready:
        st.info("Your library is ready. Start Ollama and install the models shown in the sidebar to begin chatting.")
    else:
        st.caption("Try “What are the key points?” or ask about a specific fact in your documents.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))

    if question := st.chat_input("Ask something about your documents…",
                                 disabled=not documents or not embedding_ready or not chat_ready,
                                 max_chars=4000):
        history = list(st.session_state.messages)
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            try:
                with st.spinner("Finding relevant passages…"):
                    excerpts = library.retrieve(question, document_ids=None if selected == "all" else [selected], k=top_k)
                answer = st.write_stream(stream_answer(settings, question, excerpts, history))
                if not answer:
                    raise ValueError("The local model returned an empty response. Try asking again.")
                sources = source_records(excerpts)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                render_sources(sources)
            except Exception as error:
                show_error(error, "generate an answer")

    if st.session_state.messages:
        st.download_button("Download conversation", data=json.dumps(st.session_state.messages, indent=2, ensure_ascii=False),
                           file_name="local-document-conversation.json", mime="application/json")
    st.caption("Local models can make mistakes. Use the retrieved excerpts to check important answers. Chat history lasts for this session.")
