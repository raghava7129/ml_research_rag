import streamlit as st
import os
import shutil
from rag.chain import RAGChain
from rag.ingest import DocumentIngester

from rag.config import validate_model_access, LLM_MODEL_NAME, DEFAULT_PROMPT_TEMPLATE

st.set_page_config(
    page_title="ML Research RAG Demo",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 ML Research Paper Assistant")
st.markdown("Ask questions grounded in your uploaded ML research papers.")
st.divider()

# ────────────────────── Session state initialization ─────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None # RAGChain object, built after ingestion

if "ingested" not in st.session_state:
    st.session_state.ingested = False # tracks whether papers have been ingested

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ──────────────────────── Model access check at startup ────────────────────
if "model_validated" not in st.session_state:
    with st.spinner("Checking model access..."):
        if validate_model_access():
            st.session_state.model_validated = True
        else:
            st.error(
                f"Model `{LLM_MODEL_NAME}` is not accessible with your API key. "
                "Check your `config.py` and `.env` file."
            )
            st.stop()

# ────────────────────── Sidebar ─────────────────────
with st.sidebar:
    st.title("📄 Upload Research Papers")
    st.markdown("Upload one or more ML research papers in PDF format.")

    uploaded_files = st.file_uploader(
        label="Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.uploader_key}",
    )

    if st.button("🚀 Ingest Papers", disabled=not uploaded_files):
        # save uploaded PDFs to data/papers/
        with st.spinner("Saving uploaded papers..."):
            for file in uploaded_files:
                save_path = f"data/papers/{file.name}"
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())

        # run ingestion pipeline
        with st.spinner("Ingesting papers into vectorstore (this may take a minute)..."):
            try:
                ingester = DocumentIngester()
                ingester.ingest()
                st.session_state.ingested = True

                #build RAG chain once after ingestion
                st.session_state.rag_chain = RAGChain(prompt_template=DEFAULT_PROMPT_TEMPLATE)
                st.success(f"✅ {len(uploaded_files)} paper(s) ingested successfully!")

            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")

    if st.session_state.ingested:
        st.info("🟢 Vectorstore is ready — ask your questions!")
    else:
        st.warning("⚠️ No papers ingested yet. Upload PDFs and click Ingest.")

    st.divider()
    if st.button("🗑️ Clear & Reset", disabled=not st.session_state.ingested):
        # clear session state
        st.session_state.ingested = False
        st.session_state.rag_chain = None
        st.session_state.messages = []

        # force file_uploader to reset selected files
        st.session_state.uploader_key += 1

        # delete saved PDFs from disk and recreate folder
        if os.path.exists("data/papers/"):
            shutil.rmtree("data/papers/")
        os.makedirs("data/papers/", exist_ok=True)

        # delete chromadb from disk
        if os.path.exists("data/chroma_db/"):
            shutil.rmtree("data/chroma_db/")

        st.success("✅ Reset complete. Upload new papers to start again.")
        st.rerun()

    # Prompt template editor
    st.divider()
    st.markdown("### ⚙️ Prompt Template")
    st.caption("Customize assistant instructions.")

    # initialize prompt in session state
    if "prompt_template" not in st.session_state:
        st.session_state.prompt_template = DEFAULT_PROMPT_TEMPLATE

    edited_prompt = st.text_area(
        label="Prompt Template",
        value=st.session_state.prompt_template,
        height=200,
        label_visibility="collapsed",
    )

    if st.button("Apply Prompt"):
        # validate prompt has meaningful instruction text
        if not edited_prompt.strip():
            st.error("Prompt instructions cannot be empty.")
        else:
            st.session_state.prompt_template = edited_prompt
            # rebuild chain with new prompt if already ingested
            if st.session_state.ingested:
                with st.spinner("Rebuilding chain with new prompt..."):
                    st.session_state.rag_chain = RAGChain(
                        prompt_template=edited_prompt
                    )
            st.success("Prompt updated!")

# ────────────────────── Chat history display ─────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # if assistant message has sources, show them in an expander
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 View Sources"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}** — `{doc.metadata.get('source', 'Unknown')}`")
                    st.caption(doc.page_content[:300] + "...")
                    st.divider()

# ────────────────────── Chat ─────────────────────
if not st.session_state.ingested:
    st.info("Upload research papers in the sidebar and click **Ingest Papers** to start chatting.")

if prompt := st.chat_input(
    placeholder="Ask a question about your research papers...",
    disabled=not st.session_state.ingested,  # locked until papers are ingested
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from RAG chain.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.rag_chain.ask(prompt)
            answer = result["answer"]
            sources = result["sources"]

        st.markdown(answer)

        # Display sources in expander
        if sources:
            with st.expander("📄 View Sources"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1}** — `{doc.metadata.get('source', 'Unknown')}`")
                    st.caption(doc.page_content[:300] + "...")
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
