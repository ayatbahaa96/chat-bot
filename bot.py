# Updated bot.py -- edits based on user's eval report
import os
import time
import streamlit as st
from typing import List, Dict, Optional
import re

from langchain_huggingface import (
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
    ChatHuggingFace,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document

# ---------------------------
# Streamlit page config + CSS
# ---------------------------
st.set_page_config(
    page_title="Enhanced AI Chatbot (Hugging Face)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header { font-size: 2.2rem; color: #2E86AB; text-align: center; margin-bottom: 1rem; }
    .source-doc { background-color: #FFF3E0; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; font-size: 0.9rem;}
    .warn { color: #B00020; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)

DB_FAISS_PATH = "vectorstore/db_faiss"

# ---------------------------
# Cached: load your FAISS vectorstore
# ---------------------------
@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        db = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

# ---------------------------
# Prompt template (stricter)
# ---------------------------
def create_enhanced_prompt() -> PromptTemplate:
    # Enforce: every technical claim MUST have source in form: [DocTitle | page N | chunk_id]
    # Enforce output sections: Short summary, Sources table, Service/Test Conditions Table, Safety example, Mini-calculation, Verification matrix
    template = """
You are an engineering assistant with access to document context. Use the provided context to answer precisely and with traceable sources.

RULES:
1) For EVERY technical claim include a source reference EXACTLY in the form: [DocumentName | page <n> | chunk_id=<id>]. If claim cannot be sourced from context, state: "(no source in context)" next to the claim.
2) If you reference figures (e.g., "Fig. 10.25" or "Şekil 10.25"), check source documents and include the exact document+page. If not found, write: "**HALÜSİNASYON: citation not found in KB**".
3) Provide numeric thresholds where possible. For safety factors, show a worked mini-calculation example.
4) Include these sections (use Markdown headings): 
   - **Short Summary**
   - **Key Claims & Sources** (bulleted; each item ends with [Doc | page N | chunk_id=X])
   - **Service/Test Conditions Table** (Markdown table: Area | Load spectrum | Temperature | RH | Impact | Standard | Target life)
   - **Safety Philosophy (example calc)**: show σ_allow calculation with numbers
   - **Verification Matrix** (table: Coupon | Condition | n | Metric | Acceptance band)
   - **Notes & Uncertainties** (explicitly list any assumptions)
5) Be concise, copy-paste friendly, and avoid vague "may" without numbers.

Context from documents:
{context}

Chat History:
{chat_history}

Human Question: {question}

Assistant Response (provide all sections; if info missing, explicitly say what is missing and which documents would be needed):"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"],
    )

# ---------------------------------------
# Hugging Face model registry (repo IDs)
# ---------------------------------------
def available_hf_models() -> dict:
    # corrected mappings; update repo ids to ones you have access to
    return {
        "Mistral 7B Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
        "Llama 3 8B Instruct": "meta-llama/Llama-3-8b-instruct",          # example corrected id
        "Llama 3 70B Instruct": "meta-llama/Llama-3-70b-instruct",        # corrected id placeholder
        # Add/adjust to match your HF account access
    }

# ------------------------------------------------------
# Build a ConversationalRetrievalChain with HF endpoint
# ------------------------------------------------------
def create_qa_chain(
    repo_id: str,
    vectorstore,
    temperature: float = 0.1,
    k: int = 4,
):
    try:
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        if not hf_token:
            st.error(
                "❌ Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN)."
            )
            return None

        endpoint = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=temperature,
            huggingfacehub_api_token=hf_token,
            max_new_tokens=1024,
        )
        llm = ChatHuggingFace(llm=endpoint)

        memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
        )

        prompt = create_enhanced_prompt()

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": k}
            ),
            memory=memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": prompt},
        )
        return chain
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

# ---------------------------
# Helpers
# ---------------------------
def format_source_documents(source_docs: List[Document]) -> str:
    if not source_docs:
        return "No source documents found."
    formatted = []
    for i, doc in enumerate(source_docs, 1):
        content = doc.page_content
        preview = (content[:400] + "...") if len(content) > 400 else content
        source = doc.metadata.get("source", "Unknown source")
        raw_page = doc.metadata.get("page", None)
        page = f"{int(raw_page)+1}" if raw_page is not None and str(raw_page).isdigit() else raw_page
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        formatted.append(
            f"**Source {i}:** {source} (page {page}, chunk_id={chunk_id})\n\n*Preview:* {preview}"
        )
    return "\n\n---\n\n".join(formatted)

def find_figure_in_sources(source_docs: List[Document], figure_str: str) -> Optional[Document]:
    # search for figure string (e.g., "Fig. 10.25" or "Şekil 10.25") in source_docs content/pages
    for doc in source_docs:
        if figure_str.lower() in doc.page_content.lower():
            return doc
    return None

def hallucination_check(answer_text: str, source_docs: List[Document]) -> List[str]:
    flags = []
    # look for "Fig." or "Şekil" mentions and verify presence in sources
    for m in re.finditer(r'(Fig\.|Şekil|Figure)\s*\d+(\.\d+)?', answer_text, flags=re.IGNORECASE):
        fig = m.group(0)
        found = find_figure_in_sources(source_docs, fig)
        if not found:
            flags.append(f"HALÜSİNASYON: referenced {fig} not found in retrieved sources.")
    # look for DOI-like patterns and check presence
    for doi in re.finditer(r'doi:\s*10\.\d{4,9}/\S+', answer_text, flags=re.IGNORECASE):
        doi_s = doi.group(0)
        # naive check: see if any source contains the doi string
        if not any(doi_s.lower() in (doc.page_content or "").lower() for doc in source_docs):
            flags.append(f"HALÜSİNASYON: referenced {doi_s} not found in sources.")
    return flags

# Mini safety calculation (example) - implements the sample from your report
def compute_allowable(f_t0: float, hot_wet_knockdown_pct: float, gamma_M: float) -> Dict[str, float]:
    """
    Example: F_t0(dry)=800 MPa; hot-wet −%20 → 640 MPa; γ_M=1.5 ⇒ σ_allow≈427 MPa
    hot_wet_knockdown_pct: e.g., 20 for 20%
    """
    f_hot_wet = f_t0 * (1.0 - hot_wet_knockdown_pct / 100.0)
    sigma_allow = f_hot_wet / gamma_M
    return {"f_hot_wet": f_hot_wet, "sigma_allow": sigma_allow}

# Evaluation function: scores an assistant response using the 10-item scale & weights from your report
EVAL_WEIGHTS = [1.5, 2.0, 1.5, 1.0, 1.5, 1.0, 0.5, 1.5, 0.5, 1.0]  # weights for items 1..10
def score_response_manual(grades: List[int]) -> Dict:
    """
    grades: list of 10 integers each in {0,1,2}
    returns weighted total and breakdown
    """
    if len(grades) != 10:
        raise ValueError("grades must be length 10")
    breakdown = []
    total = 0.0
    for i, g in enumerate(grades):
        contrib = g * EVAL_WEIGHTS[i]
        breakdown.append({"item": i+1, "grade": g, "weight": EVAL_WEIGHTS[i], "contrib": contrib})
        total += contrib
    return {"breakdown": breakdown, "total": total, "max_score": 24.0}

def display_chat(role: str, content: str, sources: str | None = None):
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            st.markdown(content)
            if sources and st.session_state.get("show_sources", True):
                with st.expander("📚 Source Documents", expanded=False):
                    st.markdown(sources)

# ---------------------------
# Main App
# ---------------------------
def main():
    st.markdown("<h1 class='main-header'>🤖 Enhanced AI Chatbot (Hugging Face) — RAG with eval tooling</h1>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        models = available_hf_models()
        selected_model_name = st.selectbox("Select Hugging Face Model:", options=list(models.keys()), index=0)
        selected_repo = models[selected_model_name]

        with st.expander("🔧 Advanced Settings"):
            temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.1, 0.1)
            k_docs = st.slider("Number of source documents (k)", 1, 11, 4)
            show_sources = st.checkbox("Show source documents", value=True)
        st.session_state["show_sources"] = show_sources

        st.header("📊 Session Stats")
        if "messages" in st.session_state:
            st.metric("Messages", len(st.session_state.messages))
            user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
            st.metric("Questions Asked", user_msgs)

    # Init session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "vectorstore" not in st.session_state:
        with st.spinner("Loading knowledge base..."):
            st.session_state.vectorstore = get_vectorstore()

    # Show history
    for m in st.session_state.messages:
        display_chat(m["role"], m["content"], m.get("sources"))

    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        display_chat("user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if st.session_state.vectorstore is None:
            st.error("Vector store not loaded. Check your data directory.")
            return

        # (Re)build chain if needed
        if (
            st.session_state.qa_chain is None
            or st.session_state.get("current_repo") != selected_repo
            or st.session_state.get("current_temp") != temperature
            or st.session_state.get("current_k") != k_docs
        ):
            with st.spinner("Initializing Hugging Face model..."):
                st.session_state.qa_chain = create_qa_chain(
                    repo_id=selected_repo,
                    vectorstore=st.session_state.vectorstore,
                    temperature=temperature,
                    k=k_docs,
                )
                st.session_state.current_repo = selected_repo
                st.session_state.current_temp = temperature
                st.session_state.current_k = k_docs

        if st.session_state.qa_chain is None:
            st.error("Failed to initialize the model. Ensure your HF token is set.")
            return

        try:
            with st.spinner("Thinking..."):
                t0 = time.time()
                response = st.session_state.qa_chain.invoke({"question": prompt})
                dt = time.time() - t0

                answer = response.get("answer", "I couldn't generate an answer.")
                source_docs = response.get("source_documents", [])

                # run hallucination check
                hflags = hallucination_check(answer, source_docs)
                if hflags:
                    hall_msg = "\n\n".join([f"> **{f}**" for f in hflags])
                    answer += f"\n\n**!!! HALÜSİNASYON UYARILARI !!!**\n\n{hall_msg}"

                # add example mini-calculation if user asked about allowable (we do a simple detection)
                if re.search(r'allowable|σ_allow|sigma_allow|σ_allow', prompt, flags=re.IGNORECASE):
                    # default example numbers (these could be interactive inputs later)
                    example = compute_allowable(800.0, 20.0, 1.5)
                    answer += f"\n\n**Örnek Hesap:** F_t0(dry)=800 MPa; hot-wet -20% → F_hot_wet={example['f_hot_wet']:.0f} MPa; γ_M=1.5 ⇒ σ_allow≈{example['sigma_allow']:.0f} MPa"

                sources_md = format_source_documents(source_docs)
                answer_md = f"{answer}\n\n*Response generated in {dt:.2f}s using {selected_model_name}*"

                display_chat("assistant", answer_md, sources_md)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer_md, "sources": sources_md}
                )
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.info("Make sure your HUGGINGFACEHUB_API_TOKEN is set and the selected model is available to your token.")

    # Footer controls
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.qa_chain = None
            st.rerun()
    with c2:
        if st.button("🔄 Reload Vector Store"):
            st.session_state.vectorstore = None
            st.cache_resource.clear()
            st.rerun()
    with c3:
        st.markdown("**Model:** " + selected_model_name)

if __name__ == "__main__":
    main()
