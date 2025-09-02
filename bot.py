# bot_hf.py
import os
import time
import streamlit as st
from typing import List

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
import os
hf_token = os.getenv("HF_TOKEN")

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
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-doc {
        background-color: #FFF3E0;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Paths / Constants
# ---------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

# -----------------------------------
# Cached: load your FAISS vectorstore
# -----------------------------------
@st.cache_resource
def get_vectorstore():
    """Load and cache the vector store."""
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
# Prompt template
# ---------------------------
def create_enhanced_prompt() -> PromptTemplate:
    template = """
You are an intelligent assistant with access to relevant document context. 
Use the provided context to give comprehensive, accurate answers.

Guidelines:
- Answer based primarily on the provided context
- If information is not in the context, clearly state that
- Provide detailed explanations when possible
- Use examples from the context when relevant
- Be conversational but informative
- If asked about previous conversation, use the chat history

Context from documents:
{context}

Chat History:
{chat_history}

Human Question: {question}

Assistant Response:"""
    return PromptTemplate(
        template=template, input_variables=["context", "chat_history", "question"]
    )

# ---------------------------------------
# Hugging Face model registry (repo IDs)
# ---------------------------------------
def available_hf_models() -> dict:
    # These are Hugging Face Inference API repo IDs.
    # You can add/remove models here as you like.
    return {
        "Mistral 7B Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
        "Llama 3 8B Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Llama 3 70B Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
       
        # Add more if you have access to them via your token
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

        # Create a raw HF endpoint (text or chat), then wrap as Chat model
        endpoint = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=temperature,
            huggingfacehub_api_token=hf_token,
            max_new_tokens=512,  # pass explicitly (not in model_kwargs)
        )
        llm = ChatHuggingFace(llm=endpoint)

        # Memory must know which output key to store
        memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            output_key="answer",  # IMPORTANT: ConversationalRetrievalChain returns {'answer', 'source_documents'}
            return_messages=True,
        )

        # Custom prompt
        prompt = create_enhanced_prompt()

        # Build ConversationalRetrievalChain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": k}
            ),
            memory=memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": prompt},  # ensures our prompt is used
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
        page = doc.metadata.get("page", "Unknown page")
        formatted.append(
            f"**Source {i}:** {source} (Page {page})\n\n*Preview:* {preview}"
        )
    return "\n\n---\n\n".join(formatted)

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
    st.markdown("<h1 class='main-header'>🤖 Enhanced AI Chatbot (Hugging Face)</h1>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        models = available_hf_models()
        selected_model_name = st.selectbox(
            "Select Hugging Face Model:",
            options=list(models.keys()),
            index=0,
        )
        selected_repo = models[selected_model_name]

        with st.expander("🔧 Advanced Settings"):
            temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.1, 0.1)
            k_docs = st.slider("Number of source documents (k)", 1, 8, 4)
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

        # (Re)build chain if model changed or not yet built
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

        # Generate response
        try:
            with st.spinner("Thinking..."):
                t0 = time.time()
                response = st.session_state.qa_chain.invoke({"question": prompt})
                dt = time.time() - t0

                answer = response.get("answer", "I couldn't generate an answer.")
                source_docs = response.get("source_documents", [])

                sources_md = format_source_documents(source_docs)
                answer_md = f"{answer}\n\n*Response generated in {dt:.2f}s using {selected_model_name}*"

                display_chat("assistant", answer_md, sources_md)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer_md, "sources": sources_md}
                )
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.info(
                "Make sure your HUGGINGFACEHUB_API_TOKEN is set and the selected model is available to your token."
            )

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

