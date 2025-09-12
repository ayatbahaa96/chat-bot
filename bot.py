# bot_improved.py
import os
import time
import streamlit as st
from typing import List, Dict, Any
import json

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
    page_title="Gelişmiş RAG Chatbot (Teknik Doğruluk Odaklı)",
    page_icon="🔬",
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
    .accuracy-warning {
        background-color: #FFE5E5;
        border-left: 5px solid #FF6B6B;
        padding: 1rem;
        margin: 1rem 0;
    }
    .technical-table {
        background-color: #F8F9FA;
        border: 1px solid #DEE2E6;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
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
# Enhanced prompt template with technical accuracy focus
# ---------------------------
def create_enhanced_prompt() -> PromptTemplate:
    template = """
You are a TECHNICAL EXPERT assistant with access to engineering documents. Your responses must be ACCURATE and TRACEABLE.

CRITICAL REQUIREMENTS:
1. **ACCURACY FIRST**: Never make technical claims without document support. If unsure, state limitations clearly.
2. **PRECISE CITATIONS**: For each technical claim, cite EXACT source (book name + page/section).
3. **STRUCTURED FORMAT**: Use tables, formulas, and clear sections for technical content.
4. **SAFETY FOCUS**: Always consider safety factors, environmental conditions, and failure modes.
5. **NO HALLUCINATION**: If information isn't in context, explicitly state "Not found in provided documents".

RESPONSE STRUCTURE:
## Direct Answer
[Concise technical answer with key parameters]

## Technical Details
[Detailed explanation with formulas, tables if applicable]

## Source References
[Exact citations: Document name, Page X, Section Y]

## Safety Considerations
[Environmental factors, failure modes, compliance standards]

## Limitations
[What cannot be determined from available context]

Context from documents:
{context}

Chat History:
{chat_history}

Technical Question: {question}

Expert Response (structured, cited, safety-conscious):"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"],
    )

# ---------------------------------------
# Enhanced model registry with quality focus
# ---------------------------------------
def available_hf_models() -> dict:
    return {
        "Mistral 7B Instruct (v0.3)": "mistralai/Mistral-7B-Instruct-v0.3",
        "Llama 3.1 8B Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Llama 3.1 70B Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "CodeLlama 7B Instruct": "codellama/CodeLlama-7b-Instruct-hf",
        # Technical/scientific models if available
    }

# ------------------------------------------------------
# Enhanced QA chain with better retrieval
# ------------------------------------------------------
def create_qa_chain(
    repo_id: str,
    vectorstore,
    temperature: float = 0.1,
    k: int = 6,  # Increased for better context
):
    try:
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        if not hf_token:
            st.error("❌ Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN).")
            return None

        # Enhanced endpoint configuration
        endpoint = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=temperature,
            huggingfacehub_api_token=hf_token,
            max_new_tokens=2048,  # Increased for detailed technical responses
            repetition_penalty=1.1,
            return_full_text=False,
        )
        llm = ChatHuggingFace(llm=endpoint)

        # Enhanced memory with more context
        memory = ConversationBufferWindowMemory(
            k=8,  # Increased context window
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
        )

        # Enhanced prompt
        prompt = create_enhanced_prompt()

        # Enhanced retriever with hybrid search
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": 0.3,  # Filter low-relevance results
            }
        )

        # Build chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True,  # Enable for debugging
            combine_docs_chain_kwargs={"prompt": prompt},
        )
        return chain
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

# ---------------------------
# Enhanced source document formatting
# ---------------------------
def format_source_documents(source_docs: List[Document]) -> str:
    if not source_docs:
        return "⚠️ No source documents found - response may not be reliable."
    
    formatted = []
    for i, doc in enumerate(source_docs, 1):
        content = doc.page_content.strip()
        preview = (content[:600] + "...") if len(content) > 600 else content
        
        # Enhanced metadata extraction
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        
        # Clean up source name
        source_name = os.path.basename(source) if source != "Unknown source" else source
        
        # Add relevance score if available
        score = doc.metadata.get("score", "N/A")
        score_text = f" (Score: {score:.3f})" if isinstance(score, float) else ""
        
        formatted.append(
            f"**📄 Source {i}:** {source_name}\n"
            f"**📍 Page:** {page}{score_text}\n\n"
            f"**📖 Content Preview:**\n{preview}"
        )
    
    return "\n\n" + "="*50 + "\n\n".join([""] + formatted)

def validate_response_quality(response: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and enhance response quality"""
    answer = response.get("answer", "")
    source_docs = response.get("source_documents", [])
    
    quality_issues = []
    
    # Check for potential hallucination indicators
    hallucination_keywords = [
        "şekil", "figure", "tablo", "table", "grafik", "chart",
        "sayfa", "page", "bölüm", "chapter", "örnek", "example"
    ]
    
    for keyword in hallucination_keywords:
        if keyword.lower() in answer.lower():
            # Check if this reference is actually supported by sources
            if not any(keyword.lower() in doc.page_content.lower() for doc in source_docs):
                quality_issues.append(f"⚠️ Potential hallucination: Reference to '{keyword}' not found in sources")
    
    # Check for vague technical claims
    vague_terms = ["genellikle", "typically", "usually", "often", "sometimes"]
    for term in vague_terms:
        if term in answer.lower():
            quality_issues.append(f"⚠️ Vague statement detected: '{term}' - consider requesting more specific information")
    
    return {
        "answer": answer,
        "source_documents": source_docs,
        "quality_issues": quality_issues
    }

def display_chat(role: str, content: str, sources: str | None = None, quality_issues: List[str] = None):
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            # Display quality issues first if any
            if quality_issues:
                with st.expander("⚠️ Quality Assessment", expanded=True):
                    st.markdown('<div class="accuracy-warning">', unsafe_allow_html=True)
                    st.warning("**Response Quality Issues Detected:**")
                    for issue in quality_issues:
                        st.write(f"• {issue}")
                    st.write("\n**Recommendation:** Verify claims against original documents or request more specific information.")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown(content)
            
            if sources and st.session_state.get("show_sources", True):
                with st.expander("📚 Source Documents & Citations", expanded=False):
                    st.markdown(sources)

# ---------------------------
# Technical validation helpers
# ---------------------------
def create_technical_summary_table():
    """Create a technical summary table for common engineering topics"""
    st.markdown('<div class="technical-table">', unsafe_allow_html=True)
    st.write("### 🔧 Common Technical Parameters")
    
    # Example table structure - you can customize based on your domain
    technical_data = {
        "Load Type": ["Static", "Dynamic", "Fatigue", "Impact"],
        "Safety Factor (γ)": ["1.35-2.0", "1.5-3.0", "Variable", "2.0-4.0"],
        "Environment": ["Dry", "Hot-Wet", "Corrosive", "Extreme"],
        "Test Standard": ["ISO/ASTM", "MIL-STD", "EN", "Custom"]
    }
    
    st.table(technical_data)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Main App with enhanced features
# ---------------------------
def main():
    st.markdown("<h1 class='main-header'>🔬 Gelişmiş RAG Chatbot (Teknik Doğruluk Odaklı)</h1>", unsafe_allow_html=True)
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("⚙️ Model Ayarları")

        models = available_hf_models()
        selected_model_name = st.selectbox(
            "Hugging Face Modeli:",
            options=list(models.keys()),
            index=0,
        )
        selected_repo = models[selected_model_name]

        with st.expander("🔧 Gelişmiş Ayarlar"):
            temperature = st.slider("Temperature (Yaratıcılık)", 0.0, 1.0, 0.05, 0.05)  # Lower for accuracy
            k_docs = st.slider("Kaynak doküman sayısı (k)", 3, 15, 6)
            score_threshold = st.slider("Relevance threshold", 0.1, 0.9, 0.3, 0.1)
            show_sources = st.checkbox("Kaynak dokümanları göster", value=True)
            enable_quality_check = st.checkbox("Kalite kontrolü aktif", value=True)
        
        st.session_state["show_sources"] = show_sources
        st.session_state["enable_quality_check"] = enable_quality_check

        st.header("📊 Oturum İstatistikleri")
        if "messages" in st.session_state:
            st.metric("Toplam Mesaj", len(st.session_state.messages))
            user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
            st.metric("Sorulan Soru", user_msgs)
            
            # Quality metrics
            if "quality_scores" in st.session_state:
                avg_quality = sum(st.session_state.quality_scores) / len(st.session_state.quality_scores)
                st.metric("Ortalama Kalite", f"{avg_quality:.1f}/10")

        st.header("📖 Teknik Referans")
        create_technical_summary_table()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "vectorstore" not in st.session_state:
        with st.spinner("Bilgi bankası yükleniyor..."):
            st.session_state.vectorstore = get_vectorstore()
    if "quality_scores" not in st.session_state:
        st.session_state.quality_scores = []

    # Display chat history
    for m in st.session_state.messages:
        display_chat(
            m["role"], 
            m["content"], 
            m.get("sources"),
            m.get("quality_issues", [])
        )

    # Enhanced chat input with suggestions
    st.write("### 💡 Örnek Teknik Sorular:")
    example_questions = [
        "Kompozit malzemelerde emniyet katsayısı nasıl hesaplanır?",
        "Hot-wet koşullarında mukavemet değerleri nasıl etkilenir?",
        "Yorgunluk testlerinde kullanılan standartlar nelerdir?",
        "Raylı sistem yüklerinde dinamik faktörler nelerdir?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        if cols[i % 2].button(f"❓ {question[:50]}...", key=f"q_{i}"):
            st.session_state.suggested_question = question

    # Main chat input
    suggested = st.session_state.get("suggested_question", "")
    if prompt := st.chat_input("Teknik sorunuzu sorun...", value=suggested):
        if "suggested_question" in st.session_state:
            del st.session_state.suggested_question
            
        display_chat("user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if st.session_state.vectorstore is None:
            st.error("Vector store yüklenemedi. Data dizinini kontrol edin.")
            return

        # Rebuild chain if parameters changed
        if (
            st.session_state.qa_chain is None
            or st.session_state.get("current_repo") != selected_repo
            or st.session_state.get("current_temp") != temperature
            or st.session_state.get("current_k") != k_docs
        ):
            with st.spinner("Model hazırlanıyor..."):
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
            st.error("Model başlatılamadı. HF token'ınızı kontrol edin.")
            return

        # Generate enhanced response
        try:
            with st.spinner("Teknik analiz yapılıyor..."):
                t0 = time.time()
                response = st.session_state.qa_chain.invoke({"question": prompt})
                dt = time.time() - t0

                # Enhanced response validation
                if enable_quality_check:
                    validated_response = validate_response_quality(response)
                    answer = validated_response["answer"]
                    source_docs = validated_response["source_documents"]
                    quality_issues = validated_response["quality_issues"]
                else:
                    answer = response.get("answer", "Cevap oluşturulamadı.")
                    source_docs = response.get("source_documents", [])
                    quality_issues = []

                sources_md = format_source_documents(source_docs)
                
                # Enhanced answer with metadata
                answer_md = f"{answer}\n\n---\n**⏱️ Analiz süresi:** {dt:.2f}s | **🤖 Model:** {selected_model_name} | **📄 Kaynak sayısı:** {len(source_docs)}"

                display_chat("assistant", answer_md, sources_md, quality_issues)
                
                # Store with quality info
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer_md, 
                    "sources": sources_md,
                    "quality_issues": quality_issues
                })

        except Exception as e:
            st.error(f"Cevap oluşturulurken hata: {str(e)}")
            st.info("HF token ve model erişiminizi kontrol edin.")

    # Enhanced footer controls
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        if st.button("🗑️ Sohbeti Temizle"):
            st.session_state.messages = []
            st.session_state.qa_chain = None
            st.session_state.quality_scores = []
            st.rerun()
    
    with c2:
        if st.button("🔄 Vector Store Yenile"):
            st.session_state.vectorstore = None
            st.cache_resource.clear()
            st.rerun()
    
    with c3:
        if st.button("📊 Kalite Raporu"):
            if st.session_state.messages:
                # Generate quality report
                st.info("Kalite raporu özelliği geliştirme aşamasında...")
    
    with c4:
        if st.button("📥 Sohbeti İndir"):
            if st.session_state.messages:
                chat_export = json.dumps(st.session_state.messages, indent=2, ensure_ascii=False)
                st.download_button(
                    label="JSON olarak indir",
                    data=chat_export,
                    file_name=f"chat_export_{int(time.time())}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
