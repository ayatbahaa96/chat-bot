import os
import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader, 
    DirectoryLoader, 
    TextLoader, 
    CSVLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, data_path: str = "data/", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor with improved chunking parameters
        
        Args:
            data_path: Path to directory containing documents
            chunk_size: Size of text chunks (increased for better context)
            chunk_overlap: Overlap between chunks (increased for continuity)
        """
        self.data_path = Path(data_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_path = "vectorstore/db_faiss"
        
        # Create directories if they don't exist
        os.makedirs(self.db_path, exist_ok=True)
        
    def load_documents(self) -> List[Document]:
        """Load documents from multiple file types"""
        documents = []
        
        if not self.data_path.exists():
            logger.error(f"Data path {self.data_path} does not exist!")
            return documents
        
        # Define loaders for different file types
        loaders = {
            '*.pdf': PyPDFLoader,
            '*.txt': TextLoader,
            '*.csv': CSVLoader,
            '*.docx': UnstructuredWordDocumentLoader,
            '*.doc': UnstructuredWordDocumentLoader,
        }
        
        # Load each file type
        for pattern, loader_class in loaders.items():
            try:
                loader = DirectoryLoader(
                    str(self.data_path),
                    glob=pattern,
                    loader_cls=loader_class,
                    show_progress=True
                )
                file_documents = loader.load()
                
                if file_documents:
                    logger.info(f"Loaded {len(file_documents)} documents with pattern {pattern}")
                    documents.extend(file_documents)
                    
            except Exception as e:
                logger.warning(f"Error loading {pattern} files: {str(e)}")
                continue
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def format_source_documents(source_docs: List[Document]) -> str:
        if not source_docs:
            return "No source documents found."
        formatted = []
        for i, doc in enumerate(source_docs, 1):
            content = doc.page_content or ""
            preview = (content[:1000] + "...") if len(content) > 1000 else content
            source = doc.metadata.get("source", "Unknown source")
            # robust page number handling (handles None, str, int)
            raw_page = doc.metadata.get("page", None)
            try:
                page_num = int(raw_page) + 1 if raw_page is not None and str(raw_page).isdigit() else raw_page
            except Exception:
                page_num = raw_page
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            formatted.append(
                f"**Source {i}:** {source} (Page {page_num}, chunk_id={chunk_id})\n\n*Preview:* {preview}"
            )
        return "\n\n---\n\n".join(formatted)


   def create_chunks(self, documents: List[Document]) -> List[Document]:
        """Create text chunks with improved splitting strategy and robust metadata"""
        if not documents:
            logger.warning("No documents to chunk!")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            add_start_index=True
        )

        chunks = text_splitter.split_documents(documents)

        # Add chunk metadata (ensure source and page are preserved and chunk_id set)
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata or {}
            # If original loader included 'page' or 'source', keep them; else try to set defaults
            source = meta.get('source', meta.get('file', 'Unknown'))
            page = meta.get('page', meta.get('page_number', None))
            chunk.metadata.update({
                'chunk_id': i,
                'chunk_size': len(chunk.page_content),
                'total_chunks': len(chunks),
                'source': source,
                'page': page
            })

        logger.info(f"Created {len(chunks)} text chunks")
        return chunks
    
    def get_embedding_model(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Get embedding model with better configuration"""
        try:
            # Alternative models for better performance:
            # "sentence-transformers/all-mpnet-base-v2" - Better quality but slower
            # "sentence-transformers/multi-qa-MiniLM-L6-cos-v1" - Good for Q&A
            # "sentence-transformers/all-MiniLM-L12-v2" - Balance of speed and quality
            
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
                encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
            )
            
            logger.info(f"Loaded embedding model: {model_name}")
            return embedding_model
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            # Fallback to basic model
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    def create_vector_store(self, chunks: List[Document], embedding_model) -> FAISS:
        """Create FAISS vector store with chunks"""
        if not chunks:
            raise ValueError("No chunks provided for vector store creation!")
        
        logger.info("Creating FAISS vector store...")
        
        # Create vector store in batches to handle large datasets
        batch_size = 100
        db = None
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            
            if db is None:
                # Create initial database
                db = FAISS.from_documents(batch, embedding_model)
            else:
                # Add to existing database
                batch_db = FAISS.from_documents(batch, embedding_model)
                db.merge_from(batch_db)
        
        return db
    
    def save_vector_store(self, db: FAISS):
        """Save vector store to disk"""
        try:
            db.save_local(self.db_path)
            logger.info(f"Vector store saved to {self.db_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def process_documents(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Full pipeline to process documents and create vector store"""
        try:
            # Step 1: Load documents
            logger.info("Step 1: Loading documents...")
            documents = self.load_documents()
            
            if not documents:
                raise ValueError("No documents found to process!")
            
            # Step 2: Create chunks
            logger.info("Step 2: Creating text chunks...")
            chunks = self.create_chunks(documents)
            
            # Step 3: Get embedding model
            logger.info("Step 3: Loading embedding model...")
            embedding_model = self.get_embedding_model(embedding_model_name)
            
            # Step 4: Create vector store
            logger.info("Step 4: Creating vector store...")
            db = self.create_vector_store(chunks, embedding_model)
            
            # Step 5: Save vector store
            logger.info("Step 5: Saving vector store...")
            self.save_vector_store(db)
            
            # Print summary
            logger.info("=" * 50)
            logger.info("PROCESSING COMPLETE!")
            logger.info(f"Documents processed: {len(documents)}")
            logger.info(f"Text chunks created: {len(chunks)}")
            logger.info(f"Embedding model: {embedding_model_name}")
            logger.info(f"Vector store saved to: {self.db_path}")
            logger.info("=" * 50)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in document processing pipeline: {str(e)}")
            return False

def main():
    """Main function with Streamlit interface for document processing"""
    st.title("📚 Document Processing for AI Chatbot")
    st.write("Process your documents to create a knowledge base for the chatbot.")
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        data_path = st.text_input("Data Directory Path", value="data/")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    
    with col2:
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
        embedding_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "sentence-transformers/all-MiniLM-L12-v2"
        ]
        selected_model = st.selectbox("Embedding Model", embedding_models)
    
    # Show model descriptions
    model_descriptions = {
        "sentence-transformers/all-MiniLM-L6-v2": "Fast and lightweight (default)",
        "sentence-transformers/all-mpnet-base-v2": "Higher quality, slower processing",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": "Optimized for Q&A tasks",
        "sentence-transformers/all-MiniLM-L12-v2": "Balance of speed and quality"
    }
    st.info(f"Selected model: {model_descriptions[selected_model]}")
    
    # Process button
    if st.button("🚀 Process Documents", type="primary"):
        processor = DocumentProcessor(data_path, chunk_size, chunk_overlap)
        
        with st.spinner("Processing documents..."):
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress updates (you could make this more sophisticated)
            import time
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 20:
                    status_text.text("Loading documents...")
                elif i < 40:
                    status_text.text("Creating text chunks...")
                elif i < 60:
                    status_text.text("Loading embedding model...")
                elif i < 90:
                    status_text.text("Creating vector store...")
                else:
                    status_text.text("Saving vector store...")
                time.sleep(0.1)
            
            success = processor.process_documents(selected_model)
            
            if success:
                st.success("✅ Documents processed successfully!")
                st.balloons()
            else:
                st.error("❌ Error processing documents. Check the logs.")

if __name__ == "__main__":
    # Run as script
    processor = DocumentProcessor(
        data_path="data/",
        chunk_size=1000,
        chunk_overlap=200
    )
    
    success = processor.process_documents("sentence-transformers/all-MiniLM-L6-v2")
    
    if success:
        print("✅ Document processing completed successfully!")
    else:

        print("❌ Document processing failed!")
