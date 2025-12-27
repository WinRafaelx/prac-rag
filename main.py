import os
from langchain_postgres import PGEngine, PGVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RAGPipeline:
    def __init__(self):
        # 1. Connection Setup (psycopg3 compatible)
        self.connection = "postgresql+psycopg://myrag:mypassword@localhost:5432/rag_db"
        self.table_name = "pdf_knowledge"
        
        # 2. Model Setup
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = ChatOllama(
            model="qwen3:8b",
            base_url="http://localhost:11434",
            temperature=0  # Set to 0 for factual accuracy in RAG
        )

        # 3. Initialize High-Perf Engine & Store
        self.engine = PGEngine.from_connection_string(
            connection_string=self.connection,
        )
        
        self.vector_store = PGVectorStore(
            engine=self.engine,
            embedding_service=self.embeddings,
            table_name=self.table_name,
            use_jsonb=True  # Best performance for metadata filtering
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )

    def ingest_pdf(self, file_path: str):
        """Loads, splits, and embeds PDF data into Postgres."""
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found.")
            return

        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split(self.splitter)
        
        self.vector_store.add_documents(docs)
        print(f"✅ Ingested {len(docs)} chunks into Postgres.")

    def ask(self, query: str):
        """Retrieves context and generates an answer using Qwen3."""
        # Retrieve top 3 chunks
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        context_docs = retriever.invoke(query)
        context_text = "\n\n".join([d.page_content for d in context_docs])

        prompt = f"""You are a helpful assistant. Use the context below to answer accurately. 
If the answer isn't in the context, say you don't know.

Context:
{context_text}

Question: {query}
"""
        response = self.llm.invoke(prompt)
        return response.content

# --- Execution ---
if __name__ == "__main__":
    rag = RAGPipeline()
    
    # Only need to ingest once
    rag.ingest_pdf("Mawin.pdf")
    
    answer = rag.ask("What is this file about?")
    print(f"\n🤖 Qwen3 says: \n{answer}")