import os
from langchain_postgres import PGEngine, PGVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
            url=self.connection,
        )

        # self.engine.init_vectorstore_table(
        #     table_name=self.table_name,
        #     vector_size=768,
        # )
        
        self.vector_store = PGVectorStore.create_sync(
            engine=self.engine,
            embedding_service=self.embeddings,
            table_name=self.table_name,
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

        print(prompt)
        response = self.llm.invoke(prompt)
        return response.content

# --- Execution ---
if __name__ == "__main__":
    rag = RAGPipeline()
    
    # Only need to ingest once
    # rag.ingest_pdf("Mawin.pdf")
    
    answer = rag.ask("List all of work places that this guy has worked at?")
    print(f"\n🤖 Qwen3 says: \n{answer}")