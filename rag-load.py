from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
from dotenv import load_dotenv
import os

load_dotenv(".env")

# --- 1. Load PDF ---
pdf_path = "/home/g22/documents/NovaPulse_Product_Documentation.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"Loaded {len(documents)} pages from PDF")

# --- 2. Split using tiktoken with o200k_harmony encoding ---
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="o200k_harmony",
    chunk_size=512,
    chunk_overlap=64,
)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# --- 3. Embedding model (Ollama) ---
embeddings = OllamaEmbeddings(
    base_url=os.getenv("BASE_URL", "http://localhost:11434"),
    model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
)

# --- 4. Load into Neo4j ---
neo4j_url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

vector_store = Neo4jVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    url=neo4j_url,
    username=neo4j_user,
    password=neo4j_password,
    index_name="novapulse_docs",
    node_label="Document",
)

print(f"Loaded {len(chunks)} chunks into Neo4j vector store")

# --- 5. Verifica caricamento ---
print("\n--- Verifica caricamento ---")
query = chunks[0].page_content[:100]
results = vector_store.similarity_search(query, k=3)

assert len(results) > 0, "Nessun risultato trovato: il caricamento potrebbe essere fallito"
assert results[0].page_content, "Il contenuto del documento restituito è vuoto"

print(f"Query di test: '{query[:50]}...'")
print(f"Risultati trovati: {len(results)}")
for i, doc in enumerate(results):
    print(f"  [{i+1}] {doc.page_content[:80]}...")
print("Verifica completata con successo!")
