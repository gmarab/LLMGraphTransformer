import argparse
from typing import Any, TypedDict

import pypdf
import docx
import tiktoken
from langchain_ollama import OllamaEmbeddings
from langchain_neo4j import Neo4jVector
from neo4j import GraphDatabase
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
import io

load_dotenv(".env")

# --- Config ---

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

embeddings = OllamaEmbeddings(
    base_url=os.getenv("BASE_URL", "http://localhost:11434"),
    model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
)

neo4j_url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

enc = tiktoken.get_encoding("o200k_harmony")


# --- Extractors ---

def extract_text_from_pdf_bytes(data: bytes) -> list[tuple[str, int]]:
    """Estrae testo da PDF in memoria, restituisce lista di (testo, numero_pagina)."""
    reader = pypdf.PdfReader(io.BytesIO(data))
    pages = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append((text, page_num))
    return pages


def extract_text_from_docx_bytes(data: bytes) -> list[tuple[str, int]]:
    """Estrae testo da DOCX in memoria, restituisce lista di (testo, indice_blocco)."""
    doc = docx.Document(io.BytesIO(data))
    pages = []
    current_text = []
    page_num = 0
    for para in doc.paragraphs:
        if para.text.strip():
            current_text.append(para.text)
        if len(current_text) >= 10:
            pages.append(("\n".join(current_text), page_num))
            current_text = []
            page_num += 1
    if current_text:
        pages.append(("\n".join(current_text), page_num))
    return pages


def extract_text_from_txt_bytes(data: bytes) -> list[tuple[str, int]]:
    """Estrae testo da file TXT in memoria, restituisce lista di (testo, indice_blocco)."""
    content = data.decode("utf-8", errors="replace")
    block_size = 2000
    blocks = []
    for i in range(0, len(content), block_size):
        block = content[i:i + block_size]
        if block.strip():
            blocks.append((block, i // block_size))
    return blocks


EXTRACTORS = {
    ".pdf": extract_text_from_pdf_bytes,
    ".docx": extract_text_from_docx_bytes,
    ".txt": extract_text_from_txt_bytes,
}


# --- State ---

class LoadState(TypedDict):
    filename: str
    data: bytes
    project: str
    clean: bool
    pages: list[tuple[str, int]]
    chunks: list[str]
    metadatas: list[dict]
    deleted: int
    result: dict[str, Any]


# --- Nodes ---

def extract_text(state: LoadState) -> dict:
    """Estrae testo dal file in base all'estensione."""
    ext = os.path.splitext(state["filename"])[1].lower()
    extractor = EXTRACTORS.get(ext)
    if not extractor:
        raise ValueError(f"Formato non supportato: {ext}. Formati supportati: {', '.join(SUPPORTED_EXTENSIONS)}")

    pages = extractor(state["data"])
    if not pages:
        raise ValueError(f"Nessun testo estratto da {state['filename']}")

    return {"pages": pages}


def chunk_text(state: LoadState) -> dict:
    """Tokenizza le pagine estratte in chunk con metadati."""
    chunks = []
    metadatas = []
    for text, page_num in state["pages"]:
        tokens = enc.encode(text)
        chunk_text = enc.decode(tokens)
        chunks.append(chunk_text)
        metadatas.append({"source": state["filename"], "page": page_num})

    return {"chunks": chunks, "metadatas": metadatas}


def clean_project_node(state: LoadState) -> dict:
    """Elimina nodi e indici precedenti per il progetto (se richiesto)."""
    if not state.get("clean", False):
        return {"deleted": 0}

    project = state["project"]
    node_label = f"Document_{project}"
    index_name = f"{project}_docs"
    keyword_index_name = f"{project}_docs_fulltext"

    driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
    with driver.session() as session:
        result = session.run(f"MATCH (n:{node_label}) DETACH DELETE n RETURN count(n) AS deleted")
        deleted = result.single()["deleted"]
        session.run(f"DROP INDEX {index_name} IF EXISTS")
        session.run(f"DROP INDEX {keyword_index_name} IF EXISTS")
    driver.close()
    return {"deleted": deleted}


def store_vectors(state: LoadState) -> dict:
    """Carica i chunk nel vector store Neo4j."""
    project = state["project"]
    node_label = f"Document_{project}"
    index_name = f"{project}_docs"
    keyword_index_name = f"{project}_docs_fulltext"

    Neo4jVector.from_texts(
        texts=state["chunks"],
        metadatas=state["metadatas"],
        embedding=embeddings,
        url=neo4j_url,
        username=neo4j_user,
        password=neo4j_password,
        index_name=index_name,
        node_label=node_label,
        keyword_index_name=keyword_index_name,
        search_type="hybrid",
    )

    return {
        "result": {
            "filename": state["filename"],
            "project": project,
            "chunks": len(state["chunks"]),
            "cleaned": state.get("deleted") if state.get("clean", False) else None,
        }
    }


# --- Graph ---

graph_builder = StateGraph(LoadState)

graph_builder.add_node("extract_text", extract_text)
graph_builder.add_node("chunk_text", chunk_text)
graph_builder.add_node("clean_project", clean_project_node)
graph_builder.add_node("store_vectors", store_vectors)

graph_builder.add_edge(START, "extract_text")
graph_builder.add_edge("extract_text", "chunk_text")
graph_builder.add_edge("chunk_text", "clean_project")
graph_builder.add_edge("clean_project", "store_vectors")
graph_builder.add_edge("store_vectors", END)

graph = graph_builder.compile()
app = graph


# --- Helper per compatibilità con rag_load.py ---

def _clean_project(project: str) -> int:
    """Elimina nodi e indici precedenti per il progetto."""
    node_label = f"Document_{project}"
    index_name = f"{project}_docs"
    keyword_index_name = f"{project}_docs_fulltext"
    driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
    with driver.session() as session:
        result = session.run(f"MATCH (n:{node_label}) DETACH DELETE n RETURN count(n) AS deleted")
        deleted = result.single()["deleted"]
        session.run(f"DROP INDEX {index_name} IF EXISTS")
        session.run(f"DROP INDEX {keyword_index_name} IF EXISTS")
    driver.close()
    return deleted


def clean_project(project: str) -> dict:
    deleted = _clean_project(project)
    return {"project": project, "deleted": deleted}


def upload_file(filename: str, data: bytes, project: str, clean: bool = False) -> dict:
    """Stessa interfaccia di rag_load.upload_file, ma eseguita tramite LangGraph."""
    result = app.invoke({
        "filename": filename,
        "data": data,
        "project": project,
        "clean": clean,
    })
    return result["result"]


def upload_path(filepath: str, project: str, clean: bool = False) -> dict:
    """Carica un file dal filesystem locale dato il suo percorso."""
    if not os.path.isfile(filepath):
        raise ValueError(f"File non trovato: {filepath}")

    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        data = f.read()

    return upload_file(filename, data, project, clean)


def upload_folder(folder: str, project: str, clean: bool = False) -> dict:
    """Processa tutti i file supportati in una cartella e li carica in Neo4j."""
    if not os.path.isdir(folder):
        raise ValueError(f"Cartella non trovata: {folder}")

    deleted = 0
    if clean:
        deleted = _clean_project(project)

    files_results = []
    errors = []
    total_chunks = 0

    for root, _, filenames in os.walk(folder):
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            filepath = os.path.join(root, fname)
            rel_path = os.path.relpath(filepath, folder)

            try:
                with open(filepath, "rb") as f:
                    data = f.read()
                result = upload_file(rel_path, data, project, clean=False)
                files_results.append(result)
                total_chunks += result["chunks"]
            except Exception as e:
                errors.append({"filename": rel_path, "error": str(e)})

    if not files_results and not errors:
        raise ValueError(f"Nessun documento supportato trovato in {folder}")

    return {
        "folder": folder,
        "project": project,
        "total_chunks": total_chunks,
        "files": files_results,
        "errors": errors if errors else None,
        "cleaned": deleted if clean else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Document Loader (LangGraph)")
    parser.add_argument("--folder", help="Path to folder to load", required=True)
    parser.add_argument("--project", help="Project name for Neo4j", required=True)
    parser.add_argument("--clean", default=True, help="Clean previous project's documents")
    return parser.parse_args()


def main():
    args = parse_args()
    upload_folder(args.folder, args.project, args.clean)


if __name__ == "__main__":
    main()
