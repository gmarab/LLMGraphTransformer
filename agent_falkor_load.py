import argparse
from typing import Any, TypedDict

import pypdf
import docx
import tiktoken
import falkordb
from langchain_ollama import OllamaEmbeddings
from langchain_falkordb.vectorstores import FalkorDBVector, SearchType
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

falkordb_host = os.getenv("FALKORDB_HOST", "localhost")
falkordb_port = int(os.getenv("FALKORDB_PORT", "6379"))
falkordb_username = os.getenv("FALKORDB_USERNAME", None)
falkordb_password = os.getenv("FALKORDB_PASSWORD", None)

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
    try:
        ext = os.path.splitext(state["filename"])[1].lower()
        extractor = EXTRACTORS.get(ext)
        if not extractor:
            raise ValueError(f"Formato non supportato: {ext}. Formati supportati: {', '.join(SUPPORTED_EXTENSIONS)}")

        pages = extractor(state["data"])
        if not pages:
            raise ValueError(f"Nessun testo estratto da {state['filename']}")

        return {"pages": pages}
    except Exception as e:
        raise type(e)(f"[extract_text] {e}") from e


def chunk_text(state: LoadState) -> dict:
    """Tokenizza le pagine estratte in chunk con metadati."""
    try:
        chunks = []
        metadatas = []
        for text, page_num in state["pages"]:
            tokens = enc.encode(text)
            chunk_text = enc.decode(tokens)
            chunks.append(chunk_text)
            metadatas.append({"source": state["filename"], "page": page_num})
            print(f"  Chunk tokens: {len(tokens)}")
            print(f"  Chunk text: {len(chunk_text)}")

        return {"chunks": chunks, "metadatas": metadatas}
    except Exception as e:
        raise type(e)(f"[chunk_text] {e}") from e


def clean_project_node(state: LoadState) -> dict:
    """Elimina nodi e indici precedenti per il progetto (se richiesto)."""
    try:
        if not state.get("clean", False):
            return {"deleted": 0}

        project = state["project"]
        node_label = f"Document_{project}"
        graph_name = f"{project}_docs"

        db = falkordb.FalkorDB(
            host=falkordb_host,
            port=falkordb_port,
            username=falkordb_username,
            password=falkordb_password,
        )
        graph = db.select_graph(graph_name)
        result = graph.query(f"MATCH (n:{node_label}) WITH n, count(n) AS cnt DETACH DELETE n RETURN cnt")
        deleted = result.result_set[0][0] if result.result_set else 0
        db.close()
        return {"deleted": deleted}
    except Exception as e:
        raise type(e)(f"[clean_project] {e}") from e


def store_vectors(state: LoadState) -> dict:
    """Carica i chunk nel vector store FalkorDB."""
    try:
        project = state["project"]
        node_label = f"Document_{project}"
        graph_name = f"{project}_docs"

        FalkorDBVector.from_texts(
            texts=state["chunks"],
            metadatas=state["metadatas"],
            embedding=embeddings,
            host=falkordb_host,
            port=falkordb_port,
            username=falkordb_username,
            password=falkordb_password,
            database=graph_name,
            node_label=node_label,
            search_type=SearchType.HYBRID,
        )

        return {
            "result": {
                "filename": state["filename"],
                "project": project,
                "chunks": len(state["chunks"]),
                "cleaned": state.get("deleted") if state.get("clean", False) else None,
            }
        }
    except Exception as e:
        raise type(e)(f"[store_vectors] {e}") from e


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
    graph_name = f"{project}_docs"

    db = falkordb.FalkorDB(
        host=falkordb_host,
        port=falkordb_port,
        username=falkordb_username,
        password=falkordb_password,
    )
    graph = db.select_graph(graph_name)
    result = graph.query(f"MATCH (n:{node_label}) WITH n, count(n) AS cnt DETACH DELETE n RETURN cnt")
    deleted = result.result_set[0][0] if result.result_set else 0
    db.close()
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
    """Processa tutti i file supportati in una cartella e li carica in FalkorDB."""
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
    parser = argparse.ArgumentParser(description="RAG Document Loader (LangGraph + FalkorDB)")
    parser.add_argument("--folder", help="Path to folder to load", required=True)
    parser.add_argument("--project", help="Project name for FalkorDB", required=True)
    parser.add_argument("--clean", default=True, help="Clean previous project's documents")
    return parser.parse_args()


def main():
    args = parse_args()
    upload_folder(args.folder, args.project, args.clean)


if __name__ == "__main__":
    main()
