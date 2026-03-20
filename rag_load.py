import pypdf
import docx
import tiktoken
from langchain_ollama import OllamaEmbeddings
from langchain_neo4j import Neo4jVector
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import io

load_dotenv(".env")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


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

embeddings = OllamaEmbeddings(
    base_url=os.getenv("BASE_URL", "http://localhost:11434"),
    model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
)

neo4j_url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

enc = tiktoken.get_encoding("o200k_harmony")


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
#        session.run(f"MATCH (n:{node_label}) REMOVE")
    driver.close()
    return deleted


def clean_project(project: str) -> dict:
    deleted = _clean_project(project)
    return {"project": project, "deleted": deleted}


def upload_file(filename: str, data: bytes, project: str, clean: bool = False) -> dict:
    """Processa un singolo file e lo carica nel vector store Neo4j.

    Returns:
        dict con chiavi: filename, project, chunks, cleaned
    """
    ext = os.path.splitext(filename)[1].lower()
    extractor = EXTRACTORS.get(ext)
    if not extractor:
        raise ValueError(f"Formato non supportato: {ext}. Formati supportati: {', '.join(SUPPORTED_EXTENSIONS)}")

    pages = extractor(data)
    if not pages:
        raise ValueError(f"Nessun testo estratto da {filename}")

    chunks = []
    metadatas = []
    for text, page_num in pages:
        tokens = enc.encode(text)
        chunk_text = enc.decode(tokens)
        chunks.append(chunk_text)
        metadatas.append({"source": filename, "page": page_num})

    deleted = 0
    if clean:
        deleted = _clean_project(project)

    node_label = f"Document_{project}"
    index_name = f"{project}_docs"
    keyword_index_name = f"{project}_docs_fulltext"

    Neo4jVector.from_texts(
        texts=chunks,
        metadatas=metadatas,
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
        "filename": filename,
        "project": project,
        "chunks": len(chunks),
        "cleaned": deleted if clean else None,
    }


def upload_path(filepath: str, project: str, clean: bool = False) -> dict:
    """Carica un file dal filesystem locale dato il suo percorso."""
    if not os.path.isfile(filepath):
        raise ValueError(f"File non trovato: {filepath}")

    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        data = f.read()

    return upload_file(filename, data, project, clean)


def upload_folder(folder: str, project: str, clean: bool = False) -> dict:
    """Processa tutti i file supportati in una cartella (ricorsiva) e li carica in Neo4j.

    Returns:
        dict con chiavi: folder, project, total_chunks, files, errors, cleaned
    """
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
                # Non pulire di nuovo: la pulizia è già stata fatta sopra
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
