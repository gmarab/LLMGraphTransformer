from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_neo4j import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import re
from functools import lru_cache

load_dotenv(".env")

# --- Init vector store & QA chain ---

embeddings = OllamaEmbeddings(
    base_url=os.getenv("BASE_URL", "http://localhost:11434"),
    model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
)

neo4j_url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")


@lru_cache(maxsize=16)
def get_vector_store(proj: str) -> Neo4jVector:
    """Restituisce il vector store per il progetto specificato (con cache)."""
    return Neo4jVector(
        embedding=embeddings,
        url=neo4j_url,
        username=neo4j_user,
        password=neo4j_password,
        index_name=f"{proj}_docs",
        node_label=f"Document_{proj}",
        keyword_index_name=f"{proj}_docs_fulltext",
        search_type="hybrid",
    )

llm = OllamaLLM(
    base_url=os.getenv("BASE_URL", "http://localhost:11434"),
    model=os.getenv("LLM_MODEL", "gpt-oss:20b-cloud"),
    temperature=0.0,
)

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))

prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant that answers questions based strictly on the provided context.\n\n"
    "Rules:\n"
    "- Answer ONLY using information found in the context below.\n"
    "- If the context does not contain enough information to answer, say \"I don't have enough information to answer this question.\"\n"
    "- Do not make up or infer information beyond what is explicitly stated.\n"
    "- Be concise and direct.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

answer_chain = prompt | llm | StrOutputParser()


def _strip_markdown(text: str) -> str:
    """Rimuove la formattazione markdown bold/italic dal testo."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    return text


def _format_docs(docs_with_scores) -> str:
    return "\n\n".join(doc.page_content for doc, _score in docs_with_scores)


def _format_references(docs_with_scores) -> list[dict]:
    """Estrai riferimenti univoci (documento, pagina, score) dai documenti recuperati."""
    seen = set()
    refs = []
    for doc, score in docs_with_scores:
        source = doc.metadata.get("source", doc.metadata.get("fileName", "Sconosciuto"))
        raw_page = doc.metadata.get("page", doc.metadata.get("page_number", None))
        page = raw_page + 1 if isinstance(raw_page, int) else "N/A"
        key = (source, page)
        if key not in seen:
            seen.add(key)
            refs.append({"nome_documento": source, "pagina": page, "score": round(score, 4)})
    return refs


def query_answer(question: str, project: str, strip_markdown: bool = True) -> tuple[str, list[dict]]:
    """Esegue la query RAG e restituisce (risposta, lista_fonti)."""
    vs = get_vector_store(project)
    docs_with_scores = vs.similarity_search_with_relevance_scores(question, k=3)
    docs_with_scores = [(doc, score) for doc, score in docs_with_scores if score >= SIMILARITY_THRESHOLD]
    context = _format_docs(docs_with_scores)
    answer = answer_chain.invoke({"context": context, "question": question})
    if strip_markdown:
        answer = _strip_markdown(answer)
    sources = _format_references(docs_with_scores)
    return answer, sources
