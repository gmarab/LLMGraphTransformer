from typing import Any, TypedDict

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_neo4j import Neo4jVector
from neo4j_graphrag.types import SearchType
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
import re
from functools import lru_cache

load_dotenv(".env")

# --- Config ---

embeddings = OllamaEmbeddings(
    base_url=os.getenv("BASE_URL", "http://localhost:11434"),
    model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
)

neo4j_url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

llm = OllamaLLM(
    base_url=os.getenv("BASE_URL", "http://localhost:11434"),
    model=os.getenv("LLM_MODEL", "gpt-oss:20b-cloud"),
    temperature=0.0,
)

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "3"))

decompose_prompt = ChatPromptTemplate.from_template(
    "You are a question analyzer. Your job is to break down complex, multi-part questions "
    "into simple, independent sub-questions that can each be answered with a single retrieval.\n\n"
    "Rules:\n"
    "- If the question is already simple and focuses on a single topic, return it unchanged.\n"
    "- If the question contains multiple distinct parts (joined by 'and', 'or', commas, etc.), "
    "split them into separate sub-questions.\n"
    "- Each sub-question must be self-contained and understandable on its own.\n"
    "- Return ONLY the sub-questions, one per line, with no numbering, bullets, or extra text.\n\n"
    "Question: {question}\n\n"
    "Sub-questions:"
)

answer_prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant that answers questions based strictly on the provided context.\n\n"
    "Rules:\n"
    "- Answer ONLY using information found in the context below.\n"
    "- If the context does not contain enough information to answer, say "
    "\"I don't have enough information to answer this question.\"\n"
    "- Do not make up or infer information beyond what is explicitly stated.\n"
    "- When the question has multiple parts, address each part clearly.\n"
    "- Be concise and direct.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

decompose_chain = decompose_prompt | llm | StrOutputParser()
answer_chain = answer_prompt | llm | StrOutputParser()


@lru_cache(maxsize=16)
def get_vector_store(proj: str) -> Neo4jVector:
    return Neo4jVector(
        embedding=embeddings,
        url=neo4j_url,
        username=neo4j_user,
        password=neo4j_password,
        index_name=f"{proj}_docs",
        node_label=f"Document_{proj}",
        keyword_index_name=f"{proj}_docs_fulltext",
        search_type=SearchType.HYBRID,
    )


# --- State ---

class RAGState(TypedDict):
    question: str
    project: str
    strip_markdown: bool
    sub_questions: list[str]
    docs_with_scores: list[tuple[Any, float]]
    context: str
    answer: str
    sources: list[dict]


# --- Nodes ---

def decompose(state: RAGState) -> dict:
    """Scompone domande multi-hop in sotto-domande indipendenti."""
    raw = decompose_chain.invoke({"question": state["question"]})
    sub_qs = [q.strip() for q in raw.strip().splitlines() if q.strip()]
    if not sub_qs:
        sub_qs = [state["question"]]
    return {"sub_questions": sub_qs}


def retrieve(state: RAGState) -> dict:
    """Recupera documenti per ogni sotto-domanda e deduplica i risultati."""
    vs = get_vector_store(state["project"])
    seen_contents: set[str] = set()
    all_docs: list[tuple[Any, float]] = []

    for sub_q in state["sub_questions"]:
        results = vs.similarity_search_with_relevance_scores(sub_q, k=RETRIEVE_K)
        for doc, score in results:
            if score >= SIMILARITY_THRESHOLD and doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                all_docs.append((doc, score))

    all_docs.sort(key=lambda x: x[1], reverse=True)
    context = "\n\n".join(doc.page_content for doc, _ in all_docs)
    return {"docs_with_scores": all_docs, "context": context}


def generate(state: RAGState) -> dict:
    """Genera la risposta usando l'LLM con il contesto recuperato."""
    answer = answer_chain.invoke({"context": state["context"], "question": state["question"]})
    if state.get("strip_markdown", True):
        answer = re.sub(r"\*\*(.+?)\*\*", r"\1", answer)
        answer = re.sub(r"\*(.+?)\*", r"\1", answer)
    return {"answer": answer}


def format_sources(state: RAGState) -> dict:
    """Estrai riferimenti univoci dai documenti recuperati."""
    seen: set[tuple] = set()
    refs: list[dict] = []
    for doc, score in state["docs_with_scores"]:
        source = doc.metadata.get("source", doc.metadata.get("fileName", "Sconosciuto"))
        raw_page = doc.metadata.get("page", doc.metadata.get("page_number", None))
        page = raw_page + 1 if isinstance(raw_page, int) else "N/A"
        key = (source, page)
        if key not in seen:
            seen.add(key)
            refs.append({"nome_documento": source, "pagina": page, "score": round(score, 4)})
    return {"sources": refs}


# --- Graph ---

graph_builder = StateGraph(RAGState)

graph_builder.add_node("decompose", decompose)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("format_sources", format_sources)

graph_builder.add_edge(START, "decompose")
graph_builder.add_edge("decompose", "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("retrieve", "format_sources")
graph_builder.add_edge("generate", END)
graph_builder.add_edge("format_sources", END)

graph = graph_builder.compile()
app = graph


# --- Helper per compatibilità con rag_qa.py ---

def query_answer(question: str, project: str, strip_markdown: bool = True) -> tuple[str, list[dict]]:
    """Stessa interfaccia di rag_qa.query_answer, ma eseguita tramite LangGraph."""
    result = app.invoke({
        "question": question,
        "project": project,
        "strip_markdown": strip_markdown,
    })
    return result["answer"], result["sources"]
