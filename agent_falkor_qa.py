from typing import Any, TypedDict

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_falkordb.vectorstores import (
    FalkorDBVector, SearchType, _get_search_index_query,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
import re
from functools import lru_cache

load_dotenv(".env")


# --- Monkey-patch: fix langchain-falkordb bug con ID non-integer ---
# similarity_search_with_score_by_vector fa int(id) sugli ID metadata,
# ma gli ID generati da from_texts() sono stringhe hex (es. "c0cb551b...").
# Questo patch wrappa il metodo originale e, in caso di ValueError,
# ricostruisce i Document senza forzare int().

_orig_search_by_vector = FalkorDBVector.similarity_search_with_score_by_vector


def _patched_search_by_vector(self, embedding, k=4, **kwargs):  # type: ignore
    try:
        return _orig_search_by_vector(self, embedding, k=k, **kwargs)
    except ValueError as e:
        if "invalid literal for int()" not in str(e):
            raise

    # Riesegui la query e costruisci i Document senza int(id)
    filter_params = kwargs.get("filter", {}) or {}
    params = kwargs.get("params", {})
    query_text = kwargs.get("query", "")

    index_query = _get_search_index_query(self.search_type, self._index_type)
    retrieval_query = self.retrieval_query if self.retrieval_query else (
        f"RETURN node.{self.text_node_property} AS text, score, "
        f"{{text: node.{self.text_node_property}, "
        f"id: node.id, source: node.source}} AS metadata"
    )
    read_query = index_query + retrieval_query
    parameters = {
        "entity_property": self.embedding_node_property,
        "k": k,
        "embedding": embedding,
        "query": query_text,
        "entity_label": self.node_label,
        **params,
        **filter_params,
    }
    results = self._query(read_query, params=parameters)
    if not results:
        return []

    docs = []
    for result in results:
        metadata = {
            mk: mv for mk, mv in result[2].items()
            if mk != "text" and mv is not None
        }
        docs.append((
            Document(
                page_content=result[0],
                metadata=metadata,
                id=result[2].get("id"),
            ),
            result[1],
        ))
    return docs


FalkorDBVector.similarity_search_with_score_by_vector = _patched_search_by_vector  # type: ignore[assignment]

# --- Config ---

embeddings = OllamaEmbeddings(
    base_url=os.getenv("BASE_URL", "http://localhost:11434"),
    model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
)

falkordb_host = os.getenv("FALKORDB_HOST", "localhost")
falkordb_port = int(os.getenv("FALKORDB_PORT", "6379"))
falkordb_username = os.getenv("FALKORDB_USERNAME", None)
falkordb_password = os.getenv("FALKORDB_PASSWORD", None)

llm = OllamaLLM(
    base_url=os.getenv("BASE_URL", "http://localhost:11434"),
    model=os.getenv("LLM_MODEL", "gpt-oss:20b-cloud"),
    temperature=0.0,
)

SIMILARITY_THRESHOLD = float(os.getenv("FALKORDB_SIMILARITY_THRESHOLD", "0.40"))
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

sub_answer_prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant that answers questions based strictly on the provided context.\n\n"
    "Rules:\n"
    "- Answer ONLY using information found in the context below.\n"
    "- Do not make up or infer information beyond what is explicitly stated.\n"
    "- Be concise and direct.\n"
    "- If the question ends with '?', your answer MUST begin with exactly one of these prefixes:\n"
    "  'YES, ' if the answer is affirmative\n"
    "  'NO, ' if the answer is negative\n"
    "  '??, ' if the context does not contain enough information to answer\n"
    "- After the prefix, provide a brief explanation.\n\n"
    "Context:\n{context}\n\n"
    "Question: {sub_question}\n\n"
    "Answer:"
)

decompose_chain = decompose_prompt | llm | StrOutputParser()
sub_answer_chain = sub_answer_prompt | llm | StrOutputParser()


@lru_cache(maxsize=16)
def get_vector_store(proj: str) -> FalkorDBVector:
    node_label = f"Document_{proj}"
    return FalkorDBVector.from_existing_index(
        embedding=embeddings,
        host=falkordb_host,
        port=falkordb_port,
        username=falkordb_username,
        password=falkordb_password,
        database=f"{proj}_docs",
        node_label=node_label,
        search_type=SearchType.HYBRID,
        retrieval_query=(
            f"RETURN node.text AS text, score, "
            f"{{text: node.text, id: node.id, source: node.source, page: node.page}} AS metadata"
        ),
    )


# --- State ---

class RAGState(TypedDict):
    question: str
    project: str
    strip_markdown: bool
    sub_questions: list[str]
    docs_with_scores: list[tuple[Any, float]]
    context: str
    sub_answers: list[str]
    yes_no: str
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


def generate_sub_answers(state: RAGState) -> dict:
    """Genera una risposta per ogni sotto-domanda con prefisso YES/NO/??."""
    context = state["context"]
    sub_answers = []
    for sub_q in state["sub_questions"]:
        answer = sub_answer_chain.invoke({"context": context, "sub_question": sub_q})
        sub_answers.append(answer)
    return {"sub_answers": sub_answers}


def aggregate(state: RAGState) -> dict:
    """Aggrega le sotto-risposte: yes_no separato, italic per i NO, strip prefissi."""
    strip_md = state.get("strip_markdown", True)
    parts = []
    all_si = True

    for sa in state["sub_answers"]:
        if sa.startswith("YES, "):
            content = sa[5:]
            is_negative = False
        elif sa.startswith("NO, "):
            content = sa[4:]
            is_negative = True
            all_si = False
        elif sa.startswith("??, "):
            content = sa[4:]
            is_negative = True
            all_si = False
        else:
            content = sa
            is_negative = False
            all_si = False

        if strip_md:
            content = re.sub(r"\*\*(.+?)\*\*", r"\1", content)
            content = re.sub(r"\*(.+?)\*", r"\1", content)

        if is_negative:
            content = f"*{content}*"

        parts.append(content)

    yes_no = "YES" if all_si else "NO"
    answer = "\n\n".join(parts)
    return {"yes_no": yes_no, "answer": answer}


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
graph_builder.add_node("generate_sub_answers", generate_sub_answers)
graph_builder.add_node("aggregate", aggregate)
graph_builder.add_node("format_sources", format_sources)

graph_builder.add_edge(START, "decompose")
graph_builder.add_edge("decompose", "retrieve")
graph_builder.add_edge("retrieve", "generate_sub_answers")
graph_builder.add_edge("retrieve", "format_sources")
graph_builder.add_edge("generate_sub_answers", "aggregate")
graph_builder.add_edge("aggregate", END)
graph_builder.add_edge("format_sources", END)

graph = graph_builder.compile()
app = graph


# --- Helper per compatibilità con rag_qa.py ---

def query_answer(question: str, project: str, strip_markdown: bool = True) -> tuple[str, str, list[dict]]:
    """Stessa interfaccia di rag_qa.query_answer, ma eseguita tramite LangGraph."""
    result = app.invoke({
        "question": question,
        "project": project,
        "strip_markdown": strip_markdown,
    })
    return result["yes_no"], result["answer"], result["sources"]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="RAG QA Agent (LangGraph + FalkorDB)")
    parser.add_argument("--question", help="Question to ask", required=True)
    parser.add_argument("--project", help="Project name for FalkorDB", required=True)
    parser.add_argument("--no-strip-markdown", dest="strip_markdown", action="store_false", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    yes_no, answer, sources = query_answer(args.question, args.project, args.strip_markdown)
    print(f"\n<yes_no>{yes_no}</yes_no>")
    print(f"\nAnswer: {answer}")
    print(f"\nSources:")
    for s in sources:
        print(f"  - {s['nome_documento']} (p. {s['pagina']}, score: {s['score']})")


if __name__ == "__main__":
    main()
