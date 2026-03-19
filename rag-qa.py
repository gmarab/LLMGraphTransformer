from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import time
import uuid

load_dotenv(".env")

app = FastAPI(title="RAG Q&A Service")

# --- OpenAI-compatible request/response models ---

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "rag-qa"
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int | None = None

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage

# --- Init vector store & QA chain ---

embeddings = OllamaEmbeddings(
    base_url=os.getenv("BASE_URL", "http://localhost:11434"),
    model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
)

vector_store = Neo4jVector(
    embedding=embeddings,
    url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    username=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "password"),
    index_name="novapulse_docs",
    node_label="Document",
)

llm = OllamaLLM(
    base_url=os.getenv("BASE_URL", "http://localhost:11434"),
    model=os.getenv("LLM_MODEL", "gpt-oss:20b-cloud"),
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template(
    "Use the following context to answer the question.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_references(docs):
    """Estrai riferimenti univoci (documento, pagina) dai documenti recuperati."""
    seen = set()
    refs = []
    for doc in docs:
        source = doc.metadata.get("source", doc.metadata.get("fileName", "Sconosciuto"))
        page = doc.metadata.get("page", doc.metadata.get("page_number", "N/A"))
        key = (source, page)
        if key not in seen:
            seen.add(key)
            refs.append(f"- {source}, pag. {page}")
    return "\n".join(refs)

answer_chain = prompt | llm | StrOutputParser()

# --- Endpoint ---

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    # Estrai l'ultimo messaggio utente
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="Nessun messaggio 'user' nella richiesta")

    question = user_messages[-1].content

    try:
        docs = retriever.invoke(question)
        context = format_docs(docs)
        answer = answer_chain.invoke({"context": context, "question": question})
        references = format_references(docs)
        if references:
            answer = f"{answer}\n\nFonti:\n{references}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    prompt_tokens = sum(len(m.content.split()) for m in request.messages)
    completion_tokens = len(answer.split())

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=answer),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
