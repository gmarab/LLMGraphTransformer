from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
import time
import uuid
import argparse

from agent_qa import query_answer
from agent_load import upload_file, upload_folder, upload_path, clean_project

#from agent_falkor_qa import query_answer
#from agent_falkor_load import upload_file, upload_folder, upload_path, clean_project

load_dotenv(".env")

app = FastAPI(title=f"RAG Q&A Service")

# --- OpenAI-compatible request/response models ---

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int | None = None
    strip_markdown: bool = True
    project: str

class Source(BaseModel):
    nome_documento: str
    pagina: int | str
    score: float

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
    yes_no: str
    choices: list[Choice]
    usage: Usage
    sources: list[Source] = []

# --- Endpoint ---

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="Nessun messaggio 'user' nella richiesta")

    question = user_messages[-1].content

    try:
        yes_no, answer, sources = query_answer(question, request.project, request.strip_markdown)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    prompt_tokens = sum(len(m.content.split()) for m in request.messages)
    completion_tokens = len(answer.split())

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        yes_no=yes_no,
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
        sources=[Source(nome_documento=s["nome_documento"], pagina=s["pagina"], score=s["score"]) for s in sources],
    )


@app.post("/v1/upload/file")
async def upload_document(
    file: UploadFile = File(...),
    project: str = Form(default=None),
    clean: bool = Form(default=False),
):
    req_project = project or globals()["project"]
    data = await file.read()

    try:
        result = upload_file(file.filename, data, req_project, clean)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


class UploadPathRequest(BaseModel):
    path: str
    project: str
    clean: bool = True


@app.post("/v1/upload/path")
async def upload_path_endpoint(request: UploadPathRequest):
    try:
        result = upload_path(request.path, request.project, request.clean)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


class UploadFolderRequest(BaseModel):
    folder: str
    project: str
    clean: bool = True


@app.post("/v1/upload/folder")
async def upload_folder_endpoint(request: UploadFolderRequest):
    try:
        result = upload_folder(request.folder, request.project, request.clean)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


class CleanProjectRequest(BaseModel):
    project: str


@app.post("/v1/clean")
async def clean_project_endpoint(request: CleanProjectRequest):
    try:
        result = clean_project(request.project)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
