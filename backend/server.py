import asyncio
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, List, Dict

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel
from tavily import TavilyClient

from context import prompt
from rag import (
    chunk_text,
    crawl_website,
    embed_in_batches,
    retrieve_context,
    save_rag_index,
)

# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()

app = FastAPI()

origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))

USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "")
MEMORY_DIR = os.getenv("MEMORY_DIR", "../memory")

s3_client = boto3.client("s3") if USE_S3 else None

# Thread pool for blocking I/O (Tavily, OpenAI embeddings) in async endpoints
_executor = ThreadPoolExecutor(max_workers=4)


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    training_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class TrainRequest(BaseModel):
    url: str


# ──────────────────────────────────────────────────────────────────────────────
# Conversation memory helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mem_key(session_id: str) -> str:
    return f"{session_id}.json"


def load_conversation(session_id: str) -> List[Dict]:
    if USE_S3 and s3_client:
        try:
            resp = s3_client.get_object(Bucket=S3_BUCKET, Key=_mem_key(session_id))
            return json.loads(resp["Body"].read().decode("utf-8"))
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "NoSuchKey":
                return []
            raise
    else:
        path = os.path.join(MEMORY_DIR, _mem_key(session_id))
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
        return []


def save_conversation(session_id: str, messages: List[Dict]) -> None:
    if USE_S3 and s3_client:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=_mem_key(session_id),
            Body=json.dumps(messages, indent=2),
            ContentType="application/json",
        )
    else:
        os.makedirs(MEMORY_DIR, exist_ok=True)
        path = os.path.join(MEMORY_DIR, _mem_key(session_id))
        with open(path, "w") as fh:
            json.dump(messages, fh, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Routes — health / root
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "AI Digital Twin API",
        "memory_enabled": True,
        "rag_enabled": True,
        "storage": "S3" if USE_S3 else "local",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "use_s3": USE_S3}


# ──────────────────────────────────────────────────────────────────────────────
# Route — train (SSE streaming)
# ──────────────────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/train")
async def train(request: TrainRequest):
    """
    Crawl a website with Tavily, chunk + embed its content, persist under a
    new UUID training_id.  Streams SSE progress events back to the client.
    """
    async def generate():
        # Use get_running_loop() — correct API for Python 3.10+
        loop = asyncio.get_running_loop()
        training_id = str(uuid.uuid4())

        try:
            from langchain_rag import crawl_and_index
            import queue
            q = queue.Queue()
            
            def sse_cb(payload):
                q.put(payload)
                
            future = loop.run_in_executor(
                _executor,
                lambda: crawl_and_index(request.url, training_id, sse_cb)
            )
            
            while not future.done():
                while not q.empty():
                    yield _sse(q.get())
                yield ": heartbeat\n\n"
                await asyncio.sleep(1)
                
            while not q.empty():
                yield _sse(q.get())
                
            chunks_count, pages_count = future.result()
            
            yield _sse({
                "status": "done",
                "training_id": training_id,
                "chunks_count": chunks_count,
                "pages_count": pages_count,
                "url": request.url,
            })

        except Exception as exc:
            import traceback
            traceback.print_exc()
            yield _sse({"status": "error", "message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Route — chat
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        conversation = load_conversation(session_id)

        # Build system prompt — optionally inject RAG context
        system_content = prompt()

        if request.training_id:
            try:
                from langchain_rag import chat_with_rag
                from langchain_openai import ChatOpenAI
                
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    max_tokens=5000,
                    top_p=0.95,
                    frequency_penalty=1.2,
                    stop_sequences=['INST']
                )
                
                assistant_response = chat_with_rag(request.message, request.training_id, llm)
            except Exception as exc:
                print(f"[chat] RAG retrieval error: {exc}")
                assistant_response = f"Sorry, I encountered the following error: \n {exc}"
        else:
            messages = [{"role": "system", "content": system_content}]
            for msg in conversation[-10:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": request.message})

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            assistant_response = response.choices[0].message.content

        conversation.append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat(),
        })
        conversation.append({
            "role": "assistant",
            "content": assistant_response,
            "timestamp": datetime.now().isoformat(),
        })
        save_conversation(session_id, conversation)

        return ChatResponse(response=assistant_response, session_id=session_id)

    except Exception as exc:
        print(f"[chat] error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    try:
        conversation = load_conversation(session_id)
        return {"session_id": session_id, "messages": conversation}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
