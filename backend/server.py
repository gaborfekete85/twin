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

USE_S3    = os.getenv("USE_S3", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "")
MEMORY_DIR = os.getenv("MEMORY_DIR", "../memory")

s3_client = boto3.client("s3") if USE_S3 else None

# Thread pool for blocking I/O (Tavily, OpenAI embeddings) in async endpoints
_executor = ThreadPoolExecutor(max_workers=4)

# Detect Lambda runtime — SSE/streaming doesn't work through API Gateway
IS_LAMBDA = bool(os.getenv("AWS_LAMBDA_FUNCTION_NAME"))


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
    return {"status": "healthy", "use_s3": USE_S3, "is_lambda": IS_LAMBDA}


# ──────────────────────────────────────────────────────────────────────────────
# Route — train
# ──────────────────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _run_train(request: TrainRequest) -> dict:
    """Core training logic shared by both sync and SSE paths."""
    loop = asyncio.get_running_loop()
    training_id = str(uuid.uuid4())

    pages = await loop.run_in_executor(
        _executor, lambda: crawl_website(request.url, tavily_client))
    if not pages:
        raise ValueError("Could not crawl the URL. Check it and try again.")

    all_chunks: list[dict] = []
    for page in pages:
        for text in chunk_text(page["content"]):
            all_chunks.append({"text": text, "source_url": page["url"], "embedding": []})

    if not all_chunks:
        raise ValueError("No usable content found on the page.")

    embeddings = await loop.run_in_executor(
        _executor, lambda: embed_in_batches([c["text"] for c in all_chunks], openai_client))
    for i, chunk in enumerate(all_chunks):
        chunk["embedding"] = embeddings[i]

    await loop.run_in_executor(
        _executor, lambda: save_rag_index(
            training_id, request.url, all_chunks, USE_S3, s3_client, S3_BUCKET))

    return {
        "status": "done",
        "training_id": training_id,
        "chunks_count": len(all_chunks),
        "pages_count": len(pages),
        "url": request.url,
    }


@app.post("/train")
async def train(request: TrainRequest):
    """
    Crawl a website, chunk + embed its content, persist under a UUID.
    - Local dev : streams SSE progress events so the UI shows live steps.
    - Lambda    : returns a single JSON response (API Gateway doesn't support SSE).
    """
    if IS_LAMBDA:
        # ── Synchronous JSON response for Lambda / API Gateway ────────────
        try:
            result = await _run_train(request)
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            import traceback; traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(exc))

    # ── SSE streaming for local dev ───────────────────────────────────────
    async def generate():
        loop = asyncio.get_running_loop()
        training_id = str(uuid.uuid4())
        try:
            yield _sse({"status": "crawling", "message": "Crawling website…"})
            pages = await loop.run_in_executor(
                _executor, lambda: crawl_website(request.url, tavily_client))
            if not pages:
                yield _sse({"status": "error", "message": "Could not crawl the URL. Check it and try again."})
                return

            yield _sse({"status": "chunking", "message": f"Processing {len(pages)} page(s)…"})
            all_chunks: list[dict] = []
            for page in pages:
                for text in chunk_text(page["content"]):
                    all_chunks.append({"text": text, "source_url": page["url"], "embedding": []})
            if not all_chunks:
                yield _sse({"status": "error", "message": "No usable content found on the page."})
                return

            yield _sse({"status": "embedding", "message": f"Generating embeddings for {len(all_chunks)} chunks…"})

            embed_future = loop.run_in_executor(
                _executor, lambda: embed_in_batches([c["text"] for c in all_chunks], openai_client))
            while not embed_future.done():
                yield ": heartbeat\n\n"
                await asyncio.sleep(2)

            embeddings = await embed_future
            for i, chunk in enumerate(all_chunks):
                chunk["embedding"] = embeddings[i]

            yield _sse({"status": "saving", "message": "Saving index…"})
            await loop.run_in_executor(
                _executor, lambda: save_rag_index(
                    training_id, request.url, all_chunks, USE_S3, s3_client, S3_BUCKET))

            yield _sse({
                "status": "done",
                "training_id": training_id,
                "chunks_count": len(all_chunks),
                "pages_count": len(pages),
                "url": request.url,
            })

        except Exception as exc:
            import traceback; traceback.print_exc()
            yield _sse({"status": "error", "message": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ──────────────────────────────────────────────────────────────────────────────
# Route — chat
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        conversation = load_conversation(session_id)

        system_content = prompt()

        if request.training_id:
            try:
                rag_chunks = retrieve_context(
                    request.message, request.training_id,
                    openai_client, USE_S3, s3_client, S3_BUCKET, top_k=3)
                if rag_chunks:
                    system_content += (
                        "\n\n## Retrieved Context from Trained Website\n"
                        "Use the following passages to answer questions about the trained website.\n\n"
                        + "\n\n---\n".join(rag_chunks)
                    )
            except Exception as exc:
                print(f"[chat] RAG retrieval error: {exc}")

        messages = [{"role": "system", "content": system_content}]
        for msg in conversation[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": request.message})

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=messages)
        assistant_response = response.choices[0].message.content

        conversation.append({"role": "user", "content": request.message,
                              "timestamp": datetime.now().isoformat()})
        conversation.append({"role": "assistant", "content": assistant_response,
                              "timestamp": datetime.now().isoformat()})
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
