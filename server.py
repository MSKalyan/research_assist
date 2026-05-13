from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import aiofiles
import os
from datetime import datetime

app = FastAPI()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_DIR = Path("templates")
TEMPLATE_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home():
    template_path = TEMPLATE_DIR / "index.html"
    if template_path.exists():
        return FileResponse(template_path)
    return HTMLResponse(content="<h1>Templates not found</h1>", status_code=500)


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File()):
    saved = []
    for f in files:
        if f.filename:
            path = UPLOAD_DIR / f.filename
            content = await f.read()
            async with aiofiles.open(path, "wb") as out:
                await out.write(content)
            saved.append(f.filename)
    return {"status": "success", "files": saved}


@app.post("/reingest")
async def reingest():
    from app.ingestion.loader import load_files
    from app.ingestion.splitter import split_documents
    from app.ingestion.vector_store import delete_collection, add_documents

    delete_collection()
    files = [UPLOAD_DIR / f for f in os.listdir(UPLOAD_DIR) if f.endswith(('.pdf', '.txt', '.md'))]
    if files:
        docs = load_files([str(p) for p in files])
        chunks = split_documents(docs)
        result = add_documents(chunks)
        return {"status": "success", "chunks": result.get("chunks_added", 0)}
    return {"status": "success", "chunks": 0}


@app.post("/ask")
async def ask_question(data: dict):
    from graph_service import graph, GraphState
    import time

    question = data.get("question", "")
    thread_id = data.get("thread_id", f"user-{int(time.time())}")

    config = {"configurable": {"thread_id": thread_id}}

    existing_messages = []
    try:
        checkpoint = graph.get_state(config)
        if checkpoint and checkpoint.values:
            existing_messages = list(checkpoint.values.get("messages", []))
    except Exception:
        existing_messages = []

    existing_messages.append({"role": "user", "content": question})

    initial_state = GraphState(question=question, messages=existing_messages)

    try:
        result = graph.invoke(initial_state, config=config)
    except Exception as e:
        return {"error": str(e), "answer": None}

    final_messages = []
    try:
        final_state = graph.get_state(config)
        if final_state and final_state.values:
            final_messages = list(final_state.values.get("messages", []))
    except Exception:
        final_messages = existing_messages.copy()
        if result.get("final_answer"):
            final_messages.append({"role": "assistant", "content": result["final_answer"]})

    contrib = result.get("source_contribution", {})
    answer = result.get("final_answer") or result.get("draft_answer") or "No answer generated"

    return {
        "answer": answer,
        "llm_pct": contrib.get("llm_pct", 0),
        "rag_pct": contrib.get("rag_pct", 0),
        "web_pct": contrib.get("web_pct", 0),
        "thread_id": thread_id,
        "messages": final_messages
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)