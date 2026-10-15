"""Local React application API. Existing parsers, Chroma, and Ollama stay local."""

from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from pathlib import Path
from threading import RLock
import time

from config import Settings
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from chat_store import ChatStore
from conversation import add_version, ensure_versions, export_messages, history_before, new_version, sync_selected, switch_version
from generation import ACTIVE, GenerationJob
from ingest import DocumentLibrary
from local_models import inspect_ollama, normalized_model


class MessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    model: str | None = None
    document_ids: list[str] | None = Field(default=None, max_length=100)
    top_k: int = Field(default=3, ge=1, le=6)


class ModelRequest(BaseModel):
    model: str | None = None


class VersionRequest(BaseModel):
    index: int = Field(ge=0)


class TitleRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class Workspace:
    def __init__(self, settings, library=None):
        self.settings = settings
        self.library = library or DocumentLibrary(settings)
        self.store = ChatStore(settings.data_dir)
        self.lock = RLock()
        self.jobs = {}
        self.writing = False
        self._status = None
        self._status_time = 0

    def status(self, refresh=False):
        if refresh or self._status is None or time.monotonic() - self._status_time > 5:
            self._status = inspect_ollama(self.settings)
            self._status_time = time.monotonic()
        return self._status

    def get(self, chat_id):
        try:
            return self.store.get(chat_id)
        except KeyError:
            raise HTTPException(404, "Conversation not found.")

    def refresh(self, chat_id):
        chat = self.get(chat_id)
        job = self.jobs.get(chat_id)
        if job:
            state = job.snapshot()
            for message in chat["messages"]:
                if message["id"] == job.message_id:
                    version = next((v for v in message["versions"] if v["id"] == job.version_id), None)
                    if version:
                        if any(version.get(key) != value for key, value in state.items()):
                            version.update(state)
                            sync_selected(message)
                            self.store.save(chat)
                        break
            if state["status"] not in ACTIVE:
                self.jobs.pop(chat_id, None)
        chat["active"] = chat_id in self.jobs
        return chat

    def refresh_all(self):
        for chat_id in list(self.jobs):
            self.refresh(chat_id)

    def idle(self):
        self.refresh_all()
        if self.writing or self.jobs:
            raise HTTPException(409, "Wait for indexing or stop the current response before continuing.")

    def model_settings(self, model):
        try:
            settings = replace(self.settings, chat_model=model or self.settings.chat_model)
        except ValueError as error:
            raise HTTPException(422, str(error))
        status = self.status()
        if not status.online:
            raise HTTPException(503, "Ollama is offline. Start the local Ollama service and try again.")
        if not status.has(settings.chat_model) or not status.has(settings.embedding_model):
            raise HTTPException(409, "Download the selected chat model and embedding model in Ollama first.")
        return settings

    def start(self, chat, index, settings):
        message = chat["messages"][index]
        version = message["versions"][message["selected_version"]]
        request = message["request"]
        version.update(document_ids=request["document_ids"], top_k=request["top_k"])
        self.store.save(chat)
        job = GenerationJob(message["id"], version["id"])
        self.jobs[chat["id"]] = job
        job.start(self.library, settings, chat["messages"][index - 1]["content"], history_before(chat["messages"], index),
                  request["document_ids"], request["top_k"])
        return self.refresh(chat["id"])

    def stop(self, chat_id):
        if job := self.jobs.get(chat_id):
            job.cancel()
        return self.refresh(chat_id)

    def close(self):
        with self.lock:
            for chat_id in list(self.jobs):
                self.stop(chat_id)


def create_app(settings=None, library=None, frontend_dir=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.workspace = Workspace(settings or Settings.from_env(), library)
        try:
            yield
        finally:
            app.state.workspace.close()

    app = FastAPI(title="Local Chat", lifespan=lifespan, docs_url="/api/docs", redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        origin = request.headers.get("origin")
        same_origin = str(request.base_url).rstrip("/")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin and origin not in {
            same_origin, "http://127.0.0.1:5173", "http://localhost:5173"
        }:
            return JSONResponse({"detail": "Only the local application can make changes."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def workspace():
        return app.state.workspace

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/status")
    def status(refresh: bool = False):
        ws = workspace()
        state = ws.status(refresh)
        models = [ws.settings.chat_model]
        for model in state.models:
            if "embed" not in model.lower() and normalized_model(model) not in {normalized_model(name) for name in models}:
                models.append(model)
        return {"online": state.online, "models": models, "installed_models": state.models,
                "default_model": ws.settings.chat_model, "embedding_model": ws.settings.embedding_model,
                "embedding_ready": state.online and state.has(ws.settings.embedding_model),
                "endpoint": ws.settings.ollama_url, "error": state.error}

    @app.get("/api/documents")
    def documents():
        return workspace().library.list_documents()

    @app.post("/api/documents")
    async def upload(file: UploadFile = File(...)):
        ws = workspace()
        with ws.lock:
            ws.idle()
            ws.writing = True
        try:
            content = await file.read(ws.settings.max_upload_bytes + 1)
            result = await run_in_threadpool(ws.library.ingest, file.filename or "document.txt", content)
            return asdict(result)
        except ValueError as error:
            raise HTTPException(422, str(error))
        except Exception:
            raise HTTPException(503, "Could not index the document. Check the local embedding model and try again.")
        finally:
            await file.close()
            with ws.lock:
                ws.writing = False

    @app.delete("/api/documents/{document_id}")
    def delete_document(document_id: str):
        ws = workspace()
        with ws.lock:
            ws.idle()
            ws.library.delete(document_id)
            deleted = ws.store.remove_document_chats(document_id)
        return {"deleted_chat_ids": deleted}

    @app.get("/api/chats")
    def chats():
        ws = workspace()
        with ws.lock:
            ws.refresh_all()
            return [{**chat, "active": chat["id"] in ws.jobs} for chat in ws.store.list()]

    @app.post("/api/chats")
    def create_chat():
        ws = workspace()
        with ws.lock:
            return {**ws.store.create(), "active": False}

    @app.get("/api/chats/{chat_id}")
    def get_chat(chat_id: str):
        ws = workspace()
        with ws.lock:
            return ws.refresh(chat_id)

    @app.patch("/api/chats/{chat_id}")
    def rename_chat(chat_id: str, body: TitleRequest):
        ws = workspace()
        with ws.lock:
            chat = ws.refresh(chat_id)
            if not body.title.strip():
                raise HTTPException(422, "Enter a conversation name.")
            chat["title"] = body.title.strip()
            ws.store.save(chat)
            return chat

    @app.delete("/api/chats/{chat_id}")
    def delete_chat(chat_id: str):
        ws = workspace()
        with ws.lock:
            ws.stop(chat_id)
            ws.store.delete(chat_id)
        return {"deleted": True}

    @app.post("/api/chats/{chat_id}/messages")
    def send_message(chat_id: str, body: MessageRequest):
        ws = workspace()
        settings = ws.model_settings(body.model)
        with ws.lock:
            ws.idle()
            if not body.content.strip():
                raise HTTPException(422, "Enter a question.")
            available = {doc["id"] for doc in ws.library.list_documents()}
            if not available:
                raise HTTPException(409, "Add a document to start chatting.")
            if body.document_ids is not None and (not body.document_ids or not set(body.document_ids) <= available):
                raise HTTPException(422, "The selected document is no longer available.")
            chat = ws.get(chat_id)
            version = new_version(settings.chat_model)
            chat["messages"].extend([
                {"role": "user", "content": body.content.strip()},
                {"id": version["id"], "role": "assistant", "content": "", "sources": [],
                 "versions": [version], "selected_version": 0,
                 "request": {"document_ids": body.document_ids, "top_k": body.top_k}},
            ])
            ensure_versions(chat["messages"])
            if chat["title"] == "New chat":
                chat["title"] = " ".join(body.content.split())[:70]
            return ws.start(chat, len(chat["messages"]) - 1, settings)

    def response_index(chat, message_id):
        for index, message in enumerate(chat["messages"]):
            if message["id"] == message_id and message["role"] == "assistant":
                return index
        raise HTTPException(404, "Response not found.")

    @app.post("/api/chats/{chat_id}/responses/{message_id}/regenerate")
    def regenerate(chat_id: str, message_id: str, body: ModelRequest):
        ws = workspace()
        settings = ws.model_settings(body.model)
        with ws.lock:
            ws.idle()
            chat = ws.get(chat_id)
            index = response_index(chat, message_id)
            add_version(chat["messages"], index, settings.chat_model)
            return ws.start(chat, index, settings)

    @app.post("/api/chats/{chat_id}/responses/{message_id}/version")
    def choose_version(chat_id: str, message_id: str, body: VersionRequest):
        ws = workspace()
        with ws.lock:
            ws.idle()
            chat = ws.get(chat_id)
            index = response_index(chat, message_id)
            try:
                switch_version(chat["messages"], index, body.index)
            except ValueError as error:
                raise HTTPException(422, str(error))
            ws.store.save(chat)
            return ws.refresh(chat_id)

    @app.post("/api/chats/{chat_id}/stop")
    def stop(chat_id: str):
        ws = workspace()
        with ws.lock:
            return ws.stop(chat_id)

    @app.get("/api/chats/{chat_id}/export")
    def export(chat_id: str):
        ws = workspace()
        with ws.lock:
            chat = ws.refresh(chat_id)
            return {"schema_version": 2, "title": chat["title"], "messages": export_messages(chat["messages"])}

    frontend = Path(frontend_dir) if frontend_dir else Path(__file__).parent / "frontend" / "dist"
    if (frontend / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

    @app.get("/")
    def index():
        if not (frontend / "index.html").is_file():
            raise HTTPException(503, "Build the React interface: cd frontend, npm install, npm run build.")
        return FileResponse(frontend / "index.html", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
