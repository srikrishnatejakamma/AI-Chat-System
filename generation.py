"""Cancellable background generation. Workers never touch Streamlit session state."""

import asyncio
from contextlib import aclosing
from threading import Event, Lock, Thread
import time

from chat import astream_answer, source_records

ACTIVE = {"retrieving", "generating"}


class GenerationJob:
    def __init__(self, message_id: str, version_id: str):
        self.message_id = message_id
        self.version_id = version_id
        self.cancelled = Event()
        self.finished = Event()
        self._lock = Lock()
        self._state = {"content": "", "sources": [], "status": "retrieving", "error": ""}
        self._last_seen = time.monotonic()

    def snapshot(self) -> dict:
        with self._lock:
            self._last_seen = time.monotonic()
            return dict(self._state)

    def cancel(self):
        with self._lock:
            if self._state["status"] in ACTIVE:
                self._state["status"] = "stopped"
                self.cancelled.set()

    def _update(self, **values):
        with self._lock:
            if not self.cancelled.is_set():
                self._state.update(values)

    def start(self, library, settings, question: str, history: list[dict], document_ids, top_k: int):
        Thread(target=self._run, args=(library, settings, question, history, document_ids, top_k),
               daemon=True, name=f"answer-{self.version_id[:8]}").start()
        return self

    def _run(self, library, settings, question, history, document_ids, top_k):
        try:
            excerpts = library.retrieve(question, document_ids=document_ids, k=top_k)
            if self.cancelled.is_set():
                return
            self._update(sources=source_records(excerpts), status="generating")
            asyncio.run(self._generate(settings, question, excerpts, history))
            if not self.cancelled.is_set():
                if not self.snapshot()["content"]:
                    raise ValueError("The local model returned an empty response. Try again.")
                self._update(status="complete")
        except Exception as error:
            self._update(status="failed", error=str(error))
        finally:
            self.finished.set()

    async def _generate(self, settings, question, excerpts, history):
        async def consume():
            async with aclosing(astream_answer(settings, question, excerpts, history)) as stream:
                async for token in stream:
                    with self._lock:
                        if not self.cancelled.is_set():
                            self._state["content"] += token

        async def watch():
            while not self.cancelled.is_set():
                # A closed browser/session must not leave generation running forever.
                with self._lock:
                    abandoned = time.monotonic() - self._last_seen > 120
                if abandoned:
                    self.cancel()
                    break
                await asyncio.sleep(0.05)
            producer.cancel()

        producer = asyncio.create_task(consume())
        watcher = asyncio.create_task(watch())
        try:
            await producer
        except asyncio.CancelledError:
            if not self.cancelled.is_set():
                raise
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
