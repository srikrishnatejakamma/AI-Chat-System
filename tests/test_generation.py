"""Cancellation must disconnect the actual HTTP stream, not just hide its text."""

import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import select
from threading import Event, Thread
import unittest
from unittest.mock import Mock, patch

from config import Settings
from generation import GenerationJob
from langchain_core.documents import Document


def sample_library():
    return Mock(retrieve=Mock(return_value=[Document(page_content="Budget: 30,000.",
                metadata={"source": "policy.txt", "document_id": "policy"})]))


@contextmanager
def blocked_ollama(partial):
    connected, disconnected = Event(), Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Connection", "close")
            self.end_headers()
            if partial:
                self.wfile.write((json.dumps({"model": "phi3", "message": {"role": "assistant", "content": partial},
                                             "done": False}) + "\n").encode())
            self.wfile.flush()
            connected.set()
            # Detect the client's FIN while the response intentionally remains open.
            readable, _, _ = select.select([self.connection], [], [], 8)
            if readable and self.connection.recv(1) == b"":
                disconnected.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", connected, disconnected
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


class GenerationTests(unittest.TestCase):
    def test_stop_disconnects_http_before_first_token_and_mid_response(self):
        for partial in ("", "Partial answer"):
            with self.subTest(partial=partial), blocked_ollama(partial) as (url, connected, disconnected):
                job = GenerationJob("message", "version").start(
                    sample_library(), Settings(ollama_url=url, chat_model="phi3"), "Budget?", [], None, 3)
                self.assertTrue(connected.wait(10), "Model HTTP request was not received.")
                if partial:
                    import time
                    deadline = time.monotonic() + 3
                    while job.snapshot()["content"] != partial and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertEqual(job.snapshot()["content"], partial)
                job.cancel()
                self.assertTrue(job.finished.wait(3), "Worker did not stop promptly.")
                self.assertTrue(disconnected.wait(3), "Cancellation must close Ollama's HTTP connection.")
                self.assertEqual(job.snapshot()["status"], "stopped")
                self.assertEqual(job.snapshot()["content"], partial)

    def test_cancel_during_retrieval_never_starts_model_or_publishes_late_result(self):
        started, release = Event(), Event()

        def retrieve(*args, **kwargs):
            started.set()
            release.wait(3)
            return sample_library().retrieve()

        with patch("generation.astream_answer") as stream:
            job = GenerationJob("message", "version").start(Mock(retrieve=retrieve), Settings(), "Budget?", [], None, 3)
            self.assertTrue(started.wait(2))
            job.cancel()
            self.assertEqual(job.snapshot()["status"], "stopped")
            release.set()
            self.assertTrue(job.finished.wait(2))
            self.assertEqual(job.snapshot()["sources"], [])
            stream.assert_not_called()

    def test_failure_preserves_partial_content_and_sources(self):
        async def broken(*args):
            yield "Partial"
            raise ConnectionError("Connection lost")

        with patch("generation.astream_answer", side_effect=broken):
            job = GenerationJob("message", "version").start(sample_library(), Settings(), "Budget?", [], ["policy"], 2)
            self.assertTrue(job.finished.wait(2))
            state = job.snapshot()
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["content"], "Partial")
            self.assertEqual(state["error"], "Connection lost")
            self.assertEqual(state["sources"][0]["document_id"], "policy")

    def test_no_context_answer_completes_without_an_http_call(self):
        from chat import NOT_FOUND

        with patch("chat.chat_model", side_effect=AssertionError("Model should not be called")):
            job = GenerationJob("message", "version").start(Mock(retrieve=Mock(return_value=[])), Settings(), "Budget?", [], None, 3)
            self.assertTrue(job.finished.wait(2))
            self.assertEqual(job.snapshot()["status"], "complete")
            self.assertEqual(job.snapshot()["content"], NOT_FOUND)


if __name__ == "__main__":
    unittest.main()
