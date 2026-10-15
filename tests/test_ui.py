"""Exercise Streamlit's offline startup and document UI without downloading AI models."""

from pathlib import Path
import asyncio
from contextlib import contextmanager
import os
import tempfile
import time
from threading import Event
import unittest
from unittest.mock import patch

from config import Settings
from ingest import DocumentLibrary
from local_models import ModelStatus
from streamlit.testing.v1 import AppTest
from test_system import TestEmbeddings
from support import temporary_database_directory

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def settle(app):
    deadline = time.monotonic() + 10
    while "generation" in app.session_state:
        if time.monotonic() > deadline:
            raise AssertionError("Background response did not finish.")
        time.sleep(0.01)
        app.run()
    return app


@contextmanager
def ready_app(answer_stream):
    with temporary_database_directory() as directory:
        settings = Settings(data_dir=Path(directory))
        library = DocumentLibrary(settings, embeddings=TestEmbeddings())
        library.ingest("handbook.txt", b"The annual budget is INR 30,000.")
        ready = ModelStatus(True, (settings.chat_model, settings.embedding_model))
        with patch.dict(os.environ, {"DATA_DIR": directory, "LLM_MODEL": settings.chat_model,
                                     "EMBEDDING_MODEL": settings.embedding_model}), \
             patch("local_models.inspect_ollama", return_value=ready), \
             patch("ingest.DocumentLibrary", return_value=library), \
             patch("generation.astream_answer", side_effect=answer_stream):
            app = AppTest.from_file(APP_PATH, default_timeout=60).run()
            try:
                yield app
            finally:
                if "generation" in app.session_state:
                    job = app.session_state["generation"]
                    job.cancel()
                    job.finished.wait(2)


class InterfaceTests(unittest.TestCase):
    def test_offline_empty_library_and_new_chat(self):
        with temporary_database_directory() as directory:
            with patch.dict(os.environ, {"DATA_DIR": directory}), \
                 patch("local_models.inspect_ollama", return_value=ModelStatus(False)):
                app = AppTest.from_file(APP_PATH, default_timeout=60).run()
                self.assertFalse(list(app.exception))
                self.assertTrue(app.chat_input[0].disabled)
                self.assertEqual(app.metric[0].value, "0")
                next(button for button in app.button if button.label == "New chat").click().run()
                self.assertFalse(list(app.exception))

    def test_library_scope_and_remove(self):
        with temporary_database_directory() as directory:
            settings = Settings(data_dir=Path(directory))
            library = DocumentLibrary(settings, embeddings=TestEmbeddings())
            result = library.ingest("handbook.txt", b"The annual budget is 30,000.")
            with patch.dict(os.environ, {"DATA_DIR": directory}), \
                 patch("local_models.inspect_ollama", return_value=ModelStatus(False)):
                app = AppTest.from_file(APP_PATH, default_timeout=60).run()
                self.assertFalse(list(app.exception))
                self.assertEqual(app.metric[0].value, "1")
                scope = next(box for box in app.selectbox if box.label == "Search in")
                scope.select(result.document_id).run()
                self.assertFalse(list(app.exception))
                next(button for button in app.button if button.label == "Remove").click().run()
                self.assertFalse(list(app.exception))
                self.assertEqual(app.metric[0].value, "0")

    def test_chat_stream_and_source_excerpts(self):
        async def answer(*args):
            yield "The budget is INR "
            yield "30,000 [1]."

        with ready_app(answer) as app:
            self.assertFalse(list(app.exception))
            self.assertFalse(app.chat_input[0].disabled)
            settle(app.chat_input[0].set_value("What is the annual budget?").run())
            self.assertFalse(list(app.exception))
            self.assertTrue(any("30,000 [1]" in message.value for message in app.markdown))
            self.assertEqual(app.session_state["messages"][-1]["sources"][0]["name"], "handbook.txt")

    def test_stop_partial_response_and_retry(self):
        started, closed = Event(), Event()
        calls = 0

        async def answer(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                try:
                    yield "A partial answer"
                    started.set()
                    await asyncio.Event().wait()
                finally:
                    closed.set()
            else:
                yield "The complete budget answer [1]."

        with ready_app(answer) as app:
            app.chat_input[0].set_value("Budget?").run()
            self.assertTrue(started.wait(2))
            app.run()
            self.assertTrue(app.chat_input[0].disabled)
            next(button for button in app.button if button.label == "Stop response").click().run()
            self.assertTrue(closed.wait(2), "Stopping must close the running answer stream.")
            message = app.session_state["messages"][-1]
            self.assertEqual(message["content"], "A partial answer")
            self.assertEqual(message["versions"][0]["status"], "stopped")
            self.assertFalse(app.chat_input[0].disabled)
            settle(next(button for button in app.button if button.label == "Retry response").click().run())
            self.assertFalse(list(app.exception))
            self.assertEqual(len(message["versions"]), 2)
            self.assertEqual(message["content"], "The complete budget answer [1].")
            next(box for box in app.selectbox if box.label == "Response version").select(0).run()
            self.assertEqual(message["content"], "A partial answer")

    def test_failed_response_can_retry_and_retains_error_version(self):
        calls = 0

        async def answer(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield "Partial before failure"
                raise ConnectionError("Connection interrupted")
            yield "Recovered answer [1]."

        with ready_app(answer) as app:
            settle(app.chat_input[0].set_value("Budget?").run())
            self.assertFalse(list(app.exception))
            message = app.session_state["messages"][-1]
            self.assertEqual(message["versions"][0]["status"], "failed")
            settle(next(button for button in app.button if button.label == "Retry response").click().run())
            self.assertEqual(message["versions"][0]["content"], "Partial before failure")
            self.assertEqual(message["versions"][0]["error"], "Connection interrupted")
            self.assertEqual(message["content"], "Recovered answer [1].")

    def test_regenerate_switches_follow_up_branch_and_restores_it(self):
        calls = []

        async def answer(settings, question, docs, history):
            calls.append((question, history))
            yield f"Answer {len(calls)} [1]."

        with ready_app(answer) as app:
            settle(app.chat_input[0].set_value("Budget?").run())
            settle(app.chat_input[0].set_value("What does it cover?").run())
            self.assertEqual(len(app.session_state["messages"]), 4)
            settle(next(button for button in app.button if button.label == "Regenerate").click().run())
            self.assertFalse(list(app.exception))
            self.assertEqual(len(app.session_state["messages"]), 2)
            self.assertEqual(calls[-1], ("Budget?", []), "Regeneration must exclude the old answer and its follow-ups.")
            next(box for box in app.selectbox if box.label == "Response version").select(0).run()
            self.assertEqual(len(app.session_state["messages"]), 4)
            self.assertEqual(app.session_state["messages"][-1]["content"], "Answer 2 [1].")
            next(box for box in app.selectbox if box.label == "Response version").select(1).run()
            self.assertEqual(len(app.session_state["messages"]), 2)
            self.assertEqual(app.session_state["messages"][-1]["content"], "Answer 3 [1].")

    def test_new_chat_cancels_pending_response(self):
        started, closed = Event(), Event()

        async def answer(*args):
            try:
                started.set()
                await asyncio.Event().wait()
                yield "Should never appear"
            finally:
                closed.set()

        with ready_app(answer) as app:
            app.chat_input[0].set_value("Budget?").run()
            self.assertTrue(started.wait(2))
            next(button for button in app.button if button.label == "New chat").click().run()
            self.assertTrue(closed.wait(2))
            self.assertEqual(app.session_state["messages"], [])
            self.assertFalse(list(app.exception))


if __name__ == "__main__":
    unittest.main()
