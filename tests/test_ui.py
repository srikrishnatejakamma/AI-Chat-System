"""Exercise Streamlit's offline startup and document UI without downloading AI models."""

from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from config import Settings
from ingest import DocumentLibrary
from local_models import ModelStatus
from streamlit.testing.v1 import AppTest
from test_system import TestEmbeddings
from support import temporary_database_directory

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


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
        with temporary_database_directory() as directory:
            settings = Settings(data_dir=Path(directory))
            library = DocumentLibrary(settings, embeddings=TestEmbeddings())
            library.ingest("handbook.txt", b"The annual budget is INR 30,000.")
            ready = ModelStatus(True, (settings.chat_model, settings.embedding_model))
            with patch.dict(os.environ, {"DATA_DIR": directory, "LLM_MODEL": settings.chat_model,
                                         "EMBEDDING_MODEL": settings.embedding_model}), \
                 patch("local_models.inspect_ollama", return_value=ready), \
                 patch("ingest.DocumentLibrary", return_value=library), \
                 patch("chat.stream_answer", side_effect=lambda *args: iter(["The budget is INR ", "30,000 [1]."])):
                app = AppTest.from_file(APP_PATH, default_timeout=60).run()
                self.assertFalse(list(app.exception))
                self.assertFalse(app.chat_input[0].disabled)
                app.chat_input[0].set_value("What is the annual budget?").run()
                self.assertFalse(list(app.exception))
                self.assertTrue(any("30,000 [1]" in message.value for message in app.markdown))
                self.assertEqual(app.session_state["messages"][-1]["sources"][0]["name"], "handbook.txt")


if __name__ == "__main__":
    unittest.main()
