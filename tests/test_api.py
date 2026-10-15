"""Exercise the React API with real local storage and deterministic embeddings."""
import asyncio
from pathlib import Path
from threading import Event
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from api import create_app
from chat_store import ChatStore
from config import Settings
from conversation import ensure_versions, new_version
from ingest import DocumentLibrary
from local_models import ModelStatus
from support import temporary_database_directory
from test_system import TestEmbeddings


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(temporary_database_directory())
        self.settings = Settings(data_dir=Path(self.directory))
        self.library = DocumentLibrary(self.settings, embeddings=TestEmbeddings())
        self.first = self.library.ingest("policy.txt", b"The annual learning budget is INR 30,000.")
        self.other = self.library.ingest("travel.txt", b"Travel expenses require manager approval.")
        self.calls = []
        self.block = False
        self.fail = False
        self.started, self.closed = Event(), Event()
        ready = ModelStatus(True, (self.settings.chat_model, "phi3", self.settings.embedding_model))
        self.enterContext(patch("api.inspect_ollama", return_value=ready))
        self.enterContext(patch("generation.astream_answer", side_effect=self.answer))
        self.app = create_app(self.settings, self.library)
        self.client = self.enterContext(TestClient(self.app))

    async def answer(self, settings, question, docs, history):
        self.calls.append((settings.chat_model, question, history))
        try:
            yield f"Answer {len(self.calls)}: INR 30,000 [1]."
            self.started.set()
            if self.fail:
                raise ConnectionError("Model connection interrupted")
            if self.block:
                await asyncio.Event().wait()
        finally:
            self.closed.set()

    def chat(self):
        response = self.client.post("/api/chats")
        self.assertEqual(response.status_code, 200)
        return response.json()

    def ask(self, chat_id, question="Budget?", **values):
        response = self.client.post(f"/api/chats/{chat_id}/messages", json={"content": question, **values})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def settle(self, chat_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            chat = self.client.get(f"/api/chats/{chat_id}").json()
            if not chat["active"]:
                return chat
            time.sleep(.01)
        self.fail("Response did not finish")

    def test_send_rename_export_delete_and_reload(self):
        chat = self.chat()
        self.ask(chat["id"], document_ids=[self.first.document_id])
        saved = self.settle(chat["id"])
        self.assertEqual(saved["title"], "Budget?")
        version = saved["messages"][-1]["versions"][0]
        self.assertEqual(version["status"], "complete")
        self.assertEqual(version["sources"][0]["name"], "policy.txt")
        self.assertEqual(ChatStore(Path(self.directory)).get(chat["id"])["messages"], saved["messages"])
        renamed = self.client.patch(f"/api/chats/{chat['id']}", json={"title": "Learning policy"}).json()
        self.assertEqual(renamed["title"], "Learning policy")
        exported = self.client.get(f"/api/chats/{chat['id']}/export").json()
        self.assertEqual(exported["schema_version"], 2)
        self.assertEqual(exported["messages"][-1]["versions"][0]["content"], version["content"])
        self.assertEqual(self.client.delete(f"/api/chats/{chat['id']}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/chats/{chat['id']}").status_code, 404)

    def test_stop_and_retry_preserves_partial_version(self):
        self.block = True
        chat = self.chat()
        self.ask(chat["id"])
        self.assertTrue(self.started.wait(2))
        stopped = self.client.post(f"/api/chats/{chat['id']}/stop").json()
        self.assertFalse(stopped["active"])
        first = stopped["messages"][-1]
        self.assertEqual(first["versions"][0]["status"], "stopped")
        self.assertIn("INR 30,000", first["content"])
        self.assertTrue(self.closed.wait(2))
        self.block = False
        retry = self.client.post(f"/api/chats/{chat['id']}/responses/{first['id']}/regenerate", json={"model": "phi3"})
        self.assertEqual(retry.status_code, 200)
        completed = self.settle(chat["id"])["messages"][-1]
        self.assertEqual(len(completed["versions"]), 2)
        self.assertEqual(completed["versions"][0]["status"], "stopped")
        self.assertEqual(completed["versions"][1]["model"], "phi3")
        self.assertEqual(completed["versions"][1]["status"], "complete")

    def test_regenerate_preserves_scope_and_follow_up_branches(self):
        chat = self.chat()
        self.ask(chat["id"], document_ids=[self.first.document_id], top_k=2)
        first = self.settle(chat["id"])["messages"][-1]
        self.ask(chat["id"], "Follow up?")
        self.settle(chat["id"])
        path = f"/api/chats/{chat['id']}/responses/{first['id']}"
        response = self.client.post(path + "/regenerate", json={"model": "phi3"})
        self.assertEqual(response.status_code, 200)
        revised = self.settle(chat["id"])
        self.assertEqual(len(revised["messages"]), 2)
        self.assertEqual(self.calls[-1], ("phi3", "Budget?", []))
        current = revised["messages"][-1]["versions"][1]
        self.assertEqual(current["document_ids"], [self.first.document_id])
        self.assertEqual(current["top_k"], 2)
        restored = self.client.post(path + "/version", json={"index": 0}).json()
        self.assertEqual(len(restored["messages"]), 4)
        self.assertEqual(restored["messages"][-2]["content"], "Follow up?")
        self.assertEqual(self.client.post(path + "/version", json={"index": 99}).status_code, 422)
        self.assertEqual(len(self.client.post(path + "/version", json={"index": 1}).json()["messages"]), 2)

    def test_upload_validation_and_delete_only_affected_chats(self):
        self.assertEqual(self.client.post("/api/documents", files={"file": ("photo.png", b"bad")}).status_code, 422)
        uploaded = self.client.post("/api/documents", files={"file": ("note.txt", b"Meeting time is 10 AM.")})
        self.assertEqual(uploaded.status_code, 200)
        self.assertTrue(any(doc["name"] == "note.txt" for doc in self.client.get("/api/documents").json()))
        affected, unaffected = self.chat(), self.chat()
        self.ask(affected["id"], document_ids=[self.first.document_id]); self.settle(affected["id"])
        self.ask(unaffected["id"], document_ids=[self.other.document_id]); self.settle(unaffected["id"])
        result = self.client.delete(f"/api/documents/{self.first.document_id}").json()
        self.assertEqual(result["deleted_chat_ids"], [affected["id"]])
        self.assertEqual(self.client.get(f"/api/chats/{affected['id']}").status_code, 404)
        self.assertEqual(self.client.get(f"/api/chats/{unaffected['id']}").status_code, 200)

    def test_busy_mutations_rejected_and_delete_chat_cancels_worker(self):
        self.block = True
        chat = self.chat()
        self.ask(chat["id"])
        self.assertTrue(self.started.wait(2))
        self.assertEqual(self.client.post(f"/api/chats/{chat['id']}/messages", json={"content": "Another?"}).status_code, 409)
        self.assertEqual(self.client.delete(f"/api/documents/{self.first.document_id}").status_code, 409)
        self.assertEqual(self.client.post("/api/documents", files={"file": ("note.txt", b"Note")}).status_code, 409)
        self.assertEqual(self.client.delete(f"/api/chats/{chat['id']}").status_code, 200)
        self.assertTrue(self.closed.wait(2))
        self.assertEqual(self.client.get("/api/chats").json(), [])

    def test_failed_response_saved_and_local_origin_enforced(self):
        self.fail = True
        chat = self.chat()
        self.ask(chat["id"])
        version = self.settle(chat["id"])["messages"][-1]["versions"][0]
        self.assertEqual(version["status"], "failed")
        self.assertIn("interrupted", version["error"])
        self.assertTrue(version["content"])
        self.assertEqual(self.client.post("/api/chats", headers={"Origin": "https://example.com"}).status_code, 403)
        self.assertEqual(self.client.get("/api/health", headers={"Host": "example.com"}).status_code, 400)
        self.assertEqual(self.client.get("/api/health").json(), {"status": "ok"})

    def test_invalid_model_and_scope_are_not_silently_used(self):
        chat = self.chat()
        self.assertEqual(self.client.post(f"/api/chats/{chat['id']}/messages", json={"content": "Budget?", "model": "test:cloud"}).status_code, 422)
        self.assertEqual(self.client.post(f"/api/chats/{chat['id']}/messages", json={"content": "Budget?", "model": "not-installed"}).status_code, 409)
        self.assertEqual(self.client.post(f"/api/chats/{chat['id']}/messages", json={"content": "Budget?", "document_ids": ["missing"]}).status_code, 422)
        self.assertEqual(self.client.get(f"/api/chats/{chat['id']}").json()["messages"], [])


class ChatRecoveryTests(unittest.TestCase):
    def test_interrupted_versions_are_restored_as_stopped(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStore(Path(directory))
            chat = store.create()
            version = new_version("phi3")
            version.update(content="Partial text", status="generating")
            chat["messages"] = [{"role": "user", "content": "Budget?"},
                                {"role": "assistant", "content": "Partial text", "sources": [], "versions": [version], "selected_version": 0}]
            ensure_versions(chat["messages"]); store.save(chat)
            restored = ChatStore(Path(directory)).get(chat["id"])
            self.assertEqual(restored["messages"][-1]["versions"][0]["status"], "stopped")
            self.assertEqual(restored["messages"][-1]["content"], "Partial text")


if __name__ == "__main__":
    unittest.main()
