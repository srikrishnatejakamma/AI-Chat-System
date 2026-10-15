"""SQLite storage for local conversations, versions, and follow-up branches."""

from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from conversation import ensure_versions, sync_selected


def now():
    return datetime.now(timezone.utc).isoformat()


def contains_document(messages, document_id):
    for message in messages:
        for version in message.get("versions", []):
            if any(source.get("document_id") == document_id for source in version["sources"]):
                return True
            if contains_document(version.get("following_messages", []), document_id):
                return True
    return False


class ChatStore:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "conversations.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, messages TEXT NOT NULL)""")
        # An interrupted process cannot resume its old socket/model request.
        for summary in self.list():
            chat = self.get(summary["id"])
            if self._repair(chat["messages"]):
                self.save(chat, touch=False)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def list(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT id, title, created_at, updated_at FROM chats ORDER BY updated_at DESC")]

    def create(self, title="New chat"):
        timestamp = now()
        chat = {"id": uuid4().hex, "title": title, "created_at": timestamp,
                "updated_at": timestamp, "messages": []}
        with self.connect() as db:
            db.execute("INSERT INTO chats VALUES (?, ?, ?, ?, ?)",
                       (chat["id"], title, timestamp, timestamp, "[]"))
        return chat

    def get(self, chat_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
        if row is None:
            raise KeyError(chat_id)
        chat = dict(row)
        chat["messages"] = json.loads(chat["messages"])
        return chat

    def save(self, chat, *, touch=True):
        if touch:
            chat["updated_at"] = now()
        with self.connect() as db:
            changed = db.execute("UPDATE chats SET title=?, updated_at=?, messages=? WHERE id=?",
                                 (chat["title"], chat["updated_at"], json.dumps(chat["messages"], ensure_ascii=False), chat["id"]))
            if changed.rowcount != 1:
                raise KeyError(chat["id"])

    def delete(self, chat_id):
        with self.connect() as db:
            changed = db.execute("DELETE FROM chats WHERE id=?", (chat_id,))
            if changed.rowcount != 1:
                raise KeyError(chat_id)

    def remove_document_chats(self, document_id):
        deleted = []
        for summary in self.list():
            chat = self.get(summary["id"])
            if contains_document(chat["messages"], document_id):
                self.delete(chat["id"])
                deleted.append(chat["id"])
        return deleted

    @classmethod
    def _repair(cls, messages):
        changed = False
        ensure_versions(messages)
        for message in messages:
            if message["role"] != "assistant":
                continue
            for version in message["versions"]:
                if version["status"] in {"retrieving", "generating"}:
                    version.update(status="stopped", error="Generation was interrupted by an app restart.")
                    changed = True
                changed = cls._repair(version.get("following_messages", [])) or changed
            sync_selected(message)
        return changed
