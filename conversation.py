"""Session conversation versions and their follow-up branches. No Streamlit dependencies."""

from datetime import datetime, timezone
from uuid import uuid4


def new_version(model: str) -> dict:
    return {"id": uuid4().hex, "content": "", "sources": [], "status": "retrieving",
            "model": model, "created_at": datetime.now(timezone.utc).isoformat(),
            "error": "", "following_messages": []}


def ensure_versions(messages: list[dict]):
    """Keep sessions opened before version support usable after a code reload."""
    for message in messages:
        message.setdefault("id", uuid4().hex)
        if message["role"] == "assistant" and "versions" not in message:
            version = new_version("")
            version.update(content=message["content"], sources=message.get("sources", []), status="complete")
            message.update(versions=[version], selected_version=0)


def sync_selected(message: dict):
    version = message["versions"][message["selected_version"]]
    message.update(content=version["content"], sources=version["sources"])


def switch_version(messages: list[dict], index: int, selected: int):
    message = messages[index]
    if selected == message["selected_version"]:
        return
    if not 0 <= selected < len(message["versions"]):
        raise ValueError("Response version does not exist.")
    message["versions"][message["selected_version"]]["following_messages"] = messages[index + 1:]
    message["selected_version"] = selected
    messages[index + 1:] = message["versions"][selected]["following_messages"]
    sync_selected(message)


def add_version(messages: list[dict], index: int, model: str) -> dict:
    message = messages[index]
    message["versions"][message["selected_version"]]["following_messages"] = messages[index + 1:]
    version = new_version(model)
    message["versions"].append(version)
    message["selected_version"] = len(message["versions"]) - 1
    del messages[index + 1:]
    sync_selected(message)
    return version


def history_before(messages: list[dict], index: int) -> list[dict]:
    # The immediately preceding user question is sent separately to the model.
    return [{"role": message["role"], "content": message["content"]}
            for message in messages[:index - 1] if message["content"]]


def export_messages(messages: list[dict]) -> list[dict]:
    """Export every saved branch once; the active branch is the containing list."""
    exported = []
    for message in messages:
        item = {key: value for key, value in message.items() if key != "versions"}
        if message["role"] == "assistant":
            item["versions"] = []
            for index, version in enumerate(message["versions"]):
                copy = {key: value for key, value in version.items() if key != "following_messages"}
                copy["following_messages"] = (export_messages(version["following_messages"])
                                               if index != message["selected_version"] else [])
                item["versions"].append(copy)
        exported.append(item)
    return exported
