"""Version switching preserves follow-ups and the exact history sent to the model."""

import json
import unittest

from conversation import add_version, ensure_versions, export_messages, history_before, switch_version, sync_selected


def turn(question, answer):
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer, "sources": []}]
    ensure_versions(messages)
    return messages


class ConversationTests(unittest.TestCase):
    def test_regeneration_preserves_independent_follow_up_branches(self):
        messages = turn("Budget?", "30,000") + turn("What does it cover?", "Courses")
        original_followup = messages[2:]
        version = add_version(messages, 1, "phi3")
        version.update(content="INR 30,000 [1].", status="complete")
        sync_selected(messages[1])
        messages.extend(turn("Who approves?", "Your manager"))
        switch_version(messages, 1, 0)
        self.assertEqual(messages[2:], original_followup)
        self.assertEqual(messages[1]["content"], "30,000")
        switch_version(messages, 1, 1)
        self.assertEqual(messages[2]["content"], "Who approves?")
        self.assertEqual(messages[1]["content"], "INR 30,000 [1].")
        self.assertEqual(history_before(messages, 3), [{"role": "user", "content": "Budget?"},
                                                      {"role": "assistant", "content": "INR 30,000 [1]."}])
        self.assertEqual(history_before(messages, 1), [])

    def test_export_includes_inactive_versions_sources_and_saved_followups(self):
        messages = turn("Budget?", "Original") + turn("Followup?", "Earlier followup")
        messages[1]["versions"][0]["sources"] = [{"name": "original.txt"}]
        version = add_version(messages, 1, "phi3")
        version.update(content="New partial", status="stopped", sources=[{"name": "new.txt"}])
        sync_selected(messages[1])
        messages.extend(turn("New followup?", "New followup answer"))
        exported = json.loads(json.dumps(export_messages(messages)))
        versions = exported[1]["versions"]
        self.assertEqual(versions[0]["following_messages"][0]["content"], "Followup?")
        self.assertEqual(versions[0]["sources"], [{"name": "original.txt"}])
        self.assertEqual(versions[1]["status"], "stopped")
        self.assertEqual(versions[1]["sources"], [{"name": "new.txt"}])
        self.assertEqual(versions[1]["following_messages"], [])
        self.assertEqual(exported[2]["content"], "New followup?")
        self.assertEqual(messages[1]["versions"][0]["following_messages"][0]["content"], "Followup?")


if __name__ == "__main__":
    unittest.main()
